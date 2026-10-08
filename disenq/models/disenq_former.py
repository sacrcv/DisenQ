"""DisenQ: the Disentangling Querying Transformer (Section 3.2 of the paper).

DisenQ is adapted from the BLIP-2 Q-Former. The only structural change is that
a *single* Q-Former -- one set of self-attention and cross-attention weights --
is driven by **three** independent sets of learnable queries:

    z_b     biometrics       (identity traits: body shape, posture)
    z_m     motion           (action label and movement)
    z_b_hat non-biometrics   (clothing, accessories)

From the paper: "Each query set shares the same self-attention and
cross-attention layers while leveraging textual guidance... However, they
explicitly attend to different information without interaction, preserving
distinct feature representations."

Each branch self-attends only within its own query set, then cross-attends to
its own context ``[F, T_x]`` -- the video-level visual feature concatenated
with that branch's textual supervision, followed by a linear projection
(Eq. 1-3). Because the three branches never interact, they can be evaluated as
three independent rows of one batch, which is what :meth:`DisenQFormer.forward`
does: one Q-Former call on a ``3B``-sized batch.

At inference no text is available, so the context degenerates to ``F`` alone
and the queries rely on what they learned during training.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from disenq.models.Qformer import BertConfig, BertLMHeadModel
from disenq.utils.logger import get_logger

#: Fixed branch ordering used for the batched Q-Former call.
BRANCHES = ("biometrics", "motion", "non_biometrics")


def init_qformer(
    num_query_token: int,
    encoder_width: int,
    cross_attention_freq: int = 2,
    bert_model: str = "bert-base-uncased",
    gradient_checkpointing: bool = False,
    pretrained_bert: bool = True,
    local_files_only: bool = False,
    num_layers: Optional[int] = None,
    bert_config_overrides: Optional[Dict] = None,
):
    """Build a BLIP-2 Q-Former and one set of query tokens.

    Mirrors ``Blip2Base.init_Qformer`` from LAVIS so that published BLIP-2 /
    InstructBLIP Q-Former checkpoints load by name.

    ``num_layers`` overrides the BERT depth. It exists for the "deeper DisenQ"
    design-choice ablation; layers beyond the pretrained 12 start randomly
    initialised.
    """
    if bert_model in (None, "", "none"):
        # Build the config from scratch: no network access, and the dimensions
        # can be shrunk for fast tests or quick architecture experiments.
        config = BertConfig(**(bert_config_overrides or {}))
        pretrained_bert = False
    else:
        config = BertConfig.from_pretrained(bert_model, local_files_only=local_files_only)
        for key, value in (bert_config_overrides or {}).items():
            setattr(config, key, value)
    config.encoder_width = encoder_width
    config.add_cross_attention = True
    config.cross_attention_freq = cross_attention_freq
    config.query_length = num_query_token
    config.gradient_checkpointing = bool(gradient_checkpointing)
    if num_layers is not None:
        config.num_hidden_layers = int(num_layers)

    if pretrained_bert:
        # BERT has no cross-attention or query-FFN weights, so transformers
        # lists all of them as "newly initialized". That is expected -- they
        # come from the BLIP-2 checkpoint afterwards -- so mute the wall of
        # warnings while loading.
        import transformers

        previous_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            qformer = BertLMHeadModel.from_pretrained(
                bert_model, config=config, local_files_only=local_files_only
            )
        finally:
            transformers.logging.set_verbosity(previous_verbosity)
    else:
        qformer = BertLMHeadModel(config=config)

    query_tokens = nn.Parameter(torch.zeros(1, num_query_token, config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=config.initializer_range)
    return qformer, query_tokens


class ContextFusion(nn.Module):
    """Builds the cross-attention context ``W[F, T_x]`` of Eq. 1-3.

    The visual feature and the textual feature are each projected to the
    Q-Former's ``encoder_width``, concatenated along the token axis, and passed
    through a shared linear projection. The shared projection is identity
    initialised so that at step 0 the visual half of the context is exactly
    what a pretrained BLIP-2 Q-Former expects.
    """

    def __init__(
        self,
        vis_width: int,
        text_width: int,
        encoder_width: int,
        text_gate_init: float = 1.0,
    ):
        super().__init__()
        self.encoder_width = encoder_width

        if vis_width == encoder_width:
            self.vis_proj = nn.Identity()
        else:
            self.vis_proj = nn.Linear(vis_width, encoder_width)
            nn.init.trunc_normal_(self.vis_proj.weight, std=0.02)
            nn.init.zeros_(self.vis_proj.bias)

        self.text_proj = nn.Linear(text_width, encoder_width)
        nn.init.trunc_normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)
        self.text_ln = nn.LayerNorm(encoder_width)
        # Scales the text half of the context. 1.0 reproduces the paper; a
        # smaller value eases the warm start from BLIP-2 weights, which never
        # saw text in the cross-attention context.
        self.text_gate = nn.Parameter(torch.tensor(float(text_gate_init)))

        # Shared "followed by a linear projection" of Eq. 1, identity init.
        self.fuse_proj = nn.Linear(encoder_width, encoder_width)
        nn.init.eye_(self.fuse_proj.weight)
        nn.init.zeros_(self.fuse_proj.bias)

    def forward(
        self,
        visual: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            visual: ``[B, N, vis_width]`` video-level visual feature ``F``.
            text: ``[B, L, text_width]`` branch textual feature ``T_x``, or
                ``None`` at inference.
            text_mask: ``[B, L]`` 1 for real tokens, 0 for padding.

        Returns:
            ``(context [B, N(+L), encoder_width], mask [B, N(+L)])``.
        """
        vis = self.vis_proj(visual)
        mask = visual.new_ones(vis.shape[:2])

        if text is not None:
            txt = self.text_ln(self.text_proj(text.to(vis.dtype))) * self.text_gate
            vis = torch.cat([vis, txt], dim=1)
            if text_mask is None:
                text_mask = txt.new_ones(txt.shape[:2])
            mask = torch.cat([mask, text_mask.to(mask.dtype)], dim=1)

        return self.fuse_proj(vis), mask


class DisenQFormer(nn.Module):
    """Three query sets over one shared Q-Former."""

    def __init__(
        self,
        vis_width: int,
        text_width: int = 768,
        num_query_token: int = 32,
        encoder_width: Optional[int] = None,
        cross_attention_freq: int = 2,
        bert_model: str = "bert-base-uncased",
        gradient_checkpointing: bool = False,
        pretrained_bert: bool = True,
        local_files_only: bool = False,
        text_gate_init: float = 1.0,
        prune_text_branch: bool = True,
        branches: Sequence[str] = BRANCHES,
        num_layers: Optional[int] = None,
        bert_config_overrides: Optional[Dict] = None,
    ):
        super().__init__()
        encoder_width = encoder_width or vis_width
        self.num_query_token = num_query_token
        self.encoder_width = encoder_width
        # A subset reproduces the Table 3 (middle) ablation over which feature
        # types are disentangled.
        self.branches = tuple(branches)
        unknown = [b for b in self.branches if b not in BRANCHES]
        if unknown:
            raise ValueError(f"unknown branch(es) {unknown}; expected a subset of {BRANCHES}")

        self.qformer, query_tokens = init_qformer(
            num_query_token=num_query_token,
            encoder_width=encoder_width,
            cross_attention_freq=cross_attention_freq,
            bert_model=bert_model,
            gradient_checkpointing=gradient_checkpointing,
            pretrained_bert=pretrained_bert,
            local_files_only=local_files_only,
            num_layers=num_layers,
            bert_config_overrides=bert_config_overrides,
        )
        if prune_text_branch:
            self._prune_text_branch()

        self.hidden_size = self.qformer.config.hidden_size

        # One independent query set per feature type; all share the layers above.
        self.query_tokens = nn.ParameterDict(
            {name: nn.Parameter(query_tokens.data.clone()) for name in self.branches}
        )
        # Break the tie between the sets so they do not start identical.
        with torch.no_grad():
            for i, name in enumerate(self.branches):
                if i > 0:
                    self.query_tokens[name].add_(
                        torch.randn_like(self.query_tokens[name])
                        * self.qformer.config.initializer_range
                    )

        self.context_fusion = ContextFusion(
            vis_width=vis_width,
            text_width=text_width,
            encoder_width=encoder_width,
            text_gate_init=text_gate_init,
        )

        get_logger().info(
            f"built DisenQ: {len(self.branches)} query sets {list(self.branches)} x "
            f"{num_query_token} queries, hidden={self.hidden_size}, "
            f"encoder_width={encoder_width}, layers={self.qformer.config.num_hidden_layers}, "
            f"cross_attention_freq={cross_attention_freq}"
        )

    def _prune_text_branch(self):
        """Drop the Q-Former parameters that the text/LM path would use.

        DisenQ only ever feeds ``query_embeds`` (never ``input_ids``), so the
        word/position embeddings, the MLM head and the *text* FFN of each layer
        are dead weight. ``BertEmbeddings.forward`` short-circuits to
        ``embeddings = query_embeds`` when ``input_ids is None``, and
        ``BertLayer.forward`` only touches ``intermediate``/``output`` when the
        sequence is longer than the query block -- never the case here. Removing
        them saves roughly 100M parameters (~400MB in fp32).
        """
        self.qformer.cls = None
        self.qformer.bert.embeddings.word_embeddings = None
        self.qformer.bert.embeddings.position_embeddings = None
        for layer in self.qformer.bert.encoder.layer:
            layer.intermediate = None
            layer.output = None

    @property
    def dtype(self) -> torch.dtype:
        return self.query_tokens[self.branches[0]].dtype

    def forward(
        self,
        visual: torch.Tensor,
        text_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run all three branches.

        Args:
            visual: ``[B, N, vis_width]`` video-level visual feature ``F``.
            text_features: ``{branch: {"features": [B, L, text_width],
                "mask": [B, L]}}``. Pass ``None`` for the inference path, which
                drops the textual half of the context entirely.

        Returns:
            ``{branch: [B, num_query_token, hidden_size]}`` query embeddings.
        """
        batch_size = visual.shape[0]
        visual = visual.to(self.dtype)

        contexts, masks, queries = [], [], []
        for branch in self.branches:
            branch_text, branch_mask = None, None
            if text_features is not None and branch in text_features:
                branch_text = text_features[branch]["features"].to(self.dtype)
                branch_mask = text_features[branch].get("mask")

            context, mask = self.context_fusion(visual, branch_text, branch_mask)
            contexts.append(context)
            masks.append(mask)
            queries.append(self.query_tokens[branch].expand(batch_size, -1, -1))

        # The three branches never interact, so stacking them along the batch
        # axis is exactly equivalent to three separate calls, but runs as one.
        context = torch.cat(contexts, dim=0)
        mask = torch.cat(masks, dim=0)
        query_embeds = torch.cat(queries, dim=0)

        output = self.qformer.bert(
            query_embeds=query_embeds,
            encoder_hidden_states=context,
            encoder_attention_mask=mask,
            return_dict=True,
        ).last_hidden_state  # [3B, Q, hidden]

        return {
            branch: output[i * batch_size : (i + 1) * batch_size]
            for i, branch in enumerate(self.branches)
        }

    def load_blip2_qformer_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, list]:
        """Load Q-Former weights out of a BLIP-2 / InstructBLIP checkpoint.

        Published checkpoints store the Q-Former under the ``Qformer.`` prefix
        and a single ``query_tokens`` tensor. The single query set is copied
        into all three DisenQ query sets (plus the small init jitter), so every
        branch starts from the pretrained query prior.
        """
        logger = get_logger()
        qformer_sd, pretrained_queries = {}, None

        for key, value in state_dict.items():
            if key.startswith("Qformer."):
                qformer_sd[key[len("Qformer.") :]] = value
            elif key == "query_tokens":
                pretrained_queries = value

        loaded_queries = False
        if pretrained_queries is not None:
            num_available = pretrained_queries.shape[1]
            num_wanted = self.num_query_token
            if num_available >= num_wanted:
                base = pretrained_queries[:, :num_wanted]
            else:
                # Tile then trim when more queries are requested than available.
                reps = (num_wanted + num_available - 1) // num_available
                base = pretrained_queries.repeat(1, reps, 1)[:, :num_wanted]
            with torch.no_grad():
                for i, branch in enumerate(self.branches):
                    target = self.query_tokens[branch]
                    target.copy_(base.to(target.dtype).to(target.device))
                    if i > 0:
                        target.add_(
                            torch.randn_like(target) * self.qformer.config.initializer_range
                        )
            loaded_queries = True

        msg = self.qformer.load_state_dict(qformer_sd, strict=False)
        relevant_missing = [
            k for k in msg.missing_keys
            if not k.startswith("cls.")
            and "word_embeddings" not in k
            and "position_embeddings" not in k
        ]
        logger.info(
            f"loaded BLIP-2 Q-Former weights: {len(qformer_sd)} tensors, "
            f"{len(relevant_missing)} missing, {len(msg.unexpected_keys)} unexpected, "
            f"query_tokens={'yes' if loaded_queries else 'no'}"
        )
        if relevant_missing:
            logger.info(f"  missing (first 10): {relevant_missing[:10]}")
        return {"missing": relevant_missing, "unexpected": list(msg.unexpected_keys)}
