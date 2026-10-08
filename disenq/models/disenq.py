"""The complete DisenQ framework for activity-biometrics.

Pipeline (Figure 2 of the paper)::

    RGB clip -> frozen ViT -> temporal pos. emb. -> temporal attention pooling -> F
    key-frame -> frozen VLM -> structured prompts -> frozen BERT -> T_b / T_m / T_b_hat
    (F, T_x) -> DisenQ (3 query sets, shared layers) -> F_b, F_m, F_b_hat
    F_b -> BNNeck -> L_ID + L_Tri
    F_m -> action head -> L_Act
    (F_b, F_b_hat) -> L_Orth
    retrieval: Sim = a1 * Sim_b + a2 * Sim_m   (adaptive MLP)

Only the components in the grey dashed box -- ViT, DisenQ, heads -- run at
inference; prompt generation and the text encoder are training-only.
"""

import contextlib
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from disenq.losses import (
    ActionLoss,
    CrossEntropyLabelSmooth,
    OrthogonalityLoss,
    PairwiseSimilarityLoss,
    TripletLoss,
)
from disenq.models.disenq_former import BRANCHES, DisenQFormer
from disenq.models.heads import AdaptiveSimilarity, BNNeck
from disenq.models.temporal import TemporalAttentionPooling, TemporalPositionEmbedding
from disenq.models.text_encoder import FrozenTextEncoder
from disenq.models.visual_encoder import build_visual_encoder
from disenq.utils.dist import download_cached_file, is_url
from disenq.utils.logger import get_logger

#: Named shortcuts for the Q-Former initialisations the paper mentions.
PRETRAINED_QFORMER_URLS = {
    # The paper initialises DisenQ from InstructBLIP.
    "instructblip_vicuna7b": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth",
    "instructblip_vicuna13b": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna13b_trimmed.pth",
    "instructblip_flant5xl": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxl_trimmed.pth",
    # BLIP-2 stage-1: same Q-Former architecture, ~3x smaller download.
    "blip2_pretrain": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth",
}


class DisenQModel(nn.Module):
    """Language-guided disentangling framework for activity-based person ID."""

    def __init__(
        self,
        num_identities: int,
        num_actions: int,
        # --- visual encoder ---
        visual_encoder: str = "eva_clip_g",
        image_size: int = 224,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        vit_grad_checkpoint: bool = False,
        drop_path_rate: float = 0.0,
        max_frames: int = 64,
        temporal_pooling_dropout: float = 0.0,
        vis_width: int = 1408,
        # --- DisenQ ---
        num_query_token: int = 32,
        cross_attention_freq: int = 2,
        qformer_encoder_width: Optional[int] = None,
        bert_model: str = "bert-base-uncased",
        qformer_grad_checkpoint: bool = False,
        pretrained_qformer: str = "instructblip_vicuna7b",
        pretrained_bert: bool = True,
        prune_qformer_text_branch: bool = True,
        bert_config_overrides: Optional[Dict] = None,
        branches: Sequence[str] = BRANCHES,
        qformer_num_layers: Optional[int] = None,
        text_gate_init: float = 1.0,
        # --- text supervision ---
        use_text_supervision: bool = True,
        text_encoder: str = "bert-base-uncased",
        max_text_len: int = 48,
        text_drop_prob: float = 0.0,
        local_files_only: bool = False,
        # --- heads ---
        head_dropout: float = 0.0,
        cosine_classifier: bool = False,
        adaptive_sim_hidden: int = 32,
        adaptive_sim_layers: int = 2,
        # --- losses ---
        id_weight: float = 1.0,
        triplet_weight: float = 1.0,
        orth_weight: float = 0.01,
        act_weight: float = 0.01,
        sim_weight: float = 0.1,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.0,
        orth_mode: str = "gram",
        action_mode: str = "ce",
        sim_mode: str = "ranking",
    ):
        super().__init__()
        logger = get_logger()

        self.num_identities = num_identities
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_text_supervision = use_text_supervision
        self.text_drop_prob = text_drop_prob
        self.freeze_vit = freeze_vit
        self.vit_precision = vit_precision

        # ---------------- visual branch ----------------
        self.visual_encoder, self.ln_vision, vis_width = build_visual_encoder(
            name=visual_encoder,
            image_size=image_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=vit_grad_checkpoint,
            precision=vit_precision,
            vis_width=vis_width,
        )
        self.vis_width = vis_width
        if freeze_vit and self.visual_encoder is not None:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            logger.info("visual encoder frozen")

        self.temporal_pos = TemporalPositionEmbedding(max_frames, vis_width)
        self.temporal_pool = TemporalAttentionPooling(
            vis_width, dropout=temporal_pooling_dropout
        )

        # ---------------- text branch (training only) ----------------
        self.text_encoder = None
        text_width = 768
        if use_text_supervision:
            self.text_encoder = FrozenTextEncoder(
                model_name=text_encoder,
                max_length=max_text_len,
                local_files_only=local_files_only,
            )
            text_width = self.text_encoder.width

        # ---------------- DisenQ ----------------
        self.disenq = DisenQFormer(
            vis_width=vis_width,
            text_width=text_width,
            num_query_token=num_query_token,
            encoder_width=qformer_encoder_width or vis_width,
            cross_attention_freq=cross_attention_freq,
            bert_model=bert_model,
            gradient_checkpointing=qformer_grad_checkpoint,
            pretrained_bert=pretrained_bert,
            local_files_only=local_files_only,
            text_gate_init=text_gate_init,
            prune_text_branch=prune_qformer_text_branch,
            bert_config_overrides=bert_config_overrides,
            branches=branches,
            num_layers=qformer_num_layers,
        )
        self.branches = self.disenq.branches
        hidden = self.disenq.hidden_size
        if "biometrics" not in self.branches:
            raise ValueError("the 'biometrics' branch is required for identification")

        # ---------------- heads ----------------
        self.id_head = BNNeck(
            hidden, num_identities, dropout=head_dropout,
            cosine_classifier=cosine_classifier,
        )
        self.action_head = BNNeck(
            hidden, num_actions, dropout=head_dropout,
            cosine_classifier=cosine_classifier,
        )
        # F_b_hat gets no classifier (it must not carry identity), only the
        # orthogonality constraint. A BN keeps its scale comparable to F_b.
        self.non_biometrics_norm = nn.BatchNorm1d(hidden)
        self.adaptive_similarity = AdaptiveSimilarity(
            hidden_dim=adaptive_sim_hidden, num_layers=adaptive_sim_layers
        )

        # ---------------- losses ----------------
        self.loss_id = CrossEntropyLabelSmooth(num_identities, epsilon=label_smoothing)
        self.loss_tri = TripletLoss(margin=triplet_margin)
        self.loss_orth = OrthogonalityLoss(mode=orth_mode)
        self.loss_act = ActionLoss(num_actions, mode=action_mode, epsilon=label_smoothing)
        self.loss_sim = PairwiseSimilarityLoss(mode=sim_mode)
        self.weights = {
            "id": id_weight,
            "tri": triplet_weight,
            "orth": orth_weight,
            "act": act_weight,
            "sim": sim_weight,
        }

        if pretrained_qformer and pretrained_qformer.lower() not in ("none", "null", ""):
            self.load_pretrained_qformer(pretrained_qformer)

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    def load_pretrained_qformer(self, source: str) -> None:
        """Initialise DisenQ (and ``ln_vision``) from a BLIP-2 family checkpoint."""
        logger = get_logger()
        url_or_path = PRETRAINED_QFORMER_URLS.get(source, source)

        if is_url(url_or_path):
            path = download_cached_file(url_or_path, check_hash=False, progress=True)
        elif os.path.isfile(url_or_path):
            path = url_or_path
        else:
            raise FileNotFoundError(
                f"pretrained_qformer '{source}' is neither a known alias "
                f"({sorted(PRETRAINED_QFORMER_URLS)}), a URL, nor an existing file"
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        self.disenq.load_blip2_qformer_weights(state_dict)

        # ln_vision is pretrained jointly with the Q-Former, so reuse it too.
        ln_sd = {
            k[len("ln_vision.") :]: v
            for k, v in state_dict.items()
            if k.startswith("ln_vision.")
        }
        if ln_sd:
            self.ln_vision.load_state_dict(ln_sd)
            logger.info("loaded pretrained ln_vision")
        logger.info(f"initialised DisenQ from '{source}'")

    # ------------------------------------------------------------------
    # encoders
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vit and self.visual_encoder is not None:
            self.visual_encoder.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.disenq.parameters()).device

    def _maybe_autocast(self):
        """fp16 autocast for the EVA ViT, which ships half-precision weights."""
        if self.vit_precision == "fp16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, T, 3, H, W]`` -> ``[B, T, N, vis_width]`` frozen ViT tokens."""
        if self.visual_encoder is None:
            raise RuntimeError(
                "model.visual_encoder is 'none', so raw frames cannot be encoded. "
                "Set data.cached_vit_features_root to a directory produced by "
                "tools/extract_vit_features.py, or choose a real visual encoder."
            )
        batch_size, num_frames = frames.shape[:2]
        flat = frames.flatten(0, 1)
        grad_ctx = torch.no_grad() if self.freeze_vit else contextlib.nullcontext()
        with grad_ctx, self._maybe_autocast():
            tokens = self.visual_encoder(flat)
        tokens = self.ln_vision(tokens.float())
        return tokens.view(batch_size, num_frames, *tokens.shape[1:])

    def encode_video(
        self,
        frames: Optional[torch.Tensor] = None,
        vit_features: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce the video-level feature ``F`` of shape ``[B, N, vis_width]``.

        Supply either raw ``frames`` or pre-extracted ``vit_features``
        (``[B, T, N, vis_width]``, already LayerNormed by
        ``tools/extract_vit_features.py``).
        """
        if vit_features is not None:
            tokens = vit_features.float()
        elif frames is not None:
            tokens = self.encode_frames(frames)
        else:
            raise ValueError("encode_video needs either `frames` or `vit_features`")

        tokens = self.temporal_pos(tokens)
        return self.temporal_pool(tokens, padding_mask=frame_mask)

    def encode_prompts(
        self, prompts: Dict[str, List[str]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Encode the three structured prompts with the frozen text encoder."""
        if self.text_encoder is None or prompts is None:
            return None
        usable = {b: prompts[b] for b in BRANCHES if b in prompts}
        if len(usable) != len(BRANCHES):
            return None
        return self.text_encoder.encode_branches(usable, device=self.device)

    # ------------------------------------------------------------------
    # disentangling + pooling
    # ------------------------------------------------------------------
    def disentangle(
        self,
        visual: torch.Tensor,
        text_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run DisenQ and mean-pool each query set into a single vector.

        "The learned query embeddings z_b, z_m and z_b_hat go through mean
        pooling to form single vectors, denoted as F_b, F_b_hat, and F_m."
        """
        queries = self.disenq(visual, text_features)
        return {branch: q.mean(dim=1) for branch, q in queries.items()}

    def extract_features(
        self,
        frames: Optional[torch.Tensor] = None,
        vit_features: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Inference-path feature extraction: no text, retrieval-ready vectors."""
        visual = self.encode_video(frames, vit_features, frame_mask)
        pooled = self.disentangle(visual, text_features=None)

        projections = {
            "biometrics": self.id_head.bottleneck,
            "motion": self.action_head.bottleneck,
            "non_biometrics": self.non_biometrics_norm,
        }
        out = {}
        for branch, bottleneck in projections.items():
            if branch not in pooled:
                continue
            feature = bottleneck(pooled[branch])
            out[branch] = F.normalize(feature, p=2, dim=-1) if normalize else feature
        return out

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def similarity_matrix(
        self,
        query: Dict[str, torch.Tensor],
        gallery: Dict[str, torch.Tensor],
        use_motion: bool = True,
        adaptive: bool = True,
        fixed_alpha: float = 0.5,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Eq. 8: ``Sim(A, B) = a1 * Sim_b(A, B) + a2 * Sim_m(A, B)``.

        Args:
            query / gallery: dicts of L2-normalised features from
                :meth:`extract_features`.
            use_motion: ``False`` reproduces the biometrics-only ablation.
            adaptive: ``False`` uses the fixed ``alpha_1 = alpha_2 =
                fixed_alpha`` weighting the paper compares against.
            chunk_size: rows of the probe axis processed at a time.
        """
        sim_b_full = query["biometrics"] @ gallery["biometrics"].t()
        if not use_motion or "motion" not in query or "motion" not in gallery:
            return sim_b_full

        sim_m_full = query["motion"] @ gallery["motion"].t()
        if not adaptive:
            return fixed_alpha * sim_b_full + (1.0 - fixed_alpha) * sim_m_full

        out = torch.empty_like(sim_b_full)
        for start in range(0, sim_b_full.shape[0], chunk_size):
            end = min(start + chunk_size, sim_b_full.shape[0])
            fused, _ = self.adaptive_similarity(
                sim_b_full[start:end], sim_m_full[start:end]
            )
            out[start:end] = fused
        return out

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Training forward pass. Returns the loss breakdown plus accuracies."""
        pids = batch["pid"]
        actions = batch["action"]

        text_features = None
        if self.use_text_supervision and self.training:
            drop = self.text_drop_prob > 0 and torch.rand(()).item() < self.text_drop_prob
            if not drop:
                text_features = self.encode_prompts(batch.get("prompts"))

        visual = self.encode_video(
            frames=batch.get("frames"),
            vit_features=batch.get("vit_features"),
            frame_mask=batch.get("frame_mask"),
        )
        pooled = self.disentangle(visual, text_features)

        f_b = pooled["biometrics"]
        id_logits, f_b_bn = self.id_head(f_b)

        losses = {}
        # Eq. 4 on F_b, Eq. 5 on F_b (pre-BN, as is standard for BNNeck).
        losses["id"] = self.loss_id(id_logits, pids)
        losses["tri"] = self.loss_tri(f_b, pids)

        act_logits, alpha = None, None
        if "motion" in pooled:
            act_logits, f_m_bn = self.action_head(pooled["motion"])
            # Eq. 4 on F_m.
            losses["act"] = self.loss_act(act_logits, actions)
            # Trains the Eq. 8 weighting MLP on in-batch probe/gallery pairs.
            sim_b = F.normalize(f_b_bn, dim=-1) @ F.normalize(f_b_bn, dim=-1).t()
            sim_m = F.normalize(f_m_bn, dim=-1) @ F.normalize(f_m_bn, dim=-1).t()
            fused_sim, alpha = self.adaptive_similarity(sim_b, sim_m)
            losses["sim"] = self.loss_sim(fused_sim, pids)

        if "non_biometrics" in pooled:
            # Eq. 6 between F_b and F_b_hat.
            f_bh_bn = self.non_biometrics_norm(pooled["non_biometrics"])
            losses["orth"] = self.loss_orth(f_b_bn, f_bh_bn)

        # Eq. 7
        total = sum(self.weights[name] * value for name, value in losses.items())

        with torch.no_grad():
            id_acc = (id_logits.argmax(dim=1) == pids).float().mean()
            act_acc = torch.zeros((), device=id_acc.device)
            if act_logits is not None:
                if self.action_mode == "ce":
                    valid = actions >= 0
                    if valid.any():
                        act_acc = (
                            act_logits[valid].argmax(dim=1) == actions[valid]
                        ).float().mean()
                else:
                    pred = act_logits.argmax(dim=1)
                    act_acc = actions.gather(1, pred[:, None]).float().mean()

        out = {f"loss_{k}": v for k, v in losses.items()}
        out["loss"] = total
        out["id_acc"] = id_acc
        out["act_acc"] = act_acc
        if alpha is not None:
            out["alpha_biometrics"] = alpha[..., 0].mean()
        out["text_gate"] = self.disenq.context_fusion.text_gate.detach()
        return out

    # ------------------------------------------------------------------
    # construction from config
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg, num_identities: int, num_actions: int) -> "DisenQModel":
        """Build from the ``model`` and ``loss`` blocks of a config."""
        model_cfg = cfg.model
        loss_cfg = cfg.get("loss", {})

        def m(key, default):
            """Config value, falling back to ``default`` for absent keys.

            A key present but explicitly ``null`` keeps the ``None``, so e.g.
            ``bert_model: null`` really means "build the config from scratch"
            rather than silently reverting to ``bert-base-uncased``.
            """
            if key not in model_cfg:
                return default
            return model_cfg[key]

        def l(key, default):
            if not loss_cfg or key not in loss_cfg:
                return default
            value = loss_cfg[key]
            return default if value is None else value

        return cls(
            num_identities=num_identities,
            num_actions=num_actions,
            visual_encoder=m("visual_encoder", "eva_clip_g"),
            image_size=m("image_size", 224),
            vit_precision=m("vit_precision", "fp16"),
            freeze_vit=m("freeze_vit", True),
            vit_grad_checkpoint=m("vit_grad_checkpoint", False),
            drop_path_rate=m("drop_path_rate", 0.0),
            max_frames=m("max_frames", 64),
            temporal_pooling_dropout=m("temporal_pooling_dropout", 0.0),
            vis_width=m("vis_width", 1408),
            num_query_token=m("num_query_token", 32),
            cross_attention_freq=m("cross_attention_freq", 2),
            qformer_encoder_width=model_cfg.get("qformer_encoder_width", None),
            bert_model=m("bert_model", "bert-base-uncased"),
            qformer_grad_checkpoint=m("qformer_grad_checkpoint", False),
            pretrained_qformer=m("pretrained_qformer", "instructblip_vicuna7b"),
            pretrained_bert=m("pretrained_bert", True),
            prune_qformer_text_branch=m("prune_qformer_text_branch", True),
            bert_config_overrides=(
                dict(model_cfg["bert_config"]) if model_cfg.get("bert_config") else None
            ),
            branches=tuple(m("branches", BRANCHES)),
            qformer_num_layers=model_cfg.get("qformer_num_layers", None),
            text_gate_init=m("text_gate_init", 1.0),
            use_text_supervision=m("use_text_supervision", True),
            text_encoder=m("text_encoder", "bert-base-uncased"),
            max_text_len=m("max_text_len", 48),
            text_drop_prob=m("text_drop_prob", 0.0),
            local_files_only=m("local_files_only", False),
            head_dropout=m("head_dropout", 0.0),
            cosine_classifier=m("cosine_classifier", False),
            adaptive_sim_hidden=m("adaptive_sim_hidden", 32),
            adaptive_sim_layers=m("adaptive_sim_layers", 2),
            id_weight=l("id_weight", 1.0),
            triplet_weight=l("triplet_weight", 1.0),
            orth_weight=l("orth_weight", 0.01),
            act_weight=l("act_weight", 0.01),
            sim_weight=l("sim_weight", 0.1),
            triplet_margin=l("triplet_margin", 0.3),
            label_smoothing=l("label_smoothing", 0.0),
            orth_mode=l("orth_mode", "gram"),
            action_mode=l("action_mode", "ce"),
            sim_mode=l("sim_mode", "ranking"),
        )
