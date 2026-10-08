#!/usr/bin/env python
"""Generate the structured textual descriptions with a frozen VLM.

Implements Section 3.1 and Appendix A: a frozen VLM is shown the *key-frame* of
each clip together with the activity label, and asked to fill in a fixed
three-slot template. The paper uses LLaVA-1.5 7B ("we select LLaVA for its
efficiency and open-source property"); Table 5 (bottom) shows the choice of VLM
barely matters.

Biometrics descriptions are generated once per identity and then kept stable
via the running average described in Section 3.1 (see
``disenq/data/prompts.py`` for how the averaging is realised).

    python tools/generate_prompts.py \
        --annotation data/ntu_rgb_ab/annotations.json \
        --out data/ntu_rgb_ab/prompts.json \
        --splits train

This runs the VLM, so it needs a GPU and the model weights. It is a one-off
preprocessing step: the VLM is never needed at training or inference time.
"""

import argparse
import json
import os
import re
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disenq.data.dataset import load_annotation  # noqa: E402
from disenq.data.prompts import DEFAULT_PROMPTS, IdentityBiometricsStore  # noqa: E402

#: Appendix A, verbatim.
PROMPT_TEMPLATE = (
    "Analyze the given image where action label is {action_label} and extract "
    "the following details:\n"
    "Biometrics: A <physique/body shape> person with <posture>, such as "
    "arms/legs positioning.\n"
    "Motion: Performing the action of {action_label} by <action description>.\n"
    "Non-biometrics: A <color, type of clothing> and <other accessories>."
)

#: Appendix C.1's "simplified slot-filler" variant, kept for the ablation.
PROMPT_TEMPLATE_SIMPLE = (
    "Analyze the given image where action label is {action_label} and extract "
    "the following details:\n"
    "Biometrics: a <body shape> person.\n"
    "Motion: {action_label}.\n"
    "Non-biometrics: a <color> <clothing>."
)

_FIELD_PATTERN = re.compile(
    r"(biometrics|motion|non[\s\-_]?biometrics)\s*:\s*(.*?)(?=\n\s*(?:biometrics|motion|non[\s\-_]?biometrics)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_response(text: str):
    """Pull the three slots out of the VLM response."""
    out = {}
    for match in _FIELD_PATTERN.finditer(text):
        label = match.group(1).lower().replace(" ", "").replace("-", "").replace("_", "")
        value = " ".join(match.group(2).split()).strip(" .") 
        if not value:
            continue
        value = value + "."
        if label == "biometrics":
            out["biometrics"] = value
        elif label == "motion":
            out["motion"] = value
        elif label == "nonbiometrics":
            out["non_biometrics"] = value
    return out


class LlavaPromptGenerator:
    """LLaVA-1.5 wrapper that batches key-frames through the template."""

    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.float16):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        self.device = device
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

    @torch.no_grad()
    def generate(self, images, instructions, max_new_tokens: int = 160):
        prompts = [f"USER: <image>\n{text} ASSISTANT:" for text in instructions]
        inputs = self.processor(
            images=images, text=prompts, return_tensors="pt", padding=True
        ).to(self.device, torch.float16)
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
        )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)


class SentenceEmbedder:
    """Mean-pooled BERT embeddings, used only for the running average."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, text: str):
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=64
        ).to(self.device)
        hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled[0].float().cpu().numpy()


def keyframe_path(sample, data_root, frame_template, frame_index_base):
    """Locate the key-frame image for a sample."""
    index = sample.get("keyframe")
    if index is None:
        index = max(int(sample.get("num_frames", 1)) // 2, 0)

    if sample.get("frames_dir"):
        frames_dir = sample["frames_dir"]
        if not os.path.isabs(frames_dir):
            frames_dir = os.path.join(data_root, frames_dir)
        return os.path.join(frames_dir, frame_template.format(int(index) + frame_index_base))
    return None


def load_keyframe(sample, data_root, frame_template, frame_index_base):
    path = keyframe_path(sample, data_root, frame_template, frame_index_base)
    if path is not None and os.path.exists(path):
        return Image.open(path).convert("RGB")

    if sample.get("video"):
        import decord

        video = sample["video"]
        if not os.path.isabs(video):
            video = os.path.join(data_root, video)
        reader = decord.VideoReader(video, num_threads=1)
        index = min(int(sample.get("keyframe", len(reader) // 2)), len(reader) - 1)
        return Image.fromarray(reader[index].asnumpy())
    return None


def main():
    args = parse_args()

    annotation = load_annotation(args.annotation)
    action_names = annotation.get("action_names", [])
    template = PROMPT_TEMPLATE_SIMPLE if args.simple_template else PROMPT_TEMPLATE

    samples = []
    for split in args.splits:
        samples.extend(annotation["splits"].get(split, []))
    if args.limit:
        samples = samples[: args.limit]
    print(f"{len(samples)} clips to describe from splits {args.splits}")

    # Resume support: keep anything already generated.
    existing = {"samples": {}, "identities": {}}
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        print(f"resuming from {args.out}: {len(existing['samples'])} already done")

    store = IdentityBiometricsStore.from_dict(
        existing.get("identities", {}), momentum=args.momentum
    )

    generator = LlavaPromptGenerator(args.vlm, device=args.device)
    embedder = SentenceEmbedder(args.text_encoder, device=args.device) if args.refine else None

    pending = [s for s in samples if s["id"] not in existing["samples"]]
    print(f"{len(pending)} clips still pending")

    def action_label(sample):
        action = sample.get("action")
        if isinstance(action, (list, tuple)):
            names = [action_names[a] for a in action if 0 <= a < len(action_names)]
            return ", ".join(names) if names else "an everyday activity"
        if isinstance(action, int) and 0 <= action < len(action_names):
            return action_names[action]
        return "an everyday activity"

    for start in tqdm(range(0, len(pending), args.batch_size), desc="VLM"):
        chunk = pending[start : start + args.batch_size]
        images, instructions, kept = [], [], []
        for sample in chunk:
            image = load_keyframe(
                sample, args.data_root, args.frame_template, args.frame_index_base
            )
            if image is None:
                continue
            images.append(image)
            instructions.append(template.format(action_label=action_label(sample)))
            kept.append(sample)

        if not kept:
            continue

        try:
            responses = generator.generate(images, instructions, args.max_new_tokens)
        except Exception as error:  # noqa: BLE001 - keep going past bad frames
            print(f"  generation failed for a batch of {len(kept)}: {error}")
            continue

        for sample, response in zip(kept, responses):
            fields = parse_response(response)
            biometrics = fields.get("biometrics")
            identity = str(sample["pid"])

            stored = store.get(identity)
            if biometrics and (stored is None or args.refine):
                embedding = embedder(biometrics) if embedder is not None else None
                stored = store.update(identity, biometrics, embedding)
            biometrics = stored or biometrics or DEFAULT_PROMPTS["biometrics"]

            existing["samples"][sample["id"]] = {
                "biometrics": biometrics,
                "motion": fields.get("motion") or DEFAULT_PROMPTS["motion"],
                "non_biometrics": fields.get("non_biometrics")
                or DEFAULT_PROMPTS["non_biometrics"],
                "raw": response.strip() if args.keep_raw else None,
            }

        if args.save_every and (start // args.batch_size + 1) % args.save_every == 0:
            write(args.out, existing, store)

    write(args.out, existing, store)
    print(f"wrote {args.out}: {len(existing['samples'])} clip prompts, "
          f"{len(store.to_dict())} identity biometrics descriptions")


def write(path, payload, store):
    payload["identities"] = store.to_dict()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="prompts are training-only supervision, so 'train' is enough")
    parser.add_argument("--vlm", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--text-encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--frame-template", default="frame{:06d}.jpg")
    parser.add_argument("--frame-index-base", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--simple-template", action="store_true",
                        help="use the Appendix C.1 slot-filler template")
    parser.add_argument("--refine", action="store_true",
                        help="keep refining each identity's biometrics description "
                             "via the running average instead of taking the first one")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--save-every", type=int, default=50,
                        help="checkpoint the output every N batches (0 disables)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
