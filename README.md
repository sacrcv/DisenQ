# DisenQ: Disentangling Q-Former for Activity-Biometrics

Official PyTorch implementation of [**DisenQ** (ICCV 2025)](https://arxiv.org/abs/2507.07262),
built on the BLIP-2 Q-Former from [Salesforce LAVIS](https://github.com/salesforce/LAVIS).

> Azad & Rawat. *DisenQ: Disentangling Q-Former for Activity-Biometrics.* ICCV 2025.

DisenQ identifies people across diverse everyday activities. A single querying
transformer carries **three** independent sets of learnable queries that share
one set of self-attention and cross-attention layers; structured textual
descriptions from a frozen VLM supervise them into three disjoint feature
spaces, so identity cues stay independent of clothing and motion.

| query set | feature | language supervision | used for |
|---|---|---|---|
| `z_b` | biometrics `F_b` | body shape / posture | identification (`L_ID`, `L_Tri`) |
| `z_m` | motion `F_m` | action label / movement | action recognition (`L_Act`) + retrieval |
| `z_b̂` | non-biometrics `F_b̂` | clothing / accessories | orthogonality constraint (`L_Orth`) only |

The VLM and the text encoder are **training-only**. At inference the queries
cross-attend to visual features alone.

---

## Install

Install PyTorch first, matched to your CUDA version
([other builds](https://pytorch.org/get-started/locally/)), then the rest:

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate disenq
```

All commands are run from the repository root — there is no package to install.

`requirements.txt` carries version ranges; `requirements-lock.txt` pins the
exact versions this was developed against (Python 3.10.20, torch 2.7.1+cu118,
transformers 4.45.2, one A100-40GB). There is deliberately **no timm
dependency** — the three helpers LAVIS imported from it are vendored in
`disenq/models/layers.py`, so you are not pinned to `timm==0.4.12`.

### Pretrained weights

Downloaded automatically on first use, into the torch hub cache
(`~/.cache/torch/hub/checkpoints`, override with `TORCH_HOME`) and the
HuggingFace cache:

| file | size | purpose |
|---|---|---|
| `eva_vit_g.pth` | 2.0 GB | EVA-CLIP ViT-G/14 visual encoder |
| `instruct_blip_vicuna7b_trimmed.pth` | 2.3 GB | Q-Former initialisation (paper default) |
| `blip2_pretrained.pth` | 747 MB | lighter alternative: `model.pretrained_qformer=blip2_pretrain` |
| `bert-base-uncased` | 440 MB | Q-Former backbone + frozen text encoder |
| `llava-hf/llava-1.5-7b-hf` | 14 GB | prompt generation only, one-off |

> **Note on trained checkpoints.** Trained DisenQ model weights are not
> released, as the method is covered by the patent noted at the end of this
> README. Everything needed to train the model from the publicly available
> initialisations above is provided here.

---

## Reproducing the paper

### 1. Get the data

Please download the source datasets from their original providers, following
each one's access terms:

| benchmark | source dataset | where |
|---|---|---|
| NTU RGB-AB | NTU RGB+D 120 | https://rose1.ntu.edu.sg/dataset/actionRecognition/ |
| Charades-AB | Charades | https://prior.allenai.org/projects/charades |
| MEVID | MEVID | https://github.com/Kitware/MEVID |

Decode videos into frame directories (much faster to train from than seeking
in compressed video):

```bash
python tools/extract_frames.py --videos data/ntu_rgb_ab/videos \
    --out data/ntu_rgb_ab/frames --fps 10
```

### 2. Build the annotations

`tools/prepare_dataset.py` reproduces the benchmark construction of
[ABNet (CVPR 2024)](https://arxiv.org/abs/2403.17360), which defines these
datasets and which DisenQ follows. It applies each dataset's **official**
train/test split, then divides the test portion into probe and gallery per
ABNet Appendix A. See `disenq/data/splits.py` for the rules and the quoted
protocol text.

```bash
# NTU RGB-AB — official NTU-120 cross-subject split (53/53 subjects).
# Two-person actions are dropped, leaving the 94 single-person classes.
python tools/prepare_dataset.py --dataset ntu \
    --frames-root data/ntu_rgb_ab/frames \
    --out data/ntu_rgb_ab/annotations.json

# Charades-AB — official train/test CSVs; `subject` is the actor (267),
# `actions` gives the 157 multi-label classes.
python tools/prepare_dataset.py --dataset charades \
    --charades-csv-dir data/charades_ab/annotations \
    --frames-root data/charades_ab/frames \
    --out data/charades_ab/annotations.json

# MEVID — ships its own train/query/gallery track files.
python tools/prepare_dataset.py --dataset mevid \
    --mevid-root data/mevid --out data/mevid/annotations.json
```

Add `--protocol cross_activity` for the cross-activity variant, which makes the
probe and gallery activity sets mutually exclusive. For any other dataset,
`--dataset manifest` accepts a CSV.

### 3. Generate the language supervision

A one-off preprocessing step — the VLM is not needed during training or
inference.

```bash
python tools/generate_prompts.py \
    --annotation data/ntu_rgb_ab/annotations.json \
    --out data/ntu_rgb_ab/prompts.json --splits train
```

Uses LLaVA-1.5 7B with the paper's Appendix A template, and keeps one stable
biometrics description per identity (Section 3.1).

### 4. Train

```bash
# single GPU
python train.py --cfg configs/ntu_rgb_ab.yaml

# multi GPU
torchrun --nproc_per_node=4 train.py --cfg configs/ntu_rgb_ab.yaml

# or the wrapper, which derives the GPU count and tees a log
bash scripts/train.sh configs/ntu_rgb_ab.yaml

# any config key is overridable inline
python train.py --cfg configs/ntu_rgb_ab.yaml run.max_epoch=30 loss.orth_weight=0.1
```

Defaults follow Section 4.1: 8 frames at stride 4, 224×224 with horizontal
flipping, batch 32 as 8 identities × 4 clips, AdamW at lr 1e-4 / weight decay
5e-2 / betas [0.9, 0.999], 60 epochs, triplet margin 0.3, DisenQ initialised
from InstructBLIP.

The ViT is frozen, so caching its output makes each epoch a Q-Former-only
forward pass — the single biggest speedup available:

```bash
python tools/extract_vit_features.py --cfg configs/ntu_rgb_ab.yaml \
    --out data/ntu_rgb_ab/vit_features

python train.py --cfg configs/ntu_rgb_ab.yaml \
    data.cached_vit_features_root=data/ntu_rgb_ab/vit_features \
    model.visual_encoder=none        # skips loading 2GB of ViT weights
```

### 5. Evaluate

Point `--checkpoint` at a model you trained in the previous step:

```bash
python test.py --cfg configs/ntu_rgb_ab.yaml \
    --checkpoint output/ntu_rgb_ab/checkpoint_best.pth
```

Protocol variants, from Section 4 and ABNet Appendix A:

| protocol | gallery filter | datasets |
|---|---|---|
| `same_activity_include_view` | none (ABNet's View+) | NTU RGB-AB |
| `same_activity_exclude_view` | drop probe's viewpoint (View−) | NTU RGB-AB |
| `cross_activity_include_view` | drop same-activity clips | NTU RGB-AB |
| `cross_activity_exclude_view` | both | NTU RGB-AB |
| `same_activity` / `cross_activity` | no viewpoint data | Charades-AB |
| `general` | drop same identity **and** same camera | MEVID |

Reports rank-1, rank-5, rank-10, rank-20 and mAP.

## Repository layout

```
disenq/
  models/
    Qformer.py          vendored BLIP-2 Q-Former from LAVIS (unmodified)
    eva_vit.py          vendored EVA-CLIP ViT-G/14 from LAVIS
    layers.py           drop_path / to_2tuple / trunc_normal_ (replaces timm)
    visual_encoder.py   encoder construction (EVA-CLIP, any hf: tower, or none)
    temporal.py         temporal position embedding + attention pooling -> F
    text_encoder.py     frozen BERT producing T_b / T_m / T_b̂
    disenq_former.py    DisenQ: three query sets over one shared Q-Former
    heads.py            BNNeck identification head, adaptive similarity MLP
    disenq.py           the full framework, losses and retrieval
  losses/               L_ID, L_Tri, L_Orth, L_Act, pairwise similarity
  data/
    splits.py           official splits + the ABNet probe/gallery protocol
    dataset.py          annotation schema and clip loading
    samplers.py         P x K identity-balanced sampler
    prompts.py          per-identity biometrics description store
  evaluation/           CMC / mAP under every protocol variant
  engine/               optimiser, LR schedule, training loop
  utils/                config, logging, distributed, checkpoint IO
tools/                  dataset prep, prompt generation, ViT feature caching
configs/                _base.yaml plus one config per benchmark
train.py / test.py      entry points
```

### How the paper maps onto the code

| paper | code |
|---|---|
| Visual features, temporal pos. emb., temporal attention pooling (Sec. 3.1) | `models/temporal.py`, `DisenQModel.encode_video` |
| Prompt generation from the key-frame (Sec. 3.1, App. A) | `tools/generate_prompts.py` |
| Per-identity biometrics description + running average (Sec. 3.1) | `data/prompts.py` |
| Frozen text encoder → `T_b`, `T_m`, `T_b̂` | `models/text_encoder.py` |
| DisenQ, three query sets sharing SA/CA (Sec. 3.2) | `models/disenq_former.py` |
| Cross-attention context `W[F, T_x]` (Eq. 1–3) | `ContextFusion` |
| Mean pooling → `F_b`, `F_m`, `F_b̂` (Sec. 3.3) | `DisenQModel.disentangle` |
| `L_ID`, `L_Act` (Eq. 4) | `losses/classification.py` |
| `L_Tri` (Eq. 5) | `losses/triplet.py` |
| `L_Orth` (Eq. 6) | `losses/orthogonality.py` |
| Total loss (Eq. 7) | `DisenQModel.forward` |
| Adaptive similarity weighting (Eq. 8) | `AdaptiveSimilarity` |
| Inference without text | `DisenQModel.extract_features` |
| Evaluation protocols (Sec. 4, ABNet App. A) | `evaluation/metrics.py`, `data/splits.py` |

### One Q-Former, one call

> "Each query set shares the same self-attention and cross-attention layers…
> However, they explicitly attend to different information without
> interaction."

Because the branches never interact, three separate forward passes are
mathematically identical to one pass over a batch of `3B`.
`DisenQFormer.forward` builds each branch's context, concatenates along the
batch axis, makes a single Q-Former call, and splits the output back into
three.

The published InstructBLIP checkpoint loads with **zero missing and zero
unexpected keys**, because LAVIS prunes exactly the same dead weights this does
(word/position embeddings, the MLM head, and the per-layer *text* FFN — none of
which DisenQ can reach, since it only ever passes `query_embeds`).

---

## Citation

```bibtex
@inproceedings{azad2025disenq,
  title={Disenq: Disentangling q-former for activity-biometrics},
  author={Azad, Shehreen and Rawat, Yogesh Singh},
  booktitle={2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={13502--13512},
  year={2025},
  organization={IEEE}
}

@inproceedings{azad2024activity,
  title={Activity-biometrics: Person identification from daily activities},
  author={Azad, Shehreen and Rawat, Yogesh Singh},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={287--296},
  year={2024},
  organization={IEEE}
}
```

## License

Copyright © 2025 University of Central Florida. This code is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
for **non-commercial research use only**. See `LICENSE` for the full terms and
attribution.

**Third-party exception.** `disenq/models/Qformer.py`, `disenq/models/eva_vit.py`
and the `download_cached_file` helper in `disenq/utils/dist.py` are vendored
from [Salesforce LAVIS](https://github.com/salesforce/LAVIS) and remain under
the **BSD 3-Clause** license. They are not relicensed, and the NonCommercial /
ShareAlike conditions do not apply to them.

### Patent notice

This repository implements technology covered, fully or partially, by
**US Patent 12,482,255**, *Activity-based person identification using biometric
disentanglement* (S. Azad, Y. S. Rawat), owned by the **University of Central
Florida**.

**No patent rights are granted by this license.** CC BY-NC-SA 4.0 is a
copyright license only — its Section 2(b)(2) states that "Patent and trademark
rights are not licensed under this Public License." Commercial use requires a
separate license from UCF:
[UCF Office of Technology Transfer](https://www.research.ucf.edu/tech-transfer/contact/).
