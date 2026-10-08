#!/usr/bin/env bash
# Evaluate a DisenQ checkpoint.
#
#   bash scripts/test.sh configs/ntu_rgb_ab.yaml output/ntu_rgb_ab/checkpoint_best.pth
#   bash scripts/test.sh configs/ntu_rgb_ab.yaml <ckpt> --feature biometrics
set -euo pipefail

CFG="${1:?usage: bash scripts/test.sh <config.yaml> <checkpoint.pth> [extra args ...]}"
CKPT="${2:?usage: bash scripts/test.sh <config.yaml> <checkpoint.pth> [extra args ...]}"
shift 2 || true

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

python test.py --cfg "${CFG}" --checkpoint "${CKPT}" "$@"
