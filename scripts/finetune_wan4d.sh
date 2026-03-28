#!/usr/bin/env bash
# Wan4D fine-tune entry: expects index.json + latents/ + caption_latents/ (see docs/dataset-structure.md).
# Generate index: python scripts/gen_index.py --dataset_root "$DATASET_ROOT"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${DATASET_ROOT:?Set DATASET_ROOT to dataset root (contains index.json)}"
: "${WAN_MODEL_DIR:?Set WAN_MODEL_DIR to Wan checkpoint directory}"

PROJECT_NAME="${PROJECT_NAME:-wan4d}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-run}"
INDEX_PATH="${INDEX_PATH:-}"

EXTRA_ARGS=("$@")

CMD=(python train.py
  --dataset_root "$DATASET_ROOT"
  --wan_model_dir "$WAN_MODEL_DIR"
  --project_name "$PROJECT_NAME"
  --experiment_name "$EXPERIMENT_NAME"
)

if [[ -n "$INDEX_PATH" ]]; then
  CMD+=(--index "$INDEX_PATH")
fi

[[ -n "${SPLIT:-}" ]] && CMD+=(--split "$SPLIT")
[[ -n "${MAX_EPOCHS:-}" ]] && CMD+=(--max_epochs "$MAX_EPOCHS")
[[ -n "${STEPS_PER_EPOCH:-}" ]] && CMD+=(--steps_per_epoch "$STEPS_PER_EPOCH")
[[ -n "${LEARNING_RATE:-}" ]] && CMD+=(--learning_rate "$LEARNING_RATE")
[[ -n "${RESUME_LIGHTNING_CKPT:-}" ]] && CMD+=(--resume_lightning_ckpt "$RESUME_LIGHTNING_CKPT")
[[ -n "${RESUME_CKPT_PATH:-}" ]] && CMD+=(--resume_ckpt_path "$RESUME_CKPT_PATH")

exec "${CMD[@]}" "${EXTRA_ARGS[@]}"
