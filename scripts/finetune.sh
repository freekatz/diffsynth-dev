#!/usr/bin/env bash
set -euo pipefail

# Simple Wan4D finetune launcher.
# Usage:
#   bash scripts/finetune.sh /path/to/wan_pretrained_dir [dataset_path] [output_path] [extra train args...]
#
# It first runs process.py to generate caches,
# then runs train_dev.py for training.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <WAN_PRETRAINED_DIR> [DATASET_PATH] [OUTPUT_PATH] [extra train args...]"
  exit 1
fi

WAN_PRETRAINED_DIR="$1"
DATASET_PATH="${2:-data/demo_videos}"
OUTPUT_PATH="${3:-outputs/wan4d_finetune}"
EXTRA_TRAIN_ARGS=("${@:4}")

if [[ ! -d "$WAN_PRETRAINED_DIR" ]]; then
  echo "Error: WAN_PRETRAINED_DIR does not exist: $WAN_PRETRAINED_DIR"
  exit 1
fi

echo "WAN_PRETRAINED_DIR=$WAN_PRETRAINED_DIR"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_PATH=$OUTPUT_PATH"
if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  echo "EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS[*]}"
fi

python "process.py" \
  --dataset_path "$DATASET_PATH" \
  --wan_model_dir "$WAN_PRETRAINED_DIR"

python "train_dev.py" \
  --dataset_path "$DATASET_PATH" \
  --wan_model_dir "$WAN_PRETRAINED_DIR" \
  --output_path "$OUTPUT_PATH" \
  "${EXTRA_TRAIN_ARGS[@]}"
