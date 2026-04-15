#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 修改这里的公共参数
# ============================================================
export PROJ=wan4d-new-v2-hoi4d200
export EXP=test_run
export GPU_ID=1
export CLIP_PATH=ZY20210800001_H1_C12_N41_S200_s04_T2/clip_1

export STEP=3000
export CLIP_DIR=/root/tos/datasets-processed/mydataset-v3/videos/omniworld_hoi4d/${CLIP_PATH}
export WAN_MODEL=/root/tos/models/alibaba-pai-Wan2.1-Fun-V1.1-1.3B-InP/
export CKPT=training/${PROJ}/${EXP}/checkpoints/step${STEP}_model.ckpt

# ============================================================
# 批量任务列表：每项格式 "UNITS|INDEXS"
# ============================================================
TASKS=(
    "F:40,Z:8,R:11,F:22|1,3"
    "F:81|0"
    "F:41,R:40|0,1"
)

python inference.py \
    --wan_model_dir "$WAN_MODEL" \
    --ckpt "$CKPT" \
    --output "${PROJ}-${EXP}-${STEP}" \
    --clip "$CLIP_DIR" \
    --gpu_id "$GPU_ID" \
    --tasks "${TASKS[@]}"
