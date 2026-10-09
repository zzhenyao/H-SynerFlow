#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

python main.py assess_target \
    --config config/fisv_clip_target.yaml \
    --dataset PCS \
    --num_epoch 180 \
    --opt_T_max 180 \
    --model_args.clip_num 34 \
    --model_args.overlap_ratio 0.3 \
    --head_args.args.num_targets 3 \
    "$@"
