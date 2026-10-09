#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

python main.py assess_target \
    --config config/fisv_clip_target.yaml \
    --dataset TES \
    --num_epoch 140 \
    --opt_T_max 140 \
    --model_args.clip_num 60 \
    --head_args.args.num_targets 7 \
    "$@"
