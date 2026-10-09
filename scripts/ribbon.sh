#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

python main.py assess_target \
    --config config/rg_clip_target.yaml \
    --dataset RIBBON \
    --num_epoch 180 \
    --opt_T_max 180 \
    --model_args.clip_num 50 \
    --head_args.args.num_targets 3 \
    "$@"
