#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

python main.py assess_target \
    --config config/rg_clip_target.yaml \
    --dataset HOOP \
    --num_epoch 100 \
    --opt_T_max 100 \
    --model_args.clip_num 120 \
    --head_args.args.num_targets 5 \
    "$@"
