#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

python main.py assess_target \
    --config config/rg_clip_target.yaml \
    --dataset BALL \
    --num_epoch 200 \
    --opt_T_max 200 \
    --model_args.clip_num 40 \
    --head_args.args.num_targets 8 \
    --head_args.args.t_min_dis 0.6 \
    "$@"
