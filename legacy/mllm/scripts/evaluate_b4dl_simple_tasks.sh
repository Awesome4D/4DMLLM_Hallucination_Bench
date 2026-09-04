#!/bin/bash
set -e

python evaluate_simple_tasks.py \
    --input-dir ./b4dl_dataset/test \
    --feat-folder ./b4dl/stage2_features \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-combined \
    --output-dir ./b4dl_eval/simple_tasks \
    --temperature 0.05 \
    --max-new-tokens 64 \
    "$@"
