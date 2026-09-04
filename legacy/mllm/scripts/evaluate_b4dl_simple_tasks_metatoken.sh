#!/bin/bash
set -e

python evaluate_simple_tasks_metatoken.py \
    --input-dir ./b4dl_dataset/test_metatoken \
    --feat-folder ./b4dl/stage2_features \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-metatoken \
    --output-dir ./b4dl_eval/simple_tasks_metatoken \
    --temperature 0.05 \
    --max-new-tokens 64 \
    "$@"
