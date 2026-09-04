#!/bin/bash
set -e

bash scripts/stage2.sh \
    --data_path ./b4dl_dataset/stage2.json \
    --feat_folder ./b4dl/stage2_features \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-only \
    "$@"
