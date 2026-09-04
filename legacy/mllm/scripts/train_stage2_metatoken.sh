#!/bin/bash
set -e

MODEL_VERSION=vicuna-v1-5-7b
GPU_VIS=${GPU_VIS:-${CUDA_VISIBLE_DEVICES:-0}}
MASTER_PORT=${MASTER_PORT:-29575}

# DeepSpeed ignores CUDA_VISIBLE_DEVICES when --include is present. Consume the
# requested GPU here, unset it, and pass the physical GPU id through --include.
unset CUDA_VISIBLE_DEVICES

deepspeed --include localhost:$GPU_VIS --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_combined_metatoken.json \
    --feat_folder ./b4dl/stage2_features \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-metatoken \
    --fp16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    "$@"
