#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_BASE="${MODEL_BASE:-./base_model/vicuna-v1-5-7b}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-combined}"
PRETRAIN_MM_MLP_ADAPTER="${PRETRAIN_MM_MLP_ADAPTER:-./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin}"
FEAT_FOLDER="${FEAT_FOLDER:-./b4dl/stage2_features}"
INPUT_DIR="${INPUT_DIR:-./b4dl_dataset/hallucination_v2_no_position_bias}"
OUTPUT_DIR="${OUTPUT_DIR:-./b4dl_eval/hallucination_v2_no_position_bias}"

python evaluate_hallucination_v2_no_position_bias.py \
    --input-dir "$INPUT_DIR" \
    --feat-folder "$FEAT_FOLDER" \
    --model-base "$MODEL_BASE" \
    --pretrain-mm-mlp-adapter "$PRETRAIN_MM_MLP_ADAPTER" \
    --stage2 "$STAGE2_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --temperature "${TEMPERATURE:-0.05}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
    --num-beams "${NUM_BEAMS:-1}" \
    "$@"
