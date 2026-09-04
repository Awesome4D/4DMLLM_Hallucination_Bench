#!/bin/bash
set -e

export NUSCENES_ROOT="${NUSCENES_ROOT:-/cvhci/data/nuscenes/renew/v1.0-trainval}"

SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-2000}"
SEED="${SEED:-42}"
SPLIT="${SPLIT:-val}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
METADATA_BACKEND="${METADATA_BACKEND:-devkit}"
SCENE_METADATA="${SCENE_METADATA:-../nuScenes-B4DL/metadata/scene_metadata.json}"
TIME_GROUNDING_JSON="${TIME_GROUNDING_JSON:-./b4dl_dataset/test/time_grounding.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./b4dl_dataset/hallucination_v2_no_position_bias}"
METATOKEN_OUTPUT_DIR="${METATOKEN_OUTPUT_DIR:-./b4dl_dataset/hallucination_v2_no_position_bias_metatoken}"

python prepare_b4dl_hallucination_v2_no_position_bias.py \
    --nuscenes-root "$NUSCENES_ROOT" \
    --nuscenes-version "$NUSCENES_VERSION" \
    --metadata-backend "$METADATA_BACKEND" \
    --scene-metadata "$SCENE_METADATA" \
    --time-grounding-json "$TIME_GROUNDING_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --metatoken-output-dir "$METATOKEN_OUTPUT_DIR" \
    --samples-per-category "$SAMPLES_PER_CATEGORY" \
    --seed "$SEED" \
    --split "$SPLIT" \
    --with-metatoken \
    "$@"
