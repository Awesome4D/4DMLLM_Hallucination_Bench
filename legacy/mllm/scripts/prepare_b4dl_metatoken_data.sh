#!/bin/bash
set -e

if [ -z "$NUSCENES_ROOT" ]; then
    echo "Set NUSCENES_ROOT=/path/to/nuscenes/v1.0-trainval before running this script." >&2
    exit 1
fi

python prepare_b4dl_stage2_combined_metatoken.py \
    --scene-metadata ../nuScenes-B4DL/metadata/scene_metadata.json \
    --nuscenes-root "$NUSCENES_ROOT" \
    --out ./b4dl_dataset/stage2_combined_metatoken.json

python prepare_b4dl_test_metatoken.py \
    --input-dir ../nuScenes-B4DL/dataset/test \
    --output-dir ./b4dl_dataset/test_metatoken \
    --scene-metadata ../nuScenes-B4DL/metadata/scene_metadata.json \
    --nuscenes-root "$NUSCENES_ROOT"
