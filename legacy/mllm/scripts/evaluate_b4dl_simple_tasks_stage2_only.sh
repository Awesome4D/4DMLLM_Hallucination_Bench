#!/bin/bash
set -e

bash scripts/evaluate_b4dl_simple_tasks.sh \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-only \
    --output-dir ./b4dl_eval/simple_tasks_stage2_only \
    "$@"
