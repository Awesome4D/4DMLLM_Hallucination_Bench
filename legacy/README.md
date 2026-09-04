# 4D LiDAR MLLM Hallucination Benchmark

This repository contains the related code for hallucination evaluation and training-free mitigation experiments on B4DL, a 4D LiDAR multimodal large language model.


# IMPORTANT NOTE
This GitLab repository does not include raw nuScenes data, LiDAR feature files, base LLM weights, full model checkpoints. These files are large and are accessed through the specified cluster directory.

Also, the benchmark samples and evaluation results arent provided in this repository as well due to their large size. They can be downloaded from the following drive links so the reported experiments can be inspected without rerunning the full pipeline:

benchmark samples: https://drive.google.com/file/d/1FCSGJ2uNjNngHYlKhtdqxxtUzHqlmE-j/view?usp=drive_link

evaluation results: https://drive.google.com/file/d/1i5yQCf_J8E3zs7GGCcJkc3gwVrvjD5g8/view?usp=drive_link



# Important Cluster Paths

The experiments were run on the cvhci cluster.

Project directory:
/cvhci/temp/makkoyun/projects/B4DL/mllm

nuScenes metadata/root used by generation scripts:
/cvhci/data/nuscenes/renew/v1.0-trainval

LiDAR feature directory:
/cvhci/temp/makkoyun/projects/B4DL/mllm/b4dl/stage2_features

Base LLM:
/cvhci/temp/makkoyun/projects/B4DL/mllm/base_model/vicuna-v1-5-7b

Main no-metatoken checkpoint:
/cvhci/temp/makkoyun/projects/B4DL/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-combined

Metatoken checkpoint:
/cvhci/temp/makkoyun/projects/B4DL/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-metatoken

Stage-1 projector:
/cvhci/temp/makkoyun/projects/B4DL/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin




# Benchmark Sample Locations on the cluster

Main benchmark samples:
mllm/b4dl_dataset/hallucination_v2

Metatoken benchmark samples:
mllm/b4dl_dataset/hallucination_v2_metatoken

No-position-bias prompt samples:
mllm/b4dl_dataset/hallucination_v2_no_position_bias
mllm/b4dl_dataset/hallucination_v2_no_position_bias_metatoken



# Evaluation Result Locations on the cluster

Baseline no-metatoken results:
mllm/b4dl_eval/hallucination_v2

Baseline metatoken results:
mllm/b4dl_eval/hallucination_v2_metatoken

No-position-bias prompt results:
mllm/b4dl_eval/hallucination_v2_no_position_bias
mllm/b4dl_eval/hallucination_v2_no_position_bias_metatoken

Temporal-shuffle contrastive decoding results:
mllm/b4dl_eval/hallucination_v2_temporal_shuffle_cd
mllm/b4dl_eval/hallucination_v2_temporal_shuffle_cd_metatoken
mllm/b4dl_eval/hallucination_v2_temporal_shuffle_cd_alpha05
mllm/b4dl_eval/hallucination_v2_temporal_shuffle_cd_alpha05_metatoken
mllm/b4dl_eval/hallucination_v2_temporal_shuffle_cd_alpha15
mllm/b4dl_eval/hallucination_v2_temporal_shuffle_cd_alpha15_metatoken


# Environment

Activate the existing cluster environment:
conda activate /cvhci/temp/makkoyun/conda_envs/b4dl_mllm

# Generate Benchmark Samples

From the cluster project directory:
cd /cvhci/temp/makkoyun/projects/B4DL/mllm
bash scripts/prepare_b4dl_hallucination_v2.sh

# Generate no-position-bias prompt samples:
bash scripts/prepare_b4dl_hallucination_v2_no_position_bias.sh

# Run Baseline Evaluation

No-metatoken model:
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2.sh

Metatoken model:
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_metatoken.sh

Run No-Position-Bias Prompt Evaluation
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_no_position_bias.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_no_position_bias_metatoken.sh

Run Temporal-Shuffle Contrastive Decoding Experiment
Alpha 1.0:
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_temporal_shuffle_cd.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_temporal_shuffle_cd_metatoken.sh

Alpha 0.5:
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_temporal_shuffle_cd_alpha05.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_temporal_shuffle_cd_alpha05_metatoken.sh

Alpha 1.5:
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_temporal_shuffle_cd_alpha15.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_b4dl_hallucination_v2_temporal_shuffle_cd_alpha15_metatoken.sh





