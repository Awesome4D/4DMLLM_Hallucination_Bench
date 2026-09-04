# Implementation map and reproducibility boundaries

## Preserved legacy behavior

`legacy/mllm/prepare_b4dl_hallucination_v2.py` compiles the five task families. The no-position-bias generator duplicates its task logic and changes the instruction. The new analysis does not rewrite the historical inputs.

`legacy/mllm/b4dl_metatoken.py` derives ego-motion text from poses. A separately trained metatoken checkpoint uses this text. It is not simply a prompt-only ablation of NoMeta.

`legacy/mllm/evaluate_hallucination_v2*.py` and the shell wrappers retain baseline, anti-position prompting and temporal-shuffle contrastive decoding with strengths 0.5/1/1.5. The TSCD clean/shuffled branches use independent caches and a shared answer prefix. No adaptive VCD plausibility mask is implemented.

## Reorganized analysis

`metrics.py`: explicit parsing, answer distributions, balanced recall, fixed-A decomposition, paired repair/regression and whole-scene percentile intervals.

`analyze.py`: one-to-one integrity checks and strict rescoring of 100,000 archived outputs. It validates saved permutations, actual feature lengths, source temporal options, motion predicates and the four dataset variants. It does not regenerate source nuScenes annotations from raw point clouds.

`priors.py`: leave-one-scene-out categorical controls. Ground-truth `object_class` metadata is not used as a predictor feature; class/context features are extracted from the question or options. This matters because some metadata fields encode the answer.

`conflicts.py`: descriptive model performance on the out-of-scene control's correct/incorrect strata, with paired scene intervals. These strata are not an independently collected test split.

## Inputs that remain external

Full B4DL/VTimeLLM upstream source, the imported `evaluate_simple_tasks.py`, feature-cache files, model weights/projector, nuScenes metadata and original runtime environment are not all present in the uploaded archive. Restore these with the thesis author before GPU reruns. Keep archived outputs immutable and write new runs to separate condition directories with checkpoint/input hashes and all seeds recorded.

## Rerun contract

For future causal sensor tests, use the same checkpoint, prompt, candidate order, decoding policy and sampling seed for clean and intervention runs. Apply one explicit intervention to the sequence. Track label-preserving versus label-changing transformations. Include an identity intervention and an option-permutation / candidate-likelihood diagnostic before attributing constant letter outputs to the temporal representation.
