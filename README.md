# LiDAR-Hallu / MLLM_Hallucination

Prior-aware evaluation of 4D LiDAR language model responses, accompanying **LiDAR-Hallu: Separating Answer Priors from 4D LiDAR Grounding** (ICRA 2027 submission draft, version 2).

## Layout

- `lidar_hallu/`: lightweight, CPU-only response analysis and sensor-free controls.
- `tests/`: output-contract, exact-accounting, scene-bootstrap and leakage tests.
- `results/`: recomputed machine-readable evidence for the paper.
- `legacy/`: unmodified thesis-era generators, training wrappers, inference and TSCD implementation in the complete handoff.
- `docs/`: scientific scope, migration notes and source hashes.

The local full handoff preserves all original code. The published analysis core can run from the original benchmark and prediction archives without downloading a model or using a GPU.

## Install and test

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Python 3.10+ and NumPy are sufficient for the analysis. The legacy GPU environment is separate and is not installed by this package.

## Reproduce the submitted numbers

Unzip the original benchmark and result archives into `data/b4dl_dataset` and `data/b4dl_eval`, then run:

```bash
python -m lidar_hallu.analyze --dataset data/b4dl_dataset --predictions data/b4dl_eval --output results
python -m lidar_hallu.priors --dataset data/b4dl_dataset --output results
python -m lidar_hallu.conflicts --predictions data/b4dl_eval --output results
```

Do not run the validation command with Python `-O`, which disables assertions. The default bootstrap uses 10,000 whole-scene draws with seed 20260904. The analysis verifies all four dataset variants and all 100,000 saved predictions against their exact questions, prompts, options, references and sequence bounds.

The archives linked in the original project README are:

- Benchmark: Google Drive file `1FCSGJ2uNjNngHYlKhtdqxxtUzHqlmE-j`.
- Predictions: Google Drive file `1i5yQCf_J8E3zs7GGCcJkc3gwVrvjD5g8`.

The private complete handoff also includes these exact ZIPs. Access and redistribution remain subject to their owners' and upstream dataset terms.

## What is new

The explicit parser rejects first-letter accidents such as mapping `car` to C, counts invalid outputs as incorrect, and preserves the leading answer-label contract. The position-prior audit relates constant-A accuracy to useful departures from A. The categorical control excludes the complete target scene from fitting and uses question/candidate strings only. Paired analysis distinguishes repaired errors, damaged correct predictions and wrong-to-wrong flips.

The scene-held-out control deliberately uses labels from other evaluation scenes. It diagnoses within-suite predictability; it is not a matched-supervision competitor or a question-only pass of the original checkpoint. The paper's TSCD conditions are not standalone corrupted-input ablations. There is no new neural-model inference in this reanalysis.

## Legacy inference

The complete handoff preserves the original scripts byte-for-byte. They assume the author's B4DL/VTimeLLM tree, feature cache, stage-one projector, base Vicuna model and trained checkpoints. Those assets and an imported `evaluate_simple_tasks.py` are not all present in the uploaded code archive. Legacy GPU execution has therefore not been validated here. See `docs/MIGRATION.md` rather than silently substituting different models or defaults.

## Rights

No license is newly granted over inherited code, nuScenes data, B4DL questions or checkpoints. See `NOTICE.md`. Do not place the author-identifying legacy README or thesis archive into an anonymous review supplement.
