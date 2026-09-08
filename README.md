# LiDAR-Hallu / MLLM_Hallucination

Prior-aware evaluation of 4D LiDAR language model responses, accompanying the [ICRA paper repository](https://github.com/RunyiYang/-ICRA-2027-MLLM_Hallucinations).

## Full benchmark and predictions are in this repository

**[Benchmark JSON](data/b4dl_dataset/)** · **[Per-example predictions](data/b4dl_eval/)** · **[Original ZIPs](data/archives/)** · **[Data inventory](data/README.md)** · **[nuScenes LiDAR 下载说明](docs/NUSCENES_DOWNLOAD_ZH.md)**

A normal clone now contains all scientific records from the two supplied thesis archives. No Google Drive download, GPU, raw nuScenes data, or model checkpoint is needed to analyze these saved responses.

- **10,000 unique questions across 150 scenes.** Four prompt/metatoken exports, each containing five 2,000-question category files: 20 JSON files / 40,000 exported records, not 40,000 independent questions.
- **100,000 saved predictions.** Two B4DL-based configurations and five conditions per configuration. All 50 JSONL and corresponding 50 CSV files are included. CSV and JSONL encode the same predictions, not two sets of experiments.
- **Original bytes preserved.** The two ZIPs are committed unchanged. Extracted scientific files are byte-identical to their archive members; only macOS filesystem metadata is excluded from the extracted trees.

## Install, verify, and reproduce without external data

```bash
git clone https://github.com/RunyiYang/MLLM_Hallucination.git
cd MLLM_Hallucination
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

# No network needed for the data-verification step.
python scripts/materialize_paper_data.py
python -m pytest -q

# Write fresh statistics separately, preserving the committed paper evidence.
OUT=results_reproduced/local
python -m lidar_hallu.analyze --dataset data/b4dl_dataset --predictions data/b4dl_eval --output "$OUT"
python -m lidar_hallu.priors --dataset data/b4dl_dataset --output "$OUT"
python -m lidar_hallu.conflicts --predictions data/b4dl_eval --output "$OUT"
```

Python 3.10+ and NumPy are sufficient for the analysis. Do not use Python `-O` for the legacy analysis validator, because that disables its assertions. The materialization checker uses explicit exceptions. The default scene bootstrap uses 10,000 draws and seed 20260904. Installing Python dependencies still requires a suitable package cache or network; the dataset itself is self-contained after cloning.

`data/manifests/assets.json` lists archive and file hashes, sizes and record counts. `verification.json` records exact question/prompt/options/reference matching. `benchmark_scenes.json` maps all 150 scene IDs to nuScenes tokens and retained frame/sample references. The import workflow also preserves independent reanalysis under `data/manifests/reanalysis/`.

## What is and is not included

| Asset | Status |
|---|---|
| Full benchmark and ten saved evaluation runs | Included under `data/` |
| Strict rescoring, prior controls, paired statistics | Included under `lidar_hallu/` and `results/` |
| Original thesis-era scripts | Included under `legacy/` |
| Raw nuScenes LiDAR and full annotation metadata | External; see download guide |
| Synchronized RGB for a real-data teaser | Optional external sensor files |
| Learned LiDAR feature cache (`stage2_features/*.npy`) | Not in the supplied archives |
| Base model, projector, and trained checkpoints | External; not restored in this update |
| Complete original B4DL/VTimeLLM runtime | Not fully present; see migration notes |
| Token logits, hidden states, or unrelated experiments | Not present in the supplied archives |

The historical inference entry point reads learned `.npy` features, not raw `.bin` point clouds. New inference therefore needs the feature cache or a matching encoder/preprocessing pipeline in addition to the model/runtime. Restoring checkpoints is separate from analyzing the current paper results.

## Raw nuScenes setup

Follow [docs/NUSCENES_DOWNLOAD_ZH.md](docs/NUSCENES_DOWNLOAD_ZH.md). Use **v1.0-trainval**, including metadata and the required LiDAR sensor files. The exported string `test` denotes this project's evaluation role, not the official nuScenes test split.

```bash
export NUSCENES_ROOT=/path/to/nuscenes
# Metadata only: generate exact sensor filenames for the benchmark scenes.
python scripts/check_nuscenes_assets.py --dataroot "$NUSCENES_ROOT" --manifest-only
# Once raw data are present: verify alignment and nonempty file existence.
python scripts/check_nuscenes_assets.py --dataroot "$NUSCENES_ROOT"
```

The raw-data checker has synthetic unit tests; this update did not download or run on the full raw nuScenes trainval dataset. A file-presence PASS is not a feature-reproduction or model-inference result.

## Scientific scope and scoring

Saved `correct` and `parsed_prediction` values remain the original legacy scores. Use `lidar_hallu` for the declared strict parser. It rejects first-letter accidents such as mapping `car` to C, retains invalid outputs in the denominator, and reports useful departures from a fixed answer together with repaired and regressed predictions.

Scene-held-out categorical controls deliberately use labels from other evaluation scenes. They diagnose within-suite sensor-free predictability, not matched-supervision competition or a question-only run of the original checkpoint. TSCD is not a shuffled-only sensor ablation. No new neural-model predictions were generated in this data import.

The archived QA conversations contain reference answers for evaluation/training-format compatibility. Do not pass the reference assistant turn into an inference prompt. Keep future model runs in a new directory rather than overwriting archived outputs. See [docs/MIGRATION.md](docs/MIGRATION.md).

## Layout and rights

`lidar_hallu/` contains CPU analysis; `tests/` contains correctness checks; `scripts/` contains immutable data import and nuScenes asset checks; `results/` contains the existing paper summaries; `data/` contains the source corpus and evidence; `legacy/` preserves thesis-era code; `docs/` contains provenance and setup guidance.

No license is newly granted over inherited code, nuScenes data, B4DL questions or checkpoints. See [NOTICE.md](NOTICE.md). The repository visibility was not changed. Do not upload author-identifying legacy material as an anonymous review supplement.
