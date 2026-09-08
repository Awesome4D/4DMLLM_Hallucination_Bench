# Complete paper benchmark and saved predictions

The two original user-provided archives and their scientific contents are committed to this repository, not just linked to Drive or stored in temporary CI artifacts. A normal clone includes them. No model weights, raw nuScenes point clouds, RGB frames, or feature tensors are included.

## Inventory

| Path | Contents |
|---|---|
| `archives/b4dl_dataset.zip` | Original 4,771,522-byte benchmark archive, unchanged |
| `archives/b4dl_eval.zip` | Original 15,886,960-byte predictions archive, unchanged |
| `b4dl_dataset/` | 20 JSON files: four variants, five task categories, 2,000 records per file |
| `b4dl_eval/` | 50 JSONL + 50 CSV files: two configurations x five conditions x five categories |
| `manifests/assets.json` | Archive hashes, every extracted file's SHA256/blob ID, byte size and record count |
| `manifests/verification.json` | Full import integrity summary |
| `manifests/benchmark_scenes.json` | 150 scene IDs mapped to official scene tokens and retained sample tokens |
| `manifests/reanalysis/` | Independent CPU reanalysis generated when the import workflow is run |

There are **10,000 unique benchmark questions**, not 40,000 independent questions. The four exports differ in prompt/metatoken configuration. There are **100,000 saved predictions**, not 200,000: CSV and JSONL are duplicate formats of the same responses. The archives do not contain full token logits, hidden states, or results for experiments outside these ten runs.

## Exact variants and conditions

Question directories:

```text
hallucination_v2
hallucination_v2_metatoken
hallucination_v2_no_position_bias
hallucination_v2_no_position_bias_metatoken
```

Prediction directories include the above four baseline/prompt conditions and:

```text
hallucination_v2_temporal_shuffle_cd
hallucination_v2_temporal_shuffle_cd_metatoken
hallucination_v2_temporal_shuffle_cd_alpha05
hallucination_v2_temporal_shuffle_cd_alpha05_metatoken
hallucination_v2_temporal_shuffle_cd_alpha15
hallucination_v2_temporal_shuffle_cd_alpha15_metatoken
```

Each has object existence, temporal grounding, ego-relative spatial, motion/action and distance/depth results. Original predictions, prompts, answers and historical score fields remain byte-for-byte unchanged. JSONL carries the most convenient nested options and run metadata. The original `correct`/`parsed_prediction` fields are legacy scoring; use the analysis package for strict rescoring rather than silently changing the saved records.

Only `.DS_Store` and `__MACOSX` metadata are omitted from the extracted directories. The unchanged ZIPs preserve even these entries. No scientific records are omitted or reformatted.

## Offline verification

From the repository root:

```bash
python scripts/materialize_paper_data.py
```

The command checks both ZIP hashes, compares every materialized file with its archived bytes, verifies all record counts, checks one-to-one prediction/question/prompt/options/answer correspondence, and compares CSV with JSONL. It refuses to overwrite modified source data. `--download` is only a recovery/import option when original ZIPs are absent; it is not needed after cloning this repository.

Do not treat `conversations` containing reference answers as an inference prompt. Only the human/question prompt is model input. This is an evaluation corpus, not an additional training split. Historical `split: test` strings mean project evaluation, not the official nuScenes test split.

## External raw data

See [nuScenes download and validation guide](../docs/NUSCENES_DOWNLOAD_ZH.md). Saved-response analysis needs neither raw LiDAR nor checkpoints. Rendering real examples requires raw LiDAR plus metadata; new model inference also needs the learned feature cache/runtime/weights. The original scene metadata and encoder configuration remain relevant to exact inference reproduction.

No new rights are granted over inherited B4DL or nuScenes material. Review [NOTICE.md](../NOTICE.md) before changing repository visibility or redistributing the data.
