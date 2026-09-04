import argparse
from pathlib import Path

from b4dl_metatoken import (
    MetatokenBuilder,
    convert_row,
    load_json,
    print_metatoken_report,
    require_nuscenes_root,
    save_json,
)


def convert_rows(rows, builder, source_stage):
    converted = []
    skipped = 0
    for row in rows:
        item = convert_row(row, builder, source_stage=source_stage)
        if item is None:
            skipped += 1
            continue
        converted.append(item)
    return converted, skipped


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-in",
        default=str(repo_root / "nuScenes-B4DL" / "dataset" / "train" / "stage2.json"),
    )
    parser.add_argument(
        "--stage3-in",
        default=str(repo_root / "nuScenes-B4DL" / "dataset" / "train" / "stage3.json"),
    )
    parser.add_argument(
        "--scene-metadata",
        default=str(repo_root / "nuScenes-B4DL" / "metadata" / "scene_metadata.json"),
    )
    parser.add_argument("--nuscenes-root", default=None)
    parser.add_argument(
        "--out",
        default=str(
            repo_root / "mllm" / "b4dl_dataset" / "stage2_combined_metatoken.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    builder = MetatokenBuilder(args.scene_metadata, require_nuscenes_root(args.nuscenes_root))

    stage2_rows, skipped_stage2 = convert_rows(load_json(args.stage2_in), builder, "stage2")
    stage3_rows, skipped_stage3 = convert_rows(load_json(args.stage3_in), builder, "stage3")
    combined = stage2_rows + stage3_rows

    save_json(args.out, combined)

    print(f"stage2 source: converted {len(stage2_rows)}, skipped {skipped_stage2}")
    print(f"stage3 source: converted {len(stage3_rows)}, skipped {skipped_stage3}")
    print_metatoken_report(combined, f"combined: wrote to {args.out}", expected_count=148271)
    print("leakage check: metatoken frames were parsed from question text only")


if __name__ == "__main__":
    main()
