import argparse
import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def conversation(question, answer):
    return [
        {"from": "human", "value": "<video>\n" + question},
        {"from": "gpt", "value": answer},
    ]


def convert_rows(rows, source_stage):
    converted = []
    skipped = 0
    for row in rows:
        scene_id = row.get("scene_id")
        question = row.get("question")
        answer = row.get("answer")
        if not scene_id or not question or not answer:
            skipped += 1
            continue
        converted.append(
            {
                "source": "B4DL",
                "source_stage": source_stage,
                "split": row.get("split"),
                "scene_token": row.get("scene_token"),
                "scene_id": scene_id,
                "human_annotation": row.get("human_annotation"),
                "conversations": conversation(question, answer),
            }
        )
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
        "--out",
        default=str(repo_root / "mllm" / "b4dl_dataset" / "stage2.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    stage2_rows, skipped_stage2 = convert_rows(load_json(args.stage2_in), "stage2")
    stage3_rows, skipped_stage3 = convert_rows(load_json(args.stage3_in), "stage3")
    combined = stage2_rows + stage3_rows

    save_json(args.out, combined)

    print(f"stage2 source: converted {len(stage2_rows)}, skipped {skipped_stage2}")
    print(f"stage3 source: converted {len(stage3_rows)}, skipped {skipped_stage3}")
    print(f"combined: wrote {len(combined)} rows to {args.out}")


if __name__ == "__main__":
    main()
