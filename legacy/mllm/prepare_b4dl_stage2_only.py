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


def convert_rows(rows):
    converted = []
    skipped = 0
    for row in rows:
        scene_id = row.get("scene_id")
        question = row.get("question")
        answer = row.get("answer")
        if not scene_id or not question or answer is None:
            skipped += 1
            continue

        converted.append(
            {
                "source": "B4DL",
                "source_stage": "stage2",
                "split": row.get("split"),
                "scene_token": row.get("scene_token"),
                "scene_id": scene_id,
                "human_annotation": row.get("human_annotation"),
                "question": question,
                "answer": answer,
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
        "--out",
        default=str(repo_root / "mllm" / "b4dl_dataset" / "stage2.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, skipped = convert_rows(load_json(args.stage2_in))
    save_json(args.out, rows)

    print(f"stage2 source: converted {len(rows)}, skipped {skipped}")
    print(f"stage2-only: wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
