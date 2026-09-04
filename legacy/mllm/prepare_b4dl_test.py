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


def convert_rows(rows, task, known_scene_ids=None):
    converted = []
    skipped = 0
    for row in rows:
        scene_id = row.get("scene_id")
        question = row.get("question")
        answer = row.get("answer")
        if not scene_id or not question:
            skipped += 1
            continue
        if known_scene_ids is not None and scene_id not in known_scene_ids:
            skipped += 1
            continue

        item = {
            "source": "B4DL",
            "task": task,
            "split": row.get("split", "test"),
            "scene_token": row.get("scene_token"),
            "scene_id": scene_id,
            "question": question,
            "answer": answer,
        }
        if answer is not None:
            item["conversations"] = conversation(question, answer)
        converted.append(item)
    return converted, skipped


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(repo_root / "nuScenes-B4DL" / "dataset" / "test"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "mllm" / "b4dl_dataset" / "test"),
    )
    parser.add_argument(
        "--scene-metadata",
        default=str(repo_root / "encoders" / "lidarclip" / "annotations" / "scene_metadata.json"),
    )
    parser.add_argument("--skip-unknown-scenes", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    known_scene_ids = None
    if args.skip_unknown_scenes:
        scene_metadata = load_json(args.scene_metadata)
        known_scene_ids = {scene["scene_id"] for scene in scene_metadata}

    all_rows = []
    for input_path in sorted(input_dir.glob("*.json")):
        task = input_path.stem
        rows = load_json(input_path)
        converted, skipped = convert_rows(rows, task, known_scene_ids)
        save_json(output_dir / input_path.name, converted)
        all_rows.extend(converted)
        print(f"{task}: wrote {len(converted)} rows, skipped {skipped}")

    save_json(output_dir / "all.json", all_rows)
    print(f"all: wrote {len(all_rows)} rows to {output_dir / 'all.json'}")


if __name__ == "__main__":
    main()
