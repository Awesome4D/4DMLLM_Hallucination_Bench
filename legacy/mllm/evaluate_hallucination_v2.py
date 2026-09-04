import argparse
import csv
import json
import re
import string
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from evaluate_simple_tasks import load_model  # noqa: E402
from infer_b4dl import generate_answer, load_features  # noqa: E402


CATEGORY_FILES = {
    "object_existence": "object_existence.json",
    "temporal_grounding": "temporal_grounding.json",
    "ego_relative_spatial": "ego_relative_spatial.json",
    "motion_action": "motion_action.json",
    "distance_depth": "distance_depth.json",
}

OPTION_LETTERS = ["A", "B", "C", "D"]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "category",
        "question_type",
        "scene_id",
        "frame",
        "start_frame",
        "end_frame",
        "object_class",
        "question",
        "reference_answer",
        "correct_answer",
        "correct_option",
        "prediction",
        "parsed_prediction",
        "correct",
        "invalid",
        "error",
    ]
    extra = sorted({key for row in rows for key in row if key not in fieldnames})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = text.strip(string.whitespace + string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_yes_no(text):
    normalized = normalize_text(text)
    if normalized == "yes" or normalized.startswith("yes "):
        return "Yes."
    if normalized == "no" or normalized.startswith("no "):
        return "No."
    match = re.search(r"\b(?:answer|choice|response)\s*(?:is|:|-)?\s*(yes|no)\b", normalized)
    if match:
        return "Yes." if match.group(1) == "yes" else "No."
    return None


def parse_mcq(text, options):
    raw = str(text or "").strip()
    normalized = normalize_text(raw)

    # Prefer an explicit option letter at the beginning.
    match = re.match(r"^\s*[\(\[]?\s*([abcd])\s*[\)\].:\-]?", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Accept "option A", "answer is B", "choice: C", etc.
    match = re.search(r"\b(?:option|answer|choice)\s*(?:is|:|-)?\s*([abcd])\b", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Accept a single standalone choice letter when the answer has light wrapping text.
    letter_matches = re.findall(r"\b([abcd])\b", raw, flags=re.IGNORECASE)
    unique_letters = sorted({letter.upper() for letter in letter_matches})
    if len(unique_letters) == 1:
        return unique_letters[0]

    # Fall back to option text matching.
    matches = []
    for letter, option in options.items():
        option_norm = normalize_text(option)
        if normalized == option_norm:
            matches.append(letter)
        elif option_norm and re.search(r"\b" + re.escape(option_norm) + r"\b", normalized):
            matches.append(letter)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def get_prompt(item):
    if item.get("prompt"):
        return item["prompt"]
    conversations = item.get("conversations", [])
    for turn in conversations:
        if turn.get("from") == "human":
            return turn.get("value")
    return item.get("question", "")


def score_prediction(item, prediction, error):
    question_type = item.get("question_type")
    invalid = bool(error)
    parsed = None

    if question_type == "binary":
        parsed = parse_yes_no(prediction)
        if parsed is None:
            invalid = True
        correct = parsed == item.get("answer")
    elif question_type == "mcq":
        parsed = parse_mcq(prediction, item.get("options", {}))
        if parsed is None:
            invalid = True
        correct = parsed == item.get("correct_option")
    else:
        invalid = True
        correct = False

    return {
        "parsed_prediction": parsed or "",
        "correct": bool(correct),
        "invalid": bool(invalid),
    }


def summarize_rows(rows):
    total = len(rows)
    invalid = sum(1 for row in rows if row.get("invalid"))
    correct = sum(1 for row in rows if row.get("correct"))
    summary = {
        "total": total,
        "valid_predictions": total - invalid,
        "invalid_predictions": invalid,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }

    by_type = {}
    for question_type in sorted({row.get("question_type") for row in rows}):
        subset = [row for row in rows if row.get("question_type") == question_type]
        type_total = len(subset)
        type_invalid = sum(1 for row in subset if row.get("invalid"))
        type_correct = sum(1 for row in subset if row.get("correct"))
        type_summary = {
            "total": type_total,
            "valid_predictions": type_total - type_invalid,
            "invalid_predictions": type_invalid,
            "correct": type_correct,
            "accuracy": type_correct / type_total if type_total else 0.0,
        }
        if question_type == "binary":
            yes_answers = sum(1 for row in subset if row.get("parsed_prediction") == "Yes.")
            yes_rows = [row for row in subset if row.get("reference_answer") == "Yes."]
            no_rows = [row for row in subset if row.get("reference_answer") == "No."]
            type_summary.update(
                {
                    "yes_answer_rate": yes_answers / type_total if type_total else 0.0,
                    "yes_accuracy": (
                        sum(1 for row in yes_rows if row.get("correct")) / len(yes_rows)
                        if yes_rows
                        else 0.0
                    ),
                    "no_accuracy": (
                        sum(1 for row in no_rows if row.get("correct")) / len(no_rows)
                        if no_rows
                        else 0.0
                    ),
                }
            )
        by_type[question_type] = type_summary
    summary["by_question_type"] = by_type
    return summary


def run_category(category, input_path, args, tokenizer, model, feature_cache):
    rows = load_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    evaluated = []
    for idx, item in enumerate(tqdm(rows, desc=category)):
        scene_id = item.get("scene_id")
        prompt = get_prompt(item)
        output = {
            "index": idx,
            "category": category,
            "question_type": item.get("question_type"),
            "scene_id": scene_id,
            "frame": item.get("frame"),
            "start_frame": item.get("start_frame"),
            "end_frame": item.get("end_frame"),
            "object_class": item.get("object_class"),
            "question": item.get("question"),
            "prompt": prompt,
            "reference_answer": item.get("answer"),
            "correct_answer": item.get("correct_answer"),
            "correct_option": item.get("correct_option"),
            "options": item.get("options"),
            "prediction": "",
            "error": "",
        }
        try:
            if scene_id not in feature_cache:
                feature_cache[scene_id] = load_features(args.feat_folder, scene_id, model.device)
            output["prediction"] = generate_answer(
                model,
                tokenizer,
                feature_cache[scene_id],
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.num_beams,
            )
        except Exception as exc:
            output["error"] = str(exc)
        output.update(score_prediction(item, output["prediction"], output["error"]))
        evaluated.append(output)
    return evaluated


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", default="./base_model/vicuna-v1-5-7b")
    parser.add_argument(
        "--pretrain-mm-mlp-adapter",
        dest="pretrain_mm_mlp_adapter",
        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin",
    )
    parser.add_argument("--stage2", default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", default="")
    parser.add_argument("--feat-folder", default="./b4dl/stage2_features")
    parser.add_argument("--input-dir", default="./b4dl_dataset/hallucination_v2")
    parser.add_argument("--output-dir", default="./b4dl_eval/hallucination_v2")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(CATEGORY_FILES),
        default=list(CATEGORY_FILES),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--num-beams", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    tokenizer, model = load_model(args)
    feature_cache = {}
    summary = {}

    for category in args.categories:
        rows = run_category(
            category,
            input_dir / CATEGORY_FILES[category],
            args,
            tokenizer,
            model,
            feature_cache,
        )
        write_jsonl(output_dir / f"{category}_predictions.jsonl", rows)
        write_csv(output_dir / f"{category}_predictions.csv", rows)
        summary[category] = summarize_rows(rows)

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
