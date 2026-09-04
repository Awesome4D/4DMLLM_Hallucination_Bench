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
from infer_b4dl import load_features, normalize_question  # noqa: E402
from vtimellm.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from vtimellm.conversation import SeparatorStyle, conv_templates  # noqa: E402
from vtimellm.mm_utils import tokenizer_image_token  # noqa: E402


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


def shuffled_features(features, seed):
    if features.shape[0] <= 1:
        return features, list(range(features.shape[0]))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(features.shape[0], generator=generator).tolist()
    index = torch.tensor(permutation, device=features.device, dtype=torch.long)
    return features.index_select(0, index), permutation


def build_input_ids(tokenizer, question, device):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], normalize_question(question))
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return input_ids, stop_str


def next_token_logits(model, input_ids, images, past_key_values, attention_mask):
    outputs = model(
        input_ids=input_ids,
        images=images,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    return outputs.logits[:, -1, :], outputs.past_key_values


def sample_next_token(logits, temperature):
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_answer_temporal_shuffle_cd(
    model,
    tokenizer,
    features,
    question,
    max_new_tokens,
    temperature,
    contrastive_alpha,
    shuffle_seed,
):
    shuffled, permutation = shuffled_features(features, shuffle_seed)
    input_ids, stop_str = build_input_ids(tokenizer, question, model.device)
    clean_images = features.unsqueeze(0)
    shuffled_images = shuffled.unsqueeze(0)

    clean_past = None
    shuffled_past = None
    generated = []
    next_input_ids = input_ids

    with torch.inference_mode():
        for step in range(max_new_tokens):
            logical_length = input_ids.shape[1] + step
            attention_mask = torch.ones(
                (1, logical_length),
                dtype=torch.long,
                device=model.device,
            )
            clean_logits, clean_past = next_token_logits(
                model,
                next_input_ids,
                clean_images,
                clean_past,
                attention_mask,
            )
            shuffled_logits, shuffled_past = next_token_logits(
                model,
                next_input_ids,
                shuffled_images,
                shuffled_past,
                attention_mask,
            )
            logits = clean_logits + contrastive_alpha * (clean_logits - shuffled_logits)
            next_token = sample_next_token(logits, temperature)
            token_id = int(next_token.item())
            generated.append(token_id)

            if token_id == tokenizer.eos_token_id:
                break

            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            if stop_str in decoded:
                break

            next_input_ids = next_token

    output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if stop_str in output:
        output = output.split(stop_str, 1)[0].strip()
    return output, permutation


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
            "mitigation": "temporal_shuffle_contrastive_decoding",
            "contrastive_alpha": args.contrastive_alpha,
            "shuffle_seed": args.shuffle_seed + idx,
            "shuffle_permutation": "",
        }
        try:
            if scene_id not in feature_cache:
                feature_cache[scene_id] = load_features(args.feat_folder, scene_id, model.device)
            torch.manual_seed(args.generation_seed + idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.generation_seed + idx)
            output["prediction"], permutation = generate_answer_temporal_shuffle_cd(
                model,
                tokenizer,
                feature_cache[scene_id],
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.contrastive_alpha,
                args.shuffle_seed + idx,
            )
            output["shuffle_permutation"] = " ".join(str(x) for x in permutation)
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
    parser.add_argument("--output-dir", default="./b4dl_eval/hallucination_v2_temporal_shuffle_cd")
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
    parser.add_argument("--contrastive-alpha", type=float, default=1.0)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_beams != 1:
        raise ValueError("Temporal shuffle contrastive decoding currently supports --num-beams 1 only.")
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
