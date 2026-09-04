import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from vtimellm.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from vtimellm.conversation import SeparatorStyle, conv_templates  # noqa: E402
from vtimellm.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token  # noqa: E402
from vtimellm.model.builder import load_pretrained_model  # noqa: E402
from vtimellm.utils import disable_torch_init  # noqa: E402


def load_features(feat_folder, scene_id, device):
    path = Path(feat_folder) / f"{scene_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature file: {path}")
    features = torch.from_numpy(np.load(path))
    if features.ndim == 1:
        features = features.unsqueeze(0)
    return features.to(device=device, dtype=torch.float16)


def normalize_question(question):
    question = question.strip()
    if question.startswith("<video>"):
        return question
    return "<video>\n" + question


def generate_answer(model, tokenizer, features, question, max_new_tokens, temperature, num_beams):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], normalize_question(question))
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=features.unsqueeze(0),
            do_sample=True,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    output = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0].strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()
    return output


def get_question_and_answer(item):
    if "question" in item:
        return item["question"], item.get("answer")
    conversations = item.get("conversations", [])
    question = None
    answer = None
    for turn in conversations:
        if turn.get("from") == "human" and question is None:
            question = turn.get("value", "").replace("<video>", "").strip()
        elif turn.get("from") == "gpt" and answer is None:
            answer = turn.get("value")
    return question, answer


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output-jsonl", default="./b4dl_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--num-beams", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.question) == bool(args.input_json):
        raise ValueError("Use either --question for one custom query or --input-json for batch inference.")
    if args.question and not args.scene_id:
        raise ValueError("--scene-id is required with --question.")

    disable_torch_init()
    stage3 = args.stage3 or None
    tokenizer, model, _ = load_pretrained_model(args, args.stage2, stage3)
    model = model.cuda().eval()
    model.to(torch.float16)

    if args.question:
        features = load_features(args.feat_folder, args.scene_id, model.device)
        answer = generate_answer(
            model,
            tokenizer,
            features,
            args.question,
            args.max_new_tokens,
            args.temperature,
            args.num_beams,
        )
        print(json.dumps({"scene_id": args.scene_id, "question": args.question, "prediction": answer}, indent=2))
        return

    rows = json.load(open(args.input_json, "r"))
    if args.limit is not None:
        rows = rows[: args.limit]

    outputs = []
    feature_cache = {}
    for idx, item in enumerate(tqdm(rows)):
        scene_id = item.get("scene_id") or item.get("id")
        question, reference = get_question_and_answer(item)
        result = {
            "index": idx,
            "task": item.get("task"),
            "scene_id": scene_id,
            "question": question,
            "reference_answer": reference,
        }
        try:
            if scene_id not in feature_cache:
                feature_cache[scene_id] = load_features(args.feat_folder, scene_id, model.device)
            result["prediction"] = generate_answer(
                model,
                tokenizer,
                feature_cache[scene_id],
                question,
                args.max_new_tokens,
                args.temperature,
                args.num_beams,
            )
        except Exception as exc:
            result["error"] = str(exc)
        outputs.append(result)

    write_jsonl(args.output_jsonl, outputs)
    print(f"Wrote {len(outputs)} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
