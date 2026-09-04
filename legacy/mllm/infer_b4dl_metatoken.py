import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from b4dl_metatoken import MetatokenBuilder, require_nuscenes_root  # noqa: E402
from infer_b4dl import generate_answer, load_features, write_jsonl  # noqa: E402
from vtimellm.model.builder import load_pretrained_model  # noqa: E402
from vtimellm.utils import disable_torch_init  # noqa: E402


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def get_prompt_question_and_answer(item):
    raw_question = item.get("question")
    prompt = item.get("prompt")
    answer = item.get("answer")

    conversations = item.get("conversations", [])
    if prompt is None:
        for turn in conversations:
            if turn.get("from") == "human":
                prompt = turn.get("value")
                break

    if raw_question is None:
        raw_question = prompt
        if raw_question:
            raw_question = raw_question.replace("<video>", "").replace("<4DLiDAR>", "")
            raw_question = raw_question.replace("<meta>", "").strip()

    if answer is None:
        for turn in conversations:
            if turn.get("from") == "gpt":
                answer = turn.get("value")
                break

    return prompt, raw_question, answer


def load_model(args):
    disable_torch_init()
    stage3 = args.stage3 or None
    tokenizer, model, _ = load_pretrained_model(args, args.stage2, stage3)
    model = model.cuda().eval()
    model.to(torch.float16)
    return tokenizer, model


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", default="./base_model/vicuna-v1-5-7b")
    parser.add_argument(
        "--pretrain-mm-mlp-adapter",
        dest="pretrain_mm_mlp_adapter",
        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin",
    )
    parser.add_argument(
        "--stage2",
        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-metatoken",
    )
    parser.add_argument("--stage3", default="")
    parser.add_argument("--feat-folder", default="./b4dl/stage2_features")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output-jsonl", default="./b4dl_predictions_metatoken.jsonl")
    parser.add_argument(
        "--scene-metadata",
        default=str(repo_root / "nuScenes-B4DL" / "metadata" / "scene_metadata.json"),
    )
    parser.add_argument("--nuscenes-root", default=None)
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

    tokenizer, model = load_model(args)

    if args.question:
        builder = MetatokenBuilder(args.scene_metadata, require_nuscenes_root(args.nuscenes_root))
        prompt, metatoken = builder.prompt(args.scene_id, args.question)
        features = load_features(args.feat_folder, args.scene_id, model.device)
        answer = generate_answer(
            model,
            tokenizer,
            features,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.num_beams,
        )
        print(
            json.dumps(
                {
                    "scene_id": args.scene_id,
                    "question": args.question,
                    "prompt": prompt,
                    "metatoken": metatoken,
                    "prediction": answer,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    rows = load_json(args.input_json)
    if args.limit is not None:
        rows = rows[: args.limit]

    outputs = []
    feature_cache = {}
    for idx, item in enumerate(tqdm(rows)):
        scene_id = item.get("scene_id") or item.get("id")
        prompt, raw_question, reference = get_prompt_question_and_answer(item)
        result = {
            "index": idx,
            "task": item.get("task"),
            "scene_id": scene_id,
            "question": raw_question,
            "prompt": prompt,
            "metatoken": item.get("metatoken"),
            "reference_answer": reference,
        }
        try:
            if scene_id not in feature_cache:
                feature_cache[scene_id] = load_features(args.feat_folder, scene_id, model.device)
            result["prediction"] = generate_answer(
                model,
                tokenizer,
                feature_cache[scene_id],
                prompt,
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
