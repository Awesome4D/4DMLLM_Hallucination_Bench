import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from evaluate_simple_tasks import (  # noqa: E402
    TASK_FILES,
    evaluate_prediction,
    load_json,
    load_model,
    print_summary,
    summarize_task,
    write_csv,
    write_json,
    write_jsonl,
)
from infer_b4dl import generate_answer, load_features  # noqa: E402
from infer_b4dl_metatoken import get_prompt_question_and_answer  # noqa: E402


def run_task(task, input_path, args, tokenizer, model, feature_cache):
    rows = load_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    evaluated = []
    progress = tqdm(rows, desc=task)
    for idx, item in enumerate(progress):
        scene_id = item.get("scene_id") or item.get("id")
        prompt, raw_question, reference = get_prompt_question_and_answer(item)
        output = {
            "index": idx,
            "task": task,
            "scene_id": scene_id,
            "question": raw_question,
            "prompt": prompt,
            "metatoken": item.get("metatoken"),
            "reference_answer": reference,
            "prediction": "",
            "error": "",
        }
        try:
            if scene_id not in feature_cache:
                feature_cache[scene_id] = load_features(
                    args.feat_folder, scene_id, model.device
                )
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

        output.update(
            evaluate_prediction(
                task,
                output["reference_answer"],
                output["prediction"],
                error=output["error"],
            )
        )
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
    parser.add_argument(
        "--stage2",
        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-metatoken",
    )
    parser.add_argument("--stage3", default="")
    parser.add_argument("--feat-folder", default="./b4dl/stage2_features")
    parser.add_argument("--input-dir", default="./b4dl_dataset/test_metatoken")
    parser.add_argument("--output-dir", default="./b4dl_eval/simple_tasks_metatoken")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_FILES),
        default=["existence", "binary", "time_grounding"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--num-beams", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)

    tokenizer, model = load_model(args)
    feature_cache = {}
    summary = {}

    for task in args.tasks:
        input_path = input_dir / TASK_FILES[task]
        task_rows = run_task(task, input_path, args, tokenizer, model, feature_cache)
        write_jsonl(output_dir / f"{task}_predictions.jsonl", task_rows)
        write_csv(output_dir / f"{task}_predictions.csv", task_rows)
        summary[task] = summarize_task(task, task_rows)

    write_json(output_dir / "summary.json", summary)
    print_summary(summary)
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
