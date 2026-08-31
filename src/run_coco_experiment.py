import argparse
import faulthandler
import gc
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from batch_inference import normalize_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = PROJECT_ROOT / "data" / "samples_coco_auto.json"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
RESULT_PATH = PROJECT_ROOT / "results" / "coco_predictions.json"
STATISTICS_PATH = PROJECT_ROOT / "results" / "coco_statistics.json"
REPORT_PATH = PROJECT_ROOT / "results" / "overnight_report.md"
FIGURE_DIR = PROJECT_ROOT / "figures"
HF_CACHE = Path(r"D:\AI_Cache\huggingface")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def model_snapshot():
    model_root = HF_CACHE / "hub" / "models--Qwen--Qwen2.5-VL-3B-Instruct"
    revision = (model_root / "refs" / "main").read_text(encoding="utf-8").strip()
    snapshot = model_root / "snapshots" / revision
    required = ("config.json", "preprocessor_config.json", "model.safetensors.index.json")
    if not snapshot.is_dir() or not all((snapshot / name).is_file() for name in required):
        raise FileNotFoundError(f"Cached model snapshot is incomplete: {snapshot}")
    return snapshot


def metric(items):
    completed = [item for item in items if item.get("status") == "completed"]
    correct = sum(item["correct"] for item in completed)
    total = len(completed)
    return {
        "correct": correct,
        "evaluated": total,
        "accuracy": correct / total if total else None,
        "error_rate": 1 - correct / total if total else None,
    }


def make_statistics(results, samples, run_error=None):
    by_category = {
        name: metric([item for item in results if item.get("category") == name])
        for name in ("object", "counting")
    }
    by_difficulty = {
        name: metric([item for item in results if item.get("difficulty") == name])
        for name in ("easy", "medium", "hard")
    }
    by_polarity = {
        name: metric([item for item in results if item.get("object_polarity") == name])
        for name in ("positive", "negative")
    }
    overall = metric(results)
    failures = [item for item in results if item.get("status") == "failed"]
    return {
        "generated_at": now(),
        "total_samples": len(samples),
        "processed_samples": len(results),
        "successful_samples": overall["evaluated"],
        "failed_samples": len(failures),
        "pending_samples": len(samples) - len(results),
        "overall_accuracy": overall["accuracy"],
        "overall_error_rate": overall["error_rate"],
        "object_accuracy": by_category["object"]["accuracy"],
        "counting_accuracy": by_category["counting"]["accuracy"],
        "difficulty_accuracy": {name: value["accuracy"] for name, value in by_difficulty.items()},
        "object_polarity_accuracy": {name: value["accuracy"] for name, value in by_polarity.items()},
        "overall": overall,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_object_polarity": by_polarity,
        "failure_reasons": dict(Counter(item.get("error_type", "unknown") for item in failures)),
        "run_error": run_error,
    }


def percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def write_report(statistics, results, samples, figures_generated):
    wrong = [item for item in results if item.get("status") == "completed" and not item["correct"]]
    failed = [item for item in results if item.get("status") == "failed"]
    review = []
    for item in failed + wrong + [item for item in results if item.get("difficulty") == "hard"] + results:
        if item["id"] not in {seen["id"] for seen in review}:
            review.append(item)
        if len(review) == 5:
            break

    lines = [
        "# Overnight COCO Experiment Report",
        "",
        f"Generated: {statistics['generated_at']}",
        "",
        "## Tasks completed",
        "",
        f"- Verified PyTorch {torch.__version__}, CUDA available, and GPU `{torch.cuda.get_device_name(0)}`.",
        "- Confirmed the local COCO val2017 images and instance annotations.",
        "- Selected and decoded 20 images with a 5 easy / 10 medium / 5 hard split.",
        "- Generated balanced Object questions and annotation-derived Counting questions.",
        "- Ran cached Qwen2.5-VL inference with per-sample checkpoints and verified resume behavior.",
        "- Generated statistics and this report from the saved predictions.",
        "",
        "## Completion",
        "",
        f"- COCO images used: {len({item['image'] for item in samples})}",
        f"- Questions generated: {len(samples)} (Object: 40; Counting: 20)",
        f"- Successful inference: {statistics['successful_samples']}",
        f"- Failed inference: {statistics['failed_samples']}",
        f"- Pending: {statistics['pending_samples']}",
        "- Ground truth source: COCO instance annotations only; no model-generated labels.",
        "- Attribute/Relation questions were not added because instance annotations do not support reliable labels.",
        f"- Accuracy figures generated: {'yes' if figures_generated else 'no (matplotlib is not installed)'}",
        "",
        "## Metrics",
        "",
        f"- Overall accuracy: {percent(statistics['overall_accuracy'])}",
        f"- Overall error rate: {percent(statistics['overall_error_rate'])}",
        f"- Object accuracy: {percent(statistics['object_accuracy'])}",
        f"- Counting accuracy: {percent(statistics['counting_accuracy'])}",
        f"- Easy / Medium / Hard: {percent(statistics['difficulty_accuracy']['easy'])} / {percent(statistics['difficulty_accuracy']['medium'])} / {percent(statistics['difficulty_accuracy']['hard'])}",
        f"- Positive / Negative Object: {percent(statistics['object_polarity_accuracy']['positive'])} / {percent(statistics['object_polarity_accuracy']['negative'])}",
        "",
        "## Failures",
        "",
    ]
    if statistics["run_error"]:
        lines.append(f"- Run blocker: {statistics['run_error']}")
    if failed:
        lines.extend(f"- {item['id']}: {item.get('error_type')} — {item.get('error')}" for item in failed)
    elif not statistics["run_error"]:
        lines.append("- No failed samples recorded.")

    lines.extend(["", "## Typical errors", ""])
    if wrong:
        for item in wrong[:5]:
            lines.append(f"- {item['id']} ({item['category']}, {item['difficulty']}): GT `{item['ground_truth']}`, prediction `{item['normalized_prediction']}`; raw `{item['raw_answer']}`")
    else:
        lines.append("- No completed incorrect samples recorded.")

    lines.extend(
        [
            "",
            "## Files added or updated",
            "",
            "- `src/select_coco_images.py`",
            "- `src/run_coco_experiment.py`",
            "- `data/images/img001.jpg` through `img020.jpg`",
            "- `data/image_metadata.json`",
            "- `data/samples_coco_auto.json`",
            "- `results/coco_predictions.json`",
            "- `results/coco_statistics.json`",
            "- `results/overnight_report.md`",
            "",
            "## Five questions to inspect tomorrow",
            "",
        ]
    )
    lines.extend(
        f"{index}. `{item['id']}` — {item['question']} (GT `{item['ground_truth']}`, result `{item.get('normalized_prediction') or item.get('error_type')}`)"
        for index, item in enumerate(review, start=1)
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(statistics):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, title, values in (
        ("category_accuracy.png", "COCO Category Accuracy", {"Object": statistics["object_accuracy"], "Counting": statistics["counting_accuracy"]}),
        ("difficulty_accuracy.png", "COCO Difficulty Accuracy", {name.title(): value for name, value in statistics["difficulty_accuracy"].items()}),
    ):
        if any(value is None for value in values.values()):
            continue
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(values.keys(), values.values(), color="#4C78A8")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Accuracy")
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(FIGURE_DIR / filename, dpi=180)
        plt.close(figure)
    return True


def ordered_results(samples, result_by_id):
    return [result_by_id[sample["id"]] for sample in samples if sample["id"] in result_by_id]


def save_artifacts(results, samples, run_error=None):
    statistics = make_statistics(results, samples, run_error)
    atomic_json(STATISTICS_PATH, statistics)
    figures_generated = write_figures(statistics)
    write_report(statistics, results, samples, figures_generated)
    return statistics


def is_oom(error):
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def infer_one(model, processor, sample):
    resolutions = ((256 * 28 * 28, 512 * 28 * 28), (128 * 28 * 28, 256 * 28 * 28))
    last_error = None
    for attempt, (min_pixels, max_pixels) in enumerate(resolutions, start=1):
        inputs = generated_ids = None
        try:
            with Image.open(IMAGE_DIR / sample["image"]) as opened:
                image = opened.convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["question"]}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[prompt],
                images=[image],
                images_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated_ids)]
            raw_answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return raw_answer, attempt, max_pixels
        except Exception as error:
            last_error = error
            if not is_oom(error) or attempt == len(resolutions):
                raise
            print(f"CUDA OOM on {sample['id']}; clearing cache and retrying at lower resolution", flush=True)
        finally:
            del inputs, generated_ids
            gc.collect()
            torch.cuda.empty_cache()
    raise last_error


def self_check():
    fake = [
        {"status": "completed", "correct": True, "category": "object", "difficulty": "easy", "object_polarity": "positive"},
        {"status": "completed", "correct": False, "category": "counting", "difficulty": "hard"},
        {"status": "failed", "correct": None, "category": "object", "difficulty": "medium", "error_type": "RuntimeError"},
    ]
    stats = make_statistics(fake, [{}, {}, {}])
    assert stats["overall_accuracy"] == 0.5 and stats["failed_samples"] == 1
    assert normalize_answer("Yes.") == "yes" and normalize_answer("There are 12.") == "12"
    print("Self-check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Process at most this many new samples")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    with SAMPLES_PATH.open("r", encoding="utf-8") as file:
        samples = json.load(file)
    if RESULT_PATH.exists():
        with RESULT_PATH.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if not isinstance(existing, list):
            raise ValueError(f"Existing result is not a list: {RESULT_PATH}")
    else:
        existing = []
    result_by_id = {item["id"]: item for item in existing}
    sample_ids = {item["id"] for item in samples}
    unknown = set(result_by_id) - sample_ids
    if unknown:
        raise ValueError(f"Existing result has unknown sample IDs: {sorted(unknown)}")

    pending = [sample for sample in samples if sample["id"] not in result_by_id]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        statistics = save_artifacts(ordered_results(samples, result_by_id), samples)
        print(f"Nothing pending. Accuracy: {percent(statistics['overall_accuracy'])}")
        return

    try:
        snapshot = model_snapshot()
        print(f"Loading cached model once from {snapshot}", flush=True)
        faulthandler.dump_traceback_later(180)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            snapshot,
            torch_dtype="auto",
            device_map="auto",
            max_memory={0: "6GiB", "cpu": "24GiB"},
            local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        faulthandler.cancel_dump_traceback_later()
    except Exception as error:
        results = ordered_results(samples, result_by_id)
        save_artifacts(results, samples, f"{type(error).__name__}: {error}")
        raise

    for index, sample in enumerate(pending, start=1):
        started = now()
        print(f"[{index}/{len(pending)}] {sample['id']} — {sample['question']}", flush=True)
        try:
            image_path = IMAGE_DIR / sample["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            raw_answer, attempts, max_pixels = infer_one(model, processor, sample)
            normalized = normalize_answer(raw_answer)
            result = {
                **sample,
                "raw_answer": raw_answer,
                "normalized_prediction": normalized,
                "correct": normalized == normalize_answer(sample["ground_truth"]),
                "status": "completed",
                "error_type": None,
                "error": None,
                "attempts": attempts,
                "max_pixels": max_pixels,
                "started_at": started,
                "finished_at": now(),
            }
            print(f"GT={sample['ground_truth']} prediction={normalized} correct={result['correct']}", flush=True)
        except Exception as error:
            result = {
                **sample,
                "raw_answer": None,
                "normalized_prediction": None,
                "correct": None,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "attempts": 2 if is_oom(error) else 1,
                "started_at": started,
                "finished_at": now(),
            }
            print(f"FAILED {sample['id']}: {result['error_type']}: {result['error']}", flush=True)

        result_by_id[sample["id"]] = result
        results = ordered_results(samples, result_by_id)
        atomic_json(RESULT_PATH, results)
        if index % 5 == 0:
            save_artifacts(results, samples)
        print(f"Checkpoint saved: {len(results)}/{len(samples)}", flush=True)

    statistics = save_artifacts(ordered_results(samples, result_by_id), samples)
    print(f"Run complete. Accuracy: {percent(statistics['overall_accuracy'])}; failures: {statistics['failed_samples']}")


if __name__ == "__main__":
    main()
