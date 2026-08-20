import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLES_PATH = PROJECT_ROOT / "data" / "samples.json"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
RESULT_PATH = PROJECT_ROOT / "results" / "predictions.json"


def normalize_answer(text):
    """
    对答案进行简单标准化：
    Yes -> yes
    No. -> no
    2 -> 2
    """
    text = text.strip().lower()

    text = re.sub(r"[^\w\s]", "", text)

    if text.startswith("yes"):
        return "yes"

    if text.startswith("no"):
        return "no"

    numbers = re.findall(r"\d+", text)

    if numbers:
        return numbers[0]

    return text


def main():

    # -------------------------
    # 读取题目
    # -------------------------

    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"读取到 {len(samples)} 道测试题。")

    # -------------------------
    # 模型只加载一次
    # -------------------------

    print("正在加载 Qwen2.5-VL……")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    min_pixels = 256 * 28 * 28
    max_pixels = 768 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    results = []

    correct_count = 0

    # -------------------------
    # 开始逐题推理
    # -------------------------

    for index, sample in enumerate(samples, start=1):

        print("\n" + "=" * 60)
        print(f"[{index}/{len(samples)}]")
        print("ID:", sample["id"])
        print("Category:", sample["category"])
        print("Question:", sample["question"])

        image_path = IMAGE_DIR / sample["image"]

        if not image_path.exists():
            print("找不到图片：", image_path)
            continue

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": sample["question"]
                    }
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        with torch.inference_mode():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids
            in zip(inputs.input_ids, generated_ids)
        ]

        answer = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        prediction = normalize_answer(answer)
        ground_truth = normalize_answer(
            sample["ground_truth"]
        )

        correct = prediction == ground_truth

        if correct:
            correct_count += 1

        result = {
            "id": sample["id"],
            "image": sample["image"],
            "category": sample["category"],
            "question": sample["question"],
            "ground_truth": ground_truth,
            "raw_answer": answer,
            "prediction": prediction,
            "correct": correct
        }

        results.append(result)

        print("Ground Truth:", ground_truth)
        print("Model Answer:", answer)
        print("Normalized:", prediction)

        if correct:
            print("Result: ✅ Correct")
        else:
            print("Result: ❌ Wrong")

    # -------------------------
    # 保存结果
    # -------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4,
        )

    # -------------------------
    # 输出统计
    # -------------------------

    total = len(results)

    accuracy = (
        correct_count / total
        if total > 0
        else 0
    )

    print("\n" + "=" * 60)

    print("实验完成")

    print(
        f"正确题数：{correct_count}/{total}"
    )

    print(
        f"Accuracy：{accuracy:.2%}"
    )

    print(
        f"结果已保存到：{RESULT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()