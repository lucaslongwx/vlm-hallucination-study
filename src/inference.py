from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "test.jpg"

QUESTION = "Are both people riding bicycles? Answer only yes or no."

def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"没有找到图片：{IMAGE_PATH}"
        )

    print("正在读取图片……")
    image = Image.open(IMAGE_PATH).convert("RGB")

    print("正在加载 Qwen2.5-VL 模型……")

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

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": QUESTION,
                },
            ],
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

    print("问题：", QUESTION)
    print("正在生成回答……")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n模型回答：")
    print(output_text[0])


if __name__ == "__main__":
    main()