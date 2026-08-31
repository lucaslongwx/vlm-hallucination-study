import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_IMAGE_DIR = Path(r"D:\COCO\val2017")
COCO_ANNOTATIONS = Path(r"D:\COCO\annotations\instances_val2017.json")
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
METADATA_PATH = PROJECT_ROOT / "data" / "image_metadata.json"
SAMPLES_PATH = PROJECT_ROOT / "data" / "samples_coco_auto.json"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def count_bin(count):
    if count <= 3:
        return str(count)
    if count <= 5:
        return "4-5"
    return "6+"


def evenly_spaced(items, count):
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def select_records(coco):
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)

    records = []
    for image in coco["images"]:
        source = COCO_IMAGE_DIR / image["file_name"]
        if not source.is_file():
            continue
        image_annotations = annotations[image["id"]]
        usable = [
            item
            for item in image_annotations
            if not item.get("iscrowd") and item.get("area", 0) > 0
        ]
        if not usable:
            continue

        image_area = image["width"] * image["height"]
        small_count = sum(item["area"] / image_area < 0.02 for item in usable)
        category_counts = Counter(item["category_id"] for item in usable)
        crowd_categories = {
            item["category_id"] for item in image_annotations if item.get("iscrowd")
        }
        countable_counts = {
            category_id: count
            for category_id, count in category_counts.items()
            if category_id not in crowd_categories
        }
        if not countable_counts:
            continue

        category_areas = defaultdict(float)
        for item in usable:
            category_areas[item["category_id"]] += item["area"]

        records.append(
            {
                "image": image,
                "source": source,
                "present_category_ids": {item["category_id"] for item in image_annotations},
                "category_counts": dict(category_counts),
                "countable_counts": countable_counts,
                "category_areas": dict(category_areas),
                "annotation_count": len(usable),
                "small_object_count": small_count,
                "selection_score": len(usable) + 2 * small_count + len(category_counts),
            }
        )

    records.sort(key=lambda item: (item["selection_score"], item["image"]["id"]))
    if len(records) < 20:
        raise RuntimeError(f"Only {len(records)} usable COCO images were found")

    easy = records[:5]
    hard = records[-5:]
    hard_ids = {item["image"]["id"] for item in hard}
    middle = [item for item in records[5:] if item["image"]["id"] not in hard_ids]
    medium = evenly_spaced(middle, 10)
    return [(item, difficulty) for difficulty, group in (("easy", easy), ("medium", medium), ("hard", hard)) for item in group]


def build_samples(selected, categories):
    common_negative_ids = [
        category_id
        for name in ("person", "car", "chair", "dog", "bicycle", "truck", "bus", "cat", "bottle", "dining table")
        for category_id, category_name in categories.items()
        if category_name == name
    ]
    samples = []
    used_count_bins = Counter()

    for index, (record, difficulty) in enumerate(selected, start=1):
        image_name = f"img{index:03d}.jpg"
        positive_id = max(record["category_areas"], key=record["category_areas"].get)
        negative_pool = common_negative_ids + sorted(categories)
        negative_id = next(
            category_id
            for category_id in negative_pool
            if category_id not in record["present_category_ids"]
        )

        count_id, count = min(
            record["countable_counts"].items(),
            key=lambda item: (used_count_bins[count_bin(item[1])], -record["category_areas"][item[0]], item[0]),
        )
        used_count_bins[count_bin(count)] += 1

        base = {
            "image": image_name,
            "image_id": record["image"]["id"],
            "difficulty": difficulty,
        }
        samples.extend(
            [
                {
                    **base,
                    "id": f"coco-{index:03d}-object-positive",
                    "category": "object",
                    "object_polarity": "positive",
                    "target_category": categories[positive_id],
                    "question": f'Is any object of the category "{categories[positive_id]}" present in the image? Answer only yes or no.',
                    "ground_truth": "yes",
                },
                {
                    **base,
                    "id": f"coco-{index:03d}-object-negative",
                    "category": "object",
                    "object_polarity": "negative",
                    "target_category": categories[negative_id],
                    "question": f'Is any object of the category "{categories[negative_id]}" present in the image? Answer only yes or no.',
                    "ground_truth": "no",
                },
                {
                    **base,
                    "id": f"coco-{index:03d}-counting",
                    "category": "counting",
                    "target_category": categories[count_id],
                    "count_bin": count_bin(count),
                    "question": f'How many objects of the category "{categories[count_id]}" are in the image? Answer with a number only.',
                    "ground_truth": str(count),
                },
            ]
        )
    return samples


def self_check(metadata, samples):
    assert len(metadata) == 20
    assert Counter(item["difficulty"] for item in metadata) == {"easy": 5, "medium": 10, "hard": 5}
    assert len(samples) == 60
    assert Counter(item["category"] for item in samples) == {"object": 40, "counting": 20}
    assert Counter(item.get("object_polarity") for item in samples if item["category"] == "object") == {"positive": 20, "negative": 20}
    metadata_by_image = {item["output_image"]: item for item in metadata}
    for sample in samples:
        item = metadata_by_image[sample["image"]]
        present = item["present_categories"]
        if sample["category"] == "object":
            assert (sample["target_category"] in present) == (sample["ground_truth"] == "yes")
        else:
            assert item["countable_category_counts"][sample["target_category"]] == int(sample["ground_truth"])


def main():
    if not COCO_IMAGE_DIR.is_dir() or not COCO_ANNOTATIONS.is_file():
        raise FileNotFoundError("COCO val2017 images or instances_val2017.json are missing")

    with COCO_ANNOTATIONS.open("r", encoding="utf-8") as file:
        coco = json.load(file)
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    selected = select_records(coco)
    samples = build_samples(selected, categories)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index, (record, difficulty) in enumerate(selected, start=1):
        destination = IMAGE_DIR / f"img{index:03d}.jpg"
        shutil.copy2(record["source"], destination)
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            image.convert("RGB").load()

        metadata.append(
            {
                "output_image": destination.name,
                "image_id": record["image"]["id"],
                "source_file": record["source"].name,
                "width": record["image"]["width"],
                "height": record["image"]["height"],
                "difficulty": difficulty,
                "annotation_count": record["annotation_count"],
                "small_object_count": record["small_object_count"],
                "selection_score": record["selection_score"],
                "present_categories": sorted(categories[item] for item in record["present_category_ids"]),
                "countable_category_counts": {
                    categories[item]: count for item, count in sorted(record["countable_counts"].items())
                },
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    self_check(metadata, samples)
    atomic_json(METADATA_PATH, metadata)
    atomic_json(SAMPLES_PATH, samples)
    print(f"Selected {len(metadata)} readable images: 5 easy, 10 medium, 5 hard")
    print(f"Generated {len(samples)} COCO-only questions: 40 object, 20 counting")
    print("Counting bins:", dict(sorted(Counter(item["count_bin"] for item in samples if item["category"] == "counting").items())))


if __name__ == "__main__":
    main()
