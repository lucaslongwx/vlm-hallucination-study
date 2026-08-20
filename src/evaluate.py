import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULT_PATH = PROJECT_ROOT / "results" / "predictions.json"


def main():

    with open(
        RESULT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        results = json.load(f)

    total = len(results)

    correct = sum(
        1 for item in results
        if item["correct"]
    )

    print("=" * 60)
    print("Overall Results")
    print("=" * 60)

    print(f"Total Questions: {total}")
    print(f"Correct: {correct}")

    if total > 0:
        accuracy = correct / total
        error_rate = 1 - accuracy

        print(f"Accuracy: {accuracy:.2%}")
        print(f"Error Rate: {error_rate:.2%}")

    # -------------------------
    # 按类别统计
    # -------------------------

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0
        }
    )

    for item in results:

        category = item["category"]

        category_stats[category]["total"] += 1

        if item["correct"]:
            category_stats[category]["correct"] += 1

    print("\n" + "=" * 60)
    print("Category Results")
    print("=" * 60)

    for category, stats in category_stats.items():

        category_total = stats["total"]
        category_correct = stats["correct"]

        category_accuracy = (
            category_correct / category_total
            if category_total > 0
            else 0
        )

        print(
            f"{category:12s} "
            f"{category_correct}/{category_total} "
            f"Accuracy: {category_accuracy:.2%}"
        )

    # -------------------------
    # 输出错误案例
    # -------------------------

    wrong_cases = [
        item
        for item in results
        if not item["correct"]
    ]

    print("\n" + "=" * 60)
    print("Wrong Cases")
    print("=" * 60)

    if len(wrong_cases) == 0:

        print("No wrong cases.")

    else:

        for item in wrong_cases:

            print("\nID:", item["id"])
            print("Category:", item["category"])
            print("Question:", item["question"])
            print("Ground Truth:", item["ground_truth"])
            print("Model Answer:", item["raw_answer"])


if __name__ == "__main__":
    main()