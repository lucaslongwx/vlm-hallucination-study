# Overnight COCO Experiment Report

Generated: 2026-08-31T00:48:09+08:00

## Tasks completed

- Verified PyTorch 2.11.0+cu128, CUDA available, and GPU `NVIDIA GeForce RTX 5070 Laptop GPU`.
- Confirmed the local COCO val2017 images and instance annotations.
- Selected and decoded 20 images with a 5 easy / 10 medium / 5 hard split.
- Generated balanced Object questions and annotation-derived Counting questions.
- Ran cached Qwen2.5-VL inference with per-sample checkpoints and verified resume behavior.
- Generated statistics and this report from the saved predictions.

## Completion

- COCO images used: 20
- Questions generated: 60 (Object: 40; Counting: 20)
- Successful inference: 60
- Failed inference: 0
- Pending: 0
- Ground truth source: COCO instance annotations only; no model-generated labels.
- Attribute/Relation questions were not added because instance annotations do not support reliable labels.
- Accuracy figures generated: no (matplotlib is not installed)

## Metrics

- Overall accuracy: 85.00%
- Overall error rate: 15.00%
- Object accuracy: 100.00%
- Counting accuracy: 55.00%
- Easy / Medium / Hard: 100.00% / 86.67% / 66.67%
- Positive / Negative Object: 100.00% / 100.00%

## Failures

- No failed samples recorded.

## Typical errors

- coco-012-counting (counting, medium): GT `5`, prediction `4`; raw `4`
- coco-013-counting (counting, medium): GT `3`, prediction `2`; raw `2`
- coco-014-counting (counting, medium): GT `4`, prediction `2`; raw `2`
- coco-015-counting (counting, medium): GT `8`, prediction `0`; raw `0`
- coco-016-counting (counting, hard): GT `13`, prediction `10`; raw `10`

## Files added or updated

- `src/select_coco_images.py`
- `src/run_coco_experiment.py`
- `data/images/img001.jpg` through `img020.jpg`
- `data/image_metadata.json`
- `data/samples_coco_auto.json`
- `results/coco_predictions.json`
- `results/coco_statistics.json`
- `results/overnight_report.md`

## Five questions to inspect tomorrow

1. `coco-012-counting` — How many objects of the category "person" are in the image? Answer with a number only. (GT `5`, result `4`)
2. `coco-013-counting` — How many objects of the category "cup" are in the image? Answer with a number only. (GT `3`, result `2`)
3. `coco-014-counting` — How many objects of the category "person" are in the image? Answer with a number only. (GT `4`, result `2`)
4. `coco-015-counting` — How many objects of the category "car" are in the image? Answer with a number only. (GT `8`, result `0`)
5. `coco-016-counting` — How many objects of the category "person" are in the image? Answer with a number only. (GT `13`, result `10`)
