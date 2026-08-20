# VLM Hallucination Study

A preliminary experimental study on hallucination and visual reasoning errors in Vision-Language Models (VLMs).

This project is currently under active development as an undergraduate research practice project. The current goal is to build a reproducible evaluation pipeline for analyzing different types of errors in multimodal large language models.

## Overview

Vision-Language Models have demonstrated strong capabilities in image understanding and visual question answering. However, these models may generate answers that are inconsistent with the actual visual content.

This project aims to conduct a small-scale empirical study of such errors, with a particular focus on four types of visual question answering tasks:

* **Object**: whether a specific object exists in the image
* **Attribute**: whether an object or person has a specific visual attribute
* **Counting**: the number of objects or people in an image
* **Relation**: spatial or semantic relationships between objects

The current implementation uses **Qwen2.5-VL-3B-Instruct** as the initial evaluation model.

## Project Structure

```text
vlm-hallucination-study/
│
├── data/
│   ├── images/
│   └── samples.json
│
├── figures/
│
├── notes/
│
├── results/
│   └── predictions.json
│
├── src/
│   ├── inference.py
│   ├── batch_inference.py
│   └── evaluate.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Current Pipeline

The current evaluation pipeline supports:

1. Loading a local image
2. Running multimodal inference with Qwen2.5-VL
3. Reading multiple questions from `samples.json`
4. Performing batch evaluation without repeatedly loading the model
5. Normalizing model outputs
6. Comparing predictions with manually annotated ground-truth answers
7. Saving predictions to JSON files
8. Computing overall and category-wise accuracy

## Preliminary Sanity Check

The current pipeline has been tested on a small sanity-check example containing one image and four questions.

| Category  | Number of Questions | Correct |
| --------- | ------------------: | ------: |
| Object    |                   1 |       1 |
| Attribute |                   1 |       1 |
| Counting  |                   1 |       1 |
| Relation  |                   1 |       1 |
| **Total** |               **4** |   **4** |

Current sanity-check accuracy:

**4 / 4 = 100%**

> This result is only used to verify that the evaluation pipeline works correctly. Due to the extremely small sample size, it should not be interpreted as a statistically meaningful evaluation result.

## Example Questions

### Object

```text
Is there a car in the image?
Answer only yes or no.
```

### Attribute

```text
Is the front cyclist wearing a black top?
Answer only yes or no.
```

### Counting

```text
How many people are in the image?
Answer with a number only.
```

### Relation

```text
Are both people riding bicycles?
Answer only yes or no.
```

## Model

The current experiment uses:

**Qwen/Qwen2.5-VL-3B-Instruct**

The model is loaded using Hugging Face Transformers and automatically distributed between GPU and CPU when necessary.

The current local experimental environment includes an NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM.

## Installation

It is recommended to create an isolated Python environment before installing dependencies.

Example:

```bash
conda create -n vlm-study python=3.11
conda activate vlm-study
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Single-image inference

```bash
python src/inference.py
```

### Batch evaluation

```bash
python src/batch_inference.py
```

The predictions will be saved to:

```text
results/predictions.json
```

### Evaluation

```bash
python src/evaluate.py
```

## Dataset Format

Questions are stored in `data/samples.json`.

Example:

```json
{
    "id": "001",
    "image": "test.jpg",
    "category": "object",
    "question": "Is there a car in the image? Answer only yes or no.",
    "ground_truth": "no"
}
```

## Evaluation Metrics

The current implementation primarily reports:

* Overall Accuracy
* Overall Error Rate
* Category-wise Accuracy

At the current stage, general prediction errors are **not directly treated as hallucinations**.

A more rigorous definition of object hallucination and corresponding hallucination-specific metrics will be introduced in later experiments.

## Planned Experiments

The next stage of this project will focus on expanding the evaluation dataset and improving the experimental methodology.

Planned work includes:

* Expanding the dataset to approximately 80–100 visual questions
* Balancing positive and negative questions
* Evaluating different difficulty levels
* Comparing Object, Attribute, Counting, and Relation performance
* Analyzing representative failure cases
* Introducing dedicated hallucination metrics
* Visualizing category-wise results
* Comparing different prompting strategies
* Evaluating additional Vision-Language Models

## Research Questions

Several preliminary research questions will be explored:

1. Which type of visual question is most likely to produce incorrect answers?
2. Are counting tasks more difficult than object-presence questions?
3. Do small or partially occluded objects increase model error rates?
4. Does question formulation influence hallucination behavior?
5. Can simple prompting strategies reduce visual reasoning errors?

## Status

🚧 **Work in Progress**

This repository currently contains an early-stage experimental pipeline and is being continuously improved.

## Notes

Images used during early development are intended only for pipeline testing.

Future formal experiments will preferentially use images from public datasets or data with clearly defined usage permissions to ensure reproducibility and appropriate data usage.

## Author

Long Wuxin
Major: Cryptography Science and Technology
