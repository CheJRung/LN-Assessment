# LIMA

## Project introduction

LIMA (Lymph Node Immune and Metastasis Assessment) is a computational pathology framework for lymph node analysis. It combines lymph node segmentation, metastasis classification, and germinal center quantification.

## Available code

This release provides a standalone demo for single-image germinal center instance segmentation. Full pipeline code will be released separately.

## Installation and model preparation

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Model weights are not included. Prepare a fine-tuned Mask2Former checkpoint for the single `center` class with the following files:

```text
best/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

Place the checkpoint in `best/`, or use `--model` to specify another local checkpoint directory.

## Usage

Run an example with the default model directory:

```bash
python demo.py examples/inputs/1.png
```

Specify a model directory and output path:

```bash
python demo.py examples/inputs/2.png --model path/to/checkpoint --output demo_outputs/2.png
```

| Input | Prediction |
| --- | --- |
| ![Input 1](examples/inputs/1.png) | ![Prediction 1](examples/outputs/1.png) |
| ![Input 2](examples/inputs/2.png) | ![Prediction 2](examples/outputs/2.png) |
