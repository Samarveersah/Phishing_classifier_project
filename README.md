# Phishing URL Detector

This project implements a phishing URL classifier using a hybrid deep learning pipeline in PyTorch, with a Streamlit interface for interactive prediction.

## Overview

The model combines two signal sources:

- Character-level CNN features learned directly from raw URL strings
- Handcrafted URL risk features based on domain and path patterns

These representations are fused in a single classifier for binary prediction (`phishing` vs `legitimate`).

## Core capabilities

- URL preprocessing and character vocabulary building
- Hybrid CNN model training with validation tracking
- Test-set evaluation (accuracy, precision, recall, F1, ROC-AUC)
- Model artifact export for inference
- Streamlit app for real-time scoring and feature display

## Dataset format

Input data must include:

- `url`
- one label column: `label`, `status`, `target`, or `class`

Supported label values:

- `0`, `legitimate`, `benign`, `good`
- `1`, `phishing`, `malicious`, `bad`

Accepted file types:

- `.csv`
- `.parquet`

Default lookup order:

- `data/raw/phishing_dataset.csv`
- `phishing_dataset.csv`
- `Training.parquet`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --data-path Training.parquet --epochs 8 --batch-size 128
```

Training outputs:

- `artifacts/hybrid_cnn_model.pt`
- `artifacts/vocab.json`
- `artifacts/config.json`
- `artifacts/metrics.json`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## Run the app

```bash
streamlit run app/streamlit_app.py
```
