# Binary Classifier

This repository now contains a working binary classifier project implemented from
scratch in Python (no scikit-learn).

It is focused on a rocket stove setting: predicting whether an operating condition
is likely to produce a negative outcome (`label = 1`) such as excessive smoke or
inefficient combustion.

## What is included

- Logistic regression implementation written from scratch
- CSV data loading and preprocessing
- Train/test split and classification metrics
- Model save/load as JSON
- CLI for training and batch prediction
- Unit tests
- Sample rocket stove dataset

## Quick start

1. Train a model:

```bash
python -m binary_classifier.cli train ^
  --data data/rocket_stove_sample.csv ^
  --target label ^
  --model-out model.json
```

2. Run predictions:

```bash
python -m binary_classifier.cli predict ^
  --model model.json ^
  --input data/rocket_stove_sample.csv ^
  --output predictions.csv
```

3. Run tests:

```bash
python -m unittest discover -s tests -v
```

## Data format

Training CSV must include numeric features and a binary target column.

Example:

```csv
temperature_c,smoke_ppm,fuel_moisture_pct,airflow_mps,label
430,48,10,2.2,0
320,160,24,0.8,1
```

Target values accepted by the loader: `0/1`, `true/false`, `yes/no`,
`negative/positive` (case-insensitive).

## CLI reference

Train:

```bash
python -m binary_classifier.cli train --help
```

Predict:

```bash
python -m binary_classifier.cli predict --help
```

## Notes

- This project is intentionally lightweight and dependency-free.
- It is suitable as a baseline model and can be replaced later with more advanced
  methods once more data is available.
