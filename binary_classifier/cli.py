from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import load_labeled_csv, split_train_test
from .metrics import classification_metrics
from .model import BinaryLogisticRegression


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary classifier CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from labeled CSV data")
    train_parser.add_argument("--data", required=True, help="Path to CSV training data")
    train_parser.add_argument("--target", default="label", help="Target column name")
    train_parser.add_argument("--model-out", default="model.json", help="Output model JSON path")
    train_parser.add_argument("--epochs", type=int, default=1500, help="Number of training epochs")
    train_parser.add_argument("--learning-rate", type=float, default=0.1, help="Gradient descent step size")
    train_parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization weight")
    train_parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    train_parser.add_argument("--test-size", type=float, default=0.25, help="Test split fraction")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    train_parser.set_defaults(func=_run_train)

    predict_parser = subparsers.add_parser("predict", help="Generate predictions from a model")
    predict_parser.add_argument("--model", required=True, help="Path to model JSON")
    predict_parser.add_argument("--input", required=True, help="CSV input with feature columns")
    predict_parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    predict_parser.set_defaults(func=_run_predict)

    return parser


def _run_train(args: argparse.Namespace) -> int:
    features, labels, feature_names = load_labeled_csv(args.data, target_column=args.target)
    x_train, y_train, x_test, y_test = split_train_test(
        features,
        labels,
        test_size=args.test_size,
        seed=args.seed,
    )

    if not x_train:
        raise ValueError("Train set is empty. Reduce --test-size.")

    model = BinaryLogisticRegression.train(
        x_train,
        y_train,
        feature_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        threshold=args.threshold,
    )

    eval_x = x_test if x_test else x_train
    eval_y = y_test if y_test else y_train
    predictions = model.predict(eval_x)
    metrics = classification_metrics(eval_y, predictions)
    model.save(args.model_out)

    summary = {
        "model_path": str(Path(args.model_out).resolve()),
        "feature_count": len(feature_names),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    model = BinaryLogisticRegression.load(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")

        missing = [name for name in model.feature_names if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")

        rows = list(reader)

    feature_rows = [[float(row[name]) for name in model.feature_names] for row in rows]
    probabilities = model.predict_probabilities(feature_rows)
    predictions = [1 if score >= model.threshold else 0 for score in probabilities]

    base_columns = list(reader.fieldnames)
    output_columns = base_columns + ["predicted_probability", "predicted_label"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        for row, score, prediction in zip(rows, probabilities, predictions):
            merged = dict(row)
            merged["predicted_probability"] = f"{score:.6f}"
            merged["predicted_label"] = str(prediction)
            writer.writerow(merged)

    result = {
        "input_rows": len(rows),
        "output_path": str(output_path.resolve()),
        "threshold": model.threshold,
    }
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
