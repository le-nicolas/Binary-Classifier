from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Sequence


def _to_binary(value: str) -> int:
    normalized = value.strip().lower()
    true_values = {"1", "true", "yes", "positive"}
    false_values = {"0", "false", "no", "negative"}

    if normalized in true_values:
        return 1
    if normalized in false_values:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def load_labeled_csv(
    path: str | Path,
    target_column: str = "label",
) -> tuple[list[list[float]], list[int], list[str]]:
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        if target_column not in reader.fieldnames:
            raise ValueError(f"Target column {target_column!r} not found in CSV header.")

        feature_names = [name for name in reader.fieldnames if name != target_column]
        if not feature_names:
            raise ValueError("No feature columns found.")

        rows: list[list[float]] = []
        labels: list[int] = []
        for row in reader:
            labels.append(_to_binary(str(row[target_column])))
            features = [float(row[name]) for name in feature_names]
            rows.append(features)

    if not rows:
        raise ValueError("CSV contains no data rows.")
    return rows, labels, feature_names


def split_train_test(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match.")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must be within [0, 1).")
    if not features:
        raise ValueError("Cannot split an empty dataset.")

    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    test_count = int(len(indices) * test_size)
    if test_size > 0 and test_count == 0 and len(indices) > 1:
        test_count = 1

    test_indices = set(indices[:test_count])
    x_train: list[list[float]] = []
    y_train: list[int] = []
    x_test: list[list[float]] = []
    y_test: list[int] = []

    for idx, (row, label) in enumerate(zip(features, labels)):
        if idx in test_indices:
            x_test.append(list(row))
            y_test.append(int(label))
        else:
            x_train.append(list(row))
            y_train.append(int(label))

    return x_train, y_train, x_test, y_test
