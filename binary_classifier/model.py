from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _fit_standardizer(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    if not rows:
        raise ValueError("Cannot fit standardizer on empty rows.")

    column_count = len(rows[0])
    means = [0.0] * column_count
    for row in rows:
        if len(row) != column_count:
            raise ValueError("Inconsistent feature count in rows.")
        for index, value in enumerate(row):
            means[index] += value

    row_count = len(rows)
    means = [value / row_count for value in means]

    variances = [0.0] * column_count
    for row in rows:
        for index, value in enumerate(row):
            diff = value - means[index]
            variances[index] += diff * diff

    stds = [math.sqrt(value / row_count) for value in variances]
    stds = [std if std > 0 else 1.0 for std in stds]
    return means, stds


def _scale_row(row: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> list[float]:
    if len(row) != len(means) or len(row) != len(stds):
        raise ValueError("Row length does not match standardizer dimensions.")
    return [(value - mean) / std for value, mean, std in zip(row, means, stds)]


@dataclass
class BinaryLogisticRegression:
    feature_names: list[str]
    weights: list[float]
    bias: float
    means: list[float]
    stds: list[float]
    threshold: float = 0.5

    @classmethod
    def train(
        cls,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        feature_names: Sequence[str],
        *,
        epochs: int = 1500,
        learning_rate: float = 0.1,
        l2: float = 0.0,
        threshold: float = 0.5,
    ) -> "BinaryLogisticRegression":
        if len(features) != len(labels):
            raise ValueError("Feature and label counts must match.")
        if not features:
            raise ValueError("Training features are empty.")
        if not 0 < learning_rate:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if len(feature_names) != len(features[0]):
            raise ValueError("feature_names count does not match feature dimension.")

        for label in labels:
            if label not in (0, 1):
                raise ValueError("Labels must be 0 or 1.")

        means, stds = _fit_standardizer(features)
        scaled_rows = [_scale_row(row, means, stds) for row in features]

        feature_count = len(feature_names)
        sample_count = len(scaled_rows)
        weights = [0.0] * feature_count
        bias = 0.0

        for _ in range(epochs):
            gradients = [0.0] * feature_count
            bias_gradient = 0.0

            for row, target in zip(scaled_rows, labels):
                linear_output = sum(weight * value for weight, value in zip(weights, row)) + bias
                prediction = _sigmoid(linear_output)
                error = prediction - target

                for index, value in enumerate(row):
                    gradients[index] += error * value
                bias_gradient += error

            for index in range(feature_count):
                gradients[index] = gradients[index] / sample_count + (l2 * weights[index])
                weights[index] -= learning_rate * gradients[index]
            bias -= learning_rate * (bias_gradient / sample_count)

        return cls(
            feature_names=list(feature_names),
            weights=weights,
            bias=bias,
            means=means,
            stds=stds,
            threshold=threshold,
        )

    def predict_probability(self, row: Sequence[float]) -> float:
        scaled = _scale_row(row, self.means, self.stds)
        linear_output = sum(weight * value for weight, value in zip(self.weights, scaled)) + self.bias
        return _sigmoid(linear_output)

    def predict_probabilities(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_probability(row) for row in rows]

    def predict(self, rows: Sequence[Sequence[float]]) -> list[int]:
        probabilities = self.predict_probabilities(rows)
        return [1 if value >= self.threshold else 0 for value in probabilities]

    def save(self, path: str | Path) -> None:
        model_path = Path(path)
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BinaryLogisticRegression":
        model_path = Path(path)
        with model_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            feature_names=list(payload["feature_names"]),
            weights=[float(value) for value in payload["weights"]],
            bias=float(payload["bias"]),
            means=[float(value) for value in payload["means"]],
            stds=[float(value) for value in payload["stds"]],
            threshold=float(payload.get("threshold", 0.5)),
        )
