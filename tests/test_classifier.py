from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from binary_classifier.metrics import classification_metrics
from binary_classifier.model import BinaryLogisticRegression


class BinaryClassifierTests(unittest.TestCase):
    def test_model_learns_linearly_separable_data(self) -> None:
        rng = random.Random(7)
        features: list[list[float]] = []
        labels: list[int] = []
        for _ in range(300):
            x1 = rng.uniform(-2.5, 2.5)
            x2 = rng.uniform(-2.5, 2.5)
            label = 1 if (x1 + x2) > 0 else 0
            features.append([x1, x2])
            labels.append(label)

        model = BinaryLogisticRegression.train(
            features,
            labels,
            feature_names=["x1", "x2"],
            epochs=1600,
            learning_rate=0.12,
        )

        predictions = model.predict(features)
        metrics = classification_metrics(labels, predictions)
        self.assertGreater(metrics["accuracy"], 0.95)

    def test_model_round_trip_serialization(self) -> None:
        features = [[0.1, 0.2], [0.5, 0.9], [1.2, 1.3], [1.5, 1.8]]
        labels = [0, 0, 1, 1]

        model = BinaryLogisticRegression.train(
            features,
            labels,
            feature_names=["a", "b"],
            epochs=1200,
            learning_rate=0.15,
        )

        probe = [0.9, 1.0]
        original_score = model.predict_probability(probe)

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "model.json"
            model.save(target)
            loaded = BinaryLogisticRegression.load(target)

        reloaded_score = loaded.predict_probability(probe)
        self.assertAlmostEqual(original_score, reloaded_score, places=10)


if __name__ == "__main__":
    unittest.main()
