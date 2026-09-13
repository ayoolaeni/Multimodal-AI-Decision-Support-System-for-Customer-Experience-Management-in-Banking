import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import metrics, to_pred  # noqa: E402


def test_to_pred_thresholds_at_half():
    scores = np.array([0.1, 0.49, 0.5, 0.51, 0.9])
    assert to_pred(scores).tolist() == [0, 0, 1, 1, 1]


def test_metrics_perfect_prediction():
    y = [0, 1, 1, 0, 1]
    m = metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_metrics_returns_expected_keys():
    m = metrics([0, 1, 0, 1], [1, 1, 0, 0])
    assert set(m) == {"accuracy", "precision", "recall", "f1"}
