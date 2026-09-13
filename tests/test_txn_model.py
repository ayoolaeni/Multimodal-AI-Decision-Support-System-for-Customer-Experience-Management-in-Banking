import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from txn_model import TxnModel  # noqa: E402
from utils import load_config  # noqa: E402


def _synthetic_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.35).astype(int)
    feats = []
    for label in labels:
        if label == 1:
            f = {
                "failed_transfers_count": int(rng.poisson(3)),
                "declines_count": int(rng.poisson(2)),
                "txn_frequency_change": float(rng.normal(-0.3, 0.1)),
                "app_sessions_change": float(rng.normal(-0.3, 0.1)),
                "days_since_last_complaint": float(rng.exponential(9)),
                "avg_resolution_delay": float(rng.normal(70, 20)),
            }
        else:
            f = {
                "failed_transfers_count": int(rng.poisson(0.3)),
                "declines_count": int(rng.poisson(0.2)),
                "txn_frequency_change": float(rng.normal(0.05, 0.1)),
                "app_sessions_change": float(rng.normal(0.02, 0.1)),
                "days_since_last_complaint": float(rng.exponential(60)),
                "avg_resolution_delay": float(rng.normal(12, 8)),
            }
        feats.append(f)
    return feats, labels.tolist()


def test_fit_beats_majority_baseline_on_held_out_f1():
    feats, labels = _synthetic_data(n=300)
    train_feats, test_feats = feats[:200], feats[200:]
    train_labels, test_labels = labels[:200], labels[200:]

    model = TxnModel(load_config())
    model.fit(train_feats, train_labels, seed=42)
    scores = model.predict_proba(test_feats)
    preds = (scores >= 0.5).astype(int)
    model_f1 = f1_score(test_labels, preds, zero_division=0)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train_labels), 1)), train_labels)
    dummy_preds = dummy.predict(np.zeros((len(test_labels), 1)))
    dummy_f1 = f1_score(test_labels, dummy_preds, zero_division=0)

    assert model_f1 > dummy_f1


def test_process_is_null_safe_on_missing_txn():
    model = TxnModel(load_config())
    feats, labels = _synthetic_data(n=100)
    model.fit(feats, labels, seed=42)
    result = model.process(None)
    assert result == {"txn_score": None, "top_features": None, "confidence": None}


def test_process_returns_valid_score_and_top_features():
    model = TxnModel(load_config())
    feats, labels = _synthetic_data(n=100)
    model.fit(feats, labels, seed=42)
    result = model.process(feats[0])
    assert 0.0 <= result["txn_score"] <= 1.0
    assert len(result["top_features"]) == 3
    assert all(isinstance(name, str) and isinstance(contrib, float) for name, contrib in result["top_features"])
