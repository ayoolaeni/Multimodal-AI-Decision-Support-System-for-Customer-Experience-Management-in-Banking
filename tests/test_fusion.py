import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fusion import FusionModel, rank_customers  # noqa: E402
from utils import load_config  # noqa: E402


def _fitted_model():
    rng = np.random.default_rng(0)
    n = 300
    labels = (rng.random(n) < 0.35).astype(int)
    triplets = []
    for label in labels:
        base = 0.75 if label == 1 else 0.25
        triplets.append(tuple(np.clip(rng.normal(base, 0.15, 3), 0.01, 0.99)))
    model = FusionModel(load_config())
    model.fit(triplets, labels.tolist(), seed=0)
    return model


NULL_TEXT = {"text_score": None, "text_label": None, "confidence": None, "driving_text": None}
NULL_VOICE = {"voice_score": None, "voice_label": None, "confidence": None, "transcript": None}
NULL_TXN = {"txn_score": None, "top_features": None, "confidence": None}


def test_process_with_all_three_modalities_present():
    model = _fitted_model()
    text_out = {"text_score": 0.9, "text_label": "unfair_charge", "confidence": 0.9, "driving_text": "..."}
    voice_out = {"voice_score": 0.2, "voice_label": "other", "confidence": 0.8, "transcript": "..."}
    txn_out = {"txn_score": 0.3, "top_features": [("declines_count", 0.1)], "confidence": 0.7}
    out = model.process(text_out, voice_out, txn_out)
    assert 0.0 <= out["risk_score"] <= 1.0
    assert set(out["contributing"]) == {"text", "voice", "txn"}
    assert all(v is not None for v in out["contributing"].values())
    assert out["dominant_modality"] == "text"
    assert out["next_best_action"] != ""


def test_process_never_crashes_with_one_or_two_missing_modalities():
    model = _fitted_model()
    text_out = {"text_score": 0.85, "text_label": "card_declined", "confidence": 0.85, "driving_text": "..."}
    for voice_out, txn_out in [(NULL_VOICE, NULL_TXN), (NULL_VOICE, {"txn_score": 0.4, "top_features": [("declines_count", 0.1)], "confidence": 0.6})]:
        out = model.process(text_out, voice_out, txn_out)
        assert 0.0 <= out["risk_score"] <= 1.0
        assert out["contributing"]["text"] is not None


def test_missing_modality_is_redistributed_not_zeroed():
    """A customer with only a strong text signal should not be penalised for
    missing voice/txn the way a naive 'treat missing as 0' scheme would."""
    model = _fitted_model()
    text_out = {"text_score": 0.95, "text_label": "unfair_charge", "confidence": 0.95, "driving_text": "..."}
    full = model.process(text_out,
                          {"voice_score": 0.95, "voice_label": "other", "confidence": 0.9, "transcript": "..."},
                          {"txn_score": 0.95, "top_features": [("declines_count", 0.1)], "confidence": 0.9})
    text_only = model.process(text_out, NULL_VOICE, NULL_TXN)
    naive_zero = model.process(text_out,
                                {"voice_score": 0.0, "voice_label": "other", "confidence": 0.9, "transcript": "..."},
                                {"txn_score": 0.0, "top_features": [("declines_count", 0.1)], "confidence": 0.9})
    assert text_only["risk_score"] > naive_zero["risk_score"]


def test_low_risk_action_when_all_scores_low():
    model = FusionModel(load_config())
    model.weights = {"text": 1.0, "voice": 1.0, "txn": 1.0}
    model.intercept = -3.0
    out = model.process(
        {"text_score": 0.05, "text_label": "other", "confidence": 0.9, "driving_text": "..."},
        NULL_VOICE,
        {"txn_score": 0.05, "top_features": [("declines_count", 0.0)], "confidence": 0.9},
    )
    assert out["next_best_action"] == model.action_map["low_risk"]


def test_rank_customers_assigns_priority_by_descending_risk():
    results = [
        {"risk_score": 0.2, "customer_id": "A"},
        {"risk_score": 0.9, "customer_id": "B"},
        {"risk_score": 0.5, "customer_id": "C"},
    ]
    ranked = rank_customers(results)
    assert [r["customer_id"] for r in ranked] == ["B", "C", "A"]
    assert [r["priority_rank"] for r in ranked] == [1, 2, 3]
