"""Sprint 4 - Fusion + decision layer (README S6.4). Core contribution.

Late / hybrid fusion: each modality is scored independently (text_model,
voice_model, txn_model), and a small logistic-regression meta-model learns
how to weigh the three scores from training data. When a modality is
missing, its learned weight is redistributed proportionally across the
present modalities - a missing signal is never treated as a 0 (README S3.2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config

MODALITIES = ("text", "voice", "txn")

# Maps a complaint/transcript category (from text_model.categorize) to an
# action_map key. Falls back to "text_<category>" naming convention.
_TXN_FEATURE_ACTION_KEY = {
    "failed_transfers_count": "txn_failed_transfers",
    "declines_count": "txn_declines",
    "app_sessions_change": "txn_app_activity",
    "txn_frequency_change": "txn_app_activity",
    "days_since_last_complaint": "txn_failed_transfers",
    "avg_resolution_delay": "txn_failed_transfers",
}


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


class FusionModel:
    def __init__(self, config: dict | None = None):
        self.full_cfg = config or load_config()
        self.action_map: dict[str, str] = self.full_cfg.get("action_map", {})
        self.low_risk_threshold = 0.4
        self.weights = {m: 0.0 for m in MODALITIES}
        self.intercept = 0.0

    def fit(self, triplets: list[tuple[float, float, float]], labels: list[int], seed: int = 42) -> "FusionModel":
        """triplets: list of (text_score, voice_score, txn_score) for
        customers where all three modalities were present in training data."""
        X = np.array(triplets, dtype=float)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X, labels)
        self.weights = {"text": float(clf.coef_[0][0]), "voice": float(clf.coef_[0][1]), "txn": float(clf.coef_[0][2])}
        self.intercept = float(clf.intercept_[0])
        return self

    def _combine(self, scores: dict[str, float | None]) -> tuple[float, dict[str, float | None]]:
        present = {m: s for m, s in scores.items() if s is not None}
        if not present:
            return 0.5, {m: None for m in MODALITIES}

        sum_present_w = sum(self.weights[m] for m in present)
        sum_all_w = sum(self.weights.values())
        # Redistribute any missing modality's weight proportionally across
        # the present ones, rather than substituting a 0 score.
        scale = (sum_all_w / sum_present_w) if sum_present_w != 0 else 1.0

        z = self.intercept + scale * sum(self.weights[m] * present[m] for m in present)
        risk_score = _sigmoid(z)

        contributing = {}
        for m in MODALITIES:
            if m in present:
                contributing[m] = round(self.weights[m] * present[m] * scale, 4)
            else:
                contributing[m] = None
        return risk_score, contributing

    def _dominant_modality(self, contributing: dict[str, float | None]) -> str | None:
        present = {m: c for m, c in contributing.items() if c is not None}
        if not present:
            return None
        return max(present, key=present.get)

    def _next_best_action(
        self,
        dominant: str | None,
        risk_score: float,
        text_out: dict,
        voice_out: dict,
        txn_out: dict,
    ) -> str:
        if dominant is None or risk_score < self.low_risk_threshold:
            return self.action_map.get("low_risk", "No urgent action.")

        if dominant == "txn" and txn_out.get("top_features"):
            top_feature_name = txn_out["top_features"][0][0]
            key = _TXN_FEATURE_ACTION_KEY.get(top_feature_name, "txn_failed_transfers")
            return self.action_map.get(key, self.action_map.get("low_risk", ""))

        if dominant == "text" and text_out.get("text_label"):
            key = f"text_{text_out['text_label']}"
            return self.action_map.get(key, self.action_map.get("voice_negative_call", ""))

        if dominant == "voice" and voice_out.get("voice_label"):
            key = f"text_{voice_out['voice_label']}"
            return self.action_map.get(key, self.action_map.get("voice_negative_call", ""))

        return self.action_map.get("low_risk", "No urgent action.")

    def process(self, text_out: dict, voice_out: dict, txn_out: dict) -> dict:
        """Combines the three per-modality outputs (any may be the null-safe
        dict from a missing modality) into one decision-support record."""
        scores = {
            "text": text_out.get("text_score"),
            "voice": voice_out.get("voice_score"),
            "txn": txn_out.get("txn_score"),
        }
        risk_score, contributing = self._combine(scores)
        dominant = self._dominant_modality(contributing)
        action = self._next_best_action(dominant, risk_score, text_out, voice_out, txn_out)

        return {
            "risk_score": round(risk_score, 4),
            "priority_rank": None,  # filled in by rank_customers() over a batch
            "next_best_action": action,
            "contributing": contributing,
            "dominant_modality": dominant,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"weights": self.weights, "intercept": self.intercept, "low_risk_threshold": self.low_risk_threshold},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, config: dict | None = None) -> "FusionModel":
        inst = cls(config)
        state = joblib.load(path)
        inst.weights = state["weights"]
        inst.intercept = state["intercept"]
        inst.low_risk_threshold = state["low_risk_threshold"]
        return inst


def rank_customers(results: list[dict]) -> list[dict]:
    """Assigns priority_rank (1 = highest risk) over a batch, in place-ish
    (returns new list, does not mutate input dicts)."""
    ordered = sorted(results, key=lambda r: r["risk_score"], reverse=True)
    out = []
    for i, r in enumerate(ordered, start=1):
        r2 = dict(r)
        r2["priority_rank"] = i
        out.append(r2)
    return out
