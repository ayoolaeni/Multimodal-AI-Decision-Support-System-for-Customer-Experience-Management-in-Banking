"""Sprint 4 - Transaction modality (README S6.3).

Logistic regression over engineered behavioural features. Logistic
regression (rather than a black-box booster) is chosen deliberately so that
per-customer feature contributions are exact and cheap (coefficient x
standardised value), which the interpretability requirement (README S3.3)
needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config


class TxnModel:
    def __init__(self, config: dict | None = None):
        self.cfg = (config or load_config())["txn_model"]
        self.features: list[str] = self.cfg["features"]
        self.pipeline: Pipeline | None = None
        self.medians: pd.Series | None = None

    def _to_frame(self, txn_features_list: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(txn_features_list)
        for f in self.features:
            if f not in df.columns:
                df[f] = np.nan
        return df[self.features]

    def fit(self, txn_features_list: list[dict], labels: list[int], seed: int = 42) -> "TxnModel":
        X = self._to_frame(txn_features_list)
        self.medians = X.median()
        X = X.fillna(self.medians)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
        self.pipeline.fit(X, labels)
        return self

    def predict_proba(self, txn_features_list: list[dict]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("TxnModel is not fit/loaded - call fit() or load() first.")
        X = self._to_frame(txn_features_list).fillna(self.medians)
        return self.pipeline.predict_proba(X)[:, 1]

    def process(self, txn_features: dict | None) -> dict:
        """Null-safe single-customer scoring with per-feature contributions
        for the interpretability panel."""
        if txn_features is None:
            return {"txn_score": None, "top_features": None, "confidence": None}

        X = self._to_frame([txn_features]).fillna(self.medians)
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]

        x_scaled = scaler.transform(X)[0]
        contributions = clf.coef_[0] * x_scaled
        score = float(self.pipeline.predict_proba(X)[0, 1])
        confidence = float(max(score, 1 - score))

        order = np.argsort(-np.abs(contributions))
        top_features = [(self.features[i], round(float(contributions[i]), 4)) for i in order[:3]]

        return {"txn_score": score, "top_features": top_features, "confidence": confidence}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "medians": self.medians, "features": self.features}, path)

    @classmethod
    def load(cls, path: str | Path, config: dict | None = None) -> "TxnModel":
        inst = cls(config)
        state = joblib.load(path)
        inst.pipeline = state["pipeline"]
        inst.medians = state["medians"]
        inst.features = state["features"]
        return inst
