"""Sprint 5 - Evaluation harness (README S7). This is where the thesis is proven.

Trains text-only, voice-only, transaction-only, and multimodal-fusion models
on the SAME stratified split, and reports the headline comparison:
multimodal F1 vs. each unimodal baseline's F1. Also runs a fairness check
(disaggregated by the `segment` field) and an interpretability check, and
writes a fully-scored dataset for the Streamlit app.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion import FusionModel, rank_customers
from txn_model import TxnModel
from text_model import TextModel
from voice_model import VoiceModel
from utils import ensure_dir, load_config, read_jsonl, resolve_path, set_global_seed

# Palette (validated categorical order, see dataviz skill reference)
COLOR_FUSION = "#2a78d6"  # slot 1 blue
COLOR_TEXT = "#eb6834"    # slot 2 orange
COLOR_VOICE = "#eda100"   # slot 4 yellow
COLOR_TXN = "#1baf7a"     # slot 3 aqua


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def to_pred(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])

    processed_path = resolve_path(cfg["paths"]["processed_dir"]) / "customers.jsonl"
    records = read_jsonl(processed_path)
    if not records:
        raise FileNotFoundError(f"No processed data at {processed_path} - run ingest.py then preprocess.py first.")

    labels = [r["label"] for r in records]
    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        idx, test_size=cfg["data"]["test_size"], stratify=labels, random_state=cfg["seed"]
    )
    train = [records[i] for i in train_idx]
    test = [records[i] for i in test_idx]
    print(f"[evaluate] train={len(train)} test={len(test)} (stratified on label, seed={cfg['seed']})")

    # ---- train text model -------------------------------------------------
    text_train_texts, text_train_labels = [], []
    for r in train:
        for t in r["text_records"]:
            text_train_texts.append(t)
            text_train_labels.append(r["label"])
    print(f"[evaluate] training TextModel on {len(text_train_texts)} text records...")
    text_model = TextModel(cfg)
    text_model.fit(text_train_texts, text_train_labels, seed=cfg["seed"])

    voice_model = VoiceModel(text_model, cfg)

    # ---- train txn model ---------------------------------------------------
    txn_train_feats = [r["txn_features"] for r in train if r["txn_features"] is not None]
    txn_train_labels = [r["label"] for r in train if r["txn_features"] is not None]
    print(f"[evaluate] training TxnModel on {len(txn_train_feats)} customers...")
    txn_model = TxnModel(cfg)
    txn_model.fit(txn_train_feats, txn_train_labels, seed=cfg["seed"])

    # ---- score every customer with each modality ---------------------------
    def score_all(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            t = text_model.process(r["text_records"])
            v = voice_model.process(transcripts=r["voice_transcripts"])
            x = txn_model.process(r["txn_features"])
            out.append({"customer_id": r["customer_id"], "label": r["label"], "segment": r["segment"],
                        "text": t, "voice": v, "txn": x})
        return out

    print("[evaluate] scoring train/test customers with each modality...")
    train_scored = score_all(train)
    test_scored = score_all(test)

    # ---- train fusion meta-model on fully-present training rows ------------
    fusion_train = [
        (s["text"]["text_score"], s["voice"]["voice_score"], s["txn"]["txn_score"])
        for s in train_scored
        if s["text"]["text_score"] is not None and s["voice"]["voice_score"] is not None and s["txn"]["txn_score"] is not None
    ]
    fusion_train_labels = [
        s["label"] for s in train_scored
        if s["text"]["text_score"] is not None and s["voice"]["voice_score"] is not None and s["txn"]["txn_score"] is not None
    ]
    print(f"[evaluate] training FusionModel on {len(fusion_train)} fully-present customers...")
    fusion_model = FusionModel(cfg)
    fusion_model.fit(fusion_train, fusion_train_labels, seed=cfg["seed"])
    print(f"[evaluate] fusion weights: {fusion_model.weights} intercept={fusion_model.intercept:.4f}")

    # ---- fuse every test customer (robustness: must never crash) -----------
    fused_results = []
    for s in test_scored:
        out = fusion_model.process(s["text"], s["voice"], s["txn"])
        fused_results.append({**out, "customer_id": s["customer_id"], "label": s["label"], "segment": s["segment"]})
    ranked = rank_customers(fused_results)
    assert len(ranked) == len(test_scored), "fusion must produce a valid output for every customer, incl. missing modalities"
    print(f"[evaluate] fusion produced valid output for all {len(ranked)} test customers (1/2/3 modalities present).")

    # ======================================================================
    # HEADLINE COMPARISON: text-only vs voice-only vs txn-only vs multimodal
    # on the identical subset of the test set where all 3 modalities exist.
    # ======================================================================
    full_subset = [s for s in test_scored
                   if s["text"]["text_score"] is not None
                   and s["voice"]["voice_score"] is not None
                   and s["txn"]["txn_score"] is not None]
    print(f"[evaluate] headline comparison subset (all 3 modalities present in test): {len(full_subset)} customers")

    y_true = np.array([s["label"] for s in full_subset])
    text_scores = np.array([s["text"]["text_score"] for s in full_subset])
    voice_scores = np.array([s["voice"]["voice_score"] for s in full_subset])
    txn_scores = np.array([s["txn"]["txn_score"] for s in full_subset])
    fused_scores = np.array([
        fusion_model.process(s["text"], s["voice"], s["txn"])["risk_score"] for s in full_subset
    ])

    labels_train = [r["label"] for r in train]
    dummy = DummyClassifier(strategy="most_frequent", random_state=cfg["seed"])
    dummy.fit(np.zeros((len(train), 1)), labels_train)
    majority_pred = dummy.predict(np.zeros((len(full_subset), 1)))

    comparison_rows = [
        {"model": "majority_class_baseline", **metrics(y_true, majority_pred)},
        {"model": "text_only", **metrics(y_true, to_pred(text_scores))},
        {"model": "voice_only", **metrics(y_true, to_pred(voice_scores))},
        {"model": "transaction_only", **metrics(y_true, to_pred(txn_scores))},
        {"model": "multimodal_fusion", **metrics(y_true, to_pred(fused_scores))},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    print("\n[evaluate] ==== HEADLINE COMPARISON (F1-led, same held-out customers) ====")
    print(comparison_df.to_string(index=False))

    fusion_f1 = comparison_df.loc[comparison_df.model == "multimodal_fusion", "f1"].iloc[0]
    for _, row in comparison_df.iterrows():
        if row["model"] in ("multimodal_fusion",):
            continue
        verdict = "BEATS" if fusion_f1 > row["f1"] else "DOES NOT beat"
        print(f"[evaluate] multimodal F1={fusion_f1:.3f} {verdict} {row['model']} F1={row['f1']:.3f}")

    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)

    # ---- bar chart: F1 by model (fixed categorical colors, direct labels) --
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["majority_class_baseline", "text_only", "voice_only", "transaction_only", "multimodal_fusion"]
    colors = ["#c3c2b7", COLOR_TEXT, COLOR_VOICE, COLOR_TXN, COLOR_FUSION]
    plot_df = comparison_df.set_index("model").loc[order].reset_index()
    bars = ax.bar(plot_df["model"], plot_df["f1"], color=colors, width=0.6)
    for bar, val in zip(bars, plot_df["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", va="bottom",
                fontsize=10, color="#0b0b0b")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Multimodal fusion vs. unimodal baselines (held-out test set)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(axis="x", rotation=20, colors="#52514e")
    ax.tick_params(axis="y", colors="#52514e")
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    # ======================================================================
    # FAIRNESS: disaggregate the fusion model's F1 by customer segment.
    # ======================================================================
    fairness_rows = []
    for seg in sorted(set(s["segment"] for s in test_scored)):
        seg_rows = [r for r in ranked if r["segment"] == seg]
        seg_true = [next(s["label"] for s in test_scored if s["customer_id"] == r["customer_id"]) for r in seg_rows]
        seg_pred = to_pred(np.array([r["risk_score"] for r in seg_rows]))
        m = metrics(np.array(seg_true), seg_pred)
        fairness_rows.append({"segment": seg, "n": len(seg_rows), **m})
    fairness_df = pd.DataFrame(fairness_rows)
    print("\n[evaluate] ==== FAIRNESS: fusion metrics by segment ====")
    print(fairness_df.to_string(index=False))
    if len(fairness_df) > 1:
        gap = fairness_df["f1"].max() - fairness_df["f1"].min()
        flag = "FLAGGED (gap > 0.10)" if gap > 0.10 else "within tolerance"
        print(f"[evaluate] max F1 gap across segments: {gap:.3f} -> {flag}")
    fairness_df.to_csv(results_dir / "fairness_report.csv", index=False)

    # ======================================================================
    # INTERPRETABILITY CHECK: confirm `contributing` is populated & readable.
    # ======================================================================
    sample = ranked[:5]
    interpretability_samples = []
    for r in sample:
        assert r["contributing"], "contributing signals must always be populated"
        interpretability_samples.append({
            "customer_id": r["customer_id"],
            "risk_score": r["risk_score"],
            "dominant_modality": r["dominant_modality"],
            "contributing": r["contributing"],
            "next_best_action": r["next_best_action"],
        })
    with open(results_dir / "interpretability_samples.json", "w", encoding="utf-8") as f:
        json.dump(interpretability_samples, f, indent=2)
    print(f"\n[evaluate] interpretability check passed for {len(sample)} sampled customers -> results/interpretability_samples.json")

    # ======================================================================
    # Persist fully-scored dataset for the Streamlit app.
    # ======================================================================
    scored_rows = []
    for s in test_scored:
        out = fusion_model.process(s["text"], s["voice"], s["txn"])
        src = next(r for r in test if r["customer_id"] == s["customer_id"])
        scored_rows.append({
            "customer_id": s["customer_id"],
            "segment": s["segment"],
            "label": s["label"],
            "risk_score": out["risk_score"],
            "dominant_modality": out["dominant_modality"],
            "next_best_action": out["next_best_action"],
            "contributing_text": out["contributing"]["text"],
            "contributing_voice": out["contributing"]["voice"],
            "contributing_txn": out["contributing"]["txn"],
            "text_score": s["text"]["text_score"],
            "text_label": s["text"]["text_label"],
            "text_snippet": (src["text_records_raw"][0] if src["text_records_raw"] else None),
            "voice_score": s["voice"]["voice_score"],
            "voice_label": s["voice"]["voice_label"],
            "transcript_snippet": (src["voice_transcripts_raw"][0] if src["voice_transcripts_raw"] else None),
            "txn_score": s["txn"]["txn_score"],
            "top_txn_features": json.dumps(s["txn"]["top_features"]) if s["txn"]["top_features"] else None,
        })
    scored_df = pd.DataFrame(scored_rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    scored_df.insert(0, "priority_rank", np.arange(1, len(scored_df) + 1))
    scored_df.to_csv(results_dir / "scored_customers.csv", index=False)
    print(f"[evaluate] wrote scored, ranked customer list -> {results_dir / 'scored_customers.csv'}")

    # ---- persist trained models for reuse (e.g. by the Streamlit app) ------
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    text_model.save(models_dir / "text_model")
    txn_model.save(models_dir / "txn_model.joblib")
    fusion_model.save(models_dir / "fusion_model.joblib")
    print(f"[evaluate] saved trained models -> {models_dir}")


if __name__ == "__main__":
    main()
