"""Sprint 5 - Interface (README S6.5).

Reads the scored, ranked customer list produced by `python src/evaluate.py`
(results/scored_customers.csv) and presents it as a ranked worklist for a
customer-service officer, with a per-row "why" panel and an override/dismiss
control. This app never trains a model and never takes an action on a
customer - it only displays recommendations for a human to review.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_config, resolve_path  # noqa: E402

st.set_page_config(page_title="Customer CX Decision Support", layout="wide")

cfg = load_config()
results_dir = resolve_path(cfg["paths"]["results_dir"])
scored_path = results_dir / "scored_customers.csv"

st.title("Multimodal Customer Experience Decision Support")

st.warning(
    "**Recommendations only.** A customer-service officer reviews and decides. "
    "No action is taken automatically.",
    icon="⚠️",
)

if not scored_path.exists():
    st.error(
        f"No scored data found at `{scored_path}`.\n\n"
        "Run the pipeline first:\n\n"
        "```\npython src/ingest.py\npython src/preprocess.py\npython src/evaluate.py\n```"
    )
    st.stop()

df = pd.read_csv(scored_path)

DECISION_OPTIONS = ["No action yet", "Act on recommendation", "Dismiss"]


def get_decision(customer_id: str) -> str:
    return st.session_state.get(f"decision_{customer_id}", "No action yet")


with st.sidebar:
    st.header("Filters")
    min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)
    segments = st.multiselect("Segment", sorted(df["segment"].dropna().unique()), default=None)
    dominant_filter = st.multiselect(
        "Dominant signal", sorted(df["dominant_modality"].dropna().unique()), default=None
    )
    show_dismissed = st.checkbox("Show dismissed customers", value=False)
    n_dismissed = int(df["customer_id"].apply(get_decision).eq("Dismiss").sum())
    st.caption(f"{len(df)} customers in this batch (held-out evaluation set).")
    if n_dismissed:
        st.caption(f"{n_dismissed} dismissed this session.")

view = df[df["risk_score"] >= min_risk].copy()
if segments:
    view = view[view["segment"].isin(segments)]
if dominant_filter:
    view = view[view["dominant_modality"].isin(dominant_filter)]

view["_decision"] = view["customer_id"].apply(get_decision)
if not show_dismissed:
    view = view[view["_decision"] != "Dismiss"]

st.subheader(f"Ranked worklist ({len(view)} customers)")

MODALITY_LABEL = {"text": "Written complaint/chat", "voice": "Call transcript", "txn": "Transaction behaviour"}


def fmt_score(x) -> str:
    return "—" if pd.isna(x) else f"{x:.2f}"


for _, row in view.iterrows():
    officer_decision = row["_decision"]
    badge = f" · officer: {officer_decision}" if officer_decision != "No action yet" else ""
    header = (
        f"#{int(row['priority_rank'])} · {row['customer_id']} · risk {row['risk_score']:.2f} "
        f"· dominant: {MODALITY_LABEL.get(row['dominant_modality'], 'none')}{badge}"
    )
    with st.expander(header):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Recommended next-best action:** {row['next_best_action']}")
            st.markdown("**Why - per-modality contribution:**")
            st.write(
                {
                    "Text": f"score={fmt_score(row['text_score'])}, contribution={fmt_score(row['contributing_text'])}, category={row['text_label'] or '—'}",
                    "Voice": f"score={fmt_score(row['voice_score'])}, contribution={fmt_score(row['contributing_voice'])}, category={row['voice_label'] or '—'}",
                    "Transaction": f"score={fmt_score(row['txn_score'])}, contribution={fmt_score(row['contributing_txn'])}",
                }
            )
            if isinstance(row.get("text_snippet"), str):
                st.markdown(f"**Complaint/chat excerpt:** _{row['text_snippet']}_")
            if isinstance(row.get("transcript_snippet"), str):
                st.markdown(f"**Call transcript excerpt:** _{row['transcript_snippet']}_")
            if isinstance(row.get("top_txn_features"), str):
                st.markdown(f"**Top transaction signals:** `{row['top_txn_features']}`")
        with col2:
            st.metric("Risk score", f"{row['risk_score']:.2f}")
            st.caption(f"Segment: {row['segment']}")
            choice = st.radio(
                "Officer decision",
                DECISION_OPTIONS,
                key=f"decision_{row['customer_id']}",
                index=DECISION_OPTIONS.index(officer_decision),
            )
            if choice == "Act on recommendation":
                st.success(f"✅ {row['customer_id']} has been attended to and is satisfied.")
            elif choice == "Dismiss":
                st.info(
                    "This customer is dismissed and hidden from the worklist. "
                    "Tick **Show dismissed customers** in the sidebar to undo."
                )

st.divider()
st.caption(
    "This interface embodies human authority over every recommendation: the officer decision above "
    "is recorded in this session only and has no downstream effect on any customer."
)
