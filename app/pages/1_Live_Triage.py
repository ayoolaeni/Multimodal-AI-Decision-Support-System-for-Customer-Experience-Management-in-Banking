"""Live Triage page.

Demonstrates a complaint arriving and the AI agent rating it: the agent reads
the complaint, looks up the customer's transaction history, fuses the
evidence into a risk score, assigns a priority tier, and drafts a reply. The
rated cases collect in a queue sorted by urgency for an officer to approve or
dismiss. Recommendations only - nothing is sent to any customer.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from triage import CATEGORY_LABEL, TIER_EMOJI, TriageAgent  # noqa: E402

st.set_page_config(page_title="Live Triage", layout="wide")

CHANNELS = ["Mobile app", "USSD", "Call centre", "Email", "Branch", "Web chat"]
TIER_ORDER = ["Critical", "High", "Medium", "Low"]
PENDING = "Awaiting officer"


@st.cache_resource(show_spinner="Loading the trained models (first time only)...")
def load_agent() -> TriageAgent:
    return TriageAgent.from_disk()


st.title("Live Triage: complaint in, AI rating out")
st.warning(
    "**Recommendations only.** The AI agent rates and suggests; an officer approves or dismisses. "
    "Nothing is sent to any customer.",
    icon="⚠️",
)

try:
    agent = load_agent()
except FileNotFoundError as err:
    st.error(f"{err}\n\nRun the pipeline first:\n\n```\npython src/ingest.py\npython src/preprocess.py\npython src/evaluate.py\n```")
    st.stop()

st.session_state.setdefault("cases", [])
st.session_state.setdefault("next_case_id", 1)


def add_case(case: dict) -> None:
    case["id"] = st.session_state.next_case_id
    st.session_state.next_case_id += 1
    st.session_state.cases.append(case)


def run_agent(complaint: str, channel: str, customer_id: str | None, transcript: str | None, animate: bool) -> None:
    """Runs the agent inside a status box that shows its steps, then queues the case."""
    with st.status(f"AI agent is triaging a complaint from {customer_id or 'an unknown customer'}...", expanded=True) as status:
        case = agent.triage(complaint, channel, customer_id, transcript)
        for i, step in enumerate(case["steps"], start=1):
            st.markdown(f"**{i}. {step['title']}**  \n{step['detail']}")
            if animate:
                time.sleep(0.7)
        status.update(
            label=f"Done: {TIER_EMOJI[case['tier']]} {case['tier']} (risk {case['risk_score']:.2f}) - added to the queue below",
            state="complete",
            expanded=False,
        )
    add_case(case)


def set_status(case_id: int, new_status: str) -> None:
    for c in st.session_state.cases:
        if c["id"] == case_id:
            c["status"] = new_status


def clear_queue() -> None:
    st.session_state.cases = []


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Demo controls")
    animate = st.toggle("Show agent steps slowly (for presenting)", value=True)
    st.button("Clear the queue", on_click=clear_queue, disabled=not st.session_state.cases)
    st.caption(
        "The agent uses the trained text, transaction and fusion models on your laptop. "
        "No internet, no paid service."
    )

# ---------------------------------------------------------------- intake
st.subheader("1. A complaint comes in")
tab_sim, tab_type = st.tabs(["Simulate incoming complaints", "Type a complaint"])

pool = agent.directory.sample_complaints()

with tab_sim:
    if not pool:
        st.info("No sample customers found (data/processed/customers.jsonl). Run the pipeline first.")
    else:
        st.caption(
            "Draws a random customer and complaint from the synthetic dataset, as if it had just arrived. "
            "The customer's transaction history is looked up automatically."
        )
        c1, c2, _ = st.columns([1, 1, 2])
        one = c1.button("Simulate 1 complaint", type="primary")
        burst = c2.button("Simulate 5 at once")
        if one or burst:
            queued_ids = {c["customer_id"] for c in st.session_state.cases}
            fresh = [r for r in pool if r["customer_id"] not in queued_ids] or pool
            for rec in random.sample(fresh, 1 if one else 5):
                run_agent(
                    random.choice(rec["text_records_raw"]),
                    random.choice(CHANNELS),
                    rec["customer_id"],
                    None,
                    animate=animate and one,
                )

with tab_type:
    with st.form("type_complaint", clear_on_submit=False):
        f1, f2 = st.columns(2)
        customer_choice = f1.selectbox(
            "Customer",
            ["Unknown customer"] + agent.directory.ids(),
            help="A known customer lets the agent look up their transaction history.",
        )
        channel = f2.selectbox("Channel", CHANNELS)
        text = st.text_area(
            "Complaint text",
            placeholder="e.g. I did not authorise a debit of 50,000 naira from my account, please reverse it",
        )
        transcript = st.text_area("Call transcript (optional)", placeholder="Paste a call transcript here if the customer phoned in.")
        submitted = st.form_submit_button("Send to the AI agent", type="primary")
    if submitted:
        if not text.strip():
            st.warning("Please type a complaint first.")
        else:
            run_agent(
                text,
                channel,
                None if customer_choice == "Unknown customer" else customer_choice,
                transcript or None,
                animate=animate,
            )

# ---------------------------------------------------------------- queue
st.subheader("2. The AI-rated queue (most urgent first)")

cases = st.session_state.cases
pending = sorted((c for c in cases if c["status"] == PENDING), key=lambda c: c["risk_score"], reverse=True)
handled = [c for c in cases if c["status"] != PENDING]

cols = st.columns(len(TIER_ORDER) + 1)
for col, tier in zip(cols, TIER_ORDER):
    col.metric(f"{TIER_EMOJI[tier]} {tier}", sum(1 for c in pending if c["tier"] == tier))
cols[-1].metric("Handled by officer", len(handled))

if not pending:
    st.info("The queue is empty. Simulate or type a complaint above and watch it get rated.")

for rank, case in enumerate(pending, start=1):
    header = (
        f"{TIER_EMOJI[case['tier']]} #{rank} · {case['tier'].upper()} · risk {case['risk_score']:.2f} · "
        f"{case['customer_id']} · {case['channel']} · reply within {case['sla_hours']}h"
    )
    with st.expander(header, expanded=(rank == 1)):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**Complaint** ({CATEGORY_LABEL[case['category']]}, received {case['received_at']}):")
            st.markdown(f"> {case['complaint']}")
            if case["call_transcript"]:
                st.markdown(f"**Call transcript:** _{case['call_transcript']}_")
            st.markdown(f"**Why this rating:** {case['reason']}")
            st.markdown(f"**Suggested next action:** {case['next_best_action']}")
            st.markdown("**How the agent got there:**")
            st.markdown("\n".join(f"{i}. **{s['title']}** - {s['detail']}" for i, s in enumerate(case["steps"], start=1)))
        with right:
            st.text_area("Draft reply (officer can edit)", value=case["draft_reply"], key=f"reply_{case['id']}", height=160)
            b1, b2 = st.columns(2)
            b1.button(
                "Approve",
                key=f"approve_{case['id']}",
                type="primary",
                on_click=set_status,
                args=(case["id"], "Approved"),
                use_container_width=True,
            )
            b2.button(
                "Dismiss",
                key=f"dismiss_{case['id']}",
                on_click=set_status,
                args=(case["id"], "Dismissed"),
                use_container_width=True,
            )
            st.caption("Demo only: approving records the decision here and sends nothing.")

if handled:
    with st.expander(f"Handled cases ({len(handled)})"):
        for case in handled:
            row, undo = st.columns([5, 1])
            row.markdown(
                f"**{case['status']}** · {TIER_EMOJI[case['tier']]} {case['tier']} · risk {case['risk_score']:.2f} · "
                f"{case['customer_id']} · _{case['complaint'][:90]}_"
            )
            undo.button("Undo", key=f"undo_{case['id']}", on_click=set_status, args=(case["id"], PENDING))

st.divider()
st.caption(
    "Prototype on synthetic data. The ratings come from models trained on invented complaints, "
    "so wording very different from the training examples may be rated less reliably."
)
