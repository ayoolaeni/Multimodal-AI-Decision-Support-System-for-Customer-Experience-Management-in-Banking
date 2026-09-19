import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fusion import FusionModel  # noqa: E402
from triage import CustomerDirectory, TriageAgent  # noqa: E402
from utils import load_config  # noqa: E402


class FakeTextModel:
    """Scores any text containing 'fraud' high, everything else low, so the
    agent logic is tested without downloading/loading a real BERT model."""

    def process(self, texts):
        if not texts:
            return {"text_score": None, "text_label": None, "confidence": None, "driving_text": None}
        high = "fraud" in texts[0]
        return {
            "text_score": 0.95 if high else 0.1,
            "text_label": "unauthorised_deduction" if high else "other",
            "confidence": 0.9,
            "driving_text": texts[0],
        }


class FakeTxnModel:
    def process(self, txn_features):
        if txn_features is None:
            return {"txn_score": None, "top_features": None, "confidence": None}
        return {
            "txn_score": 0.8,
            "top_features": [("failed_transfers_count", 0.7), ("declines_count", -0.1)],
            "confidence": 0.8,
        }


def _agent() -> TriageAgent:
    cfg = load_config()
    rng = np.random.default_rng(0)
    labels = (rng.random(200) < 0.4).astype(int)
    triplets = [tuple(np.clip(rng.normal(0.75 if y else 0.25, 0.15, 3), 0.01, 0.99)) for y in labels]
    fusion = FusionModel(cfg).fit(triplets, labels.tolist(), seed=0)
    directory = CustomerDirectory(
        [{"customer_id": "CUST1", "segment": "mobile_app", "txn_features": {"failed_transfers_count": 5}, "text_records_raw": ["x"]}]
    )
    return TriageAgent(FakeTextModel(), FakeTxnModel(), fusion, directory, cfg)


def test_serious_complaint_is_rated_higher_than_a_benign_one():
    agent = _agent()
    serious = agent.triage("this is fraud on my account", "USSD", "CUST1")
    benign = agent.triage("thanks for your help", "USSD", "CUST1")
    assert serious["risk_score"] > benign["risk_score"]
    assert serious["category"] == "unauthorised_deduction"


def test_case_has_tier_sla_reply_and_ordered_trace():
    case = _agent().triage("this is fraud on my account", "Email", "CUST1")
    assert case["tier"] in {"Critical", "High", "Medium", "Low"}
    assert case["sla_hours"] > 0
    assert str(case["sla_hours"]) in case["draft_reply"]
    assert case["status"] == "Awaiting officer"
    titles = [s["title"] for s in case["steps"]]
    assert titles[0] == "Received complaint"
    assert titles[-1].startswith("Suggested next action")
    assert len(titles) >= 6


def test_unknown_customer_still_gets_a_rating_without_txn_history():
    case = _agent().triage("this is fraud", "Branch", None)
    assert case["customer_id"] == "Unknown customer"
    assert 0.0 <= case["risk_score"] <= 1.0
    assert "No transaction history" in case["reason"]


def test_personal_details_are_stripped_before_scoring():
    seen = {}

    class Spy(FakeTextModel):
        def process(self, texts):
            seen["texts"] = texts
            return super().process(texts)

    agent = _agent()
    agent.text_model = Spy()
    agent.triage("Call me on 08012345678 or a@b.com about fraud", "Email", None)
    assert "08012345678" not in seen["texts"][0]
    assert "a@b.com" not in seen["texts"][0]


def test_call_transcript_is_used_when_supplied():
    agent = _agent()
    case = agent.triage("hello", "Call centre", "CUST1", call_transcript="this is fraud")
    assert any(s["title"] == "Scored the call transcript" for s in case["steps"])


def test_tier_thresholds_are_ordered_highest_first():
    agent = _agent()
    assert agent.tier_for(0.99)["name"] == "Critical"
    assert agent.tier_for(0.0)["name"] == "Low"
    names = [agent.tier_for(r)["name"] for r in (0.9, 0.5, 0.35, 0.1)]
    assert names == ["Critical", "High", "Medium", "Low"]
