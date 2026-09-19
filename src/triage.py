"""Live Triage - an explainable "agent" that handles one incoming complaint.

The agent works through a case step by step, using the already-trained
models as its tools: read the complaint, look up the customer's transaction
history, score the call transcript if there is one, fuse the evidence into
one risk score, assign a priority tier, and draft a reply. Every step is
recorded in a trace so the officer (and a demo audience) can see how the
rating was reached.

It only *recommends*. Nothing here contacts a customer or changes an
account - an officer approves or dismisses each case (README S3.1).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion import FusionModel
from text_model import TextModel
from txn_model import TxnModel
from utils import load_config, normalise_text, read_jsonl, resolve_path, strip_pii
from voice_model import VoiceModel

TIER_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

CATEGORY_LABEL = {
    "failed_transaction": "Failed transaction",
    "unauthorised_deduction": "Unauthorised deduction",
    "unfair_charge": "Unfair charge",
    "delayed_refund": "Delayed refund",
    "card_declined": "Card declined",
    "app_downtime": "App / USSD downtime",
    "other": "General / other",
}

TXN_FEATURE_LABEL = {
    "failed_transfers_count": "failed transfers",
    "declines_count": "card declines",
    "txn_frequency_change": "change in transaction frequency",
    "app_sessions_change": "change in app usage",
    "days_since_last_complaint": "recent complaint history",
    "avg_resolution_delay": "slow past resolutions",
}

MODALITY_LABEL = {"text": "the written complaint", "voice": "the call transcript", "txn": "transaction behaviour"}


class CustomerDirectory:
    """Look-up of known customers (their transaction features and segment).
    Stands in for the bank's core-banking system in this prototype."""

    def __init__(self, records: list[dict]):
        self._by_id = {r["customer_id"]: r for r in records}

    @classmethod
    def from_disk(cls, cfg: dict | None = None) -> "CustomerDirectory":
        cfg = cfg or load_config()
        path = resolve_path(cfg["paths"]["processed_dir"]) / "customers.jsonl"
        return cls(read_jsonl(path))

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def get(self, customer_id: str | None) -> dict | None:
        return self._by_id.get(customer_id) if customer_id else None

    def sample_complaints(self) -> list[dict]:
        """Customers that have at least one written complaint, for the
        'simulate an incoming complaint' button."""
        return [r for r in self._by_id.values() if r.get("text_records_raw")]


class TriageAgent:
    def __init__(
        self,
        text_model: TextModel,
        txn_model: TxnModel,
        fusion_model: FusionModel,
        directory: CustomerDirectory,
        config: dict | None = None,
    ):
        self.cfg = config or load_config()
        self.text_model = text_model
        self.txn_model = txn_model
        self.fusion_model = fusion_model
        self.voice_model = VoiceModel(text_model, self.cfg)
        self.directory = directory
        self.tiers = self.cfg["triage"]["tiers"]
        self.reply_templates = self.cfg["reply_templates"]

    @classmethod
    def from_disk(cls, config: dict | None = None) -> "TriageAgent":
        cfg = config or load_config()
        models_dir = resolve_path(cfg["paths"]["models_dir"])
        needed = [models_dir / "text_model", models_dir / "txn_model.joblib", models_dir / "fusion_model.joblib"]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Trained models not found: " + ", ".join(missing) + ". Run `python src/evaluate.py` first."
            )
        return cls(
            TextModel.load(models_dir / "text_model", cfg),
            TxnModel.load(models_dir / "txn_model.joblib", cfg),
            FusionModel.load(models_dir / "fusion_model.joblib", cfg),
            CustomerDirectory.from_disk(cfg),
            cfg,
        )

    # ------------------------------------------------------------------
    def tier_for(self, risk: float) -> dict:
        for tier in self.tiers:
            if risk >= tier["min_risk"]:
                return tier
        return self.tiers[-1]

    def _reason(self, fused: dict, text_out: dict, txn_out: dict, tier: str) -> str:
        parts = [f"Rated **{tier}** (risk {fused['risk_score']:.2f})."]
        dominant = fused["dominant_modality"]
        if dominant:
            parts.append(f"The strongest signal was {MODALITY_LABEL[dominant]}.")
        if text_out.get("text_label") and text_out["text_label"] != "other":
            parts.append(f"The complaint is about: {CATEGORY_LABEL[text_out['text_label']].lower()}.")
        pushing_up = [
            TXN_FEATURE_LABEL.get(name, name) for name, contrib in (txn_out.get("top_features") or []) if contrib > 0
        ]
        if pushing_up:
            parts.append("Transaction history adds to the risk: " + ", ".join(pushing_up) + ".")
        if txn_out.get("txn_score") is None:
            parts.append("No transaction history was found, so the rating rests on the other evidence.")
        return " ".join(parts)

    def triage(
        self,
        complaint_text: str,
        channel: str,
        customer_id: str | None = None,
        call_transcript: str | None = None,
    ) -> dict:
        """Runs the full agent on one complaint. Returns a case dict that
        includes an ordered `steps` trace."""
        steps: list[dict] = []
        cleaned = normalise_text(strip_pii(complaint_text))
        steps.append(
            {
                "title": "Received complaint",
                "detail": f"Channel: {channel}. Personal details (emails, phone/account numbers) removed before analysis.",
            }
        )

        text_out = self.text_model.process([cleaned] if cleaned else [])
        if text_out["text_score"] is not None:
            steps.append(
                {
                    "title": "Read and classified the complaint (text model)",
                    "detail": f"Category: {CATEGORY_LABEL[text_out['text_label']]}. "
                    f"Dissatisfaction score: {text_out['text_score']:.2f}.",
                }
            )
        else:
            steps.append({"title": "Read the complaint", "detail": "No written text supplied."})

        customer = self.directory.get(customer_id)
        txn_features = customer["txn_features"] if customer else None
        txn_out = self.txn_model.process(txn_features)
        if txn_out["txn_score"] is not None:
            top = ", ".join(f"{TXN_FEATURE_LABEL.get(n, n)} ({c:+.2f})" for n, c in txn_out["top_features"])
            steps.append(
                {
                    "title": "Looked up the customer's transaction history",
                    "detail": f"Behaviour risk score: {txn_out['txn_score']:.2f}. Biggest factors: {top}.",
                }
            )
        else:
            steps.append(
                {
                    "title": "Looked up the customer's transaction history",
                    "detail": "No transaction history available for this customer - continuing without it.",
                }
            )

        transcript_clean = normalise_text(strip_pii(call_transcript)) if call_transcript else None
        voice_out = self.voice_model.process(transcripts=[transcript_clean] if transcript_clean else None)
        if voice_out["voice_score"] is not None:
            steps.append(
                {
                    "title": "Scored the call transcript",
                    "detail": f"Dissatisfaction score: {voice_out['voice_score']:.2f}.",
                }
            )

        fused = self.fusion_model.process(text_out, voice_out, txn_out)
        present = [m for m, c in fused["contributing"].items() if c is not None]
        steps.append(
            {
                "title": "Combined the evidence into one risk score (fusion model)",
                "detail": f"Evidence used: {', '.join(present) or 'none'}. Risk score: {fused['risk_score']:.2f}. "
                "Missing evidence is not treated as zero - the weight moves to what is present.",
            }
        )

        tier = self.tier_for(fused["risk_score"])
        steps.append(
            {
                "title": "Assigned priority",
                "detail": f"{TIER_EMOJI.get(tier['name'], '')} {tier['name']} - respond within {tier['sla_hours']} hours.",
            }
        )

        category = text_out["text_label"] or voice_out.get("voice_label") or "other"
        reply = self.reply_templates.get(category, self.reply_templates["other"]).format(sla=tier["sla_hours"])
        steps.append(
            {
                "title": "Suggested next action and drafted a reply",
                "detail": f"{fused['next_best_action']} (Draft reply is for the officer to review - nothing is sent.)",
            }
        )

        return {
            "received_at": datetime.now().strftime("%H:%M:%S"),
            "customer_id": customer_id or "Unknown customer",
            "segment": customer["segment"] if customer else None,
            "channel": channel,
            "complaint": complaint_text.strip(),
            "call_transcript": (call_transcript or "").strip() or None,
            "category": category,
            "risk_score": fused["risk_score"],
            "tier": tier["name"],
            "sla_hours": tier["sla_hours"],
            "dominant_modality": fused["dominant_modality"],
            "next_best_action": fused["next_best_action"],
            "reason": self._reason(fused, text_out, txn_out, tier["name"]),
            "draft_reply": reply,
            "steps": steps,
            "status": "Awaiting officer",
        }
