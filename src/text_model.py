"""Sprint 2 - Text modality (README S6.1).

Fine-tunes a small BERT-family model (distilbert-base-uncased by default) for
binary dissatisfaction classification, and pairs it with a transparent,
rule-based category classifier (so the "why" is always inspectable - README
S3.3). The voice modality (src/voice_model.py) reuses this exact class on
call transcripts instead of training a separate model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, set_global_seed

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "failed_transaction": [
        "failed transaction", "transaction failed", "transfer failed", "did not go through",
        "money did not enter", "did not credit", "money never got", "never got",
    ],
    "unauthorised_deduction": [
        "unauthorised", "unauthorized", "did not authorise", "did not authorize",
        "strange debit", "without my consent", "fraudulent debit", "did not do this",
        "i did not carry out",
    ],
    "unfair_charge": [
        "unfair charge", "hidden charge", "hidden fee", "extra fee", "charged twice",
        "excess charge", "maintenance charge", "wrongly charged", "sms/alert fee", "alert fee",
    ],
    "delayed_refund": [
        "refund", "reversal", "still waiting for my money", "not been returned",
        "not been refunded", "money not returned", "reversed",
    ],
    "card_declined": [
        "card declined", "card was declined", "atm rejected", "pos declined",
        "card blocked", "card got blocked", "insufficient funds",
    ],
    "app_downtime": [
        "app is down", "app not working", "cannot log in", "network error",
        "app keeps crashing", "ussd not working", "ussd is not working", "app has been down",
    ],
}


def categorize(text: str) -> str:
    """Transparent keyword-based complaint categoriser. Independent of the
    neural dissatisfaction model so the category shown to an officer is
    always directly traceable to words in the text."""
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return "other"


class TextModel:
    """Dissatisfaction scorer for free text (complaints, chat, email, and -
    via VoiceModel - call transcripts)."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or load_config())["text_model"]
        self.base_model_name = self.cfg["base_model"]
        self.max_length = self.cfg["max_length"]
        self.tokenizer = None
        self.model = None
        self._is_fit = False

    # -- lazy heavy imports so txn-only / config-only usage stays cheap -----
    def _lazy_init_model(self, num_labels: int = 2):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, num_labels=num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, texts: list[str], labels: list[int], seed: int = 42) -> "TextModel":
        set_global_seed(seed)
        if self.model is None:
            self._lazy_init_model(num_labels=2)
        torch = self._torch
        from torch.utils.data import DataLoader, TensorDataset

        enc = self.tokenizer(
            list(texts), truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        label_t = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], label_t)
        loader = DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["learning_rate"])
        self.model.train()
        for epoch in range(self.cfg["epochs"]):
            total_loss = 0.0
            for input_ids, attn_mask, y in loader:
                input_ids, attn_mask, y = input_ids.to(self.device), attn_mask.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=y)
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()
            print(f"[text_model] epoch {epoch + 1}/{self.cfg['epochs']} loss={total_loss / len(loader):.4f}")
        self.model.eval()
        self._is_fit = True
        return self

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Returns P(dissatisfied) for each text."""
        if not texts:
            return np.array([])
        if self.model is None:
            raise RuntimeError("TextModel is not fit/loaded - call fit() or load() first.")
        torch = self._torch
        probs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**enc).logits
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(p)
        return np.concatenate(probs)

    def process(self, text_records: list[str]) -> dict:
        """Aggregate a customer's text_records into one null-safe result.
        Uses the single most dissatisfaction-signalling record to drive the
        score/label so the "why" panel can show one concrete snippet."""
        if not text_records:
            return {"text_score": None, "text_label": None, "confidence": None, "driving_text": None}

        probs = self.predict_proba(text_records)
        idx = int(np.argmax(probs))
        score = float(probs[idx])
        confidence = float(max(score, 1 - score))
        label = categorize(text_records[idx])
        return {
            "text_score": score,
            "text_label": label,
            "confidence": confidence,
            "driving_text": text_records[idx],
        }

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

    @classmethod
    def load(cls, in_dir: str | Path, config: dict | None = None) -> "TextModel":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        inst = cls(config)
        inst._torch = torch
        inst.tokenizer = AutoTokenizer.from_pretrained(in_dir)
        inst.model = AutoModelForSequenceClassification.from_pretrained(in_dir)
        inst.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inst.model.to(inst.device)
        inst.model.eval()
        inst._is_fit = True
        return inst
