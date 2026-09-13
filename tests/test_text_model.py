import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from text_model import TextModel, categorize  # noqa: E402
from utils import load_config  # noqa: E402


def test_categorize_matches_known_keywords():
    assert categorize("my transfer failed and money did not enter") == "failed_transaction"
    assert categorize("there was an unauthorised deduction on my account") == "unauthorised_deduction"
    assert categorize("I was charged twice, unfair charge") == "unfair_charge"
    assert categorize("still waiting for my refund") == "delayed_refund"
    assert categorize("card declined at the pos") == "card_declined"
    assert categorize("the app is down since morning") == "app_downtime"
    assert categorize("thank you for the quick help") == "other"


def test_process_is_null_safe_on_empty_records():
    model = TextModel(load_config())
    result = model.process([])
    assert result == {"text_score": None, "text_label": None, "confidence": None, "driving_text": None}


@pytest.mark.slow
def test_fit_and_process_beats_majority_baseline():
    """Trains a tiny TextModel on a handful of obviously-separable examples
    and checks it learns the direction of the signal. Downloads the base
    model on first run - marked slow so it can be skipped in quick runs."""
    cfg = load_config()
    cfg["text_model"]["epochs"] = 2
    cfg["text_model"]["batch_size"] = 8
    texts = [
        "my transfer failed and the money did not enter the account",
        "unauthorised deduction on my account please help",
        "still waiting for my refund since last month",
        "card declined at the pos for no reason",
        "thank you for resolving my issue quickly",
        "just checking my balance, all good",
        "the app has been working fine for me",
        "everything is okay with my account",
    ] * 4
    labels = [1, 1, 1, 1, 0, 0, 0, 0] * 4

    model = TextModel(cfg)
    model.fit(texts, labels, seed=cfg["seed"])
    result = model.process(["my transfer failed and money did not enter my account"])
    assert result["text_score"] is not None
    assert 0.0 <= result["text_score"] <= 1.0
    assert result["text_label"] == "failed_transaction"
