import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import clean_text_list, inject_missingness  # noqa: E402
from utils import load_config  # noqa: E402


def test_clean_text_list_strips_pii_and_normalises():
    raw, norm = clean_text_list(["Call ME at 08031234567 NOW"])
    assert "[PHONE]" in raw[0]
    assert raw[0] != norm[0]  # raw keeps case, norm is lowercased
    assert norm[0] == norm[0].lower()


def test_clean_text_list_empty_input():
    raw, norm = clean_text_list([])
    assert raw == [] and norm == []


def _make_records(n=50):
    return [
        {
            "customer_id": f"C{i}",
            "text_records": ["a complaint"],
            "text_records_raw": ["a complaint"],
            "voice_transcripts": ["a call"],
            "voice_transcripts_raw": ["a call"],
            "txn_features": {"failed_transfers_count": 1},
            "label": i % 2,
        }
        for i in range(n)
    ]


def test_inject_missingness_never_drops_all_three():
    cfg = load_config()
    cfg["data"]["missing_text_frac"] = 1.0
    cfg["data"]["missing_voice_frac"] = 1.0
    cfg["data"]["missing_txn_frac"] = 1.0
    rng = np.random.default_rng(0)
    records = inject_missingness(_make_records(30), cfg, rng)
    for r in records:
        has_text = bool(r["text_records"])
        has_voice = bool(r["voice_transcripts"])
        has_txn = r["txn_features"] is not None
        assert has_text or has_voice or has_txn


def test_inject_missingness_zero_frac_keeps_everything():
    cfg = load_config()
    cfg["data"]["missing_text_frac"] = 0.0
    cfg["data"]["missing_voice_frac"] = 0.0
    cfg["data"]["missing_txn_frac"] = 0.0
    rng = np.random.default_rng(0)
    records = inject_missingness(_make_records(30), cfg, rng)
    assert all(r["text_records"] and r["voice_transcripts"] and r["txn_features"] is not None for r in records)
