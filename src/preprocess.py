"""Sprint 1 - Data foundation (continued).

Reads data/raw/customers.jsonl, strips PII, normalises text, and
DELIBERATELY nulls out a configured fraction of each modality per customer
so that robustness to missing modalities (README S3.2) is tested, not just
assumed. Writes the canonical schema (README S5.2) to
data/processed/customers.jsonl.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ensure_dir, load_config, normalise_text, read_jsonl, resolve_path, set_global_seed, strip_pii, write_jsonl


def clean_text_list(texts: list[str]) -> tuple[list[str], list[str]]:
    """Returns (raw_for_display, normalised_for_modeling), both PII-stripped."""
    raw, norm = [], []
    for t in texts:
        stripped = strip_pii(t)
        raw.append(stripped)
        norm.append(normalise_text(stripped))
    return raw, norm


def inject_missingness(records: list[dict], cfg: dict, rng: np.random.Generator) -> list[dict]:
    p_text = cfg["data"]["missing_text_frac"]
    p_voice = cfg["data"]["missing_voice_frac"]
    p_txn = cfg["data"]["missing_txn_frac"]

    for r in records:
        drop_text = rng.random() < p_text
        drop_voice = rng.random() < p_voice
        drop_txn = rng.random() < p_txn

        # Never drop all three - a customer with zero evidence isn't a valid case.
        if drop_text and drop_voice and drop_txn:
            keep = rng.choice(["text", "voice", "txn"])
            drop_text = keep != "text"
            drop_voice = keep != "voice"
            drop_txn = keep != "txn"

        if drop_text:
            r["text_records"] = []
            r["text_records_raw"] = []
        if drop_voice:
            r["voice_transcripts"] = []
            r["voice_transcripts_raw"] = []
            r["voice_audio_path"] = None
        if drop_txn:
            r["txn_features"] = None
    return records


def build_processed_dataset(cfg: dict) -> list[dict]:
    raw_path = resolve_path(cfg["paths"]["raw_dir"]) / "customers.jsonl"
    records = read_jsonl(raw_path)
    if not records:
        raise FileNotFoundError(f"No raw data at {raw_path} - run src/ingest.py first.")

    for r in records:
        raw_text, norm_text = clean_text_list(r.get("text_records", []))
        r["text_records_raw"] = raw_text
        r["text_records"] = norm_text

        raw_voice, norm_voice = clean_text_list(r.get("voice_transcripts", []))
        r["voice_transcripts_raw"] = raw_voice
        r["voice_transcripts"] = norm_voice

    rng = np.random.default_rng(cfg["seed"] + 1)
    records = inject_missingness(records, cfg, rng)
    return records


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    records = build_processed_dataset(cfg)

    out_dir = ensure_dir(cfg["paths"]["processed_dir"])
    out_path = out_dir / "customers.jsonl"
    write_jsonl(records, out_path)

    n = len(records)
    n_no_text = sum(1 for r in records if not r["text_records"])
    n_no_voice = sum(1 for r in records if not r["voice_transcripts"])
    n_no_txn = sum(1 for r in records if r["txn_features"] is None)
    n_all_present = sum(
        1 for r in records if r["text_records"] and r["voice_transcripts"] and r["txn_features"] is not None
    )
    print(f"[preprocess] wrote {n} processed customer records to {out_path}")
    print(f"[preprocess] missing text: {n_no_text} ({n_no_text/n:.1%})")
    print(f"[preprocess] missing voice: {n_no_voice} ({n_no_voice/n:.1%})")
    print(f"[preprocess] missing txn: {n_no_txn} ({n_no_txn/n:.1%})")
    print(f"[preprocess] all three modalities present: {n_all_present} ({n_all_present/n:.1%})")


if __name__ == "__main__":
    main()
