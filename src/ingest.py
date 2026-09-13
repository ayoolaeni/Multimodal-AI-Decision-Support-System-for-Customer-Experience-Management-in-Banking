"""Sprint 1 - Data foundation.

Generates a fully SYNTHETIC banking customer-experience dataset (no real
customer data is available or used - see README S5). Each customer gets three
independently-noisy "views" of an underlying dissatisfaction label so that no
single modality fully determines the outcome; this is what lets the fusion
layer in src/fusion.py demonstrably beat any one modality on its own
(evaluated in src/evaluate.py).

Licence note: this dataset is generated entirely by this script from hand-
written templates + numpy randomness. No external data source is downloaded,
so there is no third-party licence to verify.

Output:
  data/synthetic/complaints.jsonl        one row per generated text record
  data/synthetic/voice_transcripts.jsonl one row per generated call transcript
  data/synthetic/transactions.csv        one row per customer, txn features
  data/audio/<customer_id>.wav           a handful of TTS demo call clips
  data/raw/customers.jsonl               combined per-customer raw records
                                          (input to src/preprocess.py)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import REPO_ROOT, ensure_dir, load_config, resolve_path, set_global_seed, write_jsonl

# ---------------------------------------------------------------------------
# Complaint / chat text templates, grouped by ground-truth category.
# ---------------------------------------------------------------------------
NEGATIVE_TEMPLATES: dict[str, list[str]] = {
    "failed_transaction": [
        "I tried to send {amount} naira through the mobile app on {date} but the transfer failed and my account was still debited.",
        "My bank transfer of {amount} naira did not go through on {date}, the money left my account but the receiver never got it.",
        "The NIP transfer I did on {date} for {amount} naira failed but I have not been reversed since then.",
        "I sent money using USSD on {date} and it said the transaction failed, but my balance still went down by {amount} naira.",
    ],
    "unauthorised_deduction": [
        "There is a strange debit of {amount} naira on my account from {date} that I did not authorise.",
        "Someone deducted {amount} naira from my account without my consent, I noticed it on {date} and I am worried.",
        "I got an alert for {amount} naira debited from my account on {date} that I never did, please investigate.",
        "My account was debited {amount} naira on {date} for a transaction I did not carry out at all.",
    ],
    "unfair_charge": [
        "I was charged {amount} naira in hidden fees on {date} that nobody explained to me before now.",
        "The bank charged me twice for the same transaction on {date}, totalling an extra {amount} naira.",
        "I noticed an unfair maintenance charge of {amount} naira on {date} that seems too high for this month.",
        "Why was I charged {amount} naira in SMS/alert fees on {date}, this is more than what I was told.",
    ],
    "delayed_refund": [
        "I am still waiting for a refund of {amount} naira from a failed transaction on {date}, it has been too long.",
        "The reversal for my {amount} naira failed transfer on {date} has still not reflected in my account.",
        "My money of {amount} naira has not been returned since {date} even though the transaction did not complete.",
        "I was promised a refund of {amount} naira after {date} but nothing has come into my account yet.",
    ],
    "card_declined": [
        "My debit card was declined at the POS terminal on {date} even though I had up to {amount} naira in my account.",
        "The ATM rejected my card on {date} and kept saying insufficient funds when I had over {amount} naira available.",
        "I could not use my card to pay {amount} naira online on {date}, it kept getting declined for no reason.",
        "My card got blocked after a failed transaction on {date} and I could not withdraw {amount} naira that I needed urgently.",
    ],
    "app_downtime": [
        "The mobile banking app has been down since {date} and I cannot check my balance or send money.",
        "I keep getting a network error on the app whenever I try to transfer {amount} naira, this has been since {date}.",
        "The USSD code has not been working since {date}, I cannot do any transaction at all.",
        "Your app keeps crashing whenever I try to log in, this has been happening since {date} and it is frustrating.",
    ],
}

NEUTRAL_TEMPLATES = [
    "Thank you for resolving my last complaint quickly, I appreciate the support.",
    "Just checking my account balance and recent transactions, everything looks fine.",
    "I wanted to confirm my new debit card has been activated, thanks for the help.",
    "The app has been working well for me lately, no issues to report.",
    "I would like to update my phone number linked to my account when convenient.",
    "Please can you confirm if my salary alert came in today, thank you.",
    "I'm satisfied with how my last dispute was handled, appreciate the quick turnaround.",
    "Can I get a statement of account for the last three months please.",
]

SPOKEN_NEGATIVE_TEMPLATES: dict[str, list[str]] = {
    "failed_transaction": [
        "good day please my transfer of {amount} naira on {date} it failed but the money don leave my account",
        "i sent money since {date} the app say failed but my balance don reduce by {amount} naira please help me",
    ],
    "unauthorised_deduction": [
        "please i saw a debit of {amount} naira i did not do this on {date} i need this checked urgently",
        "somebody took {amount} naira from my account on {date} i did not authorise it at all please",
    ],
    "unfair_charge": [
        "why did you charge me {amount} naira extra on {date} nobody explained this fee to me before",
        "i was charged twice on {date} that is like {amount} naira extra please can you check it",
    ],
    "delayed_refund": [
        "please my refund of {amount} naira since {date} has not come i have been calling since",
        "i am still waiting for {amount} naira reversal from {date} please when will it enter my account",
    ],
    "card_declined": [
        "my card was declined at the pos on {date} but i have up to {amount} naira please check my card",
        "the atm rejected my card since {date} it kept saying insufficient funds but i had {amount} naira",
    ],
    "app_downtime": [
        "the app has been down since {date} i cannot send money at all please fix it",
        "the ussd is not working since {date} please when will this be fixed",
    ],
}

SPOKEN_NEUTRAL_TEMPLATES = [
    "good afternoon i just want to confirm my balance thank you",
    "thank you for helping me last time everything is fine now",
    "please can you send me my statement for last month",
    "i just want to update my phone number when you are less busy",
]

CATEGORIES = list(NEGATIVE_TEMPLATES.keys())


def _sample_amount(rng: np.random.Generator) -> str:
    amount = int(rng.choice([1500, 5000, 8000, 12000, 25000, 40000, 65000, 100000]))
    return f"{amount:,}"


def _sample_date(rng: np.random.Generator, today: dt.date) -> str:
    days_ago = int(rng.integers(1, 45))
    d = today - dt.timedelta(days=days_ago)
    return d.strftime("%d %B")


def _noisy_label(true_label: int, signal_p: float, rng: np.random.Generator) -> int:
    """With probability signal_p, return the true label; otherwise flip it.
    This is what makes each modality an independent, imperfect view of the
    same underlying outcome (see module docstring)."""
    return true_label if rng.random() < signal_p else 1 - true_label


def _gen_text_record(eff_label: int, terse: bool, rng: np.random.Generator, today: dt.date) -> tuple[str, str]:
    if eff_label == 1:
        category = rng.choice(CATEGORIES)
        template = rng.choice(NEGATIVE_TEMPLATES[category])
        text = template.format(amount=_sample_amount(rng), date=_sample_date(rng, today))
    else:
        category = "other"
        text = str(rng.choice(NEUTRAL_TEMPLATES))
    if terse:
        words = text.split()
        text = " ".join(words[: max(6, len(words) // 2)])
        if not text.endswith("."):
            text += "."
    return text, category


def _gen_spoken_record(eff_label: int, terse: bool, rng: np.random.Generator, today: dt.date) -> tuple[str, str]:
    if eff_label == 1:
        category = rng.choice(CATEGORIES)
        template = rng.choice(SPOKEN_NEGATIVE_TEMPLATES[category])
        text = template.format(amount=_sample_amount(rng), date=_sample_date(rng, today))
    else:
        category = "other"
        text = str(rng.choice(SPOKEN_NEUTRAL_TEMPLATES))
    if terse:
        words = text.split()
        text = " ".join(words[: max(5, len(words) // 2)])
    return text, category


def _gen_txn_features(eff_label: int, rng: np.random.Generator) -> dict:
    if eff_label == 1:
        failed_transfers = int(rng.poisson(3.0))
        declines = int(rng.poisson(2.2))
        txn_freq_change = float(rng.normal(-0.28, 0.15))
        app_sessions_change = float(rng.normal(-0.32, 0.20))
        days_since_complaint = float(rng.exponential(9.0))
        resolution_delay = float(np.clip(rng.normal(70.0, 22.0), 0, None))
    else:
        failed_transfers = int(rng.poisson(0.4))
        declines = int(rng.poisson(0.3))
        txn_freq_change = float(rng.normal(0.05, 0.10))
        app_sessions_change = float(rng.normal(0.02, 0.10))
        days_since_complaint = float(np.clip(rng.exponential(60.0), 0, 365))
        resolution_delay = float(np.clip(rng.normal(12.0, 8.0), 0, None))
    return {
        "failed_transfers_count": failed_transfers,
        "declines_count": declines,
        "txn_frequency_change": round(txn_freq_change, 4),
        "app_sessions_change": round(app_sessions_change, 4),
        "days_since_last_complaint": round(days_since_complaint, 1),
        "avg_resolution_delay": round(resolution_delay, 1),
    }


def generate_dataset(cfg: dict) -> pd.DataFrame:
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    n = cfg["data"]["n_customers"]
    today = dt.date.today()

    signal_p_text = 0.65
    signal_p_voice = 0.55
    signal_p_txn = 0.68

    rows = []
    complaint_rows = []
    voice_rows = []
    for i in range(n):
        customer_id = f"CUST{i:05d}"
        true_label = int(rng.random() < 0.35)
        segment = "ussd_user" if rng.random() < 0.5 else "mobile_app_user"
        terse = segment == "ussd_user"

        window_end = today - dt.timedelta(days=int(rng.integers(0, 7)))
        window_start = window_end - dt.timedelta(days=30)

        eff_text_label = _noisy_label(true_label, signal_p_text, rng)
        n_text = int(rng.integers(1, 3)) if eff_text_label == 1 else int(rng.integers(0, 2))
        text_records = []
        for _ in range(n_text):
            text, category = _gen_text_record(eff_text_label, terse, rng, today)
            text_records.append(text)
            complaint_rows.append({"customer_id": customer_id, "text": text, "category": category, "eff_label": eff_text_label})

        eff_voice_label = _noisy_label(true_label, signal_p_voice, rng)
        n_voice = int(rng.integers(1, 2)) if eff_voice_label == 1 else int(rng.integers(0, 2))
        voice_transcripts = []
        for _ in range(n_voice):
            text, category = _gen_spoken_record(eff_voice_label, terse, rng, today)
            voice_transcripts.append(text)
            voice_rows.append({"customer_id": customer_id, "transcript": text, "category": category, "eff_label": eff_voice_label})

        eff_txn_label = _noisy_label(true_label, signal_p_txn, rng)
        txn_features = _gen_txn_features(eff_txn_label, rng)

        rows.append(
            {
                "customer_id": customer_id,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "segment": segment,
                "text_records": text_records,
                "voice_transcripts": voice_transcripts,
                "txn_features": txn_features,
                "label": true_label,
            }
        )

    ensure_dir(cfg["paths"]["synthetic_dir"])
    write_jsonl(complaint_rows, resolve_path(cfg["paths"]["synthetic_dir"]) / "complaints.jsonl")
    write_jsonl(voice_rows, resolve_path(cfg["paths"]["synthetic_dir"]) / "voice_transcripts.jsonl")
    txn_df = pd.DataFrame(
        [{"customer_id": r["customer_id"], **r["txn_features"]} for r in rows]
    )
    txn_df.to_csv(resolve_path(cfg["paths"]["synthetic_dir"]) / "transactions.csv", index=False)

    return pd.DataFrame(rows)


def synthesize_demo_audio(df: pd.DataFrame, cfg: dict) -> None:
    """Generate a handful of real .wav call clips (offline TTS) so the voice
    pipeline (Whisper transcription -> text model) can be demonstrated end to
    end, per README S6.2. Falls back gracefully if no TTS engine/voice is
    available on this machine."""
    n_clips = cfg["voice_model"]["n_synthetic_clips"]
    audio_dir = ensure_dir(cfg["paths"]["audio_dir"])
    candidates = [r for _, r in df.iterrows() if len(r["voice_transcripts"]) > 0][:n_clips]

    try:
        import pyttsx3
    except ImportError:
        print("[ingest] pyttsx3 not installed - skipping demo audio synthesis.")
        return

    try:
        engine = pyttsx3.init()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[ingest] no TTS engine available ({exc}) - skipping demo audio synthesis.")
        return

    audio_paths = {}
    for row in candidates:
        cid = row["customer_id"]
        text = row["voice_transcripts"][0]
        out_path = audio_dir / f"{cid}.wav"
        try:
            engine.save_to_file(text, str(out_path))
        except Exception as exc:  # pragma: no cover
            print(f"[ingest] TTS failed for {cid}: {exc}")
            continue
        audio_paths[cid] = str(out_path.relative_to(REPO_ROOT))
    try:
        engine.runAndWait()
    except Exception:
        pass

    n_written = sum(1 for cid in audio_paths if (audio_dir / f"{cid}.wav").exists() and (audio_dir / f"{cid}.wav").stat().st_size > 0)
    print(f"[ingest] synthesized {n_written} demo call-audio clips into {audio_dir}")
    df["voice_audio_path"] = df["customer_id"].map(audio_paths)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    df = generate_dataset(cfg)
    if "voice_audio_path" not in df.columns:
        df["voice_audio_path"] = None
    synthesize_demo_audio(df, cfg)

    raw_dir = ensure_dir(cfg["paths"]["raw_dir"])
    write_jsonl(df.to_dict(orient="records"), raw_dir / "customers.jsonl")
    print(f"[ingest] wrote {len(df)} raw customer records to {raw_dir / 'customers.jsonl'}")
    print(f"[ingest] label prevalence (dissatisfied=1): {df['label'].mean():.3f}")


if __name__ == "__main__":
    main()
