import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ingest import generate_dataset  # noqa: E402
from utils import load_config  # noqa: E402


def _small_cfg():
    cfg = load_config()
    cfg["data"]["n_customers"] = 60
    return cfg


def test_generate_dataset_schema():
    df = generate_dataset(_small_cfg())
    assert len(df) == 60
    required = {"customer_id", "window_start", "window_end", "segment", "text_records",
                "voice_transcripts", "txn_features", "label"}
    assert required.issubset(df.columns)
    assert df["customer_id"].is_unique
    assert set(df["label"].unique()).issubset({0, 1})
    assert set(df["segment"].unique()).issubset({"mobile_app_user", "ussd_user"})


def test_generate_dataset_label_is_minority_class():
    df = generate_dataset(_small_cfg())
    prevalence = df["label"].mean()
    assert 0.15 < prevalence < 0.55


def test_generate_dataset_txn_features_always_present_pre_missingness():
    df = generate_dataset(_small_cfg())
    assert all(isinstance(f, dict) and len(f) == 6 for f in df["txn_features"])


def test_generate_dataset_is_seed_reproducible():
    cfg = _small_cfg()
    df1 = generate_dataset(cfg)
    df2 = generate_dataset(cfg)
    assert df1["label"].tolist() == df2["label"].tolist()
    assert df1["text_records"].tolist() == df2["text_records"].tolist()
