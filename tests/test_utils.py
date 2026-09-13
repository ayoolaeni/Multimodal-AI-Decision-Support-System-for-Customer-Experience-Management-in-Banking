import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import normalise_text, set_global_seed, strip_pii  # noqa: E402


def test_strip_pii_email():
    assert "[EMAIL]" in strip_pii("contact me at john.doe@example.com please")
    assert "example.com" not in strip_pii("contact me at john.doe@example.com please")


def test_strip_pii_phone():
    out = strip_pii("call me on 08031234567 today")
    assert "[PHONE]" in out
    assert "08031234567" not in out


def test_strip_pii_account_number():
    out = strip_pii("my account 0123456789 was debited")
    assert "[ACCOUNT]" in out
    assert "0123456789" not in out


def test_strip_pii_name():
    out = strip_pii("Please contact Mr. John Okafor about this")
    assert "[NAME]" in out
    assert "John Okafor" not in out


def test_strip_pii_noop_on_clean_text():
    text = "the app has been down since yesterday"
    assert strip_pii(text) == text


def test_normalise_text_collapses_whitespace_and_case():
    assert normalise_text("  Hello   WORLD  \n") == "hello world"


def test_set_global_seed_is_deterministic():
    import numpy as np

    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert (a == b).all()
