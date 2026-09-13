"""Shared helpers: config loading, seeding, PII stripping, JSON-list IO."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?234|0)[\s-]?\d{2,3}[\s-]?\d{3}[\s-]?\d{4}\b")
_ACCOUNT_RE = re.compile(r"\b\d{10}\b|\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_NAME_PREFIX_RE = re.compile(
    r"\b(?:mr|mrs|miss|ms|dr|mallam|chief|alhaji|alhaja|engr|barr)\.?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?",
    re.IGNORECASE,
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def strip_pii(text: str) -> str:
    """Redact emails, phone numbers, account/card numbers, and honorific+name
    patterns from free text. Applied before any text is persisted, logged, or
    displayed."""
    if not text:
        return text
    redacted = _EMAIL_RE.sub("[EMAIL]", text)
    redacted = _PHONE_RE.sub("[PHONE]", redacted)
    redacted = _ACCOUNT_RE.sub("[ACCOUNT]", redacted)
    redacted = _NAME_PREFIX_RE.sub("[NAME]", redacted)
    return redacted


def normalise_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def resolve_path(relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


def read_jsonl(path: str | Path) -> list[dict]:
    path = resolve_path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_dir(path: str | Path) -> Path:
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
