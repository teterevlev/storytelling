"""Content language (script) vs UI language helpers."""
from __future__ import annotations

import re
from typing import Optional

CONTENT_LANGS = ("en", "ru")
_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
_LETTER_RE = re.compile(r"[A-Za-zа-яА-ЯёЁ]")


def normalize_content_lang(value: Optional[str], default: str = "ru") -> str:
    raw = (value or "").strip().lower()
    if raw in ("", "auto", "detect"):
        return default if default in CONTENT_LANGS else "ru"
    if raw.startswith("ru"):
        return "ru"
    if raw.startswith("en"):
        return "en"
    return default if default in CONTENT_LANGS else "ru"


def detect_content_lang(text: Optional[str], default: str = "ru") -> str:
    """Heuristic: share of Cyrillic among letters. No extra dependencies."""
    sample = (text or "")[:8000]
    letters = _LETTER_RE.findall(sample)
    if not letters:
        return default if default in CONTENT_LANGS else "ru"
    cyr = sum(1 for ch in letters if _CYRILLIC_RE.match(ch))
    ratio = cyr / len(letters)
    return "ru" if ratio >= 0.25 else "en"


def resolve_content_lang(explicit: Optional[str], text: Optional[str]) -> str:
    """If explicit is empty/auto → detect from text; otherwise normalize explicit."""
    raw = (explicit or "").strip().lower()
    if raw in ("", "auto", "detect"):
        return detect_content_lang(text)
    return normalize_content_lang(raw)
