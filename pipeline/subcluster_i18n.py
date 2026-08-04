"""Bilingual subcluster fields (RU primary + EN twin columns)."""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .content_lang import detect_content_lang, normalize_content_lang

_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
_LETTER_RE = re.compile(r"[A-Za-zа-яА-ЯёЁ]")
_SKELETON_FORMULA_RE = re.compile(r"^[\sXYZ→\->\n\\]+$", re.I)

EN_FIELDS = ("name_en", "formula_en", "abstract_en")
RU_FIELDS = ("name", "formula", "abstract")
FIELD_PAIRS = (
    ("name", "name_en"),
    ("formula", "formula_en"),
    ("abstract", "abstract_en"),
)


def is_language_neutral(text: Optional[str]) -> bool:
    sample = (text or "").strip()
    if not sample:
        return True
    if _SKELETON_FORMULA_RE.match(sample):
        return True
    return not _LETTER_RE.search(sample)


def field_lang(text: Optional[str], default: str = "ru") -> str:
    """Detect language of a short catalog string."""
    sample = (text or "").strip()
    if not sample or is_language_neutral(sample):
        return default
    letters = _LETTER_RE.findall(sample)
    if not letters:
        return default
    cyr = sum(1 for ch in letters if _CYRILLIC_RE.match(ch))
    return "ru" if (cyr / len(letters)) >= 0.25 else "en"


def ensure_subcluster_i18n_columns(conn) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tag_subclusters)").fetchall()}
    for col in EN_FIELDS:
        if col not in cols:
            conn.execute(f"ALTER TABLE tag_subclusters ADD COLUMN {col} TEXT")


def normalize_existing_subcluster_langs(conn) -> dict[str, int]:
    """Copy EN text sitting in RU columns into *_en when *_en is empty.

    Does not clear RU columns (name is NOT NULL). Translator overwrites RU later
    when the RU slot still looks English.
    """
    ensure_subcluster_i18n_columns(conn)
    rows = conn.execute(
        """
        SELECT id, name, formula, abstract, name_en, formula_en, abstract_en
        FROM tag_subclusters
        WHERE sort_order > 0
        """
    ).fetchall()
    moved = 0
    for row in rows:
        updates: dict[str, str] = {}
        for ru_key, en_key in FIELD_PAIRS:
            ru_val = (row[ru_key] or "").strip() if ru_key in row.keys() else ""
            en_val = (row[en_key] or "").strip() if en_key in row.keys() else ""
            if not ru_val or en_val:
                continue
            if field_lang(ru_val, default="ru") != "en":
                continue
            updates[en_key] = ru_val
            moved += 1
        if not updates:
            continue
        sets = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE tag_subclusters SET {sets} WHERE id = ?",
            (*updates.values(), row["id"]),
        )
    return {"moved_en_from_ru_slot": moved, "rows": len(rows)}


def pick_localized_fields(row: Mapping[str, Any], lang: Optional[str]) -> dict[str, str]:
    """Return name/formula/abstract for the requested language with fallback."""
    want = normalize_content_lang(lang, default="ru")
    name_ru = (row.get("name") or "").strip()
    name_en = (row.get("name_en") or "").strip()
    formula_ru = (row.get("formula") or "").strip()
    formula_en = (row.get("formula_en") or "").strip()
    abstract_ru = (row.get("abstract") or "").strip()
    abstract_en = (row.get("abstract_en") or "").strip()

    if want == "en":
        return {
            "name": name_en or name_ru,
            "formula": formula_en or formula_ru,
            "abstract": abstract_en or abstract_ru,
        }
    return {
        "name": name_ru or name_en,
        "formula": formula_ru or formula_en,
        "abstract": abstract_ru or abstract_en,
    }


def localize_subcluster_dict(row: Mapping[str, Any], lang: Optional[str]) -> dict[str, Any]:
    """Copy row and overwrite name/formula/abstract with localized values.

    Also exposes name_ru/name_en/… for clients that want both.
    """
    out = dict(row)
    out["name_ru"] = (row.get("name") or "").strip() or None
    out["name_en"] = (row.get("name_en") or "").strip() or None
    out["formula_ru"] = (row.get("formula") or "").strip() or None
    out["formula_en"] = (row.get("formula_en") or "").strip() or None
    out["abstract_ru"] = (row.get("abstract") or "").strip() or None
    out["abstract_en"] = (row.get("abstract_en") or "").strip() or None
    picked = pick_localized_fields(row, lang)
    out["name"] = picked["name"]
    out["formula"] = picked["formula"]
    out["abstract"] = picked["abstract"]
    out["locale"] = normalize_content_lang(lang, default="ru")
    return out


def missing_translation_sides(row: Mapping[str, Any]) -> dict[str, list[str]]:
    """Which logical fields need translation, keyed by target lang ('en'|'ru')."""
    need_en: list[str] = []
    need_ru: list[str] = []
    for ru_key, en_key in FIELD_PAIRS:
        ru_val = (row.get(ru_key) or "").strip()
        en_val = (row.get(en_key) or "").strip()
        if is_language_neutral(ru_val) and is_language_neutral(en_val):
            continue
        # Language-neutral skeleton present on one side — mirror to the other
        if ru_val and is_language_neutral(ru_val) and not en_val:
            continue  # identical symbols; treat as shared, no translation job
        if en_val and is_language_neutral(en_val) and not ru_val:
            continue
        if ru_val and en_val and is_language_neutral(ru_val) and is_language_neutral(en_val):
            continue
        ru_is_en = bool(ru_val) and not is_language_neutral(ru_val) and field_lang(ru_val, default="ru") == "en"
        if ru_val and not en_val and not ru_is_en and not is_language_neutral(ru_val):
            need_en.append(ru_key)
        if en_val and (not ru_val or ru_is_en) and not is_language_neutral(en_val):
            need_ru.append(ru_key)
    return {"en": need_en, "ru": need_ru}


def detect_catalog_source_lang(row: Mapping[str, Any]) -> str:
    """Best-effort primary language of a cluster row."""
    blob = " ".join(
        [
            str(row.get("name") or ""),
            str(row.get("formula") or ""),
            str(row.get("abstract") or ""),
            str(row.get("name_en") or ""),
            str(row.get("formula_en") or ""),
            str(row.get("abstract_en") or ""),
        ]
    )
    return detect_content_lang(blob, default="ru")
