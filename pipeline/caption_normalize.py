"""
caption_normalize.py — dedupe rolling YouTube/SRT cues, sentence-split, word attach.
"""
from __future__ import annotations

import re
from typing import Optional

from pipeline.srt import format_timestamp, parse_srt

_WORD_NORM_RE = re.compile(r"[^\w\u0400-\u04FF]+", re.UNICODE)
# Avoid splitting on common abbreviations (simple)
_ABBR_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K)\.$",
    re.IGNORECASE,
)


def normalize_word(word: str) -> str:
    return _WORD_NORM_RE.sub("", word).lower()


def dedupe_rolling_cues(cues: list[dict]) -> list[dict]:
    """
    Remove overlapping rolling-caption prefixes/suffixes between consecutive cues.
    Keeps timing of the surviving cue text.
    """
    if not cues:
        return []

    result: list[dict] = []
    emitted_words: list[str] = []

    for cue in cues:
        text = (cue.get("text") or "").strip()
        if not text:
            continue
        words = text.split()
        if not words:
            continue

        skip = 0
        max_overlap = min(len(words), len(emitted_words), 12)
        for k in range(max_overlap, 0, -1):
            if emitted_words[-k:] == [normalize_word(w) for w in words[:k]]:
                skip = k
                break

        new_words = words[skip:]
        if not new_words:
            continue

        new_text = " ".join(new_words)
        start_ms = int(cue["start_ms"])
        end_ms = int(cue["end_ms"])
        # If we skipped a prefix, shift start toward end proportionally
        if skip and len(words) > 0:
            frac = skip / len(words)
            span = max(end_ms - start_ms, 1)
            start_ms = start_ms + int(span * frac)

        result.append(
            {
                **cue,
                "text": new_text,
                "start_ms": start_ms,
                "end_ms": max(end_ms, start_ms + 1),
                "start": format_timestamp(start_ms),
                "end": format_timestamp(max(end_ms, start_ms + 1)),
            }
        )
        emitted_words.extend(normalize_word(w) for w in new_words)

    return result


def words_from_cues(cues: list[dict]) -> list[dict]:
    """Approximate per-word timing by linear distribution inside each cue."""
    words: list[dict] = []
    for cue in cues:
        parts = (cue.get("text") or "").split()
        if not parts:
            continue
        start_ms = int(cue["start_ms"])
        end_ms = int(cue["end_ms"])
        span = max(end_ms - start_ms, len(parts))
        n = len(parts)
        for i, raw in enumerate(parts):
            norm = normalize_word(raw)
            if not norm:
                continue
            ws = start_ms + int(span * i / n)
            we = start_ms + int(span * (i + 1) / n)
            if we <= ws:
                we = ws + 1
            words.append(
                {
                    "text": raw,
                    "word": norm,
                    "start_ms": ws,
                    "end_ms": we,
                }
            )
    return words


def sentences_from_words(words: list[dict]) -> list[dict]:
    """
    Group words into sentences using punctuation on word text.
    Returns [{text, start_ms, end_ms, words}, ...].
    """
    if not words:
        return []

    sentences: list[dict] = []
    buf: list[dict] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        text = " ".join(w["text"] for w in buf)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            sentences.append(
                {
                    "text": text,
                    "start_ms": buf[0]["start_ms"],
                    "end_ms": buf[-1]["end_ms"],
                    "words": buf[:],
                }
            )
        buf = []

    for w in words:
        buf.append(w)
        raw = w.get("text") or ""
        # Sentence end if word ends with .?!… (optionally followed by quotes)
        if re.search(r'[.!?…]["\'»)\]】]*$', raw):
            # Skip abbreviation-like endings
            joined = " ".join(x["text"] for x in buf)
            if _ABBR_RE.search(joined.strip()):
                continue
            flush()

    flush()
    return sentences


def script_text_from_sentences(sentences: list[dict]) -> str:
    """One sentence per line — matches the successful plain-text paste shape."""
    lines = []
    for s in sentences:
        t = (s.get("text") or "").strip()
        if t:
            lines.append(t)
    return "\n".join(lines)


def script_and_words_from_srt(content: str) -> tuple[str, list[dict]]:
    """Parse SRT → dedupe → word timeline → sentence script."""
    cues = parse_srt(content)
    if not cues:
        return content, []
    cues = dedupe_rolling_cues(cues)
    words = words_from_cues(cues)
    sentences = sentences_from_words(words)
    script = script_text_from_sentences(sentences) or " ".join(c["text"] for c in cues)
    return script, words


def script_and_words_from_word_timeline(words: list[dict]) -> tuple[str, list[dict]]:
    """Word timeline (json3) → sentence script + same words."""
    sentences = sentences_from_words(words)
    script = script_text_from_sentences(sentences)
    if not script and words:
        script = " ".join(w["text"] for w in words)
    return script, words


def _find_subsequence(haystack: list[str], needle: list[str], start: int) -> Optional[int]:
    if not needle:
        return start
    n = len(needle)
    limit = len(haystack) - n + 1
    for i in range(max(0, start), max(0, limit)):
        if haystack[i : i + n] == needle:
            return i
    first = needle[0]
    for i in range(max(0, start), len(haystack)):
        if haystack[i] == first:
            return i
    return None


def attach_timecodes_from_words(blocks: list[dict], words: list[dict]) -> list[dict]:
    """
    Map segmented blocks onto a word timeline with per-word ms.
    Produces non-overlapping start/end when blocks advance the cursor.
    """
    if not blocks or not words:
        return blocks

    haystack = [w["word"] for w in words]
    cursor = 0

    for block in blocks:
        target = [
            normalize_word(w)
            for w in (block.get("text") or "").split()
            if normalize_word(w)
        ]
        if not target:
            block["start"] = None
            block["end"] = None
            block["start_ms"] = None
            block["end_ms"] = None
            continue

        match_at = _find_subsequence(haystack, target, cursor)
        if match_at is None:
            match_at = min(cursor, max(0, len(words) - 1))
        match_end = min(match_at + len(target), len(words))
        if match_at >= len(words):
            block["start"] = None
            block["end"] = None
            block["start_ms"] = None
            block["end_ms"] = None
            continue

        span = words[match_at:match_end] or [words[match_at]]
        start_ms = int(span[0]["start_ms"])
        end_ms = int(span[-1]["end_ms"])
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        block["start_ms"] = start_ms
        block["end_ms"] = end_ms
        block["start"] = format_timestamp(start_ms)
        block["end"] = format_timestamp(end_ms)
        cursor = match_end

    return blocks
