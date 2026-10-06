"""
srt.py — парсинг SubRip (.srt) и привязка таймкодов к сегментированным блокам.
"""
from __future__ import annotations

import re
from typing import Optional

_TS_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{1,3})"
)
_ARROW_RE = re.compile(r"\s*-->\s*")
_INDEX_RE = re.compile(r"^\d+$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def parse_timestamp(value: str) -> Optional[int]:
    """SRT timestamp → milliseconds."""
    m = _TS_RE.fullmatch(value.strip())
    if not m:
        return None
    ms = int(m.group("ms").ljust(3, "0")[:3])
    return (
        int(m.group("h")) * 3_600_000
        + int(m.group("m")) * 60_000
        + int(m.group("s")) * 1_000
        + ms
    )


def format_timestamp(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    if ms < 0:
        ms = 0
    h, rem = divmod(int(ms), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def _clean_cue_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_srt(content: str) -> list[dict]:
    """Parse SRT into cues: {index, start_ms, end_ms, start, end, text}."""
    if not content or not content.strip():
        return []

    text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Split on blank lines between cues
    chunks = re.split(r"\n\s*\n+", text)
    cues: list[dict] = []

    for chunk in chunks:
        lines = [ln.strip() for ln in chunk.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue

        idx_offset = 0
        cue_index = None
        if _INDEX_RE.match(lines[0]):
            cue_index = int(lines[0])
            idx_offset = 1
        if idx_offset >= len(lines):
            continue

        timing = lines[idx_offset]
        if "-->" not in timing:
            continue
        parts = _ARROW_RE.split(timing)
        if len(parts) != 2:
            continue
        start_raw = parts[0].strip().split()[0]
        end_raw = parts[1].strip().split()[0]
        start_ms = parse_timestamp(start_raw)
        end_ms = parse_timestamp(end_raw)
        if start_ms is None or end_ms is None:
            continue

        body = _clean_cue_text("\n".join(lines[idx_offset + 1 :]))
        if not body:
            continue

        cues.append(
            {
                "index": cue_index if cue_index is not None else len(cues) + 1,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start": format_timestamp(start_ms),
                "end": format_timestamp(end_ms),
                "text": body,
            }
        )

    return cues


def looks_like_srt(content: str) -> bool:
    """Heuristic: pasted/uploaded text is SubRip."""
    if not content or not content.strip():
        return False
    cues = parse_srt(content)
    if len(cues) >= 2:
        return True
    # Single-cue shorts still count if timing line is present
    if len(cues) == 1 and "-->" in content:
        return True
    return False


def cues_to_plain_text(cues: list[dict]) -> str:
    """Join cue texts into a single script for segmentation."""
    parts = [c["text"] for c in cues if c.get("text")]
    return " ".join(parts)


def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w\u0400-\u04FF]+", "", word, flags=re.UNICODE).lower()


def _cue_word_timeline(cues: list[dict]) -> list[dict]:
    """Flatten cues into a word stream carrying cue start/end times."""
    words: list[dict] = []
    for cue in cues:
        for raw in (cue.get("text") or "").split():
            norm = _normalize_word(raw)
            if not norm:
                continue
            words.append(
                {
                    "word": norm,
                    "start_ms": cue["start_ms"],
                    "end_ms": cue["end_ms"],
                }
            )
    return words


def _find_subsequence(haystack: list[str], needle: list[str], start: int) -> Optional[int]:
    if not needle:
        return start
    n = len(needle)
    limit = len(haystack) - n + 1
    for i in range(max(0, start), max(0, limit) + 1):
        if i >= len(haystack):
            break
        if haystack[i : i + n] == needle:
            return i
    # Fuzzy: allow first-word search then take len(needle) words
    first = needle[0]
    for i in range(max(0, start), len(haystack)):
        if haystack[i] == first:
            return i
    return None


def attach_timecodes_to_blocks(blocks: list[dict], cues: list[dict]) -> list[dict]:
    """
    Sequentially map segmented blocks onto the SRT word timeline.
    Sets start/end (SRT strings) and start_ms/end_ms on each block.
    """
    if not blocks or not cues:
        return blocks

    timeline = _cue_word_timeline(cues)
    if not timeline:
        return blocks

    haystack = [w["word"] for w in timeline]
    cursor = 0

    for block in blocks:
        target = [
            _normalize_word(w)
            for w in (block.get("text") or "").split()
            if _normalize_word(w)
        ]
        if not target:
            block["start"] = None
            block["end"] = None
            block["start_ms"] = None
            block["end_ms"] = None
            continue

        match_at = _find_subsequence(haystack, target, cursor)
        if match_at is None:
            match_at = min(cursor, max(0, len(timeline) - 1))
        match_end = min(match_at + len(target), len(timeline))
        if match_at >= len(timeline):
            block["start"] = None
            block["end"] = None
            block["start_ms"] = None
            block["end_ms"] = None
            continue

        span = timeline[match_at:match_end] or [timeline[match_at]]
        start_ms = span[0]["start_ms"]
        end_ms = span[-1]["end_ms"]
        block["start_ms"] = start_ms
        block["end_ms"] = end_ms
        block["start"] = format_timestamp(start_ms)
        block["end"] = format_timestamp(end_ms)
        cursor = match_end

    return blocks
