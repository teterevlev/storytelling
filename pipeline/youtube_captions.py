"""
youtube_captions.py — fetch YouTube auto/manual captions as json3 via yt-dlp.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?.*?v=|shorts/|embed/|live/)|youtu\.be/)([A-Za-z0-9_-]{6,})"
)
_URL_LINE_RE = re.compile(
    r"^\s*(https?://)?(www\.)?(youtube\.com|youtu\.be)/[^\s]+$",
    re.IGNORECASE,
)


class YoutubeCaptionsError(Exception):
    """Raised when captions cannot be fetched or parsed."""


def looks_like_youtube_url(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if "\n" in stripped or "\r" in stripped:
        # Only treat single-line paste as URL
        lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
        if len(lines) != 1:
            return False
        stripped = lines[0]
    return bool(_URL_LINE_RE.match(stripped) or extract_video_id(stripped))


def extract_video_id(url: str) -> Optional[str]:
    if not url:
        return None
    m = _VIDEO_ID_RE.search(url.strip())
    return m.group(1) if m else None


def normalize_youtube_url(url: str) -> str:
    vid = extract_video_id(url)
    if not vid:
        raise YoutubeCaptionsError(f"Not a valid YouTube URL: {url!r}")
    return f"https://www.youtube.com/watch?v={vid}"


def _lang_candidates(lang: Optional[str]) -> list[str]:
    primary = (lang or "en").lower().strip()
    if primary.startswith("ru"):
        order = ["ru", "en"]
    else:
        order = ["en", "ru"]
    # yt-dlp accepts codes; also try bare codes
    out: list[str] = []
    for code in order:
        if code not in out:
            out.append(code)
    return out


def fetch_captions_json3(url: str, lang: Optional[str] = "en") -> dict:
    """
    Download caption track as json3 via yt-dlp and return parsed JSON.
    Prefers auto-subs; falls back across language candidates.
    """
    if not shutil.which("yt-dlp"):
        raise YoutubeCaptionsError(
            "yt-dlp is not installed or not on PATH. Install: brew install yt-dlp"
        )

    watch_url = normalize_youtube_url(url)
    last_err: Optional[str] = None

    with tempfile.TemporaryDirectory(prefix="ytcap_") as tmp:
        outtmpl = str(Path(tmp) / "cap.%(ext)s")
        for code in _lang_candidates(lang):
            # Prefer auto-subs; also request manual in case only those exist
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-auto-subs",
                "--write-subs",
                "--sub-langs",
                code,
                "--sub-format",
                "json3",
                "-o",
                outtmpl,
                watch_url,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise YoutubeCaptionsError("yt-dlp timed out while fetching captions") from exc

            files = sorted(Path(tmp).glob("*.json3"))
            if files:
                data = json.loads(files[0].read_text(encoding="utf-8"))
                data["_meta"] = {
                    "video_id": extract_video_id(watch_url),
                    "url": watch_url,
                    "lang": code,
                    "source_file": files[0].name,
                }
                return data

            err = (proc.stderr or proc.stdout or "").strip()
            last_err = err[-500:] if err else f"exit {proc.returncode}"

        raise YoutubeCaptionsError(
            f"No json3 captions found for {watch_url}. yt-dlp: {last_err or 'unknown error'}"
        )


def parse_json3_words(data: dict) -> list[dict]:
    """
    Parse YouTube json3 into word timeline:
    [{text, word, start_ms, end_ms}, ...]
    """
    events = data.get("events") or []
    words: list[dict] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        segs = ev.get("segs")
        if not segs:
            continue
        t_start = int(ev.get("tStartMs") or 0)
        duration = int(ev.get("dDurationMs") or 0)
        ev_end = t_start + max(duration, 0)

        # Collect non-newline segments with offsets
        timed_segs = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            raw = seg.get("utf8") or ""
            if raw in ("\n", "\r", "\r\n"):
                continue
            # Keep whitespace-only only if it joins words — skip pure newlines already
            text = raw.replace("\n", " ").strip()
            if not text:
                continue
            offset = seg.get("tOffsetMs")
            timed_segs.append((text, int(offset) if offset is not None else None))

        if not timed_segs:
            continue

        # Expand multi-word segs; assign times
        expanded: list[tuple[str, Optional[int]]] = []
        for text, offset in timed_segs:
            parts = text.split()
            if not parts:
                continue
            if len(parts) == 1:
                expanded.append((parts[0], offset))
            else:
                # First part keeps offset; rest inherit None for interpolation later
                expanded.append((parts[0], offset))
                for p in parts[1:]:
                    expanded.append((p, None))

        # Resolve absolute start times per word within event
        n = len(expanded)
        abs_starts: list[Optional[int]] = []
        for i, (_w, offset) in enumerate(expanded):
            if offset is not None:
                abs_starts.append(t_start + offset)
            else:
                abs_starts.append(None)

        # Fill missing starts by linear interpolation between known anchors
        known = [(i, s) for i, s in enumerate(abs_starts) if s is not None]
        if not known:
            # No word offsets — distribute across event duration
            for i, (w, _) in enumerate(expanded):
                if n == 1:
                    ws, we = t_start, max(ev_end, t_start + 1)
                else:
                    ws = t_start + int(duration * i / n)
                    we = t_start + int(duration * (i + 1) / n)
                    if we <= ws:
                        we = ws + 1
                words.append(
                    {
                        "text": w,
                        "word": _norm_word(w),
                        "start_ms": ws,
                        "end_ms": we,
                    }
                )
            continue

        # Leading unknowns → from event start
        first_i, first_s = known[0]
        for i in range(0, first_i):
            abs_starts[i] = t_start + int((first_s - t_start) * i / max(first_i, 1))

        for (a_i, a_s), (b_i, b_s) in zip(known, known[1:]):
            gap = b_i - a_i
            for j in range(1, gap):
                abs_starts[a_i + j] = a_s + int((b_s - a_s) * j / gap)

        # Trailing unknowns → toward event end
        last_i, last_s = known[-1]
        rem = n - 1 - last_i
        for j in range(1, rem + 1):
            abs_starts[last_i + j] = last_s + int(
                max(ev_end - last_s, rem) * j / (rem + 1)
            )

        for i, (w, _) in enumerate(expanded):
            ws = abs_starts[i] if abs_starts[i] is not None else t_start
            if i + 1 < n and abs_starts[i + 1] is not None:
                we = abs_starts[i + 1]
            else:
                we = ev_end if ev_end > ws else ws + 1
            if we <= ws:
                we = ws + 1
            words.append(
                {
                    "text": w,
                    "word": _norm_word(w),
                    "start_ms": int(ws),
                    "end_ms": int(we),
                }
            )

    # Drop empty normalized words
    return [w for w in words if w.get("word")]


def _norm_word(word: str) -> str:
    return re.sub(r"[^\w\u0400-\u04FF]+", "", word, flags=re.UNICODE).lower()


def fetch_word_timeline(url: str, lang: Optional[str] = "en") -> tuple[list[dict], dict]:
    """Fetch captions and return (words, meta)."""
    data = fetch_captions_json3(url, lang=lang)
    words = parse_json3_words(data)
    meta = data.get("_meta") or {}
    if not words:
        raise YoutubeCaptionsError(
            f"json3 captions for {meta.get('url') or url} contained no words"
        )
    return words, meta
