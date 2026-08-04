#!/usr/bin/env python3
"""Mass-rename tag_subclusters named cluster-N → mechanism names via OpenAI.

Does not touch already-named clusters. Updates DB + clusters_canon.md.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CANON = OUT / "clusters_canon.md"
CACHE = OUT / "rename_placeholders_cache.json"

sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from main import get_db  # noqa: E402
from pipeline.subcluster_seed import (  # noqa: E402
    DOMAIN_LEAK_NAME_RE,
    demote_domain_leaky_subclusters,
    seed_tag_subclusters,
)

MODEL = "gpt-4o"
PLACEHOLDER_RE = re.compile(r"^cluster[-\s]?\d+$", re.I)
CANON_HEADING_RE = re.compile(
    r"^(## Кластер\s+)(\d+)(\s+[—\-]\s+)(.+)$",
    re.M,
)


def client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)


def chat_json(cli: OpenAI, system: str, user: str, temperature: float = 0.2) -> dict:
    for attempt in range(8):
        try:
            resp = cli.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            wait = min(2 ** attempt, 30)
            print(f"  retry {attempt + 1}: {exc} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError("chat_json failed")


def is_placeholder(name: str | None) -> bool:
    return bool(PLACEHOLDER_RE.match((name or "").strip()))


def normalize_name(name: str) -> str:
    value = re.sub(r"\s+", " ", (name or "").strip())
    value = value.strip(" «»\"'")
    return value


def name_batch(
    cli: OpenAI,
    tag: str,
    placeholders: list[dict],
    existing_names: list[str],
    reject_notes: list[str] | None = None,
) -> dict[int, str]:
    payload = {
        "tag": tag,
        "already_named_siblings": existing_names,
        "to_rename": [
            {
                "id": p["id"],
                "sort_order": p["sort_order"],
                "formula": p.get("formula") or "",
                "abstract": p.get("abstract") or "",
            }
            for p in placeholders
        ],
    }
    if reject_notes:
        payload["fix_previous"] = reject_notes

    system = """Ты даёшь короткие РУССКИЕ названия психологическим кластерам внутри одного функционального тега.

Правила:
- Название: 2–6 слов про МЕХАНИЗМ воздействия на зрителя, не про сюжет/объект/домен.
- Запрещены доменные слова: планеты, космос, заводы, войны, бренды, эпохи, конкретные объекты.
- Не используй шаблоны вроде cluster-N, «механизм 3», «кластер 5».
- Не повторяй названия из already_named_siblings и не дублируй имена внутри ответа.
- Если formula/abstract на английском — всё равно дай русское имя механизма.
- Не переписывай formula/abstract — только name.

Ответ строго JSON:
{"names":[{"id":123,"name":"..."}]}
Ровно по одной записи на каждый id из to_rename.
"""
    data = chat_json(cli, system, json.dumps(payload, ensure_ascii=False), temperature=0.25)
    out: dict[int, str] = {}
    for item in data.get("names") or []:
        try:
            cid = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        name = normalize_name(str(item.get("name") or ""))
        if name:
            out[cid] = name
    return out


def validate_names(
    proposed: dict[int, str],
    placeholders: list[dict],
    reserved: set[str],
) -> tuple[dict[int, str], list[str]]:
    ok: dict[int, str] = {}
    rejects: list[str] = []
    seen = {n.casefold() for n in reserved}
    wanted = {p["id"] for p in placeholders}

    for cid, name in proposed.items():
        if cid not in wanted:
            continue
        if is_placeholder(name):
            rejects.append(f"id={cid}: still placeholder {name!r}")
            continue
        if DOMAIN_LEAK_NAME_RE.search(name):
            rejects.append(f"id={cid}: domain leak {name!r}")
            continue
        if len(name) < 4 or len(name) > 80:
            rejects.append(f"id={cid}: bad length {name!r}")
            continue
        key = name.casefold()
        if key in seen:
            rejects.append(f"id={cid}: duplicate {name!r}")
            continue
        seen.add(key)
        ok[cid] = name

    missing = wanted - set(ok)
    for cid in sorted(missing):
        if not any(f"id={cid}:" in r for r in rejects):
            rejects.append(f"id={cid}: missing name")
    return ok, rejects


def load_cache() -> dict:
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {"by_id": {}}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def update_canon_md(renames_by_tag: dict[str, dict[int, str]]) -> int:
    """Replace ## Кластер N — cluster-N headings using sort_order → new name."""
    text = CANON.read_text(encoding="utf-8")
    current_tag = None
    changed = 0

    def on_line(line: str) -> str:
        nonlocal current_tag, changed
        if line.startswith("# ") and not line.startswith("## "):
            # "# CONTEXT (16 кластеров..." or "# OPEN_LOOP ..."
            m = re.match(r"^#\s+([A-Z_]+)(?:\s|\()", line)
            if m:
                current_tag = m.group(1)
            return line
        m = CANON_HEADING_RE.match(line)
        if not m or not current_tag:
            return line
        order = int(m.group(2))
        old_name = m.group(4).strip()
        new_name = (renames_by_tag.get(current_tag) or {}).get(order)
        if not new_name:
            return line
        if not is_placeholder(old_name) and old_name != new_name:
            # already named in canon — still allow overwrite if it was placeholder-like
            if not is_placeholder(old_name):
                return line
        changed += 1
        return f"{m.group(1)}{order}{m.group(3)}{new_name}"

    out_lines = [on_line(line) for line in text.splitlines()]
    # preserve trailing newline style
    CANON.write_text("\n".join(out_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return changed


def main() -> None:
    cli = client()
    cache = load_cache()
    cache_by_id: dict[str, str] = cache.setdefault("by_id", {})

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, parent_tag, sort_order, slug, name, formula, abstract, status
            FROM tag_subclusters
            WHERE sort_order > 0 AND slug != 'residual'
            ORDER BY parent_tag, sort_order
            """
        ).fetchall()
        all_rows = [dict(r) for r in rows]

    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_tag[r["parent_tag"]].append(r)

    placeholders_all = [r for r in all_rows if is_placeholder(r["name"])]
    print(f"placeholders: {len(placeholders_all)} / {len(all_rows)}", flush=True)

    renames: dict[int, str] = {}
    renames_by_tag_order: dict[str, dict[int, str]] = defaultdict(dict)
    report_lines = ["# Rename placeholders cluster-N", "", f"Total placeholders: {len(placeholders_all)}", ""]

    for tag in sorted(by_tag):
        rows_tag = by_tag[tag]
        placeholders = [r for r in rows_tag if is_placeholder(r["name"])]
        if not placeholders:
            continue
        existing = [
            r["name"] for r in rows_tag
            if not is_placeholder(r["name"]) and r["name"]
        ]
        print(f"[{tag}] rename {len(placeholders)}...", flush=True)

        # Apply cache hits first
        pending = []
        for p in placeholders:
            cached = cache_by_id.get(str(p["id"]))
            if cached and not is_placeholder(cached) and not DOMAIN_LEAK_NAME_RE.search(cached):
                if cached.casefold() not in {n.casefold() for n in existing}:
                    if cached.casefold() not in {n.casefold() for n in renames.values()}:
                        renames[p["id"]] = cached
                        renames_by_tag_order[tag][p["sort_order"]] = cached
                        existing.append(cached)
                        continue
            pending.append(p)

        rejects: list[str] = []
        for attempt in range(3):
            if not pending:
                break
            reserved = set(existing) | {renames[i] for i in renames if any(
                r["id"] == i and r["parent_tag"] == tag for r in rows_tag
            )}
            # also reserve names already accepted this tag
            for order, name in renames_by_tag_order[tag].items():
                reserved.add(name)

            proposed = name_batch(
                cli,
                tag,
                pending,
                sorted(reserved),
                reject_notes=rejects or None,
            )
            ok, rejects = validate_names(proposed, pending, reserved)
            for cid, name in ok.items():
                renames[cid] = name
                cache_by_id[str(cid)] = name
                row = next(p for p in pending if p["id"] == cid)
                renames_by_tag_order[tag][row["sort_order"]] = name
                existing.append(name)
            pending = [p for p in pending if p["id"] not in ok]
            if pending:
                print(f"  [{tag}] attempt {attempt + 1}: {len(ok)} ok, {len(pending)} left", flush=True)
                for r in rejects[:8]:
                    print(f"    reject: {r}", flush=True)
            save_cache(cache)

        if pending:
            # last-resort: individual rename
            for p in list(pending):
                one = name_batch(cli, tag, [p], existing, reject_notes=rejects)
                ok, rejects = validate_names(one, [p], set(existing))
                if p["id"] in ok:
                    name = ok[p["id"]]
                    renames[p["id"]] = name
                    cache_by_id[str(p["id"])] = name
                    renames_by_tag_order[tag][p["sort_order"]] = name
                    existing.append(name)
                    pending.remove(p)
                else:
                    print(f"  FAIL id={p['id']} {tag}/{p['sort_order']}: {rejects}", flush=True)
            save_cache(cache)

        report_lines.append(f"## {tag}")
        report_lines.append("")
        for p in placeholders:
            new = renames.get(p["id"], "???")
            report_lines.append(f"- {p['sort_order']}: `{p['name']}` → **{new}**")
        report_lines.append("")

    if len(renames) != len(placeholders_all):
        missing = [p for p in placeholders_all if p["id"] not in renames]
        print(f"WARNING: {len(missing)} unresolved", flush=True)
        for p in missing:
            print(f"  unresolved {p['parent_tag']}#{p['sort_order']} id={p['id']}", flush=True)

    with get_db() as conn:
        for cid, name in renames.items():
            conn.execute(
                "UPDATE tag_subclusters SET name = ? WHERE id = ?",
                (name, cid),
            )
        demoted = demote_domain_leaky_subclusters(conn)
        conn.commit()
    print(f"DB updated: {len(renames)} names; demoted after: {len(demoted)}", flush=True)

    canon_changed = update_canon_md(renames_by_tag_order)
    print(f"canon headings updated: {canon_changed}", flush=True)

    # Re-seed from canon to keep slug/examples path consistent with file source of truth
    with get_db() as conn:
        n = seed_tag_subclusters(conn, CANON)
        demoted2 = demote_domain_leaky_subclusters(conn)
        conn.commit()
    print(f"reseeded from canon: {n}; demoted: {len(demoted2)}", flush=True)

    left = 0
    with get_db() as conn:
        left = conn.execute(
            """
            SELECT COUNT(*) AS c FROM tag_subclusters
            WHERE sort_order > 0 AND name GLOB 'cluster-*'
            """
        ).fetchone()["c"]
    print(f"remaining cluster-* in DB: {left}", flush=True)

    report_path = OUT / "rename_placeholders_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    save_cache(cache)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
