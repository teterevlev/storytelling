#!/usr/bin/env python3
"""Pragmatic clustering for all review tags. One pass, checkpointed, no optimization loop."""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from main import get_db  # noqa: E402
from pipeline.subcluster_seed import (  # noqa: E402
    demote_domain_leaky_subclusters,
    numbered_blocks_for_tag,
    resolve_review_task_ids,
    seed_tag_subclusters,
)

MODEL = "gpt-4o"
SEED = 42
MAX_WORKERS = 2
DISCOVERY_CAP = 80
HOLDOUT_CAP = 30

# Already accepted v2 — reuse, do not recompute
KEEP_TAGS = ("OPEN_LOOP", "TENSION")

DOMAIN_LEAK_RE = re.compile(
    r"(?i)гравитац|нептун|юпитер|планет|космос|галактик|завод|автомобил|"
    r"танк|ракет|войн|оруж|ссср|советск|nasa|voyager|ньютон"
)


def soft_target(n_examples: int) -> tuple[int, int]:
    if n_examples <= 3:
        return (1, 1)
    if n_examples <= 8:
        return (1, 3)
    if n_examples <= 20:
        return (3, 8)
    if n_examples <= 50:
        return (5, 12)
    if n_examples <= 120:
        return (8, 16)
    return (10, 18)


def client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)


def chat_json(cli: OpenAI, system: str, user: str, temperature: float = 0.2) -> dict:
    for attempt in range(10):
        try:
            resp = cli.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate_limit" in msg.lower():
                m = re.search(r"try again in (\d+)ms", msg, re.I)
                wait = (int(m.group(1)) / 1000.0 + 0.35) if m else min(8.0, 1.5 * (attempt + 1))
                print(f"  rate-limit sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            if attempt >= 4:
                raise
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError("chat_json failed")


ABSTRACT_SYSTEM = """Ты извлекаешь психологический механизм одного story-beat.
Правила:
- Игнорируй тему, сюжет, персонажей, объекты, страны, эпохи, числа, имена.
- mechanism/steps только через X/Y/Z и общие слова.
- Запрещены домены (машины, планеты, заводы, войны, бренды).
- mechanism и steps непустые, если есть эффект на зрителя.
JSON:
{"n":1,"mechanism":"...","steps":["X → ..."],"candidate_residual":false,"residual_reason":null}
"""


def abstract_one(cli: OpenAI, tag: str, item: dict) -> dict:
    payload = {"n": item["n"], "text": (item["text"] or "")[:1200]}
    data = {}
    for _ in range(3):
        data = chat_json(
            cli,
            ABSTRACT_SYSTEM,
            f"Тег: {tag}\n{json.dumps(payload, ensure_ascii=False)}",
            temperature=0.1,
        )
        if "items" in data and data["items"]:
            data = data["items"][0]
        if (data.get("mechanism") or "").strip() and (data.get("steps") or []):
            break
        payload["hint"] = "Верни непустые mechanism и steps через X/Y."
    return {
        "n": item["n"],
        "mechanism": (data.get("mechanism") or "").strip(),
        "steps": data.get("steps") or [],
        "candidate_residual": bool(data.get("candidate_residual")),
        "residual_reason": data.get("residual_reason"),
        "filename": item.get("filename"),
    }


def stratified_split(blocks: dict[int, dict], discovery_n: int, holdout_n: int):
    by_file = defaultdict(list)
    for n, b in blocks.items():
        by_file[b.get("filename") or "?"].append(n)
    rng = random.Random(SEED)
    for ns in by_file.values():
        rng.shuffle(ns)
    files = sorted(by_file.keys())
    pointers = {f: 0 for f in files}
    discovery, holdout = [], []

    def take(target, bucket):
        while len(bucket) < target:
            progressed = False
            for f in files:
                i = pointers[f]
                if i < len(by_file[f]):
                    bucket.append(by_file[f][i])
                    pointers[f] = i + 1
                    progressed = True
                    if len(bucket) >= target:
                        break
            if not progressed:
                break

    take(discovery_n, discovery)
    take(holdout_n, holdout)
    return discovery, holdout


def enrich_filenames(conn, blocks: dict[int, dict]) -> dict[int, dict]:
    if not blocks:
        return blocks
    ids = [b["block_id"] for b in blocks.values()]
    ph = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT b.id, t.filename FROM blocks b JOIN tasks t ON t.id=b.task_id WHERE b.id IN ({ph})",
        ids,
    ).fetchall()
    m = {r["id"]: r["filename"] for r in rows}
    for b in blocks.values():
        b["filename"] = m.get(b["block_id"], "?")
    return blocks


def run_abstracts(cli: OpenAI, tag: str, selected: list[dict]) -> list[dict]:
    cache_path = OUT / f"abstracts_cache_{tag.replace(':', '_')}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        by_n = {a["n"]: a for a in cached}
        need = [it for it in selected if it["n"] not in by_n or not (by_n[it["n"]].get("mechanism") or "").strip()]
        have = [by_n[it["n"]] for it in selected if it["n"] in by_n and (by_n[it["n"]].get("mechanism") or "").strip()]
        if not need:
            print(f"  [{tag}] abstracts cache hit {len(have)}", flush=True)
            return [by_n[it["n"]] for it in selected]
        print(f"  [{tag}] abstracts cache partial {len(have)}, need {len(need)}", flush=True)
        selected_run = need
    else:
        have = []
        selected_run = selected
        by_n = {}

    print(f"  [{tag}] abstracting {len(selected_run)}...", flush=True)

    def work(it):
        return abstract_one(cli, tag, it)

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(work, it): it["n"] for it in selected_run}
        for fut in as_completed(futs):
            row = fut.result()
            by_n[row["n"]] = row
            done += 1
            if done % 10 == 0 or done == len(selected_run):
                print(f"  [{tag}] abstracts {done}/{len(selected_run)}", flush=True)
                cache_path.write_text(
                    json.dumps(list(by_n.values()), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    cache_path.write_text(json.dumps(list(by_n.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    return [by_n[it["n"]] for it in selected if it["n"] in by_n]


def cluster_once(cli: OpenAI, tag: str, abstracts: list[dict], lo: int, hi: int) -> dict:
    system = f"""Кластеризуй обезличенные абстракции по психологическому механизму.
Ориентир: примерно {lo}–{hi} кластеров (мягко).
Лучше несколько разных механизмов, чем 2–3 сверхшироких.
Residual — меньшинство. Без доменных слов. Не именовать.
JSON:
{{"clusters":[{{"temp_id":1,"because":"...","draft_steps":["X → ..."],"members":[1,2],"why_not_split":"..."}}],
  "residual":[]}}
Каждый n ровно один раз.
"""
    payload = [{"n": a["n"], "mechanism": a["mechanism"], "steps": a["steps"]} for a in abstracts if a.get("mechanism")]
    empty = [a["n"] for a in abstracts if not a.get("mechanism")]
    data = chat_json(cli, system, f"Тег {tag}\n{json.dumps(payload, ensure_ascii=False)}", 0.25)
    data.setdefault("residual", [])
    data["residual"] = list(data.get("residual") or []) + empty
    return data


def light_revise(cli: OpenAI, tag: str, draft: dict, abstracts: list[dict], lo: int, hi: int) -> dict:
    if len(abstracts) <= 3:
        return draft
    system = f"""Лёгкая ревизия (один проход): разрежь явные dual-mechanism, слей дубликаты.
Не сваливай половину в residual. Ориентир {lo}–{hi}. Без доменов.
JSON: {{"clusters":[...],"residual":[...],"notes":[]}}
Все n ровно один раз.
"""
    return chat_json(
        cli,
        system,
        json.dumps({"tag": tag, "draft": draft, "abstracts": abstracts}, ensure_ascii=False),
        0.2,
    )


def name_once(cli: OpenAI, tag: str, clusters: list[dict], abstracts: list[dict]) -> dict:
    by_n = {a["n"]: a for a in abstracts}
    enriched = []
    for i, c in enumerate(clusters, start=1):
        members = c.get("members") or []
        tid = c.get("temp_id", i)
        enriched.append({
            "temp_id": tid,
            "because": c.get("because"),
            "draft_steps": c.get("draft_steps"),
            "members": members,
            "samples": [{"n": n, "mechanism": by_n.get(n, {}).get("mechanism")} for n in members[:6]],
        })
    system = """Короткое название (2–5 слов) и формула по механизму. Без доменных слов.
Обязательно сохрани temp_id из входа.
JSON: {"clusters":[{"temp_id":1,"name":"...","formula":"X → ...\\n→ ...","abstract":"..."}]}
"""
    data = chat_json(cli, system, json.dumps({"tag": tag, "clusters": enriched}, ensure_ascii=False), 0.2)
    # ensure temp_ids present
    for i, c in enumerate(data.get("clusters") or [], start=1):
        if "temp_id" not in c:
            c["temp_id"] = enriched[i - 1]["temp_id"] if i - 1 < len(enriched) else i
    return data

def soft_holdout(cli: OpenAI, tag: str, catalog: list[dict], holdout: list[dict]) -> dict:
    if not holdout or not catalog:
        return {"assignments": [], "summary": "skip"}
    system = """Назначь n в cluster_id whitelist или residual. Новые кластеры запрещены.
JSON: {"assignments":[{"n":1,"cluster_id":2,"confidence":"high|medium|low"}],"summary":"..."}
"""
    whitelist = [{"cluster_id": c["id"], "name": c["name"], "formula": c["formula"], "abstract": c["abstract"]} for c in catalog]
    return chat_json(
        cli,
        system,
        json.dumps({"tag": tag, "whitelist": whitelist, "holdout": holdout}, ensure_ascii=False),
        0.1,
    )


def to_catalog(revised: dict, named: dict) -> list[dict]:
    name_by = {}
    for c in (named.get("clusters") or []):
        if "temp_id" in c:
            try:
                name_by[int(c["temp_id"])] = c
            except (TypeError, ValueError):
                continue
    out = []
    for i, c in enumerate(revised.get("clusters") or [], start=1):
        raw_id = c.get("temp_id", i)
        try:
            tid = int(raw_id)
        except (TypeError, ValueError):
            tid = i
        nm = name_by.get(tid) or name_by.get(i) or {}
        # fallback: match by position in named list
        if not nm and i - 1 < len(named.get("clusters") or []):
            nm = named["clusters"][i - 1]
        out.append({
            "id": tid,
            "name": (nm.get("name") or c.get("name") or f"cluster-{tid}").strip(),
            "formula": (
                nm.get("formula")
                or c.get("formula")
                or "\n".join(c.get("draft_steps") or [])
            ).strip(),
            "abstract": (nm.get("abstract") or c.get("because") or c.get("abstract") or "").strip(),
            "members": list(c.get("members") or []),
        })
    out.sort(key=lambda x: x["id"])
    for i, c in enumerate(out, start=1):
        c["sort_order"] = i
    return out

def tiny_single_cluster(cli: OpenAI, tag: str, abstracts: list[dict]) -> dict:
    """1–2 examples: one cluster, no drama."""
    a = abstracts[0]
    named = chat_json(
        cli,
        "Дай name/formula/abstract по механизму без домена. JSON: "
        '{"name":"...","formula":"X → ...","abstract":"..."}',
        json.dumps({"tag": tag, "mechanism": a.get("mechanism"), "steps": a.get("steps")}, ensure_ascii=False),
        0.2,
    )
    members = [x["n"] for x in abstracts]
    catalog = [{
        "id": 1,
        "sort_order": 1,
        "name": (named.get("name") or "механизм").strip(),
        "formula": (named.get("formula") or "X → Y").strip(),
        "abstract": (named.get("abstract") or a.get("mechanism") or "").strip(),
        "members": members,
    }]
    return {
        "tag": tag,
        "catalog": catalog,
        "residual": [],
        "holdout": {"assignments": [], "summary": "tiny tag"},
        "total_examples": len(abstracts),
        "discovery_n": len(abstracts),
        "holdout_n": 0,
        "soft_target": soft_target(len(abstracts)),
        "domain_leaks": [],
    }


def process_tag(cli: OpenAI, tag: str, blocks: dict[int, dict]) -> dict:
    out_path = OUT / f"pragmatic_v2_{tag.replace(':', '_')}.json"
    # Resume: any finished tag with a catalog
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        if data.get("catalog"):
            print(f"[{tag}] resume existing ({len(data['catalog'])} clusters)", flush=True)
            data["tag"] = tag
            data.setdefault("total_examples", len(blocks))
            data.setdefault("soft_target", list(soft_target(len(blocks))))
            data["domain_leaks"] = [
                c["name"] for c in data.get("catalog") or []
                if DOMAIN_LEAK_RE.search(f"{c.get('name','')} {c.get('formula','')}")
            ]
            return data

    total = len(blocks)
    lo, hi = soft_target(total)
    print(f"[{tag}] n={total} soft_target={lo}-{hi}", flush=True)

    if total <= 2:
        items = [{"n": n, "text": b["text"], "block_id": b["block_id"], "filename": b.get("filename")} for n, b in blocks.items()]
        abstracts = run_abstracts(cli, tag, items)
        result = tiny_single_cluster(cli, tag, abstracts)
        result["total_examples"] = total
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    if total <= 25:
        disc_n, hold_n = max(1, total - min(5, total // 4)), min(5, total // 4)
        if disc_n + hold_n > total:
            hold_n = max(0, total - disc_n)
    else:
        disc_n = min(DISCOVERY_CAP, max(20, int(total * 0.55)))
        hold_n = min(HOLDOUT_CAP, max(5, total - disc_n))
        if disc_n + hold_n > total:
            hold_n = total - disc_n

    discovery_ns, holdout_ns = stratified_split(blocks, disc_n, hold_n)
    selected_ns = discovery_ns + holdout_ns
    selected = [
        {"n": n, "text": blocks[n]["text"], "block_id": blocks[n]["block_id"], "filename": blocks[n].get("filename")}
        for n in selected_ns
    ]
    abstracts = run_abstracts(cli, tag, selected)
    by_n = {a["n"]: a for a in abstracts}
    disc_abs = [by_n[n] for n in discovery_ns if n in by_n]
    hold_abs = [by_n[n] for n in holdout_ns if n in by_n]

    print(f"[{tag}] cluster {len(disc_abs)}...", flush=True)
    draft = cluster_once(cli, tag, disc_abs, lo, hi)
    print(f"[{tag}] revise...", flush=True)
    revised = light_revise(cli, tag, draft, disc_abs, lo, hi)
    print(f"[{tag}] name...", flush=True)
    named = name_once(cli, tag, revised.get("clusters") or [], disc_abs)
    catalog = to_catalog(revised, named)
    print(f"[{tag}] holdout {len(hold_abs)}...", flush=True)
    holdout = soft_holdout(
        cli,
        tag,
        [{"id": c["sort_order"], "name": c["name"], "formula": c["formula"], "abstract": c["abstract"]} for c in catalog],
        [{"n": a["n"], "mechanism": a["mechanism"], "steps": a["steps"]} for a in hold_abs],
    )
    leaks = [
        c["name"] for c in catalog
        if DOMAIN_LEAK_RE.search(f"{c.get('name','')} {c.get('formula','')} {c.get('abstract','')}")
    ]
    result = {
        "tag": tag,
        "total_examples": total,
        "discovery_n": len(discovery_ns),
        "holdout_n": len(holdout_ns),
        "soft_target": [lo, hi],
        "catalog": catalog,
        "residual": revised.get("residual") or [],
        "holdout": holdout,
        "domain_leaks": leaks,
        "revised_notes": revised.get("notes") or [],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # short md
    lines = [
        f"# Pragmatic v2: {tag}",
        "",
        f"- examples: {total}",
        f"- soft target: {lo}–{hi}",
        f"- clusters: **{len(catalog)}**",
        f"- residual: **{len(result['residual'])}**",
        f"- domain leaks: {leaks or 'none'}",
        "",
    ]
    for c in catalog:
        lines.append(f"### {c['sort_order']} — {c['name']} ({len(c['members'])})")
        lines.append("")
        lines.append(c["formula"])
        lines.append("")
        lines.append(f"Абстрактно: {c['abstract']}")
        lines.append("")
    (OUT / f"pragmatic_v2_{tag.replace(':', '_')}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{tag}] done clusters={len(catalog)} residual={len(result['residual'])} leaks={leaks}", flush=True)
    return result


def render_canon(results: dict[str, dict]) -> str:
    lines = [
        "# Кластеризация тегов по психологическому механизму",
        "",
        "## Источники",
        "",
        "Прагматичный прогон all-tags v2. См. PRAGMATIC_FREEZE.md.",
        "",
        "## Метод",
        "",
        "Абстракция → выборка → один cluster/revise/name. Без итеративной оптимизации.",
        "",
    ]
    # stable order: larger first then alpha
    tags = sorted(results.keys(), key=lambda t: (-results[t].get("total_examples", 0), t))
    for tag in tags:
        payload = results[tag]
        catalog = payload.get("catalog") or []
        residual = payload.get("residual") or []
        lines.append("---")
        lines.append("")
        lines.append(f"# {tag} ({len(catalog)} кластеров)")
        lines.append("")
        for c in catalog:
            lines.append(f"## Кластер {c['sort_order']} — {c['name']}")
            lines.append("")
            lines.append("Формула:")
            lines.append("")
            lines.append(c.get("formula") or "")
            lines.append("")
            lines.append(f"Абстрактно: \"{c.get('abstract') or ''}\"")
            lines.append("")
            members = ", ".join(str(n) for n in (c.get("members") or []))
            lines.append(f"Номера примеров: {members}")
            lines.append("")
        if residual:
            lines.append("## Временно неклассифицированные")
            lines.append("")
            lines.append("Номера примеров: " + ", ".join(str(n) for n in residual))
            lines.append("")
    return "\n".join(lines)


def holdout_stats(payload: dict) -> tuple[int, int]:
    assigns = (payload.get("holdout") or {}).get("assignments") or []

    def is_res(a):
        cid = a.get("cluster_id")
        return cid in (None, "residual", 0, "none") or str(cid).lower() == "residual"

    return sum(1 for a in assigns if not is_res(a)), len(assigns)


def main():
    only = sys.argv[1:]  # optional tag filter
    cli = client()
    with get_db() as conn:
        task_ids = resolve_review_task_ids(conn)
        tags_rows = conn.execute(
            f"""
            SELECT b.tag AS tag, COUNT(*) AS c
            FROM blocks b
            WHERE b.task_id IN ({",".join("?" * len(task_ids))})
              AND b.tag IS NOT NULL AND TRIM(b.tag) != ''
            GROUP BY b.tag
            ORDER BY c DESC
            """,
            task_ids,
        ).fetchall()
        tags = [r["tag"] for r in tags_rows]
        if only:
            tags = [t for t in tags if t in only]
        print("tags:", len(tags), tags[:10], "...", flush=True)

        results = {}
        for tag in tags:
            blocks = enrich_filenames(conn, numbered_blocks_for_tag(conn, tag, task_ids))
            results[tag] = process_tag(cli, tag, blocks)

    canon_path = OUT / "clusters_canon.md"
    canon_path.write_text(render_canon(results), encoding="utf-8")
    print("wrote", canon_path, flush=True)

    # comparison
    lines = [
        "# Pragmatic v2 — all tags",
        "",
        "| Tag | n | soft | clusters | residual | holdout | leaks |",
        "|-----|---|------|----------|----------|---------|-------|",
    ]
    ok = warn = bad = 0
    for tag in sorted(results, key=lambda t: -results[t].get("total_examples", 0)):
        p = results[tag]
        cat = p.get("catalog") or []
        residual = p.get("residual") or []
        lo, hi = (p.get("soft_target") or [0, 0])[:2]
        if isinstance(p.get("soft_target"), dict):
            lo, hi = 0, 0
        hc, ht = holdout_stats(p)
        leaks = p.get("domain_leaks") or []
        ncl = len(cat)
        # simple verdict
        if leaks:
            verdict = "LEAK"
            bad += 1
        elif ncl == 0:
            verdict = "EMPTY"
            bad += 1
        elif ht and hc / max(ht, 1) < 0.5:
            verdict = "WEAK_HOLDOUT"
            warn += 1
        elif ncl < max(1, lo // 2) and p.get("total_examples", 0) > 40:
            verdict = "COARSE"
            warn += 1
        else:
            verdict = "OK"
            ok += 1
        lines.append(
            f"| {tag} | {p.get('total_examples')} | {lo}–{hi} | {ncl} | {len(residual)} | {hc}/{ht} | {leaks or '—'} | {verdict} |"
        )
        # fix table - I added verdict as 8th col, fix header
    lines[2] = "| Tag | n | soft | clusters | residual | holdout | leaks | verdict |"
    lines[3] = "|-----|---|------|----------|----------|---------|-------|---------|"
    # rebuild properly
    body = []
    ok = warn = bad = 0
    for tag in sorted(results, key=lambda t: -results[t].get("total_examples", 0)):
        p = results[tag]
        cat = p.get("catalog") or []
        residual = p.get("residual") or []
        st = p.get("soft_target") or [0, 0]
        lo, hi = (st[0], st[1]) if isinstance(st, (list, tuple)) else (0, 0)
        hc, ht = holdout_stats(p)
        leaks = p.get("domain_leaks") or []
        ncl = len(cat)
        total = p.get("total_examples") or 0
        if leaks:
            verdict = "LEAK"
            bad += 1
        elif ncl == 0:
            verdict = "EMPTY"
            bad += 1
        elif ht and hc / max(ht, 1) < 0.5:
            verdict = "WEAK_HOLDOUT"
            warn += 1
        elif total > 40 and ncl < max(1, int(lo) // 2):
            verdict = "COARSE"
            warn += 1
        else:
            verdict = "OK"
            ok += 1
        body.append(
            f"| {tag} | {total} | {lo}–{hi} | {ncl} | {len(residual)} | {hc}/{ht} | {'; '.join(leaks) if leaks else '—'} | {verdict} |"
        )
    report = "\n".join([
        "# Pragmatic v2 — all tags",
        "",
        f"OK={ok}  WARN={warn}  BAD={bad}",
        "",
        "| Tag | n | soft | clusters | residual | holdout | leaks | verdict |",
        "|-----|---|------|----------|----------|---------|-------|---------|",
        *body,
        "",
        "## Notes",
        "",
        "- OK: без доменных утечек, каталог непустой, holdout в основном в whitelist.",
        "- COARSE: мало кластеров относительно мягкого ориентира (терпимо).",
        "- WEAK_HOLDOUT: holdout часто в residual.",
        "- LEAK/EMPTY: смотреть вручную.",
        "",
    ])
    (OUT / "pragmatic_v2_all_comparison.md").write_text(report, encoding="utf-8")
    print(report, flush=True)

    with get_db() as conn:
        conn.execute("DELETE FROM block_subcluster_assignments")
        conn.execute("DELETE FROM tag_subclusters")
        n = seed_tag_subclusters(conn, canon_path)
        demoted = demote_domain_leaky_subclusters(conn)
        conn.commit()
        rows = conn.execute(
            """
            SELECT parent_tag, COUNT(*) c FROM tag_subclusters
            WHERE status='active' AND slug!='residual'
            GROUP BY parent_tag ORDER BY c DESC
            """
        ).fetchall()
        print("seeded", n, "demoted", len(demoted), flush=True)
        for r in rows:
            print(f"  {r['parent_tag']}: {r['c']}", flush=True)


if __name__ == "__main__":
    main()
