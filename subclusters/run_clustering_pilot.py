#!/usr/bin/env python3
"""Pilot clustering for OPEN_LOOP + TENSION per clustering_rules.md."""
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
sys.path.insert(0, str(ROOT))

# load .env
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from main import get_db  # noqa: E402
from pipeline.subcluster_seed import numbered_blocks_for_tag, resolve_review_task_ids  # noqa: E402

MODEL = "gpt-4o"
OUT_DIR = Path(__file__).resolve().parent
SEED = 42
DISCOVERY_TARGET = 80
HOLDOUT_TARGET = 30
ABSTRACT_BATCH = 1
MAX_WORKERS = 2


DOMAIN_LEAK_RE = re.compile(
    r"\b(гравитац|нептун|юпитер|планет|космос|завод|автомобил|войн|оружи|"
    r"gravity|neptune|jupiter|planet|factory|war|tank|missile|"
    r"ссср|советск|nasa|voyager)\w*",
    re.IGNORECASE,
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
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:
            msg = str(exc)
            if "rate_limit" in msg.lower() or "429" in msg:
                wait = 2.0 * (attempt + 1)
                # parse "try again in Xms" if present
                m = re.search(r"try again in (\d+)ms", msg, re.I)
                if m:
                    wait = max(wait, int(m.group(1)) / 1000.0 + 0.25)
                print(f"  rate-limit sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            if attempt >= 3:
                raise
            time.sleep(1.5 * (attempt + 1))
            print(f"  retry after error: {exc}", flush=True)
    raise RuntimeError("chat_json failed after retries")

ABSTRACT_SYSTEM = """Ты извлекаешь психологический механизм одного story-beat.

Жёсткие правила:
- Игнорируй тему, сюжет, персонажей, объекты, страны, эпохи, числа, имена.
- В mechanism и steps используй только X/Y/Z и общие слова (факт, причина, угроза, обещание, ожидание…).
- Запрещено упоминать конкретные домены (машины, планеты, заводы, войны, бренды).
- mechanism и steps ОБЯЗАТЕЛЬНО непустые, если в тексте есть хоть какой-то эффект на зрителя
  (интрига, угроза, ожидание, контраст, обещание раскрытия, эскалация и т.п.).
- candidate_residual=true только если текста недостаточно даже для обезличенного механизма.

Ответ СТРОГО JSON (один объект, не массив):
{
  "n": 1,
  "mechanism": "одно предложение с X/Y",
  "steps": ["X → …", "→ …"],
  "candidate_residual": false,
  "residual_reason": null
}
"""


def abstract_one(cli: OpenAI, tag: str, item: dict) -> dict:
    payload = {"n": item["n"], "text": (item["text"] or "")[:1200]}
    data = {}
    for attempt in range(3):
        data = chat_json(
            cli,
            ABSTRACT_SYSTEM,
            f"Тег: {tag}\nАбстрагируй пример:\n{json.dumps(payload, ensure_ascii=False)}",
            temperature=0.1,
        )
        # accept either bare object or {"items":[...]}
        if "items" in data and data["items"]:
            data = data["items"][0]
        mech = (data.get("mechanism") or "").strip()
        steps = data.get("steps") or []
        if mech and steps:
            break
        # strengthen retry
        payload = {
            "n": item["n"],
            "text": (item["text"] or "")[:1200],
            "hint": "Верни непустые mechanism и steps через X/Y. Не оставляй пустым.",
        }
    return {
        "n": item["n"],
        "mechanism": (data.get("mechanism") or "").strip(),
        "steps": data.get("steps") or [],
        "candidate_residual": bool(data.get("candidate_residual")),
        "residual_reason": data.get("residual_reason"),
        "filename": item.get("filename"),
    }


def abstract_batch(cli: OpenAI, tag: str, items: list[dict]) -> list[dict]:
    # kept for API compatibility; now 1-item batches
    return [abstract_one(cli, tag, it) for it in items]

def stratified_split(blocks: dict[int, dict], discovery_n: int, holdout_n: int, seed: int):
    """blocks: n -> {n,text,block_id,filename}"""
    by_file = defaultdict(list)
    for n, b in blocks.items():
        by_file[b.get("filename") or "?"].append(n)
    rng = random.Random(seed)
    for ns in by_file.values():
        rng.shuffle(ns)

    # round-robin take for discovery then holdout
    discovery, holdout = [], []
    files = sorted(by_file.keys())
    pointers = {f: 0 for f in files}

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
    # leftover unused
    used = set(discovery) | set(holdout)
    unused = [n for n in blocks if n not in used]
    return discovery, holdout, unused


CLUSTER_SYSTEM = """Ты кластеризуешь абстракции story-beat ТОЛЬКО по психологическому механизму.
Правила:
- На входе уже нет темы — группируй только по механизму воздействия на зрителя.
- Лучше разделить похожие механизмы, чем смешать разные.
- Старайся покрыть БОЛЬШИНСТВО примеров кластерами; residual — только для действительно
  уникальных/неясных случаев без устойчивого механизма.
- Кластер из одного примера допустим, если механизм чёткий и не совпадает с другими.
- Не именовать кластеры. Не использовать термины риторики/психологии/сценаристики.
- Не оценивать качество примеров. Не придумывать примеры.
Ответ JSON:
{
  "clusters":[
    {
      "temp_id":1,
      "because":"Все эти примеры работают одинаково, потому что...",
      "draft_steps":["X → ..."],
      "members":[1,2,3],
      "why_not_split":"..."
    }
  ],
  "residual":[4,5]
}
Каждый n из входа ровно один раз: либо в members, либо в residual.
"""


def cluster_abstracts(cli: OpenAI, tag: str, abstracts: list[dict]) -> dict:
    payload = [
        {
            "n": a["n"],
            "mechanism": a["mechanism"],
            "steps": a["steps"],
            "candidate_residual": a["candidate_residual"],
        }
        for a in abstracts
        if (a.get("mechanism") or "").strip()
    ]
    empty_ns = [a["n"] for a in abstracts if not (a.get("mechanism") or "").strip()]
    data = chat_json(
        cli,
        CLUSTER_SYSTEM,
        f"Тег: {tag}\nКластеризуй абстракции:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        temperature=0.2,
    )
    data.setdefault("residual", [])
    data["residual"] = list(data.get("residual") or []) + empty_ns
    return data


REVISE_SYSTEM = """Ты ревизуешь каталог кластеров механизмов.
Сделай:
1) разрежь неоднородные кластеры на разные механизмы;
2) слей кластеры с одной и той же формулой механизма;
3) псевдокластеры без устойчивого механизма → residual;
4) НЕ сваливай в residual примеры, у которых есть ясный mechanism — найди им кластер
   или создай отдельный кластер механизма.
Не добавляй доменных слов. Не именовать.
Цель: каталог из нескольких содержательных механизмов, residual небольшой.
JSON:
{
  "clusters":[{"temp_id":1,"because":"...","draft_steps":["..."],"members":[...],"why_not_split":"..."}],
  "residual":[...],
  "notes":["..."]
}
Все n из входа ровно один раз.
"""

def revise_clusters(cli: OpenAI, tag: str, draft: dict, abstracts: list[dict]) -> dict:
    abs_by_n = {a["n"]: a for a in abstracts}
    return chat_json(
        cli,
        REVISE_SYSTEM,
        json.dumps({
            "tag": tag,
            "draft": draft,
            "abstracts": [
                {"n": a["n"], "mechanism": a["mechanism"], "steps": a["steps"]}
                for a in abstracts
            ],
        }, ensure_ascii=False),
        temperature=0.2,
    )


NAME_SYSTEM = """Для каждого кластера дай короткую формулу (шаги X/Y) и короткое название по механизму.
Запрещены доменные слова (планеты, заводы, войны, бренды, эпохи) как смысл названия.
JSON:
{"clusters":[{"temp_id":1,"name":"...","formula":"X → ...\\n→ ...","abstract":"одно предложение"}]}
"""


def name_clusters(cli: OpenAI, tag: str, clusters: list[dict], abstracts: list[dict]) -> dict:
    abs_by_n = {a["n"]: a for a in abstracts}
    enriched = []
    for c in clusters:
        members = c.get("members") or []
        enriched.append({
            **c,
            "member_abstracts": [
                {"n": n, "mechanism": abs_by_n.get(n, {}).get("mechanism")}
                for n in members[:8]
            ],
        })
    return chat_json(
        cli,
        NAME_SYSTEM,
        json.dumps({"tag": tag, "clusters": enriched}, ensure_ascii=False),
        temperature=0.2,
    )


HOLDOUT_SYSTEM = """Мягкая проверка каталога. Для каждой абстракции выбери ближайший cluster_id из whitelist
ИЛИ residual. Новые кластеры создавать ЗАПРЕЩЕНО.
JSON:
{"assignments":[{"n":1,"cluster_id":3,"confidence":"high|medium|low","note":"..."}],
 "summary":"кратко"}
"""


def holdout_check(cli: OpenAI, tag: str, catalog: list[dict], holdout_abs: list[dict]) -> dict:
    whitelist = [
        {
            "cluster_id": c["id"],
            "name": c.get("name"),
            "formula": c.get("formula"),
            "abstract": c.get("abstract_line"),
        }
        for c in catalog
    ]
    return chat_json(
        cli,
        HOLDOUT_SYSTEM,
        json.dumps({
            "tag": tag,
            "whitelist": whitelist,
            "holdout": [
                {"n": a["n"], "mechanism": a["mechanism"], "steps": a["steps"]}
                for a in holdout_abs
            ],
        }, ensure_ascii=False),
        temperature=0.1,
    )


def enrich_blocks_with_filename(conn, blocks: dict[int, dict]) -> dict[int, dict]:
    if not blocks:
        return blocks
    ids = [b["block_id"] for b in blocks.values()]
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT b.id, t.filename
        FROM blocks b JOIN tasks t ON t.id=b.task_id
        WHERE b.id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    id_to_file = {r["id"]: r["filename"] for r in rows}
    for b in blocks.values():
        b["filename"] = id_to_file.get(b["block_id"], "?")
    return blocks


def run_abstracts(cli: OpenAI, tag: str, selected: list[dict]) -> list[dict]:
    batches = [selected[i:i + ABSTRACT_BATCH] for i in range(0, len(selected), ABSTRACT_BATCH)]
    results = []
    print(f"[{tag}] abstracting {len(selected)} examples in {len(batches)} batches...", flush=True)

    def work(batch):
        return abstract_batch(cli, tag, batch)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(work, b): idx for idx, b in enumerate(batches)}
        done = 0
        ordered = [None] * len(batches)
        for fut in as_completed(futs):
            idx = futs[fut]
            ordered[idx] = fut.result()
            done += 1
            if done % 5 == 0 or done == len(batches):
                print(f"  [{tag}] abstracts {done}/{len(batches)}", flush=True)
    for part in ordered:
        results.extend(part or [])
    return results


def domain_leak_hits(text: str) -> list[str]:
    return DOMAIN_LEAK_RE.findall(text or "")


def render_report(
    tag: str,
    meta: dict,
    abstracts: list[dict],
    discovery_ns: list[int],
    holdout_ns: list[int],
    revised: dict,
    named: dict,
    holdout: dict,
) -> str:
    name_by_temp = {int(c["temp_id"]): c for c in (named.get("clusters") or []) if "temp_id" in c}
    lines = [
        f"# Pilot clustering: {tag}",
        "",
        "По правилам `clustering_rules.md` (A→F, без полного assignment).",
        "",
        "## Meta",
        "",
        f"- Всего примеров в review-источниках: **{meta['total']}**",
        f"- Discovery sample: **{len(discovery_ns)}**",
        f"- Holdout sample: **{len(holdout_ns)}**",
        f"- Абстракций сделано: **{len(abstracts)}**",
        f"- Candidate residual на абстракции: **{sum(1 for a in abstracts if a.get('candidate_residual'))}**",
        "",
        "## Каталог после ревизии + имён",
        "",
    ]
    clusters = revised.get("clusters") or []
    residual = revised.get("residual") or []
    leak_clusters = []
    for c in sorted(clusters, key=lambda x: int(x.get("temp_id") or 0)):
        tid = int(c["temp_id"])
        nm = name_by_temp.get(tid, {})
        name = nm.get("name") or f"cluster-{tid}"
        formula = nm.get("formula") or "\n".join(c.get("draft_steps") or [])
        abstract_line = nm.get("abstract") or c.get("because") or ""
        members = c.get("members") or []
        blob = f"{name}\n{formula}\n{abstract_line}"
        leaks = domain_leak_hits(blob)
        if leaks:
            leak_clusters.append((tid, name, leaks))
        lines.append(f"### Кластер {tid} — {name} ({len(members)})")
        lines.append("")
        lines.append("Формула:")
        lines.append("")
        lines.append(formula)
        lines.append("")
        lines.append(f"Абстрактно: {abstract_line}")
        lines.append("")
        lines.append(f"Because: {c.get('because')}")
        lines.append("")
        lines.append(f"Why not split: {c.get('why_not_split')}")
        lines.append("")
        lines.append(f"Members (discovery n): {', '.join(str(x) for x in members)}")
        if leaks:
            lines.append("")
            lines.append(f"**DOMAIN LEAK?** hits: {leaks}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"## Residual (discovery): {len(residual)}")
    lines.append("")
    lines.append(", ".join(str(x) for x in residual) if residual else "_(пусто)_")
    lines.append("")
    if revised.get("notes"):
        lines.append("### Revise notes")
        for n in revised["notes"]:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Domain-leak audit (имена/формулы)")
    lines.append("")
    if leak_clusters:
        for tid, name, leaks in leak_clusters:
            lines.append(f"- cluster {tid} «{name}»: {leaks}")
    else:
        lines.append("- Утечек по эвристике доменных слов **не найдено**.")
    lines.append("")

    lines.append("## Holdout (мягкая проверка)")
    lines.append("")
    assigns = holdout.get("assignments") or []
    to_res = sum(1 for a in assigns if a.get("cluster_id") in (None, "residual", 0, "none"))
    # residual as string
    def is_res(a):
        cid = a.get("cluster_id")
        return cid in (None, "residual", 0, "none") or str(cid).lower() == "residual"

    to_res = sum(1 for a in assigns if is_res(a))
    to_cat = len(assigns) - to_res
    lines.append(f"- назначено в каталог: **{to_cat}/{len(assigns)}**")
    lines.append(f"- в residual / none: **{to_res}/{len(assigns)}**")
    lines.append(f"- summary: {holdout.get('summary')}")
    lines.append("")
    for a in assigns:
        lines.append(
            f"- n={a.get('n')}: cluster={a.get('cluster_id')} "
            f"({a.get('confidence')}) — {a.get('note')}"
        )
    lines.append("")

    lines.append("## Abstracts (для трассировки)")
    lines.append("")
    abs_by_n = {a["n"]: a for a in abstracts}
    for n in sorted(abs_by_n):
        a = abs_by_n[n]
        role = "discovery" if n in discovery_ns else ("holdout" if n in holdout_ns else "?")
        lines.append(f"### n={n} [{role}] file={a.get('filename')}")
        lines.append(f"- mechanism: {a.get('mechanism')}")
        lines.append(f"- steps: {a.get('steps')}")
        if a.get("candidate_residual"):
            lines.append(f"- candidate_residual: {a.get('residual_reason')}")
        lines.append("")

    return "\n".join(lines)


def process_tag(cli: OpenAI, tag: str, blocks: dict[int, dict]) -> Path:
    total = len(blocks)
    # OPEN_LOOP small: use most for discovery; TENSION: 80/30
    if total <= 120:
        disc_n = min(DISCOVERY_TARGET, max(40, int(total * 0.7)))
        hold_n = min(HOLDOUT_TARGET, total - disc_n)
    else:
        disc_n = DISCOVERY_TARGET
        hold_n = HOLDOUT_TARGET

    discovery_ns, holdout_ns, _unused = stratified_split(blocks, disc_n, hold_n, SEED)
    selected_ns = discovery_ns + holdout_ns
    selected = [
        {
            "n": n,
            "text": blocks[n]["text"],
            "block_id": blocks[n]["block_id"],
            "filename": blocks[n].get("filename"),
        }
        for n in selected_ns
    ]

    abstracts = run_abstracts(cli, tag, selected)
    abs_by_n = {a["n"]: a for a in abstracts}
    discovery_abs = [abs_by_n[n] for n in discovery_ns if n in abs_by_n]
    holdout_abs = [abs_by_n[n] for n in holdout_ns if n in abs_by_n]

    print(f"[{tag}] clustering {len(discovery_abs)} abstracts...", flush=True)
    draft = cluster_abstracts(cli, tag, discovery_abs)
    print(f"[{tag}] revising...", flush=True)
    revised = revise_clusters(cli, tag, draft, discovery_abs)
    print(f"[{tag}] naming...", flush=True)
    named = name_clusters(cli, tag, revised.get("clusters") or [], discovery_abs)

    # build catalog with stable ids
    catalog = []
    name_by_temp = {int(c["temp_id"]): c for c in (named.get("clusters") or []) if "temp_id" in c}
    for c in revised.get("clusters") or []:
        tid = int(c["temp_id"])
        nm = name_by_temp.get(tid, {})
        catalog.append({
            "id": tid,
            "name": nm.get("name") or f"cluster-{tid}",
            "formula": nm.get("formula") or "\n".join(c.get("draft_steps") or []),
            "abstract_line": nm.get("abstract") or c.get("because"),
            "members": c.get("members") or [],
        })

    print(f"[{tag}] holdout check {len(holdout_abs)}...", flush=True)
    holdout = holdout_check(cli, tag, catalog, holdout_abs) if holdout_abs and catalog else {
        "assignments": [], "summary": "no holdout or empty catalog"
    }

    report = render_report(
        tag,
        {"total": total},
        abstracts,
        discovery_ns,
        holdout_ns,
        revised,
        named,
        holdout,
    )
    # attach machine JSON sidecar
    sidecar = {
        "tag": tag,
        "total": total,
        "discovery_ns": discovery_ns,
        "holdout_ns": holdout_ns,
        "abstracts": abstracts,
        "draft": draft,
        "revised": revised,
        "named": named,
        "catalog": catalog,
        "holdout": holdout,
    }
    out_md = OUT_DIR / f"pilot_{tag}.md"
    out_json = OUT_DIR / f"pilot_{tag}.json"
    out_md.write_text(report, encoding="utf-8")
    out_json.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] wrote {out_md}", flush=True)
    return out_md


def main():
    cli = client()
    tags = sys.argv[1:] or ["OPEN_LOOP", "TENSION"]
    with get_db() as conn:
        task_ids = resolve_review_task_ids(conn)
        print("review tasks:", task_ids)
        print("tags:", tags)
        results = {}
        for tag in tags:
            blocks = numbered_blocks_for_tag(conn, tag, task_ids)
            blocks = enrich_blocks_with_filename(conn, blocks)
            print(f"{tag}: {len(blocks)} blocks")
            process_tag(cli, tag, blocks)

        # load all available pilot json for comparison
        for tag in ("OPEN_LOOP", "TENSION"):
            path = OUT_DIR / f"pilot_{tag}.json"
            if path.exists():
                results[tag] = json.loads(path.read_text())

    # comparison report
    lines = [
        "# Pilot comparison: OPEN_LOOP vs TENSION",
        "",
        "После удаления старых `tag_subclusters`. Пайплайн: `clustering_rules.md`.",
        "",
        "## Вопрос",
        "",
        "1. Не ухудшился ли «нормальный» тег (OPEN_LOOP) относительно эталона `open_loop.md`?",
        "2. Решена ли проблема тематической утечки на «проблемном» TENSION (~487 сырых абзацев)?",
        "",
    ]
    for tag, data in results.items():
        cat = data.get("catalog") or []
        residual = (data.get("revised") or {}).get("residual") or []
        hold = data.get("holdout") or {}
        assigns = hold.get("assignments") or []
        def is_res(a):
            cid = a.get("cluster_id")
            return cid in (None, "residual", 0, "none") or str(cid).lower() == "residual"
        to_res = sum(1 for a in assigns if is_res(a))
        leaks = []
        for c in cat:
            hits = domain_leak_hits(f"{c.get('name')} {c.get('formula')} {c.get('abstract_line')}")
            if hits:
                leaks.append((c.get("id"), c.get("name"), hits))
        lines += [
            f"## {tag}",
            "",
            f"- discovery/holdout: {len(data.get('discovery_ns') or [])}/{len(data.get('holdout_ns') or [])}",
            f"- кластеров в каталоге: **{len(cat)}**",
            f"- residual на discovery: **{len(residual)}**",
            f"- holdout → catalog / residual: **{len(assigns)-to_res}/{to_res}** (из {len(assigns)})",
            f"- domain-leak в именах/формулах: **{len(leaks)}**",
            "",
        ]
        if leaks:
            for i, name, hits in leaks:
                lines.append(f"  - {i} «{name}»: {hits}")
            lines.append("")
        lines.append("Каталог:")
        for c in cat:
            lines.append(f"- **{c['id']}. {c['name']}** (n={len(c.get('members') or [])})")
            lines.append(f"  - {c.get('abstract_line')}")
        lines.append("")

    lines += [
        "## Вердикт (заполняется после ручной сверки с отчётами)",
        "",
        "См. `pilot_OPEN_LOOP.md`, `pilot_TENSION.md` и автоматические метрики выше.",
        "Эталон OPEN_LOOP: ~16 механизмов в `open_loop.md` (скрытая деталь, эскалация, …).",
        "Старый TENSION review: 19 кластеров, включая тематический «Гравитационный обратный отсчёт».",
        "",
    ]
    (OUT_DIR / "pilot_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote pilot_comparison.md")


if __name__ == "__main__":
    main()
