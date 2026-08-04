#!/usr/bin/env python3
"""One pragmatic re-cluster pass from saved pilot abstracts. No iteration loop."""
from __future__ import annotations

import json
import os
import re
import sys
import time
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
    seed_tag_subclusters,
)

MODEL = "gpt-4o"

# Soft targets — guidance only, one shot, no retry-to-optimize
SOFT_TARGETS = {
    "OPEN_LOOP": (10, 18),
    "TENSION": (10, 18),
}


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
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate_limit" in msg.lower():
                m = re.search(r"try again in (\d+)ms", msg, re.I)
                wait = (int(m.group(1)) / 1000.0 + 0.3) if m else 2.0 * (attempt + 1)
                print(f"  rate-limit sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            if attempt >= 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("chat_json failed")


def load_pilot(tag: str) -> dict:
    path = OUT / f"pilot_{tag}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def discovery_abstracts(data: dict) -> list[dict]:
    disc = set(data.get("discovery_ns") or [])
    out = []
    for a in data.get("abstracts") or []:
        if a["n"] not in disc:
            continue
        if not (a.get("mechanism") or "").strip():
            continue
        out.append({
            "n": a["n"],
            "mechanism": a["mechanism"],
            "steps": a.get("steps") or [],
        })
    return out


def holdout_abstracts(data: dict) -> list[dict]:
    hold = set(data.get("holdout_ns") or [])
    out = []
    for a in data.get("abstracts") or []:
        if a["n"] not in hold:
            continue
        if not (a.get("mechanism") or "").strip():
            continue
        out.append({
            "n": a["n"],
            "mechanism": a["mechanism"],
            "steps": a.get("steps") or [],
        })
    return out


def cluster_once(cli: OpenAI, tag: str, abstracts: list[dict]) -> dict:
    lo, hi = SOFT_TARGETS[tag]
    system = f"""Ты кластеризуешь обезличенные абстракции story-beat по психологическому механизму.

Правила:
- Только механизм воздействия на зрителя. Темы/объекты уже вычищены — не возвращай их.
- Ориентир по числу кластеров: примерно {lo}–{hi}. Это мягкая цель, не жёсткий лимит.
- Лучше 12 разных механизмов, чем 3 сверхшироких «ожидание/угроза».
- Не дроби без нужды на синглтоны: если механизмы совпадают — один кластер.
- Residual — только для неясных случаев (цель: residual заметно меньше половины примеров).
- Не именовать. Не использовать доменные слова.

JSON:
{{
  "clusters":[
    {{"temp_id":1,"because":"Все эти примеры работают одинаково, потому что...",
      "draft_steps":["X → ...","→ ..."],"members":[1,2],"why_not_split":"..."}}
  ],
  "residual":[3]
}}
Каждый n ровно один раз.
"""
    return chat_json(
        cli,
        system,
        f"Тег {tag}. Абстракции:\n{json.dumps(abstracts, ensure_ascii=False)}",
        temperature=0.25,
    )


def light_revise(cli: OpenAI, tag: str, draft: dict, abstracts: list[dict]) -> dict:
    """One light pass: only fix obvious dual-mechanism merges and true duplicates."""
    lo, hi = SOFT_TARGETS[tag]
    system = f"""Лёгкая ревизия каталога (один проход, без погони за идеалом).

Сделай ТОЛЬКО:
1) разрежь кластер, если because явно смешивает два разных механизма;
2) слей кластеры-дубликаты с одной формулой;
3) не сваливай половину в residual;
4) держись мягкого ориентира {lo}–{hi} кластеров.

Не переименовывай. Не добавляй доменных слов. Не оптимизируй дальше.

JSON как на входе:
{{"clusters":[...],"residual":[...],"notes":["кратко что сделал"]}}
Все n ровно один раз.
"""
    return chat_json(
        cli,
        system,
        json.dumps({"tag": tag, "draft": draft, "abstracts": abstracts}, ensure_ascii=False),
        temperature=0.2,
    )


def name_once(cli: OpenAI, tag: str, clusters: list[dict], abstracts: list[dict]) -> dict:
    by_n = {a["n"]: a for a in abstracts}
    enriched = []
    for c in clusters:
        members = c.get("members") or []
        enriched.append({
            "temp_id": c.get("temp_id"),
            "because": c.get("because"),
            "draft_steps": c.get("draft_steps"),
            "members": members,
            "samples": [
                {"n": n, "mechanism": by_n.get(n, {}).get("mechanism")}
                for n in members[:6]
            ],
        })
    system = """Дай каждому кластеру короткое название и формулу по механизму.
Запрещены доменные слова (планеты, заводы, войны, бренды, эпохи) в названии.
Название — 2–5 слов про механизм, не про сюжет.

JSON:
{"clusters":[{"temp_id":1,"name":"...","formula":"X → ...\\n→ ...","abstract":"..."}]}
"""
    return chat_json(
        cli,
        system,
        json.dumps({"tag": tag, "clusters": enriched}, ensure_ascii=False),
        temperature=0.2,
    )


def soft_holdout(cli: OpenAI, tag: str, catalog: list[dict], holdout: list[dict]) -> dict:
    if not holdout or not catalog:
        return {"assignments": [], "summary": "skip"}
    system = """Мягкая проверка. Назначь каждый n в cluster_id из whitelist или residual.
Новые кластеры запрещены. Один проход.

JSON:
{"assignments":[{"n":1,"cluster_id":2,"confidence":"high|medium|low"}],
 "summary":"1-2 предложения"}
"""
    whitelist = [
        {"cluster_id": c["id"], "name": c["name"], "formula": c["formula"], "abstract": c["abstract"]}
        for c in catalog
    ]
    return chat_json(
        cli,
        system,
        json.dumps({"tag": tag, "whitelist": whitelist, "holdout": holdout}, ensure_ascii=False),
        temperature=0.1,
    )


def to_catalog(revised: dict, named: dict) -> list[dict]:
    name_by = {int(c["temp_id"]): c for c in (named.get("clusters") or []) if "temp_id" in c}
    out = []
    for c in revised.get("clusters") or []:
        tid = int(c["temp_id"])
        nm = name_by.get(tid, {})
        out.append({
            "id": tid,
            "name": (nm.get("name") or f"cluster-{tid}").strip(),
            "formula": (nm.get("formula") or "\n".join(c.get("draft_steps") or [])).strip(),
            "abstract": (nm.get("abstract") or c.get("because") or "").strip(),
            "members": list(c.get("members") or []),
            "because": c.get("because"),
            "why_not_split": c.get("why_not_split"),
        })
    out.sort(key=lambda x: x["id"])
    # renumber 1..N for stable canon
    for i, c in enumerate(out, start=1):
        c["sort_order"] = i
    return out


def render_canon_md(results: dict[str, dict]) -> str:
    lines = [
        "# Кластеризация тегов по психологическому механизму",
        "",
        "## Источники",
        "",
        "Прагматичный прогон v2: абстракции из pilot_*.json → один cluster + лёгкая revise + name.",
        "Без итеративной оптимизации. См. PRAGMATIC_FREEZE.md.",
        "",
        "## Метод",
        "",
        "Группировка только по психологическому механизму; тема/объекты отброшены на шаге абстракции.",
        "",
    ]
    for tag, payload in results.items():
        catalog = payload["catalog"]
        residual = payload.get("residual") or []
        lines.append("---")
        lines.append("")
        lines.append(f"# {tag} ({len(catalog)} кластеров, discovery)")
        lines.append("")
        for c in catalog:
            lines.append(f"## Кластер {c['sort_order']} — {c['name']}")
            lines.append("")
            lines.append("Формула:")
            lines.append("")
            lines.append(c["formula"])
            lines.append("")
            lines.append(f"Абстрактно: \"{c['abstract']}\"")
            lines.append("")
            members = ", ".join(str(n) for n in c["members"])
            lines.append(f"Номера примеров: {members}")
            lines.append("")
        if residual:
            lines.append("## Временно неклассифицированные")
            lines.append("")
            lines.append("Номера примеров: " + ", ".join(str(n) for n in residual))
            lines.append("")
    return "\n".join(lines)


def render_report(tag: str, catalog: list[dict], residual: list, holdout: dict, notes: list) -> str:
    assigns = holdout.get("assignments") or []

    def is_res(a):
        cid = a.get("cluster_id")
        return cid in (None, "residual", 0, "none") or str(cid).lower() == "residual"

    to_cat = sum(1 for a in assigns if not is_res(a))
    lines = [
        f"# Pragmatic v2: {tag}",
        "",
        f"- кластеров: **{len(catalog)}**",
        f"- residual discovery: **{len(residual)}**",
        f"- holdout → catalog: **{to_cat}/{len(assigns)}**",
        f"- holdout summary: {holdout.get('summary')}",
        "",
    ]
    if notes:
        lines.append("## Revise notes")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("## Каталог")
    lines.append("")
    for c in catalog:
        lines.append(f"### {c['sort_order']} — {c['name']} ({len(c['members'])})")
        lines.append("")
        lines.append(c["formula"])
        lines.append("")
        lines.append(f"Абстрактно: {c['abstract']}")
        lines.append("")
        lines.append(f"n: {', '.join(map(str, c['members']))}")
        lines.append("")
    return "\n".join(lines)


def process_tag(cli: OpenAI, tag: str) -> dict:
    data = load_pilot(tag)
    disc = discovery_abstracts(data)
    hold = holdout_abstracts(data)
    print(f"[{tag}] discovery abstracts: {len(disc)}, holdout: {len(hold)}", flush=True)

    print(f"[{tag}] cluster...", flush=True)
    draft = cluster_once(cli, tag, disc)
    print(f"[{tag}] draft clusters={len(draft.get('clusters') or [])} residual={len(draft.get('residual') or [])}", flush=True)

    print(f"[{tag}] light revise...", flush=True)
    revised = light_revise(cli, tag, draft, disc)
    print(f"[{tag}] revised clusters={len(revised.get('clusters') or [])} residual={len(revised.get('residual') or [])}", flush=True)

    print(f"[{tag}] name...", flush=True)
    named = name_once(cli, tag, revised.get("clusters") or [], disc)
    catalog = to_catalog(revised, named)

    print(f"[{tag}] holdout...", flush=True)
    holdout = soft_holdout(cli, tag, [
        {"id": c["sort_order"], "name": c["name"], "formula": c["formula"], "abstract": c["abstract"]}
        for c in catalog
    ], hold)

    report = render_report(tag, catalog, revised.get("residual") or [], holdout, revised.get("notes") or [])
    (OUT / f"pragmatic_v2_{tag}.md").write_text(report, encoding="utf-8")
    sidecar = {
        "tag": tag,
        "draft": draft,
        "revised": revised,
        "named": named,
        "catalog": catalog,
        "residual": revised.get("residual") or [],
        "holdout": holdout,
    }
    (OUT / f"pragmatic_v2_{tag}.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] wrote pragmatic_v2_{tag}.md ({len(catalog)} clusters)", flush=True)
    return sidecar


def wipe_and_seed(canon_path: Path):
    with get_db() as conn:
        conn.execute("DELETE FROM block_subcluster_assignments")
        conn.execute("DELETE FROM tag_subclusters")
        n = seed_tag_subclusters(conn, canon_path)
        demoted = demote_domain_leaky_subclusters(conn)
        conn.commit()
        active = conn.execute(
            "SELECT parent_tag, COUNT(*) c FROM tag_subclusters WHERE status='active' AND slug!='residual' GROUP BY parent_tag"
        ).fetchall()
        print("seeded", n, "demoted", len(demoted))
        for r in active:
            print(f"  active {r['parent_tag']}: {r['c']}")


def main():
    cli = client()
    tags = ["OPEN_LOOP", "TENSION"]
    results = {}
    for tag in tags:
        results[tag] = process_tag(cli, tag)

    canon_path = OUT / "clusters_canon.md"
    canon_path.write_text(render_canon_md(results), encoding="utf-8")
    print("wrote", canon_path)

    # comparison blurb
    lines = [
        "# Pragmatic v2 comparison",
        "",
        "Один проход по сохранённым абстракциям. Без цикла оптимизаций.",
        "",
    ]
    for tag, payload in results.items():
        cat = payload["catalog"]
        residual = payload["residual"]
        assigns = (payload.get("holdout") or {}).get("assignments") or []

        def is_res(a):
            cid = a.get("cluster_id")
            return cid in (None, "residual", 0, "none") or str(cid).lower() == "residual"

        lines += [
            f"## {tag}",
            f"- clusters: **{len(cat)}** (soft target {SOFT_TARGETS[tag][0]}–{SOFT_TARGETS[tag][1]})",
            f"- residual: **{len(residual)}**",
            f"- holdout→catalog: **{sum(1 for a in assigns if not is_res(a))}/{len(assigns)}**",
            "",
        ]
        for c in cat:
            lines.append(f"- {c['sort_order']}. {c['name']} ({len(c['members'])})")
        lines.append("")
    (OUT / "pragmatic_v2_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    wipe_and_seed(canon_path)
    print("done")


if __name__ == "__main__":
    main()
