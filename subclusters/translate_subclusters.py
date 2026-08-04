#!/usr/bin/env python3
"""Fill missing RU/EN translations for tag_subclusters via OpenAI.

Primary columns name/formula/abstract = RU.
Twin columns name_en/formula_en/abstract_en = EN.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CACHE = OUT / "translate_subclusters_cache.json"

sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from main import get_db  # noqa: E402
from pipeline.subcluster_i18n import (  # noqa: E402
    FIELD_PAIRS,
    ensure_subcluster_i18n_columns,
    missing_translation_sides,
    normalize_existing_subcluster_langs,
)

MODEL = "gpt-4o"
BATCH = 12


def client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)


def chat_json(cli: OpenAI, system: str, user: str, temperature: float = 0.15) -> dict:
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
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001
            wait = min(2 ** attempt, 30)
            print(f"  retry {attempt + 1}: {exc} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError("chat_json failed")


def load_cache() -> dict:
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {"by_id": {}}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_batch(
    cli: OpenAI,
    target_lang: str,
    items: list[dict],
) -> dict[int, dict[str, str]]:
    """items: [{id, fields: {name?, formula?, abstract?}}] source texts to translate."""
    if target_lang == "en":
        system = """Translate psychological story-beat cluster catalog fields to English.
Keep mechanism meaning. Do NOT introduce domain/topic nouns (planets, factories, wars, brands).
Preserve formula arrow style (X → Y). Keep names short (2–8 words).
Return JSON: {"items":[{"id":1,"name":"...","formula":"...","abstract":"..."}]}
Only include keys that were present in the input fields for that id.
"""
    else:
        system = """Переведи поля каталога психологических кластеров story-beat на русский.
Сохрани смысл механизма. НЕ добавляй доменные/тематические слова (планеты, заводы, войны, бренды).
Сохрани стиль формул (X → Y). Названия короткие (2–8 слов).
Ответ JSON: {"items":[{"id":1,"name":"...","formula":"...","abstract":"..."}]}
Включай только те ключи, которые были во входных fields для этого id.
"""
    data = chat_json(
        cli,
        system,
        json.dumps({"target_lang": target_lang, "items": items}, ensure_ascii=False),
    )
    out: dict[int, dict[str, str]] = {}
    for item in data.get("items") or []:
        try:
            cid = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        fields = {}
        for key in ("name", "formula", "abstract"):
            if key in item and item[key] is not None:
                val = str(item[key]).strip()
                if val:
                    fields[key] = val
        if fields:
            out[cid] = fields
    return out


def apply_translation(conn, cid: int, target_lang: str, fields: dict[str, str]) -> None:
    if target_lang == "en":
        mapping = {"name": "name_en", "formula": "formula_en", "abstract": "abstract_en"}
    else:
        mapping = {"name": "name", "formula": "formula", "abstract": "abstract"}
    updates = {}
    for src_key, col in mapping.items():
        if src_key in fields and fields[src_key].strip():
            updates[col] = fields[src_key].strip()
    if not updates:
        return
    sets = ", ".join(f"{k} = ?" for k in updates)
    conn.execute(
        f"UPDATE tag_subclusters SET {sets} WHERE id = ?",
        (*updates.values(), cid),
    )


def main() -> None:
    cli = client()
    cache = load_cache()
    cache_by_id: dict = cache.setdefault("by_id", {})

    with get_db() as conn:
        ensure_subcluster_i18n_columns(conn)
        stats = normalize_existing_subcluster_langs(conn)
        conn.commit()
        print("normalize:", stats, flush=True)

        rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT id, parent_tag, sort_order, slug, name, formula, abstract,
                       name_en, formula_en, abstract_en, status
                FROM tag_subclusters
                WHERE sort_order > 0 AND slug != 'residual'
                ORDER BY parent_tag, sort_order, id
                """
            ).fetchall()
        ]

    jobs_en: list[dict] = []
    jobs_ru: list[dict] = []
    for row in rows:
        cid = row["id"]
        # apply cache
        cached = cache_by_id.get(str(cid)) or {}
        if cached:
            with get_db() as conn:
                if cached.get("en"):
                    apply_translation(conn, cid, "en", cached["en"])
                if cached.get("ru"):
                    apply_translation(conn, cid, "ru", cached["ru"])
                # refresh row
                row = dict(
                    conn.execute(
                        """
                        SELECT id, parent_tag, sort_order, name, formula, abstract,
                               name_en, formula_en, abstract_en
                        FROM tag_subclusters WHERE id = ?
                        """,
                        (cid,),
                    ).fetchone()
                )
                conn.commit()

        missing = missing_translation_sides(row)
        if missing["en"]:
            fields = {k: (row.get(k) or "").strip() for k in missing["en"] if (row.get(k) or "").strip()}
            if fields:
                jobs_en.append({"id": cid, "fields": fields, "tag": row.get("parent_tag")})
        if missing["ru"]:
            fields = {}
            for k in missing["ru"]:
                en_key = f"{k}_en"
                val = (row.get(en_key) or "").strip()
                if val:
                    fields[k] = val
            if fields:
                jobs_ru.append({"id": cid, "fields": fields, "tag": row.get("parent_tag")})

    print(f"need EN translations: {len(jobs_en)} clusters", flush=True)
    print(f"need RU translations: {len(jobs_ru)} clusters", flush=True)

    def run_jobs(target: str, jobs: list[dict]) -> int:
        done = 0
        for start in range(0, len(jobs), BATCH):
            batch = jobs[start:start + BATCH]
            payload = [{"id": j["id"], "fields": j["fields"]} for j in batch]
            print(
                f"[{target}] batch {start // BATCH + 1}/{(len(jobs) + BATCH - 1) // BATCH} "
                f"({len(batch)} clusters)...",
                flush=True,
            )
            result = translate_batch(cli, target, payload)
            with get_db() as conn:
                for j in batch:
                    cid = j["id"]
                    fields = result.get(cid) or {}
                    # keep only requested keys
                    fields = {k: v for k, v in fields.items() if k in j["fields"] and v}
                    if not fields:
                        print(f"  miss id={cid}", flush=True)
                        continue
                    apply_translation(conn, cid, target, fields)
                    entry = cache_by_id.setdefault(str(cid), {})
                    entry[target] = {**(entry.get(target) or {}), **fields}
                    done += 1
                conn.commit()
            save_cache(cache)
        return done

    n_en = run_jobs("en", jobs_en)
    n_ru = run_jobs("ru", jobs_ru)
    print(f"applied EN: {n_en}, RU: {n_ru}", flush=True)

    # verify
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, name, formula, abstract, name_en, formula_en, abstract_en
            FROM tag_subclusters WHERE sort_order > 0
            """
        ).fetchall()
        still = 0
        for row in rows:
            miss = missing_translation_sides(dict(row))
            if miss["en"] or miss["ru"]:
                still += 1
                if still <= 10:
                    print(
                        f"  still missing id={row['id']}: en={miss['en']} ru={miss['ru']}",
                        flush=True,
                    )
        print(f"clusters still incomplete: {still}/{len(rows)}", flush=True)

        both = conn.execute(
            """
            SELECT COUNT(*) c FROM tag_subclusters
            WHERE sort_order > 0
              AND name IS NOT NULL AND TRIM(name) != ''
              AND name_en IS NOT NULL AND TRIM(name_en) != ''
            """
        ).fetchone()["c"]
        print(f"clusters with both names: {both}", flush=True)


if __name__ == "__main__":
    main()
