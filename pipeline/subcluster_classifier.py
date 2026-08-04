"""Classify blocks into tag subclusters using prompts built from the DB.

Never creates tag_subclusters rows — only assigns to existing active clusters.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from openai import OpenAI

from .content_lang import detect_content_lang, normalize_content_lang

MODEL = "gpt-4o"
BATCH_SIZE = 25


def load_active_subclusters(conn, parent_tag: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, slug, name, formula, abstract, notes, examples_json, sort_order, status
        FROM tag_subclusters
        WHERE parent_tag = ?
          AND status = 'active'
          AND slug != 'residual'
          AND sort_order > 0
        ORDER BY sort_order ASC, id ASC
        """,
        (parent_tag,),
    ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["examples"] = json.loads(d["examples_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            d["examples"] = []
        out.append(d)
    return out


def build_classify_system_prompt(
    parent_tag: str,
    subclusters: list[dict],
    content_lang: str = "ru",
) -> str:
    lang = normalize_content_lang(content_lang)
    allowed_orders = [int(sc["sort_order"]) for sc in subclusters]
    allowed_slugs = [str(sc["slug"]) for sc in subclusters]
    if lang == "en":
        parts = [
            f"You classify story-beat blocks tagged {parent_tag} by psychological mechanism.",
            "",
            "SCRIPT LANGUAGE: English. Cluster catalog labels may be in another language — "
            "ignore catalog language/domain wording; match by mechanism only.",
            "",
            "Hard rules:",
            "- Ignore topic, plot, characters, objects, countries, and facts.",
            "- Look only at the psychological mechanism of impact on the viewer.",
            "- Choose EXACTLY one cluster from the whitelist below for each block.",
            "- Do NOT invent new sort_order, slug, or cluster names.",
            "- If none fits perfectly — pick the closest mechanism from the whitelist.",
            "- Do not confuse example numbers (n) with cluster numbers.",
            "",
            f"WHITELIST sort_order: {json.dumps(allowed_orders)}",
            f"WHITELIST slug: {json.dumps(allowed_slugs, ensure_ascii=False)}",
            "",
            "CLUSTER LIST (only these):",
        ]
    else:
        parts = [
            f"Ты классифицируешь story-beat блоки с тегом {parent_tag} по психологическому механизму.",
            "",
            "ЯЗЫК СКРИПТА: русский. Подписи кластеров в каталоге могут быть на другом языке — "
            "игнорируй язык/домен каталога; сопоставляй только по механизму.",
            "",
            "Правила (обязательны):",
            "- Игнорируй тему, сюжет, персонажей, объекты, страны и факты.",
            "- Смотри только на психологический механизм воздействия на зрителя.",
            "- Выбери РОВНО один кластер из whitelist ниже для каждого блока.",
            "- ЗАПРЕЩЕНО придумывать новые номера, slug или названия кластеров.",
            "- Если ни один не подходит идеально — выбери ближайший по механизму из whitelist.",
            "- Не путай номер примера (n) с номером кластера.",
            "",
            f"WHITELIST sort_order: {json.dumps(allowed_orders)}",
            f"WHITELIST slug: {json.dumps(allowed_slugs, ensure_ascii=False)}",
            "",
            "СПИСОК КЛАСТЕРОВ (только эти):",
        ]
    for sc in subclusters:
        parts.append("")
        parts.append(f"### Кластер {sc['sort_order']} — {sc['name']} (slug={sc['slug']})")
        if sc.get("formula"):
            parts.append("Formula:" if lang == "en" else "Формула:")
            parts.append(sc["formula"])
        if sc.get("abstract"):
            label = "Abstract" if lang == "en" else "Абстрактно"
            parts.append(f'{label}: "{sc["abstract"]}"')
        examples = sc.get("examples") or []
        if examples:
            parts.append("Reference examples:" if lang == "en" else "Эталонные примеры:")
            for ex in examples[:3]:
                text = (ex.get("text") or "").strip().replace("\n", " ")
                if len(text) > 320:
                    text = text[:317] + "..."
                parts.append(f"- (n={ex.get('n')}) {text}")
        if sc.get("notes"):
            parts.append(("Note: " if lang == "en" else "Заметка: ") + sc["notes"])

    if lang == "en":
        parts.extend([
            "",
            "Reply with STRICT valid JSON only:",
            '{"assignments": [{"block_id": 123, "subcluster_sort_order": 1, "subcluster_slug": "01"}]}',
            "Exactly one record per input block_id.",
            "subcluster_sort_order and subcluster_slug MUST be from the WHITELIST above.",
        ])
    else:
        parts.extend([
            "",
            "Ответь СТРОГО валидным JSON без текста вокруг:",
            '{"assignments": [{"block_id": 123, "subcluster_sort_order": 1, "subcluster_slug": "01"}]}',
            "Для каждого входного block_id должна быть ровно одна запись.",
            "subcluster_sort_order и subcluster_slug ОБЯЗАНЫ быть из WHITELIST выше.",
        ])
    return "\n".join(parts)


def _parse_assignments_payload(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    return data.get("assignments") or []


def _clean_assignments(
    assignments: list[dict],
    subclusters: list[dict],
    allowed_block_ids: set[int],
) -> list[dict]:
    order_to_slug = {int(sc["sort_order"]): str(sc["slug"]) for sc in subclusters}
    slug_to_order = {str(sc["slug"]): int(sc["sort_order"]) for sc in subclusters}
    valid_orders = set(order_to_slug)
    cleaned = []
    seen_blocks: set[int] = set()
    for a in assignments:
        try:
            bid = int(a["block_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if bid not in allowed_block_ids or bid in seen_blocks:
            continue
        order = None
        slug = a.get("subcluster_slug")
        if slug is not None and str(slug) in slug_to_order:
            order = slug_to_order[str(slug)]
        else:
            try:
                order = int(a["subcluster_sort_order"])
            except (KeyError, TypeError, ValueError):
                continue
        if order not in valid_orders:
            continue
        seen_blocks.add(bid)
        cleaned.append({
            "block_id": bid,
            "subcluster_sort_order": order,
            "subcluster_slug": order_to_slug[order],
        })
    return cleaned


def classify_blocks_batch(
    client: OpenAI,
    parent_tag: str,
    subclusters: list[dict],
    blocks: list[dict],
    content_lang: Optional[str] = None,
) -> list[dict]:
    """blocks: [{id, text}]. Returns assignments with block_id + subcluster_sort_order.

    Only returns orders that exist in `subclusters`. Never invents clusters.
    """
    if not blocks or not subclusters:
        return []
    lang = normalize_content_lang(
        content_lang
        or detect_content_lang("\n".join((b.get("text") or "")[:400] for b in blocks))
    )
    system = build_classify_system_prompt(parent_tag, subclusters, content_lang=lang)
    allowed_block_ids = {int(b["id"]) for b in blocks}
    payload = [
        {"block_id": b["id"], "text": (b.get("text") or "")[:1200]}
        for b in blocks
    ]
    if lang == "en":
        user = (
            "Classify the following blocks. Return JSON assignments.\n"
            "Use ONLY whitelist sort_order/slug from the system prompt.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
    else:
        user = (
            "Классифицируй следующие блоки. Верни JSON assignments.\n"
            "Используй ТОЛЬКО whitelist sort_order/slug из system prompt.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    cleaned = _clean_assignments(_parse_assignments_payload(raw), subclusters, allowed_block_ids)

    missing = allowed_block_ids - {a["block_id"] for a in cleaned}
    if missing:
        allowed_orders = [int(sc["sort_order"]) for sc in subclusters]
        repair_payload = [b for b in payload if int(b["block_id"]) in missing]
        if lang == "en":
            repair_user = (
                "For these blocks the previous answer was empty or used an invalid cluster.\n"
                f"Choose ONLY from whitelist sort_order={json.dumps(allowed_orders)}.\n"
                "Do not invent clusters. Return JSON assignments.\n\n"
                + json.dumps(repair_payload, ensure_ascii=False, indent=2)
            )
        else:
            repair_user = (
                "Для этих блоков предыдущий ответ был пустым или с недопустимым кластером.\n"
                f"Выбери ТОЛЬКО из whitelist sort_order={json.dumps(allowed_orders)}.\n"
                "Не создавай новые кластеры. Верни JSON assignments.\n\n"
                + json.dumps(repair_payload, ensure_ascii=False, indent=2)
            )
        repair = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": repair_user},
            ],
            response_format={"type": "json_object"},
        )
        repaired = _clean_assignments(
            _parse_assignments_payload(repair.choices[0].message.content or "{}"),
            subclusters,
            missing,
        )
        cleaned.extend(repaired)

    return cleaned


def classify_unassigned_blocks(
    client: OpenAI,
    conn,
    parent_tag: str,
    task_id: Optional[str] = None,
    on_batch_done=None,
    content_lang: Optional[str] = None,
) -> dict[str, Any]:
    """Classify unassigned blocks for parent_tag (optionally limited to one task)."""
    subclusters = load_active_subclusters(conn, parent_tag)
    if not subclusters:
        return {"assigned": 0, "pending": 0, "error": "no_subclusters"}

    order_to_id = {int(sc["sort_order"]): int(sc["id"]) for sc in subclusters}
    resolved_lang = None
    if content_lang:
        resolved_lang = normalize_content_lang(content_lang)
    elif task_id:
        row = conn.execute(
            "SELECT content_lang FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row and row["content_lang"]:
            resolved_lang = normalize_content_lang(row["content_lang"])

    params: list = [parent_tag]
    task_clause = ""
    if task_id:
        task_clause = "AND b.task_id = ?"
        params.append(task_id)
    rows = conn.execute(
        f"""
        SELECT b.id, b.text, t.content_lang AS task_content_lang
        FROM blocks b
        LEFT JOIN tasks t ON t.id = b.task_id
        WHERE b.tag = ?
          {task_clause}
          AND NOT EXISTS (
              SELECT 1 FROM block_subcluster_assignments a WHERE a.block_id = b.id
          )
        ORDER BY b.id ASC
        """,
        params,
    ).fetchall()
    blocks = [
        {
            "id": r["id"],
            "text": r["text"],
            "content_lang": normalize_content_lang(r["task_content_lang"])
            if r["task_content_lang"]
            else None,
        }
        for r in rows
    ]
    total_pending = len(blocks)
    assigned = 0

    for start in range(0, len(blocks), BATCH_SIZE):
        batch = blocks[start:start + BATCH_SIZE]
        if resolved_lang:
            batch_lang = resolved_lang
        else:
            langs = {b.get("content_lang") for b in batch if b.get("content_lang")}
            if len(langs) == 1:
                batch_lang = next(iter(langs))
            else:
                batch_lang = detect_content_lang(
                    "\n".join((b.get("text") or "")[:400] for b in batch)
                )
        results = classify_blocks_batch(
            client, parent_tag, subclusters, batch, content_lang=batch_lang
        )
        for a in results:
            sc_id = order_to_id.get(a["subcluster_sort_order"])
            if not sc_id:
                # Guard: never invent / never write unknown cluster ids
                continue
            # Ensure target row still exists and belongs to this tag
            row = conn.execute(
                """
                SELECT id FROM tag_subclusters
                WHERE id = ? AND parent_tag = ? AND status = 'active' AND sort_order > 0
                """,
                (sc_id, parent_tag),
            ).fetchone()
            if not row:
                continue
            conn.execute(
                "DELETE FROM block_subcluster_assignments WHERE block_id = ?",
                (a["block_id"],),
            )
            conn.execute(
                """
                INSERT INTO block_subcluster_assignments
                    (block_id, subcluster_id, is_primary, source)
                VALUES (?, ?, 1, 'llm')
                """,
                (a["block_id"], sc_id),
            )
            assigned += 1
        conn.commit()
        if on_batch_done:
            on_batch_done(assigned, total_pending)

    return {"assigned": assigned, "pending": total_pending, "error": None}


def classify_task_unassigned_blocks(
    client: OpenAI,
    conn,
    task_id: str,
    on_progress=None,
) -> dict[str, Any]:
    """Classify all unassigned blocks of a task, tag by tag."""
    task_row = conn.execute(
        "SELECT content_lang FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    content_lang = normalize_content_lang(
        (task_row["content_lang"] if task_row else None) or "ru"
    )
    tags = [
        row["tag"]
        for row in conn.execute(
            """
            SELECT DISTINCT b.tag AS tag
            FROM blocks b
            WHERE b.task_id = ?
              AND b.tag IS NOT NULL AND TRIM(b.tag) != ''
              AND NOT EXISTS (
                  SELECT 1 FROM block_subcluster_assignments a WHERE a.block_id = b.id
              )
            ORDER BY b.tag ASC
            """,
            (task_id,),
        ).fetchall()
    ]
    total_assigned = 0
    total_pending = 0
    errors = []
    for tag in tags:
        result = classify_unassigned_blocks(
            client,
            conn,
            tag,
            task_id=task_id,
            content_lang=content_lang,
            on_batch_done=(
                (lambda a, p, _tag=tag: on_progress(_tag, a, p)) if on_progress else None
            ),
        )
        total_assigned += result.get("assigned") or 0
        total_pending += result.get("pending") or 0
        if result.get("error") and result["error"] != "no_subclusters":
            errors.append(f"{tag}: {result['error']}")
    return {
        "assigned": total_assigned,
        "pending": total_pending,
        "tags": tags,
        "error": "; ".join(errors) if errors else None,
    }
