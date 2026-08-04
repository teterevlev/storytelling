"""Classify blocks into tag subclusters using prompts built from the DB.

Never creates tag_subclusters rows — only assigns to existing active clusters.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from openai import OpenAI

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


def build_classify_system_prompt(parent_tag: str, subclusters: list[dict]) -> str:
    allowed_orders = [int(sc["sort_order"]) for sc in subclusters]
    allowed_slugs = [str(sc["slug"]) for sc in subclusters]
    parts = [
        f"Ты классифицируешь story-beat блоки с тегом {parent_tag} по психологическому механизму.",
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
            parts.append("Формула:")
            parts.append(sc["formula"])
        if sc.get("abstract"):
            parts.append(f'Абстрактно: "{sc["abstract"]}"')
        examples = sc.get("examples") or []
        if examples:
            parts.append("Эталонные примеры:")
            for ex in examples[:3]:
                text = (ex.get("text") or "").strip().replace("\n", " ")
                if len(text) > 320:
                    text = text[:317] + "..."
                parts.append(f"- (n={ex.get('n')}) {text}")
        if sc.get("notes"):
            parts.append(f"Заметка: {sc['notes']}")

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
) -> list[dict]:
    """blocks: [{id, text}]. Returns assignments with block_id + subcluster_sort_order.

    Only returns orders that exist in `subclusters`. Never invents clusters.
    """
    if not blocks or not subclusters:
        return []
    system = build_classify_system_prompt(parent_tag, subclusters)
    allowed_block_ids = {int(b["id"]) for b in blocks}
    payload = [
        {"block_id": b["id"], "text": (b.get("text") or "")[:1200]}
        for b in blocks
    ]
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
        # One repair pass: remap only missing/invalid ids — still restricted to whitelist
        allowed_orders = [int(sc["sort_order"]) for sc in subclusters]
        repair_payload = [b for b in payload if int(b["block_id"]) in missing]
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
) -> dict[str, Any]:
    """Classify unassigned blocks for parent_tag (optionally limited to one task)."""
    subclusters = load_active_subclusters(conn, parent_tag)
    if not subclusters:
        return {"assigned": 0, "pending": 0, "error": "no_subclusters"}

    order_to_id = {int(sc["sort_order"]): int(sc["id"]) for sc in subclusters}
    params: list = [parent_tag]
    task_clause = ""
    if task_id:
        task_clause = "AND b.task_id = ?"
        params.append(task_id)
    rows = conn.execute(
        f"""
        SELECT b.id, b.text
        FROM blocks b
        WHERE b.tag = ?
          {task_clause}
          AND NOT EXISTS (
              SELECT 1 FROM block_subcluster_assignments a WHERE a.block_id = b.id
          )
        ORDER BY b.id ASC
        """,
        params,
    ).fetchall()
    blocks = [{"id": r["id"], "text": r["text"]} for r in rows]
    total_pending = len(blocks)
    assigned = 0

    for start in range(0, len(blocks), BATCH_SIZE):
        batch = blocks[start:start + BATCH_SIZE]
        results = classify_blocks_batch(client, parent_tag, subclusters, batch)
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
