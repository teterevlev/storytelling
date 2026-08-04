"""Parse clusters_review markdown and seed tag_subclusters."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

REVIEW_SOURCE_FILES = ("1.json", "2.json", "3.json", "4.txt", "5.2.txt")

TAG_HEADING_RE = re.compile(
    r"^#\s+([A-Z][A-Z0-9_:]*)(?:\s|\(|$)",
    re.MULTILINE,
)
CLUSTER_HEADING_RE = re.compile(
    r"^##\s+(?:Кластер\s+)?(\d+)\s*[—.–-]\s*(.+?)(?:\s*\(\d+\))?\s*$",
    re.IGNORECASE,
)
CONTEXT_HEADING_RE = re.compile(
    r"^##\s+(\d+)\.\s+(.+?)(?:\s*\(\d+\))?\s*$",
)
RESIDUAL_HEADING_RE = re.compile(
    r"^##\s+.*(?:Временно неклассифицированные|Остаток).*$",
    re.IGNORECASE,
)
PRIMARY_EXAMPLE_NS_RE = re.compile(
    r"(?:Примеры(?:\s+из\s+данных)?|Номера примеров(?:\s*\(\d+\))?)\s*:\s*(.+)",
    re.IGNORECASE,
)


def normalize_parent_tag(raw: str) -> str:
    tag = raw.strip()
    if tag.startswith("CUSTOM_") and ":" not in tag:
        return "CUSTOM:" + tag[len("CUSTOM_"):]
    return tag


def _parse_primary_ns(line: str) -> list[int]:
    """Extract primary example numbers; ignore secondary notes in parentheses."""
    m = PRIMARY_EXAMPLE_NS_RE.search(line)
    if not m:
        return []
    payload = m.group(1)
    # Drop parenthetical secondary membership notes
    payload = re.sub(r"\([^)]*\)", "", payload)
    nums = []
    for part in re.split(r"[,;\s]+", payload.strip()):
        part = part.strip().rstrip(".")
        if part.isdigit():
            nums.append(int(part))
    return nums


def parse_review_markdown(text: str) -> dict[str, list[dict[str, Any]]]:
    """Return {parent_tag: [cluster dicts]}."""
    lines = text.splitlines()
    by_tag: dict[str, list[dict[str, Any]]] = {}
    current_tag: Optional[str] = None
    i = 0
    # Skip intro until first tag heading that looks like a real tag (not "Кластеризация")
    while i < len(lines):
        line = lines[i]
        tag_m = re.match(r"^#\s+([A-Z][A-Z0-9_:]*)\b", line)
        if tag_m and tag_m.group(1) not in ("Кластеризация",):
            # Ignore top-level Russian titles
            if re.match(r"^#\s+[A-Z]{2,}", line):
                current_tag = normalize_parent_tag(tag_m.group(1))
                by_tag.setdefault(current_tag, [])
                i += 1
                continue

        if current_tag is None:
            i += 1
            continue

        # Next tag heading ends current
        if re.match(r"^#\s+[A-Z][A-Z0-9_:]*\b", line) and not line.startswith("##"):
            tag_m = re.match(r"^#\s+([A-Z][A-Z0-9_:]*)\b", line)
            if tag_m:
                current_tag = normalize_parent_tag(tag_m.group(1))
                by_tag.setdefault(current_tag, [])
                i += 1
                continue

        residual_m = RESIDUAL_HEADING_RE.match(line.strip())
        cluster_m = CLUSTER_HEADING_RE.match(line.strip()) or CONTEXT_HEADING_RE.match(line.strip())

        if residual_m or cluster_m:
            status = "residual" if residual_m else "active"
            if residual_m:
                sort_order = 0
                name = "временно неклассифицированные"
                slug = "residual"
            else:
                sort_order = int(cluster_m.group(1))
                name = cluster_m.group(2).strip()
                slug = f"{sort_order:02d}"

            formula_parts: list[str] = []
            abstract_parts: list[str] = []
            notes_parts: list[str] = []
            example_ns: list[int] = []
            mode: Optional[str] = None
            i += 1
            while i < len(lines):
                raw = lines[i]
                stripped = raw.strip()
                if stripped.startswith("#"):
                    break
                if stripped == "---":
                    i += 1
                    continue

                if RESIDUAL_HEADING_RE.match(stripped) or CLUSTER_HEADING_RE.match(stripped) or CONTEXT_HEADING_RE.match(stripped):
                    break
                if re.match(r"^#\s+[A-Z][A-Z0-9_:]*\b", stripped):
                    break

                lower = stripped.lower()
                # Bold markdown section headers: **Формула:** / **Абстрактно:** / **Примеры...**
                section_m = re.match(r"^\*\*(.+?)\*\*:?\s*(.*)$", stripped)
                if section_m:
                    section_name = section_m.group(1).strip().lower()
                    section_rest = section_m.group(2).strip()
                    if section_name.startswith("формула"):
                        mode = "formula"
                        if section_rest:
                            formula_parts.append(section_rest)
                        i += 1
                        continue
                    if section_name.startswith("абстрактно"):
                        mode = "abstract"
                        if section_rest:
                            abstract_parts.append(section_rest.strip('"«»'))
                        i += 1
                        continue
                    if section_name.startswith("примеры") or section_name.startswith("номера примеров"):
                        mode = None
                        ns = _parse_primary_ns(stripped) or _parse_primary_ns(
                            f"Номера примеров: {section_rest}"
                        )
                        example_ns.extend(ns)
                        i += 1
                        continue
                    # Unknown **Section:** — stop capturing formula/abstract
                    mode = None

                if lower.startswith("формула:"):
                    mode = "formula"
                    rest = re.sub(r"^формула:\s*", "", stripped, flags=re.IGNORECASE)
                    if rest:
                        formula_parts.append(rest)
                    i += 1
                    continue
                if lower.startswith("абстрактно:"):
                    mode = "abstract"
                    rest = re.sub(r"^абстрактно:\s*", "", stripped, flags=re.IGNORECASE)
                    if rest:
                        abstract_parts.append(rest.strip('"«»'))
                    i += 1
                    continue
                if PRIMARY_EXAMPLE_NS_RE.search(stripped) and "из данных" not in lower:
                    # "Примеры: 1, 2" or "Номера примеров (71): 1, 2"
                    if stripped.startswith("**Примеры") or lower.startswith("примеры из данных"):
                        mode = None
                        ns = _parse_primary_ns(stripped)
                        example_ns.extend(ns)
                        i += 1
                        continue
                    ns = _parse_primary_ns(stripped)
                    example_ns.extend(ns)
                    mode = None
                    i += 1
                    continue
                if lower.startswith("номера примеров") or lower.startswith("примеры из данных"):
                    ns = _parse_primary_ns(stripped)
                    example_ns.extend(ns)
                    mode = None
                    i += 1
                    continue
                if stripped.startswith("- [") or stripped.startswith("* ["):
                    # bullet with [n] text — optional; numbers also in Номера
                    mode = None
                    i += 1
                    continue
                if stripped.startswith("(") and stripped.endswith(")"):
                    notes_parts.append(stripped[1:-1].strip())
                    mode = None
                    i += 1
                    continue
                if lower.startswith("примечание:"):
                    notes_parts.append(stripped.split(":", 1)[1].strip())
                    mode = None
                    i += 1
                    continue

                if mode == "formula" and stripped:
                    formula_parts.append(stripped)
                elif mode == "abstract" and stripped:
                    abstract_parts.append(stripped.strip('"«»'))
                i += 1

            formula = "\n".join(formula_parts).strip()
            abstract = "\n".join(abstract_parts).strip().strip('"«»')
            # Drop markdown bold leftovers / example bleed-through
            formula = re.sub(r"^\*+\s*", "", formula).strip()
            abstract = re.sub(r"^\*+\s*", "", abstract).strip().strip('"«»')
            abstract = re.split(r"\n\s*\*\*Примеры", abstract, maxsplit=1)[0].strip()
            abstract = re.split(r"\n\s*Примеры из данных", abstract, maxsplit=1)[0].strip()

            by_tag[current_tag].append({
                "slug": slug,
                "sort_order": sort_order,
                "name": name,
                "formula": formula,
                "abstract": abstract,
                "notes": "\n".join(notes_parts).strip(),
                "status": status,
                "example_ns": example_ns,
            })
            continue

        i += 1

    return by_tag


def resolve_review_task_ids(conn) -> list[str]:
    """Pick completed task ids for REVIEW_SOURCE_FILES (most blocks wins)."""
    ids: list[str] = []
    for filename in REVIEW_SOURCE_FILES:
        row = conn.execute(
            """
            SELECT t.id
            FROM tasks t
            WHERE t.filename = ? AND t.status = 'completed'
            ORDER BY (SELECT COUNT(*) FROM blocks b WHERE b.task_id = t.id) DESC,
                     t.created_at DESC
            LIMIT 1
            """,
            (filename,),
        ).fetchone()
        if row:
            ids.append(row["id"] if isinstance(row, dict) or hasattr(row, "keys") else row[0])
    return ids


def numbered_blocks_for_tag(conn, tag: str, task_ids: list[str]) -> dict[int, dict]:
    """Map n (1-based within tag across review sources) -> {block_id, text}."""
    if not task_ids:
        return {}
    order_cases = " ".join(
        f"WHEN ? THEN {i}" for i in range(len(task_ids))
    )
    placeholders = ",".join("?" * len(task_ids))
    rows = conn.execute(
        f"""
        SELECT b.id, b.text, b.task_id, t.filename
        FROM blocks b
        JOIN tasks t ON t.id = b.task_id
        WHERE b.tag = ? AND b.task_id IN ({placeholders})
        ORDER BY CASE b.task_id {order_cases} END, b.position ASC
        """,
        (tag, *task_ids, *task_ids),
    ).fetchall()
    out: dict[int, dict] = {}
    for idx, row in enumerate(rows, start=1):
        text = (row["text"] if hasattr(row, "keys") else row[1]) or ""
        block_id = row["id"] if hasattr(row, "keys") else row[0]
        out[idx] = {"n": idx, "text": text.strip(), "block_id": block_id}
    return out


def seed_tag_subclusters(conn, review_path: Path) -> int:
    """Upsert clusters from review markdown. Returns number of upserted rows."""
    if not review_path.is_file():
        return 0
    parsed = parse_review_markdown(review_path.read_text(encoding="utf-8"))
    task_ids = resolve_review_task_ids(conn)
    # Ensure columns exist (caller may also migrate)
    count = 0
    for parent_tag, clusters in parsed.items():
        numbered = numbered_blocks_for_tag(conn, parent_tag, task_ids)
        for c in clusters:
            exemplars = []
            for n in (c.get("example_ns") or [])[:5]:
                item = numbered.get(n)
                if item and item.get("text"):
                    exemplars.append(item)
            examples_json = json.dumps(exemplars, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO tag_subclusters
                    (parent_tag, slug, name, formula, abstract, notes, examples_json, sort_order, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(parent_tag, slug) DO UPDATE SET
                    name = excluded.name,
                    formula = excluded.formula,
                    abstract = excluded.abstract,
                    notes = excluded.notes,
                    examples_json = excluded.examples_json,
                    sort_order = excluded.sort_order,
                    status = excluded.status
                """,
                (
                    parent_tag,
                    c["slug"],
                    c["name"],
                    c.get("formula") or None,
                    c.get("abstract") or None,
                    c.get("notes") or None,
                    examples_json,
                    c["sort_order"],
                    c["status"],
                ),
            )
            count += 1
    return count


# Fatal class of error from raw-batch clustering: cluster *named* by domain/topic.
# Mechanism formulas may still mention X/Y; we only judge the human-facing name.
DOMAIN_LEAK_NAME_RE = re.compile(
    r"(?i)"
    r"гравитац|нептун|юпитер|сатурн|планет|космос|галактик|орбит|"
    r"завод|автомобил|грузовик|танк|ракет|оруж|винтовк|"
    r"ссср|советск|наса|nasa|voyager|ньютон"
)


def demote_domain_leaky_subclusters(conn) -> list[dict]:
    """Mark clearly topic-named clusters inactive so classify/guide ignore them.

    Suboptimal but stops the worst failure mode without re-clustering everything.
    Returns list of demoted rows for logging.
    """
    rows = conn.execute(
        """
        SELECT id, parent_tag, sort_order, slug, name, status
        FROM tag_subclusters
        WHERE slug != 'residual' AND sort_order > 0
        """
    ).fetchall()
    demoted = []
    for row in rows:
        name = row["name"] or ""
        if not DOMAIN_LEAK_NAME_RE.search(name):
            # If previously demoted but name is clean now (reseed rename), reactivate
            if row["status"] == "inactive":
                conn.execute(
                    "UPDATE tag_subclusters SET status = 'active' WHERE id = ?",
                    (row["id"],),
                )
            continue
        if row["status"] == "inactive":
            demoted.append(dict(row))
            continue
        conn.execute(
            "UPDATE tag_subclusters SET status = 'inactive' WHERE id = ?",
            (row["id"],),
        )
        demoted.append({**dict(row), "status": "inactive"})
    # Drop assignments that pointed at inactive clusters (stale llm labels)
    conn.execute(
        """
        DELETE FROM block_subcluster_assignments
        WHERE subcluster_id IN (
            SELECT id FROM tag_subclusters WHERE status = 'inactive'
        )
        """
    )
    return demoted
