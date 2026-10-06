"""
structure_extractor.py — извлекает упрощенную абстрактную структуру сценария из tree.json.

Компактный вариант: intro / outro / stories с tag_pattern.
Расширенный: то же + blocks (tag, text, subcluster) и source (описание/URL видео).
"""
import sys
import json
from typing import Optional


def _walk_beats(node):
    """Рекурсивно собирает все beat-листья из поддерева."""
    for child in node.get("children", []):
        if child.get("type") == "beat":
            yield child
        else:
            yield from _walk_beats(child)


def _intro_beats(tree: dict) -> list:
    """Блоки до первой MAIN_STORY, включая root-level sub_arc."""
    intro_beats = []
    for child in tree.get("children", []):
        if child.get("type") == "main_story":
            break
        if child.get("type") == "beat":
            intro_beats.append(child)
        else:
            intro_beats.extend(_walk_beats(child))
    return intro_beats


def _outro_beats(tree: dict) -> list:
    """Блоки после последней MAIN_STORY, включая root-level sub_arc."""
    outro_beats = []
    last_story_idx = -1
    for i, child in enumerate(tree.get("children", [])):
        if child.get("type") == "main_story":
            last_story_idx = i

    if last_story_idx >= 0:
        for child in tree.get("children", [])[last_story_idx + 1:]:
            if child.get("type") == "beat":
                outro_beats.append(child)
            else:
                outro_beats.extend(_walk_beats(child))
    return outro_beats


def _lookup_subcluster(beat: dict, subclusters_by_index: Optional[dict]) -> Optional[dict]:
    if not subclusters_by_index:
        return None
    idx = beat.get("index")
    if idx is None:
        return None
    return subclusters_by_index.get(idx) or subclusters_by_index.get(str(idx))


def _beat_summary(beat: dict, subclusters_by_index: Optional[dict] = None) -> dict:
    summary = {
        "tag": beat.get("tag"),
        "text": beat.get("text") or "",
        "subcluster": None,
        "abstract": None,
        "tension": beat.get("tension_score"),
        "start": beat.get("start"),
        "end": beat.get("end"),
    }
    sc = _lookup_subcluster(beat, subclusters_by_index)
    if not sc:
        return summary

    name = (
        sc.get("subcluster_name")
        or sc.get("name")
        or ""
    ).strip()
    order = sc.get("subcluster_sort_order")
    if order is None:
        order = sc.get("sort_order")
    # residual / unassigned
    if order is None or int(order) == 0:
        return summary

    summary["subcluster"] = name or str(order)
    abstract = (
        sc.get("subcluster_abstract")
        or sc.get("abstract")
        or ""
    ).strip()
    summary["abstract"] = abstract or None
    return summary


def _section_from_beats(
    beats: list,
    *,
    extended: bool,
    subclusters_by_index: Optional[dict] = None,
) -> dict:
    section = {
        "tag_pattern": [b["tag"] for b in beats]
    }
    if extended:
        section["blocks"] = [
            _beat_summary(b, subclusters_by_index) for b in beats
        ]
    return section


def extract_intro(
    tree: dict,
    *,
    extended: bool = False,
    subclusters_by_index: Optional[dict] = None,
) -> dict:
    return _section_from_beats(
        _intro_beats(tree),
        extended=extended,
        subclusters_by_index=subclusters_by_index,
    )


def extract_outro(
    tree: dict,
    *,
    extended: bool = False,
    subclusters_by_index: Optional[dict] = None,
) -> dict:
    return _section_from_beats(
        _outro_beats(tree),
        extended=extended,
        subclusters_by_index=subclusters_by_index,
    )


def extract_story(
    story_node: dict,
    *,
    extended: bool = False,
    subclusters_by_index: Optional[dict] = None,
) -> dict:
    return _section_from_beats(
        list(_walk_beats(story_node)),
        extended=extended,
        subclusters_by_index=subclusters_by_index,
    )


def resolve_source_comment(
    source_note: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Optional[str]:
    """Комментарий к видео: URL источника или текстовое описание."""
    for value in (source_url, source_note):
        if value and str(value).strip():
            return str(value).strip()
    return None


def extract_structure(
    tree: dict,
    *,
    extended: bool = False,
    subclusters_by_index: Optional[dict] = None,
) -> dict:
    """Главная функция: tree.json → structure.json (компактный или расширенный)."""
    story_nodes = [c for c in tree.get("children", []) if c.get("type") == "main_story"]
    return {
        "intro": extract_intro(
            tree, extended=extended, subclusters_by_index=subclusters_by_index
        ),
        "stories": [
            extract_story(
                s, extended=extended, subclusters_by_index=subclusters_by_index
            )
            for s in story_nodes
        ],
        "outro": extract_outro(
            tree, extended=extended, subclusters_by_index=subclusters_by_index
        ),
    }


def extract_extended_structure(
    tree: dict,
    *,
    source_note: Optional[str] = None,
    source_url: Optional[str] = None,
    subclusters_by_index: Optional[dict] = None,
) -> dict:
    """Расширенная структура: source + blocks (tag, text, subcluster, abstract, tension, start/end)."""
    structure = extract_structure(
        tree, extended=True, subclusters_by_index=subclusters_by_index
    )
    structure["source"] = resolve_source_comment(source_note, source_url)
    return structure


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 structure_extractor.py <tree.json> <structure.json> [--extended]")
        sys.exit(1)

    tree_path, out_path = sys.argv[1], sys.argv[2]
    extended = len(sys.argv) == 4 and sys.argv[3] == "--extended"
    tree = json.load(open(tree_path, encoding="utf-8"))

    structure = (
        extract_extended_structure(tree) if extended else extract_structure(tree)
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

    kind = "Расширенная" if extended else "Упрощенная"
    print(f"{kind} структура сохранена в {out_path}")


if __name__ == "__main__":
    main()
