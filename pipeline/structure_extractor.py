"""
structure_extractor.py — извлекает упрощенную абстрактную структуру сценария из tree.json.

Остаются только intro, outro и stories, внутри них только tag_pattern.
"""
import sys
import json


def _walk_beats(node):
    """Рекурсивно собирает все beat-листья из поддерева."""
    for child in node.get("children", []):
        if child.get("type") == "beat":
            yield child
        else:
            yield from _walk_beats(child)


def extract_intro(tree: dict) -> dict:
    """Извлекает блоки до первой MAIN_STORY, включая root-level sub_arc."""
    intro_beats = []
    for child in tree.get("children", []):
        if child.get("type") == "main_story":
            break
        if child.get("type") == "beat":
            intro_beats.append(child)
        else:
            intro_beats.extend(_walk_beats(child))
    tags = [b["tag"] for b in intro_beats]
    return {
        "tag_pattern": tags
    }


def extract_outro(tree: dict) -> dict:
    """Извлекает блоки после последней MAIN_STORY, включая root-level sub_arc."""
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

    tags = [b["tag"] for b in outro_beats]
    return {
        "tag_pattern": tags
    }


def extract_story(story_node: dict) -> dict:
    """Извлекает структуру одной main_story."""
    beats = list(_walk_beats(story_node))
    tags = [b["tag"] for b in beats]
    return {
        "tag_pattern": tags
    }


def extract_structure(tree: dict) -> dict:
    """Главная функция: tree.json → structure.json."""
    intro = extract_intro(tree)
    outro = extract_outro(tree)

    story_nodes = [c for c in tree.get("children", []) if c.get("type") == "main_story"]
    stories = [extract_story(s) for s in story_nodes]

    return {
        "intro": intro,
        "stories": stories,
        "outro": outro
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 structure_extractor.py <tree.json> <structure.json>")
        sys.exit(1)

    tree_path, out_path = sys.argv[1], sys.argv[2]
    tree = json.load(open(tree_path, encoding="utf-8"))

    structure = extract_structure(tree)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

    print(f"Упрощенная структура сохранена в {out_path}")


if __name__ == "__main__":
    main()
