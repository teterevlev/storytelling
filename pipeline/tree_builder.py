"""
tree_builder.py — строит иерархическое дерево структуры сценария из плоского списка
тегированных блоков (результат segmenter.py).

Логика (стек контекстов):
- STORY_BOUNDARY  -> закрывает все открытые под-арки/истории, открывает новую MAIN_STORY
- ARC_OPEN        -> открывает SUB_ARC внутри текущего контекста (поддерживает вложенность)
- ARC_CLOSE       -> закрывает текущую SUB_ARC (если она открыта), иначе просто beat
- TRANSITION_BRIDGE -> закрывает все открытые контексты, возвращает на уровень видео
- всё остальное   -> обычный "beat"-лист в текущем самом глубоком открытом контексте

Использование:
    python3 tree_builder.py out.json tree.json
"""
import sys
import json


def make_node(node_type: str, label: str, index: int):
    return {
        "type": node_type,       # "video" | "main_story" | "sub_arc"
        "label": label,
        "start_index": index,
        "children": [],
    }


def build_tree(blocks):
    root = make_node("video", "VIDEO", 0)
    stack = [root]

    for i, b in enumerate(blocks):
        tag = b.get("tag", "")
        beat = {
            "type": "beat",
            "tag": tag,
            "index": i,
            "text": b.get("text", ""),
            "tension_score": b.get("tension_score"),
            "loop_id": b.get("loop_id"),
            "reasoning": b.get("reasoning"),
            "chunk_index": b.get("chunk_index"),
        }

        if tag == "STORY_BOUNDARY":
            while len(stack) > 1:
                stack.pop()
            new_story = make_node("main_story", b.get("text", "")[:70], i)
            stack[-1]["children"].append(new_story)
            stack.append(new_story)
            stack[-1]["children"].append(beat)

        elif tag == "ARC_OPEN":
            new_arc = make_node("sub_arc", b.get("text", "")[:70], i)
            stack[-1]["children"].append(new_arc)
            stack.append(new_arc)
            stack[-1]["children"].append(beat)

        elif tag == "ARC_CLOSE":
            stack[-1]["children"].append(beat)
            if len(stack) > 2 and stack[-1]["type"] == "sub_arc":
                stack.pop()

        elif tag == "TRANSITION_BRIDGE":
            stack[-1]["children"].append(beat)
            while len(stack) > 1:
                stack.pop()

        else:
            stack[-1]["children"].append(beat)

    return root


def _walk_beats(node):
    """Рекурсивно собирает все beat-листья узла (включая вложенные под-арки)."""
    for child in node["children"]:
        if child["type"] == "beat":
            yield child
        else:
            yield from _walk_beats(child)


def annotate_stats(node):
    """Добавляет в каждый не-beat узел сводную статистику (число блоков, слов, средний tension)."""
    if node["type"] == "beat":
        return
    for child in node["children"]:
        annotate_stats(child)

    beats = list(_walk_beats(node))
    word_count = sum(len(b["text"].split()) for b in beats)
    scores = [b["tension_score"] for b in beats if isinstance(b["tension_score"], (int, float))]
    node["stats"] = {
        "beat_count": len(beats),
        "word_count": word_count,
        "avg_tension": round(sum(scores) / len(scores), 2) if scores else None,
        "max_tension": max(scores) if scores else None,
        "tag_sequence": [b["tag"] for b in beats],
    }


def print_outline(node, depth=0):
    indent = "  " * depth
    if node["type"] == "video":
        print(f"{indent}VIDEO  ({node['stats']['beat_count']} blocks, "
              f"{node['stats']['word_count']} words)")
    elif node["type"] == "main_story":
        print(f"{indent}├─ MAIN_STORY: {node['label']!r}  "
              f"[{node['stats']['beat_count']} blocks, avg_tension={node['stats']['avg_tension']}]")
    elif node["type"] == "sub_arc":
        print(f"{indent}├─ sub_arc: {node['label']!r}  "
              f"[{node['stats']['beat_count']} blocks, avg_tension={node['stats']['avg_tension']}]")

    for child in node["children"]:
        if child["type"] == "beat":
            print(f"{indent}    · {child['tag']}"
                  + (f" (tension={child['tension_score']})" if child['tension_score'] is not None else "")
                  + f": {child['text'][:55]!r}")
        else:
            print_outline(child, depth + 1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 tree_builder.py <segmented.json> <tree_output.json>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    data = json.load(open(input_path, encoding="utf-8"))
    blocks = data["blocks"]

    tree = build_tree(blocks)
    annotate_stats(tree)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    main_stories = [c for c in tree["children"] if c["type"] == "main_story"]
    print(f"Построено дерево: {len(main_stories)} главных историй, "
          f"{tree['stats']['beat_count']} блоков всего.")
    print(f"Сохранено в {output_path}\n")
    print("=" * 70)
    print_outline(tree)


if __name__ == "__main__":
    main()
