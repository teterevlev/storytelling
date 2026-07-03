"""
validator.py — проверка, что блоки из segmenter.py реально соответствуют исходному тексту
и покрывают его последовательно без существенных дыр/наложений.

Использование:
    python3 validator.py story1.txt out_story1.json
"""
import sys
import json
import re


def normalize(s: str) -> str:
    s = s.replace("[музыка]", " ")
    return re.sub(r"\s+", " ", s).strip()


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 validator.py <source.txt> <segmented.json>")
        sys.exit(1)

    source = open(sys.argv[1], encoding="utf-8").read()
    norm_source = normalize(source)
    data = json.load(open(sys.argv[2], encoding="utf-8"))
    blocks = data["blocks"]

    cursor = 0
    issues = []
    found_positions = []

    for i, b in enumerate(blocks):
        block_text = normalize(b["text"])
        idx = norm_source.find(block_text, max(0, cursor - 20))
        if idx == -1:
            # попробуем найти где угодно (не по порядку) — для диагностики
            idx_any = norm_source.find(block_text)
            issues.append({
                "block_index": i,
                "tag": b.get("tag"),
                "problem": "NOT_FOUND_IN_ORDER" if idx_any == -1 else "FOUND_OUT_OF_ORDER",
                "text_preview": block_text[:80],
            })
            continue

        gap = idx - cursor
        if gap > 30:
            issues.append({
                "block_index": i,
                "tag": b.get("tag"),
                "problem": f"GAP_{gap}_CHARS_SKIPPED",
                "text_preview": block_text[:80],
            })

        found_positions.append((i, idx, idx + len(block_text)))
        cursor = idx + len(block_text)

    coverage = cursor / len(norm_source) if norm_source else 0

    print(f"Всего блоков: {len(blocks)}")
    print(f"Покрытие текста (по курсору до последнего найденного блока): {coverage:.1%}")
    print(f"Длина исходника (норм.): {len(norm_source)} символов")
    print(f"Проблем найдено: {len(issues)}")
    for iss in issues:
        print(f"  - [{iss['block_index']}] {iss['tag']}: {iss['problem']} :: {iss['text_preview']!r}")

    # сводка по тегам
    from collections import Counter
    tag_counts = Counter(b.get("tag") for b in blocks)
    print("\nЧастота тегов:")
    for tag, cnt in tag_counts.most_common():
        print(f"  {tag}: {cnt}")

    # проверка на дублирование соседних блоков (типично для шва между чанками)
    dups = []
    for i in range(1, len(blocks)):
        a = normalize(blocks[i - 1]["text"])
        b = normalize(blocks[i]["text"])
        if a and b and (a in b or b in a) and min(len(a), len(b)) > 20:
            dups.append((i - 1, i))

    if dups:
        print(f"\nВозможные дубли на стыке блоков (часто = шов между чанками): {len(dups)}")
        for i1, i2 in dups:
            print(f"  [{i1}]<->[{i2}]: {normalize(blocks[i2]['text'])[:90]!r}")


if __name__ == "__main__":
    main()