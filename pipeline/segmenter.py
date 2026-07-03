"""
segmenter.py — сегментация и тегирование сценария через OpenAI structured output.

Использование:
    OPENAI_API_KEY=sk-... python3 segmenter.py story1.txt out_story1.json
"""
import os
import sys
import json
from openai import OpenAI
from .taxonomy import TAG_DEFINITIONS, TAG_LIST
from .chunker import chunk_text


MODEL = "gpt-4o"

SYSTEM_PROMPT = f"""Ты — аналитик структуры видео-сторителлингов (жанр: "топ-N худших/лучших объектов",
длинные нарративные ролики на YouTube, автоматическая транскрипция с разметкой [музыка]).

Твоя задача — разбить присланный фрагмент текста на смысловые функциональные блоки и для каждого
блока определить его функцию (тег) из закрытого списка ниже. Блок — это не предложение и не абзац
по форме, а минимальная единица текста, выполняющая ОДНУ функцию. Один блок может быть от половины
предложения до нескольких предложений подряд, если они выполняют одну и ту же функцию.

СПИСОК ТЕГОВ (используй ТОЛЬКО эти значения для поля tag):
{json.dumps(TAG_DEFINITIONS, ensure_ascii=False, indent=2)}

Если ни один тег не подходит — используй "CUSTOM:<своё_короткое_название>" (например "CUSTOM:IRONY").
Старайся использовать CUSTOM редко, только когда блок реально не описывается списком.

ВАЖНО про STORY_BOUNDARY: это технический счётный тег, и у него самое узкое и строгое условие
срабатывания во всём списке. Используй его ТОЛЬКО в блоке, где герой истории называется по имени
В ПЕРВЫЙ РАЗ — точным названием/моделью/устоявшимся прозвищем. Если до этого был загадочный тизер
без имени ("была машина, которая..."), это OPEN_LOOP, а не STORY_BOUNDARY. Любые последующие
упоминания уже названного героя в той же истории — это НЕ STORY_BOUNDARY, даже если меняется сцена,
локация или ракурс рассказа (для смены ракурса внутри уже идущей истории используй ARC_OPEN).
Если в TRANSITION_BRIDGE упоминается герой ПРЕДЫДУЩЕЙ истории для сравнения — это тоже не
STORY_BOUNDARY. Перед финальным ответом мысленно проверь: на каждую историю должен приходиться
РОВНО один блок с STORY_BOUNDARY — тот, где имя героя называется первый и единственный раз как
"открытие".

Называние героя не всегда прямое ('это машина X'). Иногда оно подаётся косвенно — например через
вовлекающий рассказ от 2-го лица ('ты подходишь к машине... тебе выдали не пятьдесят третий, тебе
выдали пятьдесят второй') или через номер/индекс без явного 'это ГАЗ-52'. Такой косвенный, но
фактически ПЕРВЫЙ момент опознания героя — тоже STORY_BOUNDARY, даже без хрестоматийной
фразы-объявления.

ВАЖНЕЙШЕЕ ТРЕБОВАНИЕ: поле "text" каждого блока должно быть ТОЧНОЙ дословной подстрокой исходного
текста (можно с минимальной нормализацией пробелов), без перефразирования и без пропусков — блоки
должны вместе покрывать весь присланный фрагмент последовательно, без дыр и наложений.

Для каждого блока также укажи:
- tension_score: число 0-10, уровень эмоционального/драматического напряжения именно этого блока
- loop_id: если это OPEN_LOOP или CLOSE_LOOP — короткий человекочитаемый идентификатор петли
  (например "kabina_padaet"), одинаковый для open и его close. Для остальных тегов — null.
- reasoning: одна короткая фраза, почему именно эта функция

Отвечай СТРОГО валидным JSON по схеме:
{{
  "blocks": [
    {{"text": "...", "tag": "...", "tension_score": 0, "loop_id": null, "reasoning": "..."}}
  ]
}}
Никакого текста до или после JSON.
"""


def segment_text(client: OpenAI, text: str, context_note: str = "") -> dict:
    user_content = text
    if context_note:
        user_content = (
            f"[КОНТЕКСТ: это продолжение текста, не начало. {context_note}]\n\n"
            f"{text}"
        )
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 segmenter.py <input.txt> <output.json> [expected_stories]")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    expected_stories = int(sys.argv[3]) if len(sys.argv) == 4 else None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY env var")
        sys.exit(1)

    text = open(input_path, encoding="utf-8").read()
    client = OpenAI(api_key=api_key)

    chunks = chunk_text(text, target_words=400, max_words=550)
    print(f"Текст разбит на {len(chunks)} чанков")

    all_blocks = []
    open_loop_ids = set()
    introduced_heroes = []  # тексты уже подтверждённых STORY_BOUNDARY блоков
    prev_tail = ""

    for i, chunk in enumerate(chunks):
        context_note = ""
        if i > 0:
            heroes_note = (
                "; ".join(introduced_heroes) if introduced_heroes else "пока никого"
            )
            context_note = (
                f"Предыдущий фрагмент заканчивался так: \"...{prev_tail[-200:]}\". "
                f"Открытые ранее петли (loop_id), которые ещё могут закрываться здесь: "
                f"{sorted(open_loop_ids) if open_loop_ids else 'нет'}. "
                f"УЖЕ ВВЕДЁННЫЕ ГЕРОИ (через STORY_BOUNDARY) ранее в этом видео: {heroes_note}. "
                f"Если кто-то из них упоминается в этом фрагменте снова — это НЕ STORY_BOUNDARY "
                f"(используй ARC_OPEN/TENSION/CONTEXT/итд. по смыслу). STORY_BOUNDARY можно "
                f"использовать в этом фрагменте только для объекта, которого нет в списке выше.\n\n"
                f"ВАЖНЫЙ ПРИМЕР: называние героя не всегда прямое ('это машина X'). Иногда оно "
                f"подаётся косвенно — например через вовлекающий рассказ от 2-го лица "
                f"('ты подходишь к машине... тебе выдали не пятьдесят третий, тебе выдали "
                f"пятьдесят второй') или просто через номер/индекс без явного 'это ГАЗ-52'. "
                f"Такой косвенный, но фактически ПЕРВЫЙ момент опознания героя — это тоже "
                f"STORY_BOUNDARY, даже без хрестоматийной фразы-объявления."
            )

        print(f"  Обрабатываю чанк {i+1}/{len(chunks)} ({len(chunk.split())} слов)...")
        result = segment_text(client, chunk, context_note)
        blocks = result.get("blocks", [])

        for b in blocks:
            tag = b.get("tag", "")
            loop_id = b.get("loop_id")
            if tag == "OPEN_LOOP" and loop_id:
                open_loop_ids.add(loop_id)
            elif tag == "CLOSE_LOOP" and loop_id:
                open_loop_ids.discard(loop_id)
            elif tag == "STORY_BOUNDARY":
                introduced_heroes.append(b.get("text", "")[:60])
            b["chunk_index"] = i

        all_blocks.extend(blocks)
        prev_tail = chunk

    final = {"blocks": all_blocks, "num_chunks": len(chunks)}

    # Постобработка: если несколько STORY_BOUNDARY идут кучно (тизер + раскрытие имени),
    # это почти всегда одна и та же граница, разбитая на 2 шага саспенса.
    # Оставляем последний (обычно самый содержательный — с явным именем), остальные
    # в кластере понижаем до OPEN_LOOP (чем они по сути и являются).
    CLUSTER_WINDOW = 20  # макс. расстояние в блоках, чтобы считать кластером
    boundary_indices = [i for i, b in enumerate(all_blocks) if b.get("tag") == "STORY_BOUNDARY"]
    clusters = []
    for idx in boundary_indices:
        if clusters and idx - clusters[-1][-1] <= CLUSTER_WINDOW:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    demoted = []
    for cluster in clusters:
        if len(cluster) > 1:
            keep = cluster[-1]  # оставляем последний — обычно явное называние имени
            for idx in cluster[:-1]:
                all_blocks[idx]["tag"] = "OPEN_LOOP"
                all_blocks[idx]["_auto_demoted_from"] = "STORY_BOUNDARY"
                demoted.append(idx)

    if demoted:
        print(f"Автокоррекция: {len(demoted)} дублирующих STORY_BOUNDARY понижены до OPEN_LOOP "
              f"(индексы: {demoted}) — оставлен последний блок в каждом кластере.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(all_blocks)} blocks total -> {output_path}")

    boundary_count = sum(1 for b in all_blocks if b.get("tag") == "STORY_BOUNDARY")
    boundary_block_indices = [i for i, b in enumerate(all_blocks) if b.get("tag") == "STORY_BOUNDARY"]
    print(f"\nSTORY_BOUNDARY (главные герои/линии видео) встретился {boundary_count} раз(а):")
    for i in boundary_block_indices:
        print(f"    [{i}] {all_blocks[i]['text'][:80]!r}")

    if expected_stories is not None and boundary_count != expected_stories:
        print(
            f"\nℹ️  Для справки: по заголовку/твоим словам ожидалось ~{expected_stories} главных "
            f"линий, нашлось {boundary_count}. Это НЕ обязательно ошибка — структура может "
            f"легитимно иметь другое число главных линий, либо часть найденных STORY_BOUNDARY "
            f"на самом деле являются под-сюжетами (тогда им место в ARC_OPEN, а не в "
            f"STORY_BOUNDARY) — стоит посмотреть глазами на список выше, а не считать это "
            f"автоматическим браком."
        )


if __name__ == "__main__":
    main()