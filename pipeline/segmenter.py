"""
segmenter.py — сегментация и тегирование сценария через OpenAI structured output.

Использование:
    OPENAI_API_KEY=sk-... python3 segmenter.py story1.txt out_story1.json
"""
import os
import sys
import json
from openai import OpenAI
from .taxonomy import TAG_LIST, get_tag_definitions
from .chunker import chunk_text
from .content_lang import normalize_content_lang


MODEL = "gpt-4o"


def build_system_prompt(content_lang: str = "ru") -> str:
    lang = normalize_content_lang(content_lang)
    definitions = get_tag_definitions(lang)
    defs_json = json.dumps(definitions, ensure_ascii=False, indent=2)

    if lang == "en":
        return f"""You are an analyst of long-form video storytelling structure (genre: "top-N worst/best objects",
narrative YouTube videos, automatic transcripts that may include [music] markers).

The INPUT SCRIPT LANGUAGE is English. Tag codes must stay Latin (from the whitelist). Write reasoning in English.

Your task: split the given text fragment into functional meaning blocks and assign each block exactly one
function (tag) from the closed list below. A block is not a sentence or paragraph by form — it is the
minimal text unit that performs ONE function. One block may span half a sentence to several sentences
if they share the same function.

TAG LIST (use ONLY these values for the tag field):
{defs_json}

If no tag fits — use "CUSTOM:<short_name>" (e.g. "CUSTOM:IRONY"). Use CUSTOM rarely.

IMPORTANT about STORY_BOUNDARY: it is a strict technical counter tag with the narrowest firing rule.
Use it ONLY in the block where the story hero is named for the FIRST TIME — exact name/model/nickname.
A mysterious teaser without a name is OPEN_LOOP, not STORY_BOUNDARY. Later mentions of an already-named
hero are NOT STORY_BOUNDARY (use ARC_OPEN etc.). A previous-story hero inside TRANSITION_BRIDGE is also
not STORY_BOUNDARY. Mentally check: exactly one STORY_BOUNDARY per story — the first naming.

Naming may be indirect (second-person address, index without "this is GAZ-52"). The first recognition
moment still counts as STORY_BOUNDARY.

CRITICAL: each block's "text" must be an EXACT verbatim substring of the source (minimal whitespace
normalization only), covering the fragment sequentially with no gaps or overlaps.

For each block also provide:
- tension_score: 0-10 emotional/dramatic tension of this block
- loop_id: for OPEN_LOOP/CLOSE_LOOP a short human id shared by open/close; else null
- reasoning: one short phrase why this function

Respond with STRICT valid JSON only:
{{
  "blocks": [
    {{"text": "...", "tag": "...", "tension_score": 0, "loop_id": null, "reasoning": "..."}}
  ]
}}
No text before or after JSON.
"""

    return f"""Ты — аналитик структуры видео-сторителлингов (жанр: "топ-N худших/лучших объектов",
длинные нарративные ролики на YouTube, автоматическая транскрипция с разметкой [музыка]).

ЯЗЫК ВХОДНОГО СКРИПТА: русский. Коды тегов — латиницей из whitelist. Reasoning пиши по-русски.

Твоя задача — разбить присланный фрагмент текста на смысловые функциональные блоки и для каждого
блока определить его функцию (тег) из закрытого списка ниже. Блок — это не предложение и не абзац
по форме, а минимальная единица текста, выполняющая ОДНУ функцию. Один блок может быть от половины
предложения до нескольких предложений подряд, если они выполняют одну и ту же функцию.

СПИСОК ТЕГОВ (используй ТОЛЬКО эти значения для поля tag):
{defs_json}

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


def build_chunk_context_note(
    content_lang: str,
    prev_tail: str,
    open_loop_ids: set,
    introduced_heroes: list,
) -> str:
    lang = normalize_content_lang(content_lang)
    heroes_note = "; ".join(introduced_heroes) if introduced_heroes else (
        "none yet" if lang == "en" else "пока никого"
    )
    loops = sorted(open_loop_ids) if open_loop_ids else (
        "none" if lang == "en" else "нет"
    )
    tail = prev_tail[-200:]
    if lang == "en":
        return (
            f'The previous fragment ended with: "...{tail}". '
            f"Open loop_ids that may still close here: {loops}. "
            f"HEROES ALREADY INTRODUCED (via STORY_BOUNDARY) earlier in this video: {heroes_note}. "
            f"If any of them is mentioned again in this fragment — that is NOT STORY_BOUNDARY "
            f"(use ARC_OPEN/TENSION/CONTEXT/etc.). STORY_BOUNDARY in this fragment is only for an "
            f"object NOT in the list above.\n\n"
            f"IMPORTANT: naming may be indirect (second-person address, index without "
            f"'this is MODEL-X'). The first recognition moment is still STORY_BOUNDARY."
        )
    return (
        f"Предыдущий фрагмент заканчивался так: \"...{tail}\". "
        f"Открытые ранее петли (loop_id), которые ещё могут закрываться здесь: {loops}. "
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


def segment_text(
    client: OpenAI,
    text: str,
    context_note: str = "",
    content_lang: str = "ru",
) -> dict:
    lang = normalize_content_lang(content_lang)
    user_content = text
    if context_note:
        if lang == "en":
            user_content = (
                f"[CONTEXT: this is a continuation, not the start. {context_note}]\n\n{text}"
            )
        else:
            user_content = (
                f"[КОНТЕКСТ: это продолжение текста, не начало. {context_note}]\n\n{text}"
            )
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": build_system_prompt(lang)},
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
    introduced_heroes = []
    prev_tail = ""

    for i, chunk in enumerate(chunks):
        context_note = ""
        if i > 0:
            context_note = build_chunk_context_note(
                "ru", prev_tail, open_loop_ids, introduced_heroes
            )

        print(f"  Обрабатываю чанк {i+1}/{len(chunks)} ({len(chunk.split())} слов)...")
        result = segment_text(client, chunk, context_note, content_lang="ru")
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

    CLUSTER_WINDOW = 20
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
            f"линий, нашлось {boundary_count}."
        )


if __name__ == "__main__":
    main()
