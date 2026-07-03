"""
chunker.py — режет длинный текст на чанки по ~300-500 слов, не разрывая предложения.
"""
import re


def split_sentences(text: str):
    # Грубое разбиение по предложениям, сохраняем разделители
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def chunk_text(text: str, target_words: int = 400, max_words: int = 550):
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sent)
        current_words += sent_words
        if current_words >= target_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


if __name__ == "__main__":
    import sys
    text = open(sys.argv[1], encoding="utf-8").read()
    chunks = chunk_text(text)
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} ({len(c.split())} words) ---")
        print(c[:150], "...")
        print()