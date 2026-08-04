import re

input_file = "clusters_review_1-5.2.md"
output_file = "result.txt"

result = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")

        # строки, начинающиеся с #
        if line.startswith("#"):
            # убрать все # в начале и пробел после них
            text = re.sub(r"^#+\s*", "", line)

            # удалить слово "кластер" (без учета регистра)
            text = re.sub(r"\bкластер\b", "", text, flags=re.IGNORECASE)

            # удалить текст в скобках вместе со скобками
            text = re.sub(r"\s*\([^)]*\)", "", text)

            # убрать лишние пробелы
            text = re.sub(r"\s+", " ", text).strip()

            result.append(text)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(result))