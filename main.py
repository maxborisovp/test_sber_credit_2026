"""
main.py

Точка входа: прогоняет все .txt документы из data/ через три пайплайна:
    1. classify() - определение типа документа
    2. extract() - извлечение структурированных полей
    3. check_subject() - проверка предмета оплаты (из extract()) на соответствие
льготной сельхоз-программе

Результаты каждого шага выводятся в виде таблицы в консоль.

Запуск (из корня проекта, рядом с data/):
python main.py
"""

from pathlib import Path
from typing import List, Tuple

from src.classification import classify
from src.extraction import extract
from src.checking_subject import check_subject

def load_documents(data_dir: str) -> List[Tuple[str, str]]:
    """
    Загружает все .txt файлы из data_dir.
    Возвращает список (filename, text), отсортированный по имени файла.
    """
    paths = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix == ".txt"
    )
    return [(path.name, path.read_text(encoding="utf-8")) for path in paths]


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Печатает таблицу в консоль с выравниванием колонок по ширине содержимого."""
    str_rows = [
        [("-" if cell is None else str(cell)) for cell in row] 
        for row in rows
        ]

    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(cells: List[str]) -> str:
        return " | ".join(
            cell.ljust(widths[i]) for i, cell in enumerate(cells)
        )

    print(format_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in str_rows:
        print(format_row(row))
    print()


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    documents = load_documents(data_dir)
    if not documents:
        print(f"В папке {data_dir} не найдено ни одного .txt файла.")
        return

    # classify() -----------------------------------------------------
    print("=== Классификация типа документа ===")
    classify_rows = []
    for filename, text in documents:
        doc_type, _confidence = classify(text)
        classify_rows.append([filename, doc_type])
    print_table(["filename", "doc type"], classify_rows)

    # extract() ----------------------------------------------------------
    print("=== Извлечённые поля ===")
    extract_rows = []
    extracted_by_file = {}
    for filename, text in documents:
        fields = extract(text)
        extracted_by_file[filename] = fields
        extract_rows.append([
            filename,
            fields["amount"],
            fields["date"],
            fields["inn"],
            fields["contractor"],
            fields["subject"],
        ])
    print_table(
        ["filename", "amount", "date", "inn", "contractor", "subject"],
        extract_rows
    )

    # check_subject() ------------------------------------------------------
    print("=== Проверка предмета оплаты на соответствие сельхоз-программе ===")
    check_rows = []
    for filename, _text in documents:
        subject = extracted_by_file[filename]["subject"]
        eligible, confidence, explanation = check_subject(subject)
        check_rows.append([filename, f"{eligible} ({confidence:.2f}) — {explanation}"])
    print_table(["filename", "check_result"], check_rows)


if __name__ == "__main__":
    main()
