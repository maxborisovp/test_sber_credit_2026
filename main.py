"""
main.py

Точка входа: прогоняет все .txt документы из data/ через:
    1. classify() - определение типа документа
    2. extract() - извлечение структурированных полей
    3. check_subject() - проверка предмета оплаты (из extract()) на соответствие
льготной сельхоз-программе

Результаты каждого шага выводятся в виде таблицы в консоль.

Запуск (из корня проекта):
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
        doc_type, confidence = classify(text)
        classify_rows.append([filename, doc_type, confidence])
    print_table(["filename", "doc type", "confidence"], classify_rows)

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

    subject_tests = [
    # --- Должны соответствовать программе (matches=True) ---
    "Поставка минеральных удобрений (карбамид марки Б)",
    "Поставка семян подсолнечника посевная партия 2025",
    "Техническое обслуживание и ремонт зерноуборочного комбайна John Deere",
    "Поставка дизельного топлива для нужд сельхозпроизводства",
    "Выполнение агрохимических работ, внесение КАС-32",
    "Приобретение запасных частей для трактора МТЗ-82",
    "Поставка средств защиты растений (фунгицид Амистар)",
    "Страхование урожая от неблагоприятных погодных условий",
    # --- Должны не соответствовать программе (matches=False) ---
    "Аренда офисного помещения, г. Краснодар, ул. Ленина 15",
    "Юридическое сопровождение сделки, консультационные услуги",
    "Поставка офисной мебели и канцелярских товаров",
    "Разработка корпоративного сайта и SEO-продвижение",
    "Услуги клининговой компании, уборка административного здания",
    "Обучение механизаторов работе с новой техникой",
    # --- Спорные (неочевидный ответ) ---
    "Транспортные услуги по доставке удобрений до склада",
    "Аренда сельскохозяйственной техники на период уборки урожая",
    "Услуги агронома-консультанта по подбору схемы удобрений",
    ]
    for subj in subject_tests:
        eligible, confidence, explanation = check_subject(subj)
        print(f"{subj!r} -> \n{eligible}, {confidence}, {explanation}\n")

if __name__ == "__main__":
    main()
