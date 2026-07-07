import re
from typing import Dict, Tuple

KEYWORD_RULES = {
    "contract": [
        (r"договор[а-я]*\s+(поставки|подряда|оказания\s+услуг|на\s+выполнение)", 6),
        (r"именуем(ое|ый|ая)\s+в\s+дальнейшем", 5),
        (r"предмет\s+договора", 4),
        (r"заключили\s+настоящий\s+договор", 5),
        (r"действующ(его|ий)\s+на\s+основании\s+устава", 4),
        (r"\bсторон[аы]\b", 2),
        (r"\bдоговор\b", 1),
    ],
    "spec": [
        (r"спецификаци[яию]", 8),
        (r"к\s+договору\s+поставки", 3),
        (r"условия\s+поставки", 2),
        (r"срок\s+поставки", 1.5),
        (r"наименование\s+товара", 1),
    ],
    "invoice": [
        (r"сч[её]т\s+на\s+оплату", 8),
        (r"\bсч[её]т\s*№", 5),
        (r"банковские\s+реквизиты", 4),
        (r"\bр/с\b", 3),
        (r"\bбик\b", 3),
        (r"к[/\\]с", 2),
        (r"срок\s+оплаты", 2),
        (r"главный\s+бухгалтер", 2),
    ],
    "act": [
        (r"акт\s+выполненных\s+работ", 8),
        (r"универсальный\s+передаточный\s+документ", 8),
        (r"\bупд\b", 6),
        (r"работы\s+выполнены?", 3),
        (r"заказчик\s+принял", 3),
        (r"товар\s+(передан|получен)", 3),
        (r"претензий\s+по\s+качеству", 3),
        (r"\bакт\b", 1),
    ],
}

def classify_keywords(text: str) -> Tuple[str, float]:
        scores = {}
        text_lower = text.lower()
        
        for doc_type, rules in KEYWORD_RULES.items():
            score = 0.0
            for pattern, weight in rules:
                matches = re.findall(pattern, text_lower, flags=re.IGNORECASE)
                if matches:
                    score += weight * len(matches)
            scores[doc_type] = score
        
        total = sum(scores.values())
        if total == 0:
            return 'unknown', 0.0
        
        # print(scores)
        normalized_scores = {doc_type: score / total for doc_type, score in scores.items()}
        ranked = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_type, best_score = ranked[0]
        second_score = ranked[1][1]
        gap = best_score - second_score
        # if best_score < MIN_TOP1_CONFIDENCE or gap < gap_threshold:
        if best_score < 0.5 or gap < 0.2:
            return "unknown", round(best_score, 3)
        return best_type, round(best_score, 3)

if __name__ == "__main__":
    filenames = ["../data/act_001.txt", "../data/act_002.txt", "../data/contract_001.txt", "../data/invoice_001.txt", "../data/invoice_002.txt", "../data/spec_001.txt", "../data/scan_ocr_001.txt"]
    for filename in filenames: 
        with open(filename, 'r', encoding='utf-8') as file:
            doc_text = file.read()
        result = classify_keywords(doc_text)
        print(f"\nText - {filename},  result - {result}")
