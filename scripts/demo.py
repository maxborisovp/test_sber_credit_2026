"""Demo / report generator.

Runs extract(), classify() and check_subject() over the sample dataset in
data/ and prints the markdown tables used in RESULTS.md, plus (if
matplotlib is available) saves a confidence chart to output/.

Usage:
    python -m scripts.demo
"""
from __future__ import annotations

import glob
import os

from src.classifier import classify
from src.extractor import extract
from src.subject_checker import check_subject

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def extraction_table() -> str:
    rows = ["| Файл | amount | date | inn | contractor | subject |",
            "|---|---|---|---|---|---|"]
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        name = os.path.basename(path)
        if name == "subjects_test.txt":
            continue
        text = open(path, encoding="utf-8").read()
        r = extract(text)
        cells = [name] + ["OK" if r[f] is not None else "—" for f in
                           ("amount", "date", "inn", "contractor", "subject")]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def classification_table() -> list[tuple[str, str, float]]:
    results = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        name = os.path.basename(path)
        if name == "subjects_test.txt":
            continue
        text = open(path, encoding="utf-8").read()
        doc_type, confidence = classify(text)
        results.append((name, doc_type, confidence))
    return results


def subject_table() -> list[tuple[str, str, bool, float, str]]:
    path = os.path.join(DATA_DIR, "subjects_test.txt")
    results = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, subject = (p.strip() for p in line.split("|", 1))
        matches, confidence, reason = check_subject(subject)
        results.append((expected, subject, matches, confidence, reason))
    return results


def main() -> None:
    print("=== extract() field coverage ===")
    print(extraction_table())

    print("\n=== classify() results ===")
    cls_results = classification_table()
    for name, doc_type, confidence in cls_results:
        print(f"{name:20s} {doc_type:10s} {confidence:.2f}")

    print("\n=== check_subject() results ===")
    subj_results = subject_table()
    correct = total = 0
    for expected, subject, matches, confidence, reason in subj_results:
        got = "PASS" if matches else "FAIL"
        if expected in ("PASS", "FAIL"):
            total += 1
            correct += got == expected
        print(f"{expected:5s} got={got:4s} conf={confidence:.2f}  {subject}")
    print(f"\n{correct}/{total} PASS/FAIL cases matched expected label")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n, _, _ in cls_results]
        confidences = [c for _, _, c in cls_results]
        colors = ["#2e7d32" if c >= 0.6 else "#c62828" for c in confidences]
        plt.figure(figsize=(8, 4))
        plt.barh(names, confidences, color=colors)
        plt.xlabel("confidence")
        plt.title("classify() confidence by document")
        plt.xlim(0, 1)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "classification_confidence.png")
        plt.savefig(out_path, dpi=120)
        print(f"\nSaved chart: {out_path}")
    except ImportError:
        print("\n(matplotlib not installed - skipping chart)")


if __name__ == "__main__":
    main()
