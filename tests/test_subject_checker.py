import os

import pytest

from src.subject_checker import check_subject

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_subjects_test():
    path = os.path.join(DATA_DIR, "subjects_test.txt")
    cases = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, subject = (p.strip() for p in line.split("|", 1))
        cases.append((expected, subject))
    return cases


_ALL_CASES = _load_subjects_test()
_PASS_FAIL_CASES = [c for c in _ALL_CASES if c[0] in ("PASS", "FAIL")]
_EDGE_CASES = [c for c in _ALL_CASES if c[0] == "EDGE"]


# --- Required assertions from the task brief ------------------------------
@pytest.mark.unit
def test_check_subject_positive_example():
    matches, confidence, reason = check_subject("Поставка удобрений (карбамид)")
    assert matches is True
    assert confidence > 0.5
    assert reason


@pytest.mark.unit
def test_check_subject_negative_example():
    matches, confidence, reason = check_subject("Аренда офисного помещения")
    assert matches is False
    assert confidence > 0.5
    assert reason


# --- Full dataset.txt coverage (14 unambiguous PASS/FAIL cases) -----------
@pytest.mark.unit
@pytest.mark.parametrize("expected, subject", _PASS_FAIL_CASES)
def test_check_subject_dataset(expected, subject):
    matches, confidence, reason = check_subject(subject)
    want = expected == "PASS"
    assert matches is want, f"{subject!r}: expected {expected}, got matches={matches} ({reason})"
    assert 0.0 <= confidence <= 1.0
    assert reason


# --- EDGE cases: no single correct answer, but must not crash and must
# return a real explanation -------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("expected, subject", _EDGE_CASES)
def test_check_subject_edge_cases_return_reasoned_result(expected, subject):
    matches, confidence, reason = check_subject(subject)
    assert isinstance(matches, bool)
    assert 0.0 <= confidence <= 1.0
    assert reason


@pytest.mark.unit
def test_check_subject_empty_input():
    matches, confidence, reason = check_subject("")
    assert matches is False
    assert reason


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="requires GOOGLE_API_KEY"
)
def test_check_subject_via_llm_smoke():
    matches, confidence, reason = check_subject("Поставка семян пшеницы")
    assert matches is True
