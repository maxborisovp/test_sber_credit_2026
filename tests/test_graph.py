import os

import pytest

from src.graph import (
    DocState,
    build_graph,
    process_batch,
    reconcile_batch,
    route_after_extract,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


@pytest.mark.unit
def test_route_after_extract_skips_when_no_subject():
    """The conditional edge added after extract(): documents with no
    extracted subject must be routed away from check_subject()."""
    assert route_after_extract(DocState(fields={"subject": None})) == "skip_subject_check"
    assert route_after_extract(DocState(fields={"subject": ""})) == "skip_subject_check"
    assert route_after_extract(DocState(fields={"subject": "Карбамид марки Б"})) == "check_subject"


@pytest.mark.unit
def test_process_document_skip_path_gives_honest_reason():
    graph = build_graph()
    result = graph.invoke(DocState(file_path="inline", text="Просто текст без реквизитов."))
    assert result["subject_match"] is False
    assert "не извлечён" in result["subject_reason"]


@pytest.mark.unit
def test_reconcile_batch_groups_same_deal_by_inn():
    """act_001/spec_001/invoice_001/contract_001/scan_ocr_001 all describe
    the same ООО ТехАгро deal (1 250 000 руб.) and share one INN - the
    post-control step (business case TO BE, п.6) must group them together
    and confirm the amounts agree.
    """
    reports = process_batch(DATA_DIR, OUTPUT_DIR)
    groups = reconcile_batch(reports)
    techagro = next(g for g in groups if g["inn"] == "7701234567")

    assert len(techagro["documents"]) == 5
    assert techagro["amount_consistent"] is True
    assert techagro["status"] == "ok"
    assert set(techagro["amounts_by_file"].values()) == {1_250_000.0}


@pytest.mark.unit
def test_reconcile_batch_single_document_group_has_nothing_to_compare():
    reports = process_batch(DATA_DIR, OUTPUT_DIR)
    groups = reconcile_batch(reports)
    single_doc_groups = [g for g in groups if len(g["documents"]) == 1]

    assert len(single_doc_groups) == 2  # act_002.txt and invoice_002.txt
    for g in single_doc_groups:
        assert g["amount_consistent"] is True
        assert "сверять не с чем" in g["note"]


@pytest.mark.unit
def test_reconcile_batch_detects_amount_mismatch():
    """Synthetic case: same INN, different amounts -> must be flagged."""
    fake_reports = [
        {"file": "a.txt", "fields": {"inn": "1112223334", "amount": 1000.0}},
        {"file": "b.txt", "fields": {"inn": "1112223334", "amount": 2000.0}},
    ]
    groups = reconcile_batch(fake_reports)
    assert len(groups) == 1
    assert groups[0]["amount_consistent"] is False
    assert groups[0]["status"] == "mismatch"
