"""LangGraph pipeline(s) for the document-processing agent.

Two separate graphs live here, both following patterns from the LangGraph
course referenced in RESULTS.md (habr.com/ru/companies/amvera, части 1-3):

1. **Per-document graph** (``build_graph`` / ``DocState``): classify ->
   extract -> [conditional] -> check_subject | skip_subject_check ->
   finalize. The conditional edge after ``extract`` is a genuine branching
   decision (part 1, "Условные рёбра"): if no ``subject`` field was
   extracted at all, running keyword-matching on an empty string is
   pointless noise, so that document is routed straight to a lightweight
   skip node instead, with an honest reason in the report.

2. **Reconciliation graph** (``build_reconciliation_graph`` /
   ``ReconcileState``): implements the "пост-контроль" step described in
   the original business case (test_credit.pdf, TO BE p.6) - "Система
   проводит пост-контроль: сопоставляет данные из авансового платежа с
   данными закрывающих документов по сумме, количеству и номенклатуре."
   It groups already-processed documents by counterparty INN (documents
   about the same underlying deal share an INN in this dataset) and,
   using the "начни с условного ребра" fake-node trick from part 1, only
   runs the actual cross-document amount comparison when a group has 2+
   documents to compare - a single document has nothing to reconcile
   against.

If the ``langgraph`` package is not installed, both ``build_*`` functions
return a plain-Python drop-in with the same ``.invoke(state)`` interface
and identical branching logic, so the rest of the module (and the tests)
behave identically either way - this mirrors the "must run locally
without extra dependencies" requirement from the task brief.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any, TypedDict

from src.classifier import classify
from src.extractor import extract
from src.subject_checker import check_subject

# Below this confidence, a classification/extraction/subject result is
# considered too uncertain to trust automatically.
MIN_CLASSIFY_CONFIDENCE = 0.6
MIN_SUBJECT_CONFIDENCE = 0.5
CRITICAL_FIELDS = ("amount", "date", "inn")


# ==========================================================================
# Graph 1: per-document processing
# ==========================================================================
class DocState(TypedDict, total=False):
    file_path: str
    text: str
    doc_type: str
    doc_type_confidence: float
    fields: dict[str, Any]
    subject_match: bool
    subject_confidence: float
    subject_reason: str
    status: str
    notes: list[str]


def node_classify(state: DocState) -> DocState:
    doc_type, confidence = classify(state["text"])
    state["doc_type"] = doc_type
    state["doc_type_confidence"] = confidence
    return state


def node_extract(state: DocState) -> DocState:
    state["fields"] = extract(state["text"])
    return state


def route_after_extract(state: DocState) -> str:
    """Conditional edge: only bother running check_subject() when there is
    an actual subject string to check. Without this branch, a document
    where extraction failed to find a subject would call
    ``check_subject("")``, which returns a generic "not specified" verdict
    that reads like a real check happened when it didn't.
    """
    return "check_subject" if (state.get("fields") or {}).get("subject") else "skip_subject_check"


def node_check_subject(state: DocState) -> DocState:
    subject = state["fields"]["subject"]
    matches, confidence, reason = check_subject(subject)
    state["subject_match"] = matches
    state["subject_confidence"] = confidence
    state["subject_reason"] = reason
    return state


def node_skip_subject_check(state: DocState) -> DocState:
    """Reached instead of node_check_subject when extract() found no
    subject at all - honestly reports that the check couldn't run, rather
    than reusing check_subject's generic empty-input message."""
    state["subject_match"] = False
    state["subject_confidence"] = 0.0
    state["subject_reason"] = "предмет оплаты не извлечён из документа - проверка невозможна"
    return state


def node_finalize(state: DocState) -> DocState:
    notes: list[str] = []

    if state["doc_type"] == "unknown":
        notes.append("тип документа не удалось определить с достаточной уверенностью")
    elif state["doc_type_confidence"] < MIN_CLASSIFY_CONFIDENCE:
        notes.append(
            f"низкая уверенность в типе документа ({state['doc_type_confidence']:.2f})"
        )

    missing = [f for f in CRITICAL_FIELDS if state["fields"].get(f) is None]
    if missing:
        notes.append(f"не удалось извлечь обязательные поля: {', '.join(missing)}")

    if not state["subject_match"]:
        notes.append(f"предмет оплаты не подходит под программу: {state['subject_reason']}")
    elif state["subject_confidence"] < MIN_SUBJECT_CONFIDENCE:
        notes.append("низкая уверенность в соответствии предмета оплаты программе")

    state["status"] = "requires_manual_review" if notes else "ok"
    state["notes"] = notes
    return state


class _FallbackDocGraph:
    """Sequential-execution stand-in used when ``langgraph`` isn't
    installed. Mirrors the same node order and the same conditional
    branch as the compiled StateGraph below.
    """

    def invoke(self, state: DocState) -> DocState:
        state = node_classify(state)
        state = node_extract(state)
        state = (
            node_check_subject(state)
            if route_after_extract(state) == "check_subject"
            else node_skip_subject_check(state)
        )
        state = node_finalize(state)
        return state


def build_graph() -> Any:
    """Build (and compile) the per-document LangGraph pipeline. Falls back
    to a plain sequential implementation if langgraph is not installed."""
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except ImportError:
        return _FallbackDocGraph()

    workflow = StateGraph(DocState)
    workflow.add_node("classify", node_classify)
    workflow.add_node("extract", node_extract)
    workflow.add_node("check_subject", node_check_subject)
    workflow.add_node("skip_subject_check", node_skip_subject_check)
    workflow.add_node("finalize", node_finalize)

    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "extract")
    workflow.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "check_subject": "check_subject",
            "skip_subject_check": "skip_subject_check",
        },
    )
    workflow.add_edge("check_subject", "finalize")
    workflow.add_edge("skip_subject_check", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def save_graph_visualization(app: Any, out_path: str = "output/graph.png") -> bool:
    """Best-effort PNG export of the compiled graph via LangGraph's
    built-in Mermaid renderer (draw_mermaid_png). This calls a hosted
    rendering service, so it silently no-ops (returns False) without
    network access or on any error - it is a documentation nicety, never
    required for the pipeline to run.
    """
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        png_bytes = app.get_graph().draw_mermaid_png()
        with open(out_path, "wb") as fh:
            fh.write(png_bytes)
        return True
    except Exception:  # noqa: BLE001 - visualization is optional
        return False


def process_document(file_path: str) -> dict[str, Any]:
    """Run the full pipeline on a single text file and return its report
    as a plain dict (JSON-serializable)."""
    with open(file_path, encoding="utf-8") as fh:
        text = fh.read()
    graph = build_graph()
    result = graph.invoke(DocState(file_path=file_path, text=text))
    return {
        "file": os.path.basename(file_path),
        "doc_type": result["doc_type"],
        "doc_type_confidence": result["doc_type_confidence"],
        "fields": result["fields"],
        "subject_check": {
            "matches": result["subject_match"],
            "confidence": result["subject_confidence"],
            "reason": result["subject_reason"],
        },
        "status": result["status"],
        "notes": result["notes"],
    }


# ==========================================================================
# Graph 2: cross-document reconciliation ("пост-контроль")
# ==========================================================================
class ReconcileState(TypedDict, total=False):
    inn: str
    documents: list[dict[str, Any]]
    amount_consistent: bool
    amounts_by_file: dict[str, float]
    note: str
    status: str


def node_entry(state: ReconcileState) -> ReconcileState:
    """Fake/pass-through entry node. LangGraph cannot start a graph on a
    conditional edge directly (see part 1, "Если хочется начать с
    условного ребра?") - START must point at a real node first, so this
    node exists purely to give the conditional router something to hang
    off of; it does not touch the state.
    """
    return state


def route_by_group_size(state: ReconcileState) -> str:
    return "reconcile" if len(state["documents"]) >= 2 else "single_document"


def node_reconcile(state: ReconcileState) -> ReconcileState:
    """Compares the extracted `amount` across every document that shares
    this INN. In this dataset that means checking that the contract, the
    spec, the invoice and the closing act for the same deal all agree on
    the price - exactly the "сопоставляет данные ... по сумме" check from
    the business case.
    """
    amounts_by_file = {
        d["file"]: d["fields"].get("amount")
        for d in state["documents"]
        if d["fields"].get("amount") is not None
    }
    unique_amounts = set(amounts_by_file.values())
    consistent = len(unique_amounts) <= 1

    if consistent:
        note = f"суммы совпадают по всем {len(amounts_by_file)} документам с суммой"
    else:
        note = f"расхождение сумм между документами: {sorted(unique_amounts)}"

    state["amount_consistent"] = consistent
    state["amounts_by_file"] = amounts_by_file
    state["note"] = note
    return state


def node_single_document(state: ReconcileState) -> ReconcileState:
    state["amount_consistent"] = True
    state["amounts_by_file"] = {
        d["file"]: d["fields"].get("amount") for d in state["documents"]
    }
    state["note"] = "по этому контрагенту только один документ - сверять не с чем"
    return state


def node_finalize_reconciliation(state: ReconcileState) -> ReconcileState:
    state["status"] = "ok" if state.get("amount_consistent", True) else "mismatch"
    return state


class _FallbackReconcileGraph:
    def invoke(self, state: ReconcileState) -> ReconcileState:
        state = node_entry(state)
        state = (
            node_reconcile(state)
            if route_by_group_size(state) == "reconcile"
            else node_single_document(state)
        )
        state = node_finalize_reconciliation(state)
        return state


def build_reconciliation_graph() -> Any:
    try:
        from langgraph.graph import StateGraph, START, END  # type: ignore
    except ImportError:
        return _FallbackReconcileGraph()

    workflow = StateGraph(ReconcileState)
    workflow.add_node("entry", node_entry)
    workflow.add_node("reconcile", node_reconcile)
    workflow.add_node("single_document", node_single_document)
    workflow.add_node("finalize", node_finalize_reconciliation)

    workflow.add_edge(START, "entry")
    workflow.add_conditional_edges(
        "entry",
        route_by_group_size,
        {"reconcile": "reconcile", "single_document": "single_document"},
    )
    workflow.add_edge("reconcile", "finalize")
    workflow.add_edge("single_document", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def reconcile_batch(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group processed document reports by counterparty INN and run the
    reconciliation graph on each group. Groups with no INN at all
    (extraction failed completely) are skipped - there is nothing to key
    them on.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        inn = report["fields"].get("inn")
        if inn:
            groups[inn].append(report)

    graph = build_reconciliation_graph()
    results = []
    for inn, docs in groups.items():
        result = graph.invoke(ReconcileState(inn=inn, documents=docs))
        results.append(
            {
                "inn": inn,
                "documents": [d["file"] for d in docs],
                "amounts_by_file": result["amounts_by_file"],
                "amount_consistent": result["amount_consistent"],
                "note": result["note"],
                "status": result["status"],
            }
        )
    return results


# ==========================================================================
# Batch driver
# ==========================================================================
def process_batch(input_dir: str, output_dir: str) -> list[dict[str, Any]]:
    """Run the per-document pipeline over every *.txt file in *input_dir*
    (skipping the subjects_test.txt fixture), write one JSON report per
    document into *output_dir*, run cross-document reconciliation over the
    whole batch, write it to reconciliation.json, and return the list of
    per-document reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    reports = []
    for file_path in sorted(glob.glob(os.path.join(input_dir, "*.txt"))):
        if os.path.basename(file_path) == "subjects_test.txt":
            continue
        report = process_document(file_path)
        out_name = os.path.splitext(report["file"])[0] + ".json"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        reports.append(report)

    reconciliation = reconcile_batch(reports)
    with open(os.path.join(output_dir, "reconciliation.json"), "w", encoding="utf-8") as fh:
        json.dump(reconciliation, fh, ensure_ascii=False, indent=2)

    if save_graph_visualization(build_graph(), os.path.join(output_dir, "graph.png")):
        print(f"Граф сохранён: {os.path.join(output_dir, 'graph.png')}")

    return reports


if __name__ == "__main__":
    all_reports = process_batch("data", "output")
    for r in all_reports:
        print(f"{r['file']:20s} {r['doc_type']:10s} status={r['status']}")

    print("\n=== Пост-контроль (сверка по контрагентам) ===")
    for group in reconcile_batch(all_reports):
        print(f"ИНН {group['inn']}: {group['status']:8s} - {group['note']}")
