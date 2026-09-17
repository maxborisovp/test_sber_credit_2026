from src.graph import process_batch, reconcile_batch
from scripts.demo import main
if __name__ == "__main__":
    all_reports = process_batch("data", "output")
    for r in all_reports:
        print(f"{r['file']:20s} {r['doc_type']:10s} status={r['status']}")

    print("\n=== Пост-контроль (сверка по контрагентам) ===")
    for group in reconcile_batch(all_reports):
        print(f"ИНН {group['inn']}: {group['status']:8s} - {group['note']}")

    main()