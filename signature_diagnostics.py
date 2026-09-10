"""
signature_diagnostics.py
=========================
V16 (Section 4/5/6) -- audit TERPISAH utk signature_nasabah vs signature_atasan
sebelum mengubah threshold apa pun. TIDAK menduplikasi logika deteksi (reuse
postprocessing.process_signatures apa adanya) -- script ini HANYA menjalankan
pipeline penuh per record & mendata diagnostik mentah (ink_area_ratio,
spread_x, spread_y, component_count, change_ratio, status saat ini) ke CSV,
supaya threshold final dipilih dari DISTRIBUSI NYATA 25 dokumen, bukan angka
tebakan.

Jalankan:
    python signature_diagnostics.py --dataset assets/ocr_evaluation.xlsx
"""

import argparse
import csv
from pathlib import Path

import data_input
import pipeline
import preprocessing as prep

FIELDS = ("signature_nasabah", "signature_atasan")
COLUMNS = (
    "record", "field", "status", "ink_area_ratio", "ink_spread_x", "ink_spread_y",
    "component_count", "change_ratio", "roi_source", "anchor_similarity",
)
# V18 -- kumpul di eval_runs/ bareng output evaluation.py, bukan root project
# (lihat handover.md §12).
EVAL_RUNS_DIR = Path(__file__).resolve().parent / "eval_runs"


def main():
    parser = argparse.ArgumentParser(description="Audit diagnostik signature (Section 4).")
    parser.add_argument("--dataset", default="assets/ocr_evaluation.xlsx")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    EVAL_RUNS_DIR.mkdir(exist_ok=True)
    args.out = args.out or str(EVAL_RUNS_DIR / "signature_diagnostics.csv")

    records = data_input.read_excel_records(args.dataset)
    rows = []
    for record in records:
        no = record.get("Record")
        doc_path = data_input.get_document_path(record)
        if not doc_path:
            print(f"record {no}: dokumen tidak ditemukan, dilewati")
            continue
        try:
            result = pipeline.run_pipeline(doc_path, template_path=prep.TEMPLATE_PATH)
        except Exception as exc:
            print(f"record {no}: PIPELINE ERROR {type(exc).__name__}: {exc}")
            continue
        raw_results = result.get("raw_results", {})
        for field in FIELDS:
            r = raw_results.get(field, {})
            rows.append({
                "record": no, "field": field, "status": r.get("status"),
                "ink_area_ratio": r.get("ink_area_ratio"),
                "ink_spread_x": r.get("ink_spread_x"), "ink_spread_y": r.get("ink_spread_y"),
                "component_count": r.get("component_count"), "change_ratio": r.get("change_ratio"),
                "roi_source": r.get("roi_source"), "anchor_similarity": r.get("anchor_similarity"),
            })
        print(f"record {no}: done")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
