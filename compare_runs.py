"""
compare_runs.py
================
Gabungkan beberapa hasil evaluation.py / stage_evaluation.py (masing2 sudah
dijalankan dgn --label sendiri, opsional --pipeline / --vlm-fallback utk
model/versi berbeda) jadi SATU laporan perbandingan sisi-berdampingan --
supaya "model/versi mana yang lebih baik" bisa dilihat langsung tanpa buka
file JSON satu per satu atau menghitung delta manual.

TIDAK menjalankan pipeline/OCR apa pun sendiri -- HANYA membaca file
evaluation_summary_<label>.json / stage_evaluation_summary_<label>.json yang
SUDAH dihasilkan evaluator terkait. Prinsip yang sama dgn evaluation.py:
jangan menduplikasi orkestrasi, cuma menyusun ulang hasil yang sudah ada.

Contoh alur "bandingkan model" lengkap:
    # PaddleOCR sendirian vs PaddleOCR + fallback VLM (Qwen2-VL), pipeline sama
    python evaluation.py --dataset assets/ocr_evaluation.xlsx --label paddle_only --vlm-fallback off
    python evaluation.py --dataset assets/ocr_evaluation.xlsx --label paddle_vlm  --vlm-fallback on
    python compare_runs.py evaluation_summary_paddle_only.json evaluation_summary_paddle_vlm.json

    # Versi orkestrasi pipeline V12 (current) vs V10/11 (pipeline1.py)
    python stage_evaluation.py --dataset assets/ocr_evaluation.xlsx --label v12 --pipeline pipeline
    python stage_evaluation.py --dataset assets/ocr_evaluation.xlsx --label v11 --pipeline pipeline1
    python compare_runs.py stage_evaluation_summary_v12.json stage_evaluation_summary_v11.json

Bisa campur evaluation_summary_*.json dan stage_evaluation_summary_*.json
sekaligus dalam satu panggilan -- masing2 "jenis" dilaporkan di tabel
terpisah (union metrik dari semua file jenis itu). Kolom pertama (argumen
pertama per jenis) jadi BASELINE -- kolom lain menampilkan delta terhadap
baseline utk metrik numerik.

Output: tabel ke console (per jenis) + compare_runs_<timestamp>.csv (satu
sheet gabungan, kolom "kind" memisahkan jenis) + .xlsx (satu sheet per
jenis) kalau pandas/openpyxl tersedia (sudah dependency proyek).
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# V18 -- kumpul di eval_runs/ bareng output evaluation.py/stage_evaluation.py
# (lihat handover.md §12).
EVAL_RUNS_DIR = BASE_DIR / "eval_runs"

# Metadata key yang TIDAK dianggap "metrik terukur" (label run itu sendiri),
# ditampilkan sbg header, bukan baris tabel metrik.
RUN_META_KEYS = ("run_label", "pipeline_module", "vlm_fallback_enabled")


def _kind_of(summary):
    if "funnel" in summary:
        return "stage_evaluation"
    if "decision_accuracy" in summary:
        return "evaluation"
    return "unknown"


def _flatten(d, prefix=""):
    """Ratakan dict bersarang jadi {dot.path: leaf_value}. List/tuple
    diubah jadi string ringkas (bukan diratakan) supaya tabel tetap 1
    baris/metrik."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        elif isinstance(v, (list, tuple)):
            out[key] = ", ".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def _label_for(path, summary):
    """Nama kolom run: pakai run_label/pipeline_module kalau tersedia
    (evaluator versi baru, lihat evaluation.py/stage_evaluation.py
    --label/--pipeline), fallback ke nama file (evaluator versi lama yg
    belum sempat menuliskan metadata run)."""
    label = summary.get("run_label")
    pipeline_mod = summary.get("pipeline_module")
    if label and pipeline_mod and pipeline_mod != "pipeline":
        return f"{label} ({pipeline_mod})"
    if label:
        return label
    return Path(path).stem


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def _fmt_delta(base, other):
    if not (_is_number(base) and _is_number(other)):
        return ""
    d = other - base
    if abs(d) < 1e-9:
        return "="
    return f"{d:+.4f}"


def build_table(paths_and_summaries):
    """paths_and_summaries: list[(path, summary_dict)] SATU jenis (kind sama).
    Return (columns: list[str], rows: list[(metric_key, {col: value})])."""
    flat_list = [(_label_for(p, s), _flatten(s)) for p, s in paths_and_summaries]
    columns = [label for label, _ in flat_list]

    all_keys = []
    seen = set()
    for _, flat in flat_list:
        for k in flat:
            if k in RUN_META_KEYS or k in seen:
                continue
            seen.add(k)
            all_keys.append(k)

    rows = []
    for key in all_keys:
        row = {label: flat.get(key) for label, flat in flat_list}
        rows.append((key, row))
    return columns, rows


def print_table(kind, columns, rows, baseline_col):
    print(f"\n=== {kind} -- baseline: {baseline_col} ===")
    col_w = {c: max(len(c), 10) for c in columns}
    metric_w = max((len(k) for k, _ in rows), default=10)
    metric_w = max(metric_w, len("metric"))

    header = "metric".ljust(metric_w) + "  " + "  ".join(c.ljust(col_w[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for key, row in rows:
        cells = []
        base_val = row.get(baseline_col)
        for c in columns:
            val = row.get(c)
            text = _fmt(val)
            if c != baseline_col and _is_number(val) and _is_number(base_val):
                text = f"{text} ({_fmt_delta(base_val, val)})"
            cells.append(text.ljust(col_w[c]))
        print(key.ljust(metric_w) + "  " + "  ".join(cells))


def write_outputs(all_tables, output_prefix):
    """all_tables: list[(kind, columns, rows, baseline_col)]. Tulis CSV
    gabungan selalu; .xlsx (1 sheet per kind) kalau pandas/openpyxl ada."""
    import csv as csv_mod

    csv_path = f"{output_prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["kind", "metric"] + ["run_columns_see_xlsx_or_rerun"])
        for kind, columns, rows, _baseline in all_tables:
            writer.writerow([kind, "__columns__"] + columns)
            for key, row in rows:
                writer.writerow([kind, key] + [row.get(c) for c in columns])
    print(f"\nSaved: {csv_path}")

    try:
        import pandas as pd
    except ImportError:
        return
    xlsx_path = f"{output_prefix}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for kind, columns, rows, _baseline in all_tables:
            df = pd.DataFrame(
                [{"metric": key, **row} for key, row in rows],
                columns=["metric"] + columns,
            )
            sheet = kind[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Saved: {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gabungkan beberapa evaluation_summary_*.json / "
                     "stage_evaluation_summary_*.json jadi satu laporan "
                     "perbandingan model/versi sisi-berdampingan."
    )
    parser.add_argument("summaries", nargs="+",
                         help="Path ke file summary JSON (>=2 supaya perbandingan berguna, "
                              "tapi 1 file tetap jalan -- laporan tunggal)")
    parser.add_argument("--output-prefix", default=None,
                         help="Prefix file output (default: compare_runs_<timestamp>)")
    args = parser.parse_args()

    by_kind = {}
    for path in args.summaries:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: file tidak ditemukan, dilewati: {path}", file=sys.stderr)
            continue
        summary = json.loads(p.read_text(encoding="utf-8"))
        kind = _kind_of(summary)
        by_kind.setdefault(kind, []).append((str(p), summary))

    if not by_kind:
        print("Tidak ada summary valid yang bisa dibandingkan.", file=sys.stderr)
        sys.exit(1)

    EVAL_RUNS_DIR.mkdir(exist_ok=True)
    output_prefix = args.output_prefix or str(EVAL_RUNS_DIR / f"compare_runs_{int(time.time())}")

    all_tables = []
    for kind, items in by_kind.items():
        columns, rows = build_table(items)
        baseline_col = columns[0]
        print_table(kind, columns, rows, baseline_col)
        all_tables.append((kind, columns, rows, baseline_col))

    write_outputs(all_tables, output_prefix)


if __name__ == "__main__":
    main()
