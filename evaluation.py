"""
evaluation.py
=============
Evaluator (DIPERLUAS utk V9.2 -- BUKAN evaluator kedua; masih file & CLI yang
sama sejak V9.1). Dijalankan TERPISAH dari aplikasi utama (bukan endpoint
FastAPI):

    python evaluation.py --dataset ocr_evaluation.xlsx --label v9_2_baseline

Gunakan --label berbeda tiap run (mis. v9_1, v9_2_baseline, v9_2_optimized)
utk membandingkan versi tanpa file evaluator kedua -- lihat build_summary()
utk metrik V9.2 (7-check decision, tenor/reward internal-consistency,
signature, Decision Accuracy/FAR/FRR/Review Rate).

Prinsip:
  - `ocr_evaluation.xlsx` adalah dataset ground truth 25 record, TIDAK
    pernah ditulis/diubah oleh script ini.
  - TIDAK menduplikasi logika preprocessing/OCR: setiap tahap memanggil
    fungsi produksi yang sudah ada (data_input.*, preprocessing.*,
    pipeline.run_pipeline). Evaluator ini hanya mengorkestrasi & mengukur.
  - Tahap dipanggil satu-per-satu (load -> align -> semantic region ->
    full pipeline) SEBELUM run_pipeline() dipanggil, supaya kalau gagal,
    failure_stage bisa ditentukan PERSIS: INPUT | ALIGNMENT | REGION |
    PIPELINE | COMPARISON. Ini menyebabkan align_to_template dipanggil dua
    kali per dokumen (sekali langsung, sekali lagi di dalam run_pipeline) --
    trade-off yang disengaja demi diagnostik tahap yang presisi, bukan bug.
  - Perbandingan dgn ground truth di sini HANYA untuk kebutuhan evaluasi
    (metrik akurasi/baseline). Keputusan bisnis OK/TOLAK/REVIEW yang
    sesungguhnya tetap wewenang comparison.py (tidak diduplikasi di sini).
  - Field yang TIDAK punya ground truth di dataset (signature, evidence
    internal seperti reward_non_tunai/reward_tunai) TIDAK dibandingkan --
    hasil ekstraksinya tetap dicatat, tapi kolom comparison diisi "N/A".

Output:
  - evaluation_results_<label>.csv  (satu baris per record)
  - evaluation_summary_<label>.json (metrik agregat)
  - (opsional) folder debug image untuk record yang gagal/bermasalah saja
"""

import argparse
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path

import cv2

import data_input
import pipeline
import preprocessing as prep
import comparison

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DEBUG_DIR = BASE_DIR / "evaluation_debug"

# 5 field keputusan (sesuai handover.md) -> kolom ground truth di spreadsheet
# evaluasi (SPREADSHEET_COLUMNS milik data_input.py).
FIELD_GT_MAP = {
    "nama_nasabah": "Nama Nasabah",
    "nomor_rekening": "Nomor Rekening",
    "nominal_penempatan": "Nominal Penempatan",
    "tenor_penempatan": "Tenor Hold",
    "bentuk_reward": "Pilihan Hadiah",
}

FAILURE_STAGES = (
    "INPUT", "ALIGNMENT", "LOCALIZATION", "OCR", "VISUAL_CHOICE",
    "NORMALIZATION", "INTERNAL_VALIDATION", "SIGNATURE", "COMPARISON", "DECISION",
)
# V9.2 catatan: pipeline.run_pipeline() sbg satu kesatuan mencakup tahap
# LOCALIZATION/OCR/VISUAL_CHOICE/NORMALIZATION/SIGNATURE internal (tidak
# dipecah lagi jadi pemanggilan terpisah spt REGION di V9.1, supaya TIDAK
# menduplikasi orkestrasi pipeline.py). Kegagalan pada tahap2 itu dilaporkan
# sbg "PIPELINE" oleh run_pipeline (exception generik) -- evaluator memetakan
# ke label taxonomy yg paling mendekati lewat pesan exception (best-effort,
# lihat _map_pipeline_failure_stage), bukan menebak tanpa dasar.
GT_DECISION_COLUMN = "Status Verifikasi Form Pendaftaran Nasabah"


def _map_pipeline_failure_stage(exc_text):
    text = (exc_text or "").lower()
    if "ocr" in text:
        return "OCR"
    if "choice" in text or "pilihan" in text:
        return "VISUAL_CHOICE"
    if "signature" in text or "tanda tangan" in text:
        return "SIGNATURE"
    if "normal" in text:
        return "NORMALIZATION"
    if "anchor" in text or "roi" in text or "localiz" in text:
        return "LOCALIZATION"
    return "OCR"  # default best-guess: OCR/model call is the most common single point of failure


def _map_ground_truth_decision(raw_status):
    text = str(raw_status or "").strip().lower()
    if not text:
        return None
    if text.startswith("ok"):
        return "OK"
    if text.startswith("tolak"):
        return "TOLAK"
    return None


# ============================================================================
# Normalisasi & perbandingan KHUSUS EVALUASI (bukan business decision --
# comparison.py yang punya wewenang itu di production; source-nya tidak
# disertakan dalam scope task ini, jadi tidak diduplikasi/ditebak di sini).
# ============================================================================

def _digits_only(v):
    return re.sub(r"[^0-9]", "", str(v if v is not None else ""))


def _norm_text(v):
    return re.sub(r"\s+", " ", str(v if v is not None else "")).strip().upper()


def _norm_numeric(v):
    d = _digits_only(v)
    return int(d) if d else None


def _norm_tenor(v):
    m = re.search(r"\d+", str(v if v is not None else ""))
    return int(m.group()) if m else None


_CASH_KEYWORDS = ("cash", "tunai")


def _norm_reward_bucket(v):
    """Ground truth 'Pilihan Hadiah' berupa nama hadiah (mis. 'Cashback'
    vs nama barang), sedangkan hasil ekstraksi form berupa pilihan
    tunai/non_tunai (coret salah satu). Dipetakan ke bucket yang sama
    supaya bisa dibandingkan -- heuristik best-effort utk evaluasi, BUKAN
    aturan bisnis final (itu tetap milik comparison.py)."""
    text = str(v if v is not None else "").strip().lower()
    if not text:
        return None
    return "tunai" if any(k in text for k in _CASH_KEYWORDS) else "non_tunai"


def compare_field(field_name, extracted_value, ground_truth_value):
    if field_name == "nama_nasabah":
        a, b = _norm_text(extracted_value), _norm_text(ground_truth_value)
    elif field_name == "nomor_rekening":
        a, b = _digits_only(extracted_value), _digits_only(ground_truth_value)
    elif field_name == "nominal_penempatan":
        a, b = _norm_numeric(extracted_value), _norm_numeric(ground_truth_value)
    elif field_name == "tenor_penempatan":
        a, b = _norm_tenor(extracted_value), _norm_tenor(ground_truth_value)
    elif field_name == "bentuk_reward":
        a, b = _norm_reward_bucket(extracted_value), _norm_reward_bucket(ground_truth_value)
    else:
        a, b = extracted_value, ground_truth_value

    if a in (None, "") or b in (None, ""):
        return "N/A"
    return "MATCH" if a == b else "MISMATCH"


def _extract_value(fields_table, raw_results, field_name):
    """Ambil nilai hasil ekstraksi utk satu field dari struktur return
    pipeline.run_pipeline(). Defensif thd beberapa bentuk umum skema
    post.build_fields_table (dict-by-name ATAU list-of-dict) karena
    postprocessing.py tidak termasuk file yang diberikan utk task ini --
    evaluator tidak boleh berasumsi kaku pada skema yang tidak terlihat."""
    candidates = []
    if isinstance(fields_table, dict) and field_name in fields_table:
        candidates.append(fields_table[field_name])
    elif isinstance(fields_table, list):
        for item in fields_table:
            if isinstance(item, dict) and (item.get("field") == field_name or item.get("name") == field_name):
                candidates.append(item)
    if not candidates and isinstance(raw_results, dict) and field_name in raw_results:
        candidates.append(raw_results[field_name])

    for c in candidates:
        if isinstance(c, dict):
            for key in ("normalized_value", "value", "text", "label", "final_value"):
                if c.get(key) not in (None, ""):
                    return c[key], c
            return None, c
        return c, {"raw": c}
    return None, None


# ============================================================================
# Evaluasi per-record, dipecah per tahap
# ============================================================================

def _blank_result(record):
    result = {
        "record": record.get("Record"),
        "regional_office": record.get("Regional Office"),
        "branch_office": record.get("Nama Branch Office"),
        "document_url": record.get("Surat Pernyataan"),
        "load_success": False,
        "alignment_status": "N/A",
        "alignment_method": "N/A",
        "alignment_rotation_deg": "N/A",
        "alignment_inlier_ratio": "N/A",
        "region_status": "N/A",
        "pipeline_success": False,
        "processing_time_sec": "N/A",
        "all_5_fields_match": "N/A",
        "all_7_checks_match": "N/A",
        # -- V9.2: evidence tenor (2 bukti wajib) --
        "tenor_choice": "N/A", "tenor_date_range_derived": "N/A", "tenor_internal_consistent": "N/A",
        # -- V9.2: evidence reward (2 bukti wajib) --
        "reward_choice": "N/A", "reward_non_tunai_filled": "N/A", "reward_tunai_filled": "N/A",
        "reward_internal_consistent": "N/A",
        # -- V9.2: signature (presence only) --
        "signature_nasabah_status": "N/A", "signature_atasan_status": "N/A",
        # -- V9.2: 7-check deterministic decision --
        "decision": "N/A", "decision_reasons": "N/A",
        "ground_truth_decision": _map_ground_truth_decision(record.get(GT_DECISION_COLUMN)) or "N/A",
        "decision_correct": "N/A",
        "failure_stage": None,
        "failure_reason": None,
    }
    for fname, gt_col in FIELD_GT_MAP.items():
        result[f"{fname}_extracted"] = "N/A"
        result[f"{fname}_ground_truth"] = record.get(gt_col)
        result[f"{fname}_comparison"] = "N/A"
    return result


def _save_debug(debug_dir, record, aligned_img, regions):
    if debug_dir is None:
        return
    rec_id = record.get("Record", "unknown")
    out_dir = Path(debug_dir) / f"record_{rec_id}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if aligned_img is not None:
            cv2.imwrite(str(out_dir / "aligned.png"), aligned_img)
        for name, r in (regions or {}).items():
            cv2.imwrite(str(out_dir / f"{name}.png"), r["crop"])
    except Exception:
        pass  # debug image tidak boleh menggagalkan evaluasi


def evaluate_record(record, template_img, template_path, debug_dir, debug_failed_only=True):
    result = _blank_result(record)
    t0 = time.time()

    # -- Tahap INPUT --------------------------------------------------------
    try:
        doc_path = data_input.get_document_path(record)
        if not doc_path:
            result["failure_stage"] = "INPUT"
            result["failure_reason"] = "document_not_found_or_download_failed"
            return result
        doc_img = prep.load_document(doc_path)
        result["load_success"] = True
    except Exception as exc:
        result["failure_stage"] = "INPUT"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result

    # -- Tahap ALIGNMENT ------------------------------------------------------
    try:
        aligned_img, _H, align_meta = prep.prepare_and_align(doc_path, doc_img, template_img)
        result["alignment_status"] = align_meta.get("status")
        result["alignment_method"] = align_meta.get("method", "N/A")
        result["alignment_rotation_deg"] = align_meta.get("rotation_deg", "N/A")
        result["alignment_inlier_ratio"] = align_meta.get("inlier_ratio", "N/A")
        if align_meta.get("status") == "failed":
            result["failure_stage"] = "ALIGNMENT"
            result["failure_reason"] = align_meta.get("reason", "alignment_failed")
            return result
    except Exception as exc:
        result["failure_stage"] = "ALIGNMENT"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result

    # -- Tahap LOCALIZATION (semantic region, fondasi anchor+spasial V9.2) --
    regions = None
    try:
        regions = prep.extract_semantic_regions(aligned_img, aligned_img.shape)
        result["region_status"] = "ok" if regions else "empty"
    except Exception as exc:
        result["region_status"] = "failed"
        result["failure_stage"] = "LOCALIZATION"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _save_debug(debug_dir, record, aligned_img, regions)
        return result

    # -- Tahap PIPELINE penuh (LOCALIZATION+OCR+VISUAL_CHOICE+NORMALIZATION+
    # SIGNATURE terjadi di dalam run_pipeline sbg satu orkestrasi -- TIDAK
    # dipecah ulang di sini spy tidak menduplikasi pipeline.py). Kegagalan
    # dipetakan best-effort ke salah satu tahap resmi lewat pesan exception.
    try:
        pipeline_result = pipeline.run_pipeline(doc_path, template_path=template_path)
        result["pipeline_success"] = True
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        result["failure_stage"] = _map_pipeline_failure_stage(reason)
        result["failure_reason"] = reason
        result["processing_time_sec"] = round(time.time() - t0, 2)
        _save_debug(debug_dir, record, aligned_img, regions)
        return result

    result["processing_time_sec"] = round(time.time() - t0, 2)

    # -- Tahap COMPARISON (ground truth, evaluasi saja) ----------------------
    try:
        fields_table = pipeline_result.get("fields")
        raw_results = pipeline_result.get("raw_results")
        mismatch_or_na = False
        for fname, gt_col in FIELD_GT_MAP.items():
            value, _raw = _extract_value(fields_table, raw_results, fname)
            result[f"{fname}_extracted"] = value if value not in (None, "") else "N/A"
            cmp = compare_field(fname, value, record.get(gt_col))
            result[f"{fname}_comparison"] = cmp
            if cmp != "MATCH":
                mismatch_or_na = True
        result["all_5_fields_match"] = not mismatch_or_na
    except Exception as exc:
        result["failure_stage"] = "COMPARISON"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _save_debug(debug_dir, record, aligned_img, regions)
        return result

    # -- Tahap DECISION (7 validasi deterministik, comparison.py V9.2) ------
    problem = mismatch_or_na
    try:
        choice_groups = pipeline_result.get("choice_groups", {})
        decision = comparison.compute_decision_v9_2(fields_table, raw_results, choice_groups, record)
        result["decision"] = decision["decision"]
        result["decision_reasons"] = "; ".join(decision["reasons"]) if decision["reasons"] else ""
        checks = decision["checks"]
        result["all_7_checks_match"] = all(v == "OK" for v in checks.values())

        tenor_ev = decision["evidence"]["tenor_penempatan"]
        result["tenor_choice"] = tenor_ev.get("choice") if tenor_ev.get("choice") is not None else "N/A"
        result["tenor_date_range_derived"] = (
            tenor_ev.get("date_range_derived") if tenor_ev.get("date_range_derived") is not None else "N/A"
        )
        result["tenor_internal_consistent"] = checks.get("tenor_penempatan") != "TOLAK"

        reward_ev = decision["evidence"]["bentuk_reward"]
        result["reward_choice"] = reward_ev.get("choice") or "N/A"
        result["reward_non_tunai_filled"] = reward_ev.get("non_tunai_filled")
        result["reward_tunai_filled"] = reward_ev.get("tunai_filled")
        result["reward_internal_consistent"] = checks.get("bentuk_reward") != "TOLAK"

        result["signature_nasabah_status"] = (raw_results.get("signature_nasabah") or {}).get("status", "N/A")
        result["signature_atasan_status"] = (raw_results.get("signature_atasan") or {}).get("status", "N/A")

        gt_decision = result["ground_truth_decision"]
        if gt_decision != "N/A":
            result["decision_correct"] = (result["decision"] == gt_decision)
        problem = problem or (result["decision"] != "OK") or (gt_decision != "N/A" and result["decision_correct"] is False)
    except Exception as exc:
        result["failure_stage"] = "DECISION"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _save_debug(debug_dir, record, aligned_img, regions)
        return result

    if debug_dir is not None and (not debug_failed_only or problem):
        _save_debug(debug_dir, record, aligned_img, regions)

    return result


# ============================================================================
# Agregasi & output
# ============================================================================

def write_csv(results, path):
    fieldnames = list(_field_order(results))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "N/A") for k in fieldnames})


def _field_order(results):
    seen = []
    for row in results:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    return seen


def build_summary(results):
    total = len(results)

    def _count(pred):
        return sum(1 for r in results if pred(r))

    loaded = _count(lambda r: r.get("load_success"))
    aligned_ok = _count(lambda r: r.get("alignment_status") in ("ok", "warning"))
    aligned_attempted = _count(lambda r: r.get("alignment_status") not in (None, "N/A"))
    homography_escalations = _count(lambda r: r.get("alignment_method") == "homography")
    alignment_failed = _count(lambda r: r.get("failure_stage") == "ALIGNMENT")
    region_ok = _count(lambda r: r.get("region_status") == "ok")
    pipeline_ok = _count(lambda r: r.get("pipeline_success"))

    per_field = {}
    for fname in FIELD_GT_MAP:
        col = f"{fname}_comparison"
        measurable = _count(lambda r, c=col: r.get(c) in ("MATCH", "MISMATCH"))
        matched = _count(lambda r, c=col: r.get(c) == "MATCH")
        per_field[fname] = {
            "measurable": measurable,
            "match": matched,
            "accuracy": round(matched / measurable, 4) if measurable else "N/A",
        }

    all5_measurable = _count(lambda r: r.get("all_5_fields_match") in (True, False))
    all5_match = _count(lambda r: r.get("all_5_fields_match") is True)
    all7_measurable = _count(lambda r: r.get("all_7_checks_match") in (True, False))
    all7_match = _count(lambda r: r.get("all_7_checks_match") is True)

    # -- V9.2: tenor/reward internal-consistency (evidence-only, tanpa ground
    # truth khusus -- "both evidence read & agree", TIDAK invented) --
    tenor_both_evidence = _count(lambda r: r.get("tenor_choice") != "N/A" and r.get("tenor_date_range_derived") != "N/A")
    tenor_consistent = _count(lambda r: r.get("tenor_internal_consistent") is True)
    reward_both_evidence = _count(
        lambda r: r.get("reward_choice") != "N/A" and (r.get("reward_non_tunai_filled") or r.get("reward_tunai_filled")))
    reward_consistent = _count(lambda r: r.get("reward_internal_consistent") is True)

    signature_dist = {}
    for col in ("signature_nasabah_status", "signature_atasan_status"):
        signature_dist[col] = {
            s: _count(lambda r, c=col, s=s: r.get(c) == s) for s in ("present", "absent", "uncertain")
        }

    # -- V9.2: Decision Accuracy / FAR / FRR / Review Rate (ground truth dari
    # kolom "Status Verifikasi Form Pendaftaran Nasabah" jika ada di sheet;
    # kalau tidak ada -> N/A, TIDAK di-invent) --
    gt_available = _count(lambda r: r.get("ground_truth_decision") != "N/A")
    decision_correct = _count(lambda r: r.get("decision_correct") is True)
    gt_tolak = _count(lambda r: r.get("ground_truth_decision") == "TOLAK")
    false_accept = _count(lambda r: r.get("ground_truth_decision") == "TOLAK" and r.get("decision") == "OK")
    gt_ok = _count(lambda r: r.get("ground_truth_decision") == "OK")
    false_reject = _count(lambda r: r.get("ground_truth_decision") == "OK" and r.get("decision") == "TOLAK")
    review_count = _count(lambda r: r.get("decision") == "REVIEW")

    times = [r["processing_time_sec"] for r in results if isinstance(r.get("processing_time_sec"), (int, float))]
    avg_time = round(sum(times) / len(times), 2) if times else "N/A"

    failure_breakdown = {stage: _count(lambda r, s=stage: r.get("failure_stage") == s) for stage in FAILURE_STAGES}

    return {
        "version": "V9.2",
        "documents_total": total,
        "documents_processed": pipeline_ok,
        "load_success_rate": round(loaded / total, 4) if total else "N/A",
        "alignment_success_rate": round(aligned_ok / aligned_attempted, 4) if aligned_attempted else "N/A",
        "alignment_failed_count": alignment_failed,
        "alignment_homography_escalation_count": homography_escalations,
        "semantic_region_success_rate": round(region_ok / loaded, 4) if loaded else "N/A",
        "pipeline_success_rate": round(pipeline_ok / total, 4) if total else "N/A",
        "field_accuracy": per_field,
        "all_5_fields_exact_accuracy": round(all5_match / all5_measurable, 4) if all5_measurable else "N/A",
        "all_7_checks_exact_accuracy": round(all7_match / all7_measurable, 4) if all7_measurable else "N/A",
        "tenor_both_evidence_read_count": tenor_both_evidence,
        "tenor_internal_consistency_rate": round(tenor_consistent / pipeline_ok, 4) if pipeline_ok else "N/A",
        "reward_both_evidence_read_count": reward_both_evidence,
        "reward_internal_consistency_rate": round(reward_consistent / pipeline_ok, 4) if pipeline_ok else "N/A",
        "signature_status_distribution": signature_dist,
        "decision_ground_truth_available": gt_available,
        "decision_accuracy": round(decision_correct / gt_available, 4) if gt_available else "N/A",
        "false_accept_rate": round(false_accept / gt_tolak, 4) if gt_tolak else "N/A",
        "false_reject_rate": round(false_reject / gt_ok, 4) if gt_ok else "N/A",
        "review_rate": round(review_count / pipeline_ok, 4) if pipeline_ok else "N/A",
        "average_processing_time_sec": avg_time,
        "failure_stage_breakdown": failure_breakdown,
        "note": (
            "Field comparison here is evaluation-only normalization (see compare_field); "
            "production OK/TOLAK/REVIEW decision remains comparison.py's responsibility "
            "and is not duplicated here."
        ),
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluator V9.2 (25-record ground truth dataset).")
    parser.add_argument("--dataset", required=True, help="Path ke ocr_evaluation.xlsx (TIDAK diubah oleh script ini)")
    parser.add_argument("--template", default=None, help="Override path template.pdf (default: preprocessing.TEMPLATE_PATH)")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah record yang diproses (debug cepat)")
    parser.add_argument("--debug-dir", default=str(DEFAULT_DEBUG_DIR),
                         help="Folder debug image (aligned + 3 semantic region). Isi 'none' utk mematikan.")
    parser.add_argument("--debug-all", action="store_true",
                         help="Simpan debug image utk SEMUA record (default: hanya yang gagal/bermasalah)")
    parser.add_argument("--label", default="v9_2",
                         help="Suffix run (mis. v9_2_baseline / v9_2_optimized) -> nama file output berbeda per run, "
                              "supaya baseline vs optimized bisa dibandingkan tanpa evaluator kedua")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    args.output_csv = args.output_csv or str(BASE_DIR / f"evaluation_results_{args.label}.csv")
    args.output_json = args.output_json or str(BASE_DIR / f"evaluation_summary_{args.label}.json")

    template_path = args.template or prep.TEMPLATE_PATH
    template_img = prep.load_document(template_path)

    records = data_input.read_excel_records(args.dataset)
    if args.limit:
        records = records[: args.limit]

    debug_dir = None if str(args.debug_dir).strip().lower() == "none" else args.debug_dir

    results = []
    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] Record {record.get('Record')} - {record.get('Nama Branch Office')}",
              file=sys.stderr)
        try:
            res = evaluate_record(record, template_img, template_path, debug_dir,
                                   debug_failed_only=not args.debug_all)
        except Exception:
            res = _blank_result(record)
            res["failure_stage"] = "PIPELINE"
            res["failure_reason"] = f"unhandled_exception: {traceback.format_exc(limit=2)}"
        results.append(res)

    write_csv(results, args.output_csv)
    summary = build_summary(results)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved: {args.output_csv}", file=sys.stderr)
    print(f"Saved: {args.output_json}", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
