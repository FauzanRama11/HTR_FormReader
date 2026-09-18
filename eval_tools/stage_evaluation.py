"""
stage_evaluation.py
====================
Evaluator TERPISAH per-tahap (GATE), TIDAK menggantikan evaluation.py (V9.2,
ground-truth 25-record + Decision Accuracy/FAR/FRR). Dijalankan terpisah dari
aplikasi utama (bukan endpoint FastAPI):

    python stage_evaluation.py --dataset ocr_evaluation.xlsx --label baseline

TUJUAN (beda dgn evaluation.py): evaluation.py mengukur APAKAH keputusan akhir
(OK/TOLAK/REVIEW) benar. File ini mengukur DI TAHAP MANA proses mulai gagal --
supaya kegagalan preprocessing/ROI TIDAK salah didiagnosis sbg "PaddleOCR
buruk". Tiga gate berurutan, evaluasi field di gate N+1 HANYA relevan/
bermakna kalau dokumen/field itu sudah PASS gate N:

  GATE 1 -- PREPROCESSING/ALIGNMENT (per dokumen)
    Reuse persis: preprocessing.load_document, preprocessing.prepare_and_align
    (SAMA fungsi yang dipanggil pipeline.run_pipeline() -- alignment TIDAK
    diduplikasi/ditebak ulang). PASS/FAIL dinilai dari sinyal yg SUDAH
    dihasilkan fungsi itu (coverage_mask utk crop/cutoff, rotation_deg/
    scale_x/scale_y/shear_deg per method) + SATU diagnostik tambahan
    evaluasi-only (_estimate_paper_area_ratio, lihat docstring-nya) yang
    tidak ada di preprocessing.py karena tidak dipakai keputusan produksi.

  GATE 2 -- ROI/LOCALIZATION (per field, HANYA field dari dokumen yang PASS
    gate 1)
    Reuse persis hasil pipeline.run_pipeline(): roi_bbox/roi_source/status
    per field (dynamic_extraction.py + preprocessing.resolve_roi_section),
    scores per opsi choice (postprocessing.process_choices), status
    signature (postprocessing.process_signatures). Diklasifikasi ke
    PASS/TOO_NARROW/TOO_WIDE/WRONG_LOCATION/CUT_OFF/LABEL_BLEED lewat
    heuristik (lihat classify_text_roi) yang SEMUANYA dibangun dari sinyal
    produksi yang sudah ada -- tidak ada model/logic ROI baru yang
    menggantikan punya preprocessing.py. Heuristik ini bukan ground truth
    geometri ROI -- gunakan debug crop tiap field (selalu disimpan) utk
    verifikasi visual final.

  GATE 3 -- OCR/RECOGNITION (per field, HANYA field yang ROI-nya PASS gate 2)
    Bandingkan raw OCR (raw_results[field]["raw"]) & hasil normalisasi
    (fields_table) vs ground truth (dataset). Field teks pakai
    evaluation.compare_field (REUSE, tidak diduplikasi -- exact match utk
    nomor_rekening/nominal_penempatan sudah inheren dari cara field itu
    dinormalisasi jadi digit murni sebelum dibandingkan). nama_nasabah
    tambahan CER (Character Error Rate, lihat _cer). tenor_penempatan &
    bentuk_reward dievaluasi sbg VISUAL-CHOICE classification (skor
    ink-diff antar opsi dari postprocessing.process_choices, BUKAN OCR
    teks) -- confidence = margin skor opsi terpilih vs runner-up.

  Field/dokumen yang gagal di gate N TIDAK dihitung sbg kegagalan gate N+1 --
  itulah inti "evaluation pipeline terpisah per stage" yang diminta. Funnel
  akhir (lihat build_summary): dari total input -> % gagal preprocessing ->
  dari yang lolos, % gagal ROI -> dari ROI yang benar, % benar-benar gagal
  OCR/model.

Prinsip (SAMA seperti evaluation.py, lihat docstring-nya utk detail):
  - ocr_evaluation.xlsx TIDAK PERNAH ditulis/diubah oleh script ini.
  - TIDAK menduplikasi logika preprocessing/OCR/decision -- reuse fungsi
    produksi (preprocessing.*, postprocessing.*, comparison.*,
    pipeline.run_pipeline, evaluation.compare_field/_extract_value/
    FIELD_GT_MAP). Field yang tidak ada di scope perbandingan (signature)
    dicatat statusnya saja, tidak diberi skor akurasi (tidak ada ground
    truth-nya di dataset).
  - align_to_template dipanggil DUA KALI per dokumen (sekali langsung utk
    Gate 1 dgn granularitas coverage_mask/rotation/shear, sekali lagi di
    dalam run_pipeline utk Gate 2/3) -- trade-off yang SAMA & disengaja
    seperti evaluation.py (lihat docstringnya), demi diagnostik tahap yang
    presisi per field, bukan bug/pemborosan yang tidak disadari.

Output:
  - stage_evaluation_results_<label>.csv   (satu baris per dokumen, kolom
    document | preprocessing_* | roi_<field>_status/reason (semua field) |
    ocr_<field>_* (5 field decisive) | failure_stage | failure_reason)
  - stage_evaluation_results_<label>.xlsx  (sheet sama + sheet "summary")
  - stage_evaluation_summary_<label>.json  (funnel + metrik per-field)
  - stage_evaluation_debug/record_<id>/    (before.png, aligned.png, dan
    satu crop .png per field/opsi/signature -- HANYA utk dokumen
    gagal/bermasalah, kecuali --debug-all)
"""

import argparse
import csv
import importlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

# Dijalankan langsung (`python eval_tools/stage_evaluation.py ...`), jadi
# sys.path[0] = folder eval_tools/ ini sendiri -- root harus ditambah manual.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from core import data_input
import pipeline.pipeline as pipeline
from pipeline import preprocessing as prep
from pipeline import postprocessing as post
from pipeline import comparison
from eval_tools import evaluation as ev9   # REUSE V9.2 field normalization/perbandingan
                            # (compare_field, _extract_value, FIELD_GT_MAP) --
                            # TIDAK diduplikasi di sini.
from core.paths import PROJECT_ROOT, EVAL_RUNS_DIR

DEFAULT_DEBUG_DIR = PROJECT_ROOT / "stage_evaluation_debug"

# ============================================================================
# CONFIG -- field groups (REUSE definisi produksi, tidak hardcode ulang)
# ============================================================================

FIELD_GT_MAP = ev9.FIELD_GT_MAP                       # field -> kolom ground truth
DECISIVE_FIELDS = comparison.DECISIVE_FIELDS_V9_2      # 5 field keputusan
TEXT_FIELDS = list(prep.FIELD_CONFIG.keys())           # 8 field teks/currency
CHOICE_GROUP_NAMES = list(prep.CHOICE_GROUPS.keys())   # tenor_penempatan, bentuk_reward
SIGNATURE_FIELDS = list(prep.SIGNATURE_CONFIG.keys())  # signature_nasabah/atasan
OPTIONAL_TEXT_FIELDS = {n for n, c in prep.FIELD_CONFIG.items() if c["type"] == "optional_text"}

ROI_STATUSES = ("PASS", "TOO_NARROW", "TOO_WIDE", "WRONG_LOCATION", "CUT_OFF", "LABEL_BLEED")

# -- GATE 1 thresholds --
# Sebagian besar REUSE threshold produksi (preprocessing.py) apa adanya;
# beberapa BARU (evaluasi-only) karena sinyalnya juga tidak ada di produksi.
PREPROCESSING_COVERAGE_MIN_RATIO = prep.COVERAGE_MIN_RATIO   # reuse (0.85)
SHEAR_MAX_DEG_HOMOGRAPHY = 3.0     # BARU: affine TIDAK PERNAH shear (by
                                    # construction, similarity-only) -- shear
                                    # signifikan cuma bisa muncul kalau
                                    # method=="homography" (fallback lebih
                                    # destruktif, lihat preprocessing.align_to_template)
PAPER_AREA_MIN_RATIO = 0.35        # BARU: proxy "terlalu banyak background";
                                    # sedikit di bawah ambang rectify_photo
                                    # (0.30) sbg buffer karena ini estimasi
                                    # kontur terbesar, bukan syarat 4-titik ketat

# -- GATE 2 thresholds (heuristik klasifikasi ROI, evaluasi-only) --
ROI_LEN_RATIO_NARROW = 0.6   # raw_text jauh lebih pendek dari ground truth
ROI_LEN_RATIO_WIDE = 1.6     # raw_text jauh lebih panjang dari ground truth


# ============================================================================
# GATE 1 -- PREPROCESSING / ALIGNMENT
# ============================================================================

def _estimate_paper_area_ratio(raw_image):
    """Diagnostik EVALUASI-ONLY (TIDAK dipakai/mengubah keputusan produksi):
    estimasi seberapa besar kertas formulir mengisi frame foto, dgn
    pendekatan kontur YANG SAMA dgn preprocessing._find_document_quad
    (dipakai jg oleh rectify_photo produksi -- V13, diperbaiki dari Canny
    tetap ke adaptif+dilasi, lihat docstring-nya) supaya diagnostik ini
    TIDAK PERNAH menyimpang dari cara produksi sungguhan "melihat" tepi
    kertas. Return None kalau tidak ada kontur sama sekali (bukan berarti
    gagal -- rectify_photo pun memperlakukan 'quad tidak ketemu' sbg 'tidak
    direktifikasi', bukan error keras)."""
    try:
        _quad, ratio = prep._find_document_quad(raw_image)
        return ratio
    except Exception:
        return None


def evaluate_preprocessing_gate(doc_path, template_img):
    """GATE 1. Return (result_dict, raw_image_or_None, aligned_img_or_None,
    coverage_mask_or_None). result_dict berisi preprocessing_status
    (PASS/FAIL), preprocessing_reasons (list alasan spesifik), dan semua
    diagnostik mentah (rotation_deg/scale/shear/coverage per region/
    paper_area_ratio) utk kebutuhan audit di CSV."""
    out = {
        "preprocessing_status": "FAIL",
        "load_success": False,
        "alignment_status": "N/A", "alignment_method": "N/A",
        "rotation_deg": "N/A", "scale_x": "N/A", "scale_y": "N/A", "shear_deg": "N/A",
        "identity_area_coverage": "N/A", "placement_area_coverage": "N/A",
        "signature_area_coverage": "N/A", "paper_area_ratio": "N/A",
    }
    reasons = []

    try:
        raw_image = prep.load_document(doc_path)
        out["load_success"] = True
    except Exception as exc:
        reasons.append(f"load_failed: {type(exc).__name__}: {exc}")
        out["preprocessing_reasons"] = "; ".join(reasons)
        return out, None, None, None

    is_photo = not prep.is_pdf_document(doc_path)
    if is_photo:
        ratio = _estimate_paper_area_ratio(raw_image)
        out["paper_area_ratio"] = round(ratio, 4) if ratio is not None else "N/A"
        if ratio is not None and ratio < PAPER_AREA_MIN_RATIO:
            reasons.append(
                f"excessive_background: paper_area_ratio={ratio:.2f} (<{PAPER_AREA_MIN_RATIO})"
            )

    try:
        aligned_img, _H, align_meta = prep.prepare_and_align(doc_path, raw_image, template_img)
    except Exception as exc:
        reasons.append(f"alignment_exception: {type(exc).__name__}: {exc}")
        out["preprocessing_reasons"] = "; ".join(reasons)
        return out, raw_image, None, None

    coverage_mask = align_meta.get("coverage_mask")
    out["alignment_status"] = align_meta.get("status")
    out["alignment_method"] = align_meta.get("method", "N/A")
    out["rotation_deg"] = align_meta.get("rotation_deg", "N/A")
    out["scale_x"] = align_meta.get("scale_x", "N/A")
    out["scale_y"] = align_meta.get("scale_y", "N/A")
    out["shear_deg"] = align_meta.get("shear_deg", "N/A")

    if align_meta.get("status") == "failed":
        reasons.append(f"alignment_failed: {align_meta.get('reason', 'unknown')}")
        out["preprocessing_reasons"] = "; ".join(reasons)
        return out, raw_image, aligned_img, coverage_mask

    # -- orientation/perspective: rotasi & skala AKHIR harus di dalam batas
    # yg sama dipakai preprocessing.py utk kandidat affine. Affine dijamin
    # memenuhi ini by construction (lihat _estimate_affine_candidate) --
    # pengecekan ini efektif menyasar kasus homography escalation, yang
    # tidak digate dgn batas rotasi/skala yg sama di produksi.
    rot = align_meta.get("rotation_deg")
    if isinstance(rot, (int, float)) and abs(rot) > prep.AFFINE_MAX_ROTATION_DEG:
        reasons.append(
            f"orientation_out_of_tolerance: rotation={rot:.2f}deg (max {prep.AFFINE_MAX_ROTATION_DEG})"
        )
    lo, hi = prep.AFFINE_SCALE_RANGE
    sx, sy = align_meta.get("scale_x"), align_meta.get("scale_y")
    if isinstance(sx, (int, float)) and not (lo <= sx <= hi):
        reasons.append(f"scale_x_out_of_tolerance: {sx:.3f} (range {lo}-{hi})")
    if isinstance(sy, (int, float)) and not (lo <= sy <= hi):
        reasons.append(f"scale_y_out_of_tolerance: {sy:.3f} (range {lo}-{hi})")

    # -- distortion tambahan DARI PROSES ALIGNMENT itu sendiri: shear
    # signifikan hanya mungkin muncul dari method=="homography".
    shear = align_meta.get("shear_deg")
    if (align_meta.get("method") == "homography" and isinstance(shear, (int, float))
            and abs(shear) > SHEAR_MAX_DEG_HOMOGRAPHY):
        reasons.append(
            f"alignment_added_shear: shear={shear:.2f}deg "
            f"(method=homography, max {SHEAR_MAX_DEG_HOMOGRAPHY})"
        )

    # -- crop/cutoff: bagian penting formulir (identity/placement) tidak
    # boleh jatuh di luar cakupan piksel sumber. Reuse prep.region_coverage
    # + threshold SAMA (COVERAGE_MIN_RATIO) yg dipakai pipeline produksi
    # utk menandai field "out_of_frame".
    shape = aligned_img.shape
    for region_name, bbox_norm in prep.SEMANTIC_REGIONS_NORM.items():
        bbox_px = prep.norm_bbox_to_px(bbox_norm, shape)
        cov = prep.region_coverage(coverage_mask, bbox_px)
        out[f"{region_name}_coverage"] = round(cov, 4)
        if cov < PREPROCESSING_COVERAGE_MIN_RATIO:
            if region_name in ("identity_area", "placement_area"):
                reasons.append(
                    f"form_cropped: {region_name} coverage={cov:.2f} (<{PREPROCESSING_COVERAGE_MIN_RATIO})"
                )
            else:
                # signature_area TIDAK termasuk 5 field decisive (tidak ada
                # ground truth-nya) -- dicatat sbg info, TIDAK men-FAIL-kan
                # gate 1 sendirian, konsisten dgn cakupan evaluasi V9.2.
                reasons.append(f"signature_area_cropped_informational: coverage={cov:.2f}")

    blocking = [r for r in reasons if "_informational" not in r]
    out["preprocessing_status"] = "FAIL" if blocking else "PASS"
    out["preprocessing_reasons"] = "; ".join(reasons) if reasons else ""
    return out, raw_image, aligned_img, coverage_mask


# ============================================================================
# GATE 2 -- ROI / LOCALIZATION (klasifikasi heuristik, lihat docstring modul)
# ============================================================================

def _looks_like_label_bleed(text):
    """Sinyal residual label cetak MASIH ada di teks AKHIR (setelah
    postprocessing.build_text_field_result menjalankan strip-nya sendiri) --
    kalau masih lolos, itu bukti ROI menangkap label tetangga (LABEL_BLEED),
    bukan cuma OCR salah baca. Reuse postprocessing._LABEL_BLEED_FRAGMENTS
    (konstanta produksi) alih-alih menebak daftar baru."""
    if not text:
        return False
    if ":" in text:
        return True
    first = text.strip().split(" ", 1)[0].strip(".,").lower()
    return len(first) >= 5 and any(frag.endswith(first) for frag in post._LABEL_BLEED_FRAGMENTS)


def classify_text_roi(field_name, row, ground_truth):
    """Klasifikasi PASS/TOO_NARROW/TOO_WIDE/WRONG_LOCATION/CUT_OFF/
    LABEL_BLEED utk satu field teks, dari sinyal produksi (row = entry
    raw_results[field_name] hasil pipeline.run_pipeline) + ground truth
    dataset (opsional). Urutan cek = urutan prioritas diagnosis (paling
    pasti dulu). HEURISTIK evaluasi, bukan ground truth ROI -- selalu
    silangkan dgn debug crop field ybs sebelum menyimpulkan."""
    row = row or {}
    status = row.get("status")

    if status == "out_of_frame":
        return "CUT_OFF", "field_di_luar_cakupan_piksel_sumber_foto"

    gt_str = str(ground_truth).strip() if ground_truth not in (None, "") else ""
    is_optional = field_name in OPTIONAL_TEXT_FIELDS

    if gt_str and not is_optional and status in ("not_detected", "blank"):
        return "WRONG_LOCATION", "roi_tidak_menangkap_apa_pun_padahal_ground_truth_terisi"

    raw_text = row.get("raw") or ""
    if _looks_like_label_bleed(raw_text):
        preview = raw_text[:40]
        return "LABEL_BLEED", f"residual_fragmen_label_tercetak_di_teks: '{preview}'"

    if gt_str and raw_text:
        ratio = len(raw_text) / max(1, len(gt_str))
        if ratio < ROI_LEN_RATIO_NARROW:
            return "TOO_NARROW", f"panjang_capture={len(raw_text)} vs ground_truth={len(gt_str)} (ratio={ratio:.2f})"
        if ratio > ROI_LEN_RATIO_WIDE:
            return "TOO_WIDE", f"panjang_capture={len(raw_text)} vs ground_truth={len(gt_str)} (ratio={ratio:.2f})"

    return "PASS", "ok"


def classify_choice_roi(group_name, raw_results):
    """Utk grup pilihan (tenor/reward): ROI dinilai dari status per opsi
    (postprocessing.process_choices) -- tidak ada teks/panjang utk
    dibandingkan, jadi hanya CUT_OFF/WRONG_LOCATION/PASS yang relevan."""
    options = prep.CHOICE_GROUPS[group_name]["options"]
    rows = [raw_results.get(opt_name, {}) for opt_name in options]
    if any(r.get("status") == "out_of_frame" for r in rows):
        return "CUT_OFF", "area_opsi_di_luar_cakupan_foto"
    if any(r.get("roi_source") == "fallback" for r in rows):
        return "WRONG_LOCATION", "anchor_pilihan_tidak_match_ROI_pakai_posisi_template_polos"
    return "PASS", "ok"


def classify_signature_roi(field_name, raw_results):
    row = raw_results.get(field_name, {}) or {}
    if row.get("status") == "out_of_frame":
        return "CUT_OFF", "area_tanda_tangan_di_luar_cakupan_foto"
    return "PASS", "ok"


# ============================================================================
# GATE 3 -- OCR / RECOGNITION
# ============================================================================

def _cer(hypothesis, reference):
    """Character Error Rate = Levenshtein(hyp, ref) / len(ref). Tidak ada
    implementasi Levenshtein di codebase (dicek: preprocessing/postprocessing/
    ocr/comparison tidak punya) -- ditambahkan di sini SEBAGAI metrik
    evaluasi murni, tidak dipakai/mengubah jalur produksi mana pun."""
    a, b = hypothesis or "", reference or ""
    if not b:
        return None
    if not a:
        return 1.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m] / m


def _choice_confidence_margin(choice_groups_raw, group_name):
    """Confidence utk visual-choice classification = margin skor ink-diff
    (postprocessing.process_choices) antara skor tertinggi & runner-up --
    BUKAN OCR confidence teks (tenor/reward memang tidak pernah di-OCR
    teks, lihat preprocessing.py: 'Choice & signature: HANYA image-diff')."""
    scores = (choice_groups_raw.get(group_name) or {}).get("scores") or {}
    if len(scores) < 2:
        return None
    ordered = sorted(scores.values(), reverse=True)
    return round(ordered[0] - ordered[1], 4)


# ============================================================================
# ORKESTRASI PER DOKUMEN
# ============================================================================

def _blank_gate23_columns(row):
    for f in TEXT_FIELDS:
        row[f"roi_{f}_status"] = "N/A"
        row[f"roi_{f}_reason"] = "N/A"
    for g in CHOICE_GROUP_NAMES:
        row[f"roi_{g}_status"] = "N/A"
        row[f"roi_{g}_reason"] = "N/A"
    for s in SIGNATURE_FIELDS:
        row[f"roi_{s}_status"] = "N/A"
        row[f"roi_{s}_reason"] = "N/A"
    row["roi_status"] = "N/A"
    row["roi_fail_fields"] = "N/A"
    for f in DECISIVE_FIELDS:
        row[f"ocr_{f}_extracted"] = "N/A"
        row[f"ocr_{f}_ground_truth"] = "N/A"
        row[f"ocr_{f}_comparison"] = "N/A"
        row[f"ocr_{f}_confidence"] = "N/A"
    row["ocr_nama_nasabah_cer"] = "N/A"
    row["ocr_status"] = "N/A"


def _save_debug(debug_dir, record, raw_image, aligned_img, raw_results):
    """Simpan before/after + satu crop per field/opsi/signature. raw_results
    (dict field_name -> {roi_bbox,...}) SUDAH dihasilkan produksi (pipeline/
    dynamic_extraction/postprocessing) -- crop di sini hanya membaca
    roi_bbox itu, tidak menghitung ROI baru."""
    if debug_dir is None:
        return
    rec_id = record.get("Record", "unknown")
    out_dir = Path(debug_dir) / f"record_{rec_id}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if raw_image is not None:
            cv2.imwrite(str(out_dir / "before_raw.png"), raw_image)
        if aligned_img is not None:
            cv2.imwrite(str(out_dir / "after_aligned.png"), aligned_img)
        if aligned_img is not None and raw_results:
            for name, r in raw_results.items():
                bbox = (r or {}).get("roi_bbox")
                if not bbox:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = aligned_img[max(0, y1):y2, max(0, x1):x2]
                if crop.size:
                    cv2.imwrite(str(out_dir / f"field_{name}.png"), crop)
    except Exception:
        pass  # debug image tidak boleh menggagalkan evaluasi


def _blank_row(record):
    row = {
        "record": record.get("Record"),
        "document": record.get("Nama Branch Office") or record.get("Record"),
        "document_url": record.get("Surat Pernyataan"),
        "preprocessing_status": "N/A", "preprocessing_reasons": "N/A",
        "load_success": False,
        "failure_stage": None, "failure_reason": None,
        "processing_time_sec": "N/A",
    }
    _blank_gate23_columns(row)
    return row


def evaluate_record(record, template_img, template_path, debug_dir, debug_failed_only=True,
                     pipeline_module=pipeline):
    row = _blank_row(record)
    t0 = time.time()

    doc_path = data_input.get_document_path(record)
    if not doc_path:
        row["failure_stage"] = "PREPROCESSING"
        row["failure_reason"] = "document_not_found_or_download_failed"
        return row

    # ---- GATE 1 ----
    gate1, raw_image, aligned_img, coverage_mask = evaluate_preprocessing_gate(doc_path, template_img)
    row.update(gate1)

    if gate1["preprocessing_status"] != "PASS":
        row["failure_stage"] = "PREPROCESSING"
        row["failure_reason"] = gate1.get("preprocessing_reasons") or "preprocessing_gate_failed"
        row["processing_time_sec"] = round(time.time() - t0, 2)
        if debug_dir is not None:
            _save_debug(debug_dir, record, raw_image, aligned_img, None)
        return row

    # ---- Jalankan pipeline penuh (REUSE, bukan duplikasi) utk field-level
    # ROI/OCR/choice/signature. align_to_template dipanggil lagi di dalam --
    # trade-off yang disengaja, lihat docstring modul. ----
    try:
        pipeline_result = pipeline_module.run_pipeline(doc_path, template_path=template_path)
    except Exception as exc:
        row["failure_stage"] = "ROI"  # setelah preprocessing PASS, kegagalan
        # keras berikutnya paling sering di localization/orkestrasi ROI
        row["failure_reason"] = f"pipeline_exception: {type(exc).__name__}: {exc}"
        row["processing_time_sec"] = round(time.time() - t0, 2)
        if debug_dir is not None:
            _save_debug(debug_dir, record, raw_image, aligned_img, None)
        return row

    raw_results = pipeline_result.get("raw_results", {})
    fields_table = pipeline_result.get("fields")
    choice_groups_final = pipeline_result.get("choice_groups", {})   # nilai final (post evidence-merge)
    choice_groups_raw = pipeline_result.get("ocr_meta", {})           # tidak dipakai; scores ada di raw langsung
    # scores ink-diff per grup TERSEDIA di variabel lokal process_choices,
    # TIDAK diteruskan run_pipeline sbg objek terpisah -- ambil dari
    # raw_results per-opsi (SUDAH ada di sana lewat choice_raw.update, lihat
    # pipeline.py) utk hitung margin, bukan dari choice_groups_final.
    choice_scores = {}
    for gname in CHOICE_GROUP_NAMES:
        opt_scores = {
            opt: (raw_results.get(opt, {}) or {}).get("mark_score")
            for opt in prep.CHOICE_GROUPS[gname]["options"]
        }
        choice_scores[gname] = {"scores": {k: v for k, v in opt_scores.items() if v is not None}}

    # ---- GATE 2: ROI/LOCALIZATION ----
    roi_fail_fields = []
    for fname in TEXT_FIELDS:
        gt_col = FIELD_GT_MAP.get(fname)
        gt_value = record.get(gt_col) if gt_col else None
        status, reason = classify_text_roi(fname, raw_results.get(fname), gt_value)
        row[f"roi_{fname}_status"] = status
        row[f"roi_{fname}_reason"] = reason
        if status != "PASS" and fname in DECISIVE_FIELDS:
            roi_fail_fields.append(fname)

    for gname in CHOICE_GROUP_NAMES:
        status, reason = classify_choice_roi(gname, raw_results)
        row[f"roi_{gname}_status"] = status
        row[f"roi_{gname}_reason"] = reason
        if status != "PASS" and gname in DECISIVE_FIELDS:
            roi_fail_fields.append(gname)

    for sname in SIGNATURE_FIELDS:
        status, reason = classify_signature_roi(sname, raw_results)
        row[f"roi_{sname}_status"] = status
        row[f"roi_{sname}_reason"] = reason

    row["roi_status"] = "FAIL" if roi_fail_fields else "PASS"
    row["roi_fail_fields"] = ",".join(roi_fail_fields)

    # ---- GATE 3: OCR/RECOGNITION -- selalu dihitung utk SEMUA 5 field
    # decisive (transparansi), tapi field yang ROI-nya sendiri sudah FAIL
    # TIDAK dihitung sbg "kegagalan OCR murni" (lihat ocr_fail_fields di
    # bawah, hanya field dgn roi_status=="PASS" yang masuk hitungan itu). ----
    for fname in DECISIVE_FIELDS:
        gt_col = FIELD_GT_MAP[fname]
        gt_value = record.get(gt_col)
        extracted, _raw = ev9._extract_value(fields_table, raw_results, fname)  # REUSE
        cmp = ev9.compare_field(fname, extracted, gt_value)                     # REUSE
        row[f"ocr_{fname}_extracted"] = extracted if extracted not in (None, "") else "N/A"
        row[f"ocr_{fname}_ground_truth"] = gt_value if gt_value not in (None, "") else "N/A"
        row[f"ocr_{fname}_comparison"] = cmp

        if fname in CHOICE_GROUP_NAMES:
            conf = _choice_confidence_margin(choice_scores, fname)
        else:
            conf = (raw_results.get(fname) or {}).get("confidence")
        row[f"ocr_{fname}_confidence"] = round(conf, 4) if isinstance(conf, (int, float)) else "N/A"

    if row.get("ocr_nama_nasabah_ground_truth") not in (None, "N/A") and row.get("ocr_nama_nasabah_extracted") not in (None, "N/A"):
        row["ocr_nama_nasabah_cer"] = round(
            _cer(str(row["ocr_nama_nasabah_extracted"]), str(row["ocr_nama_nasabah_ground_truth"])) or 0.0, 4
        )
    else:
        row["ocr_nama_nasabah_cer"] = "N/A"

    # "gagal OCR murni" HANYA dihitung utk field yang ROI-nya sudah PASS --
    # inilah pemisahan tahap inti yang diminta.
    ocr_fail_fields_pure = [
        f for f in DECISIVE_FIELDS
        if row.get(f"roi_{f}_status") == "PASS" and row.get(f"ocr_{f}_comparison") == "MISMATCH"
    ]
    measurable = [f for f in DECISIVE_FIELDS if row.get(f"ocr_{f}_comparison") in ("MATCH", "MISMATCH")]
    row["ocr_status"] = "N/A" if not measurable else ("FAIL" if any(
        row.get(f"ocr_{f}_comparison") == "MISMATCH" for f in measurable
    ) else "PASS")
    row["ocr_pure_status"] = "N/A" if not roi_fail_fields and not measurable else (
        "FAIL" if ocr_fail_fields_pure else "PASS"
    )

    # ---- failure_stage dokumen: bottleneck PALING AWAL (Preprocessing >
    # ROI > OCR), sesuai filosofi "jangan salah menyimpulkan tahap belakang
    # gagal kalau tahap depan yang sebenarnya bermasalah". ----
    if roi_fail_fields:
        row["failure_stage"] = "ROI"
        row["failure_reason"] = "roi_localization_failed_for: " + ",".join(roi_fail_fields)
    elif ocr_fail_fields_pure:
        row["failure_stage"] = "OCR"
        row["failure_reason"] = "ocr_recognition_failed_for: " + ",".join(ocr_fail_fields_pure)
    else:
        row["failure_stage"] = "PASS"
        row["failure_reason"] = ""

    row["processing_time_sec"] = round(time.time() - t0, 2)

    problem = row["failure_stage"] != "PASS"
    if debug_dir is not None and (not debug_failed_only or problem):
        _save_debug(debug_dir, record, raw_image, aligned_img, raw_results)

    return row


# ============================================================================
# AGREGASI & OUTPUT
# ============================================================================

def _field_order(results):
    seen = []
    for row in results:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    return seen


def write_csv(results, path):
    fieldnames = _field_order(results)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "N/A") for k in fieldnames})


def write_xlsx(results, summary, path):
    """Sheet 'results' (SAMA isi dgn CSV) + sheet 'summary' (funnel & metrik
    per-field, flattened). Butuh pandas+openpyxl (SUDAH ada di
    requirements.txt, tidak menambah dependency baru)."""
    import pandas as pd
    fieldnames = _field_order(results)
    df = pd.DataFrame([{k: r.get(k, "N/A") for k in fieldnames} for r in results])

    def _flatten(d, prefix=""):
        rows = []
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                rows.extend(_flatten(v, prefix=f"{key}."))
            else:
                rows.append({"metric": key, "value": v})
        return rows

    summary_df = pd.DataFrame(_flatten(summary))
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)


def build_summary(results):
    total = len(results)

    def _count(pred):
        return sum(1 for r in results if pred(r))

    preprocessing_pass = _count(lambda r: r.get("preprocessing_status") == "PASS")
    preprocessing_fail = _count(lambda r: r.get("preprocessing_status") == "FAIL")
    loaded = _count(lambda r: r.get("load_success"))

    # -- funnel utama yang diminta: input -> preprocessing -> ROI -> OCR --
    roi_base = preprocessing_pass
    roi_pass = _count(lambda r: r.get("preprocessing_status") == "PASS" and r.get("roi_status") == "PASS")
    roi_fail = roi_base - roi_pass

    ocr_base = roi_pass
    ocr_pass = _count(
        lambda r: r.get("preprocessing_status") == "PASS"
        and r.get("roi_status") == "PASS"
        and r.get("failure_stage") == "PASS"
    )
    ocr_fail = ocr_base - ocr_pass

    def _pct(numer, denom):
        return round(100.0 * numer / denom, 2) if denom else "N/A"

    # -- per-field: Field Capture Rate (ROI PASS rate, di antara dokumen yg
    # lolos preprocessing) + OCR accuracy MURNI (di antara field yg ROI-nya
    # sendiri PASS -- inilah yang memisahkan "OCR jelek" dari "ROI jelek") --
    per_field = {}
    for fname in DECISIVE_FIELDS:
        base = _count(lambda r: r.get("preprocessing_status") == "PASS")
        roi_ok = _count(lambda r, f=fname: r.get("preprocessing_status") == "PASS" and r.get(f"roi_{f}_status") == "PASS")
        ocr_measurable = _count(
            lambda r, f=fname: r.get(f"roi_{f}_status") == "PASS" and r.get(f"ocr_{f}_comparison") in ("MATCH", "MISMATCH")
        )
        ocr_match = _count(
            lambda r, f=fname: r.get(f"roi_{f}_status") == "PASS" and r.get(f"ocr_{f}_comparison") == "MATCH"
        )
        per_field[fname] = {
            "field_capture_rate": round(roi_ok / base, 4) if base else "N/A",
            "roi_pass_count": roi_ok,
            "pure_ocr_accuracy": round(ocr_match / ocr_measurable, 4) if ocr_measurable else "N/A",
            "pure_ocr_measurable": ocr_measurable,
        }

    roi_status_breakdown = {
        status: {
            fname: _count(lambda r, f=fname, s=status: r.get(f"roi_{f}_status") == s)
            for fname in DECISIVE_FIELDS
        }
        for status in ROI_STATUSES
    }

    all_decisive_roi_pass_count = roi_pass  # dokumen yang SELURUH decisive field ROI-nya PASS
    functional_all_fields_match = _count(
        lambda r: all(r.get(f"ocr_{f}_comparison") in ("MATCH", "N/A") for f in DECISIVE_FIELDS)
        and any(r.get(f"ocr_{f}_comparison") == "MATCH" for f in DECISIVE_FIELDS)
    )

    times = [r["processing_time_sec"] for r in results if isinstance(r.get("processing_time_sec"), (int, float))]
    avg_time = round(sum(times) / len(times), 2) if times else "N/A"

    failure_stage_breakdown = {
        stage: _count(lambda r, s=stage: r.get("failure_stage") == s)
        for stage in ("PREPROCESSING", "ROI", "OCR", "PASS")
    }

    return {
        "documents_total": total,
        "load_success_count": loaded,
        "funnel": {
            "total_input": total,
            "preprocessing_pass": preprocessing_pass,
            "preprocessing_fail": preprocessing_fail,
            "PREPROCESSING_ERROR_pct": _pct(preprocessing_fail, total),
            "roi_evaluated_base_ie_preprocessing_pass": roi_base,
            "roi_pass": roi_pass,
            "roi_fail": roi_fail,
            "ROI_LOCALIZATION_ERROR_pct": _pct(roi_fail, roi_base),
            "ocr_evaluated_base_ie_preprocessing_and_roi_pass": ocr_base,
            "ocr_pass": ocr_pass,
            "ocr_fail": ocr_fail,
            "OCR_RECOGNITION_ERROR_pct": _pct(ocr_fail, ocr_base),
        },
        "field_capture_and_pure_ocr_accuracy": per_field,
        "roi_status_breakdown_per_decisive_field": roi_status_breakdown,
        "documents_all_decisive_roi_pass": all_decisive_roi_pass_count,
        "functional_all_fields_match_count": functional_all_fields_match,
        "average_processing_time_sec": avg_time,
        "failure_stage_breakdown": failure_stage_breakdown,
        "note": (
            "field_capture_rate = ROI PASS rate di antara dokumen yg lolos preprocessing (Gate 1). "
            "pure_ocr_accuracy = akurasi OCR HANYA di antara field yg ROI-nya sendiri sudah PASS (Gate 2) -- "
            "field dgn ROI FAIL tidak ikut menghitung akurasi OCR supaya kegagalan ROI tidak salah dituduhkan "
            "ke model OCR (lihat known_issues di HANDOVER_V10.md). "
            "roi_status heuristik (TOO_NARROW/TOO_WIDE/WRONG_LOCATION/LABEL_BLEED) dibangun dari sinyal produksi "
            "(status/roi_source/panjang teks vs ground truth) -- BUKAN ground truth ROI, verifikasi dgn debug crop."
        ),
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluator 3-gate (PREPROCESSING -> ROI/LOCALIZATION -> OCR/RECOGNITION)."
    )
    parser.add_argument("--dataset", required=True, help="Path ke ocr_evaluation.xlsx (TIDAK diubah oleh script ini)")
    parser.add_argument("--template", default=None, help="Override path template.pdf (default: preprocessing.TEMPLATE_PATH)")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah record yang diproses (debug cepat)")
    parser.add_argument("--debug-dir", default=str(DEFAULT_DEBUG_DIR),
                         help="Folder debug image (before/after + crop per field). Isi 'none' utk mematikan.")
    parser.add_argument("--debug-all", action="store_true",
                         help="Simpan debug image utk SEMUA record (default: hanya yang gagal/bermasalah)")
    parser.add_argument("--label", default="stage_eval",
                         help="Suffix run -> nama file output berbeda per run")
    parser.add_argument("--pipeline", default="pipeline",
                         help="Nama modul pipeline yg dipakai (mis. 'pipeline' [V12] atau 'pipeline1' [V10/11]) "
                              "utk membandingkan versi/strategi model -- lihat compare_runs.py")
    parser.add_argument("--vlm-fallback", choices=["on", "off"], default=None,
                         help="Override pipeline_module.VLM_FALLBACK_ENABLED kalau ada")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-xlsx", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--no-xlsx", action="store_true", help="Lewati output .xlsx (hanya CSV+JSON)")
    args = parser.parse_args()
    EVAL_RUNS_DIR.mkdir(exist_ok=True)
    args.output_csv = args.output_csv or str(EVAL_RUNS_DIR / f"stage_evaluation_results_{args.label}.csv")
    args.output_xlsx = args.output_xlsx or str(EVAL_RUNS_DIR / f"stage_evaluation_results_{args.label}.xlsx")
    args.output_json = args.output_json or str(EVAL_RUNS_DIR / f"stage_evaluation_summary_{args.label}.json")

    pipeline_module = importlib.import_module(args.pipeline) if args.pipeline != "pipeline" else pipeline
    if args.vlm_fallback is not None and hasattr(pipeline_module, "VLM_FALLBACK_ENABLED"):
        pipeline_module.VLM_FALLBACK_ENABLED = (args.vlm_fallback == "on")

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
                                   debug_failed_only=not args.debug_all, pipeline_module=pipeline_module)
        except Exception:
            res = _blank_row(record)
            res["failure_stage"] = "PREPROCESSING"
            res["failure_reason"] = f"unhandled_exception: {traceback.format_exc(limit=2)}"
        results.append(res)

    write_csv(results, args.output_csv)
    summary = build_summary(results)
    summary["run_label"] = args.label
    summary["pipeline_module"] = args.pipeline
    summary["vlm_fallback_enabled"] = getattr(pipeline_module, "VLM_FALLBACK_ENABLED", "N/A")
    Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_xlsx:
        try:
            write_xlsx(results, summary, args.output_xlsx)
        except Exception as exc:
            print(f"WARNING: gagal menulis .xlsx ({exc}) -- CSV/JSON tetap tersimpan.", file=sys.stderr)

    print(f"\nSaved: {args.output_csv}", file=sys.stderr)
    if not args.no_xlsx:
        print(f"Saved: {args.output_xlsx}", file=sys.stderr)
    print(f"Saved: {args.output_json}", file=sys.stderr)
    print(json.dumps(summary["funnel"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
