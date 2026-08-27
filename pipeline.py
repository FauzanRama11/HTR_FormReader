"""
pipeline.py
===========
V10 -- mengatur URUTAN proses. Implementasi detail ada di modul lain:
  preprocessing.py     -> load, enhance, align (+coverage mask), section
                           transform, per-field ROI, ink-crop
  ocr.py                -> model OCR & batching (satu/dua kali panggilan/dok)
  postprocessing.py     -> normalisasi, choice/signature, tabel akhir, debug
  dynamic_extraction.py -> region OCR gabungan + asosiasi spasial token->field
  vlm.py                -> fallback independen (Qwen2-VL), TIDAK PERNAH default

Alur:
  load -> prepare+align (+coverage mask, section transforms via
  dynamic_extraction) -> region OCR gabungan (identity+placement, SATU
  batch) -> field tempat_tanggal_surat (V9.1 lama) -> normalisasi
  -> [FALLBACK 1] Paddle tight-crop utk field yg masih blank/review (SATU
  batch tambahan bila perlu) -> choice & signature (image-diff, coverage
  aware) -> [FALLBACK 2] VLM independent-read utk field decisive yg masih
  tidak yakin/geometri tidak yakin/mismatch vs referensi (MAKS 1 panggilan)
  -> arbitrase Paddle vs VLM -> tabel -> debug image.

Field yang boxnya jatuh di luar cakupan piksel sumber (foto terpotong)
ditandai "out_of_frame" sejak awal dan TIDAK PERNAH dikirim ke OCR/VLM atau
ditafsirkan sbg blank.
"""

import cv2

import preprocessing as prep
import ocr
import postprocessing as post
import dynamic_extraction as dynamic
import vlm
from comparison import COLUMN_FIELD_MAP, FIELD_DATA_TYPES, _normalize_for_compare

# Re-export supaya app.py bisa tetap memanggil pipeline.load_document /
# pipeline.TEMPLATE_PATH seperti sebelumnya.
load_document = prep.load_document
TEMPLATE_PATH = prep.TEMPLATE_PATH

# Field decisive BERTIPE TEKS yang bisa diverifikasi VLM secara independen
# (tenor/reward tetap pakai jalur deterministic image-diff dual-evidence yg
# sudah ada di postprocessing/comparison -- tidak diubah).
DECISIVE_TEXT_FIELDS = ("nama_nasabah", "nomor_rekening", "nominal_penempatan")
PADDLE_FALLBACK_STATUSES = {"review", "not_detected"}
VLM_RECOGNITION_TRIGGER_STATUSES = {"review", "not_detected"}
_REVERSE_COLUMN_MAP = {field: col for col, field in COLUMN_FIELD_MAP.items()}

# Matikan VLM fallback total lewat flag ini kalau model tidak tersedia di
# environment (mis. server tanpa GPU/torch) -- pipeline tetap jalan dgn
# Paddle saja, field decisive yg tidak terbaca cukup jatuh ke REVIEW seperti
# V9.x, bukan error keras.
VLM_FALLBACK_ENABLED = True


def _emit_progress(callback, step, percent, message, detail=None):
    if callback is None:
        return
    payload = {"step": step, "percent": int(max(0, min(100, percent))), "message": message}
    if detail is not None:
        payload["detail"] = detail
    try:
        callback(payload)
    except Exception:
        pass


def _padded_crop(image, bbox_px, shape, pad_ratio=0.30):
    x1, y1, x2, y2 = bbox_px
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio) + 4, int(h * pad_ratio) + 4
    nx1, ny1 = max(0, x1 - px), max(0, y1 - py)
    nx2, ny2 = min(shape[1], x2 + px), min(shape[0], y2 + py)
    if nx2 <= nx1 or ny2 <= ny1:
        return image[0:0, 0:0]
    return image[ny1:ny2, nx1:nx2].copy()


# ============================================================================
# PRIORITY 2 -- Paddle tight-crop fallback (SATU batch gabungan utk semua
# field yg unresolved, bukan satu panggilan per field).
# ============================================================================

def _run_paddle_fallback(raw_results, field_errors, aligned_img, shape):
    candidates = []
    for name, cfg in prep.FIELD_CONFIG.items():
        if name in field_errors or cfg["type"] == "optional_text":
            continue  # optional_text blank = legitimately not filled, bukan target fallback
        row = raw_results.get(name, {})
        if row.get("status") not in PADDLE_FALLBACK_STATUSES:
            continue
        bbox = row.get("roi_bbox")
        if not bbox:
            continue
        crop = _padded_crop(aligned_img, bbox, shape, pad_ratio=0.30)
        if crop.size == 0:
            continue
        scaled = cv2.resize(crop, None, fx=ocr.OCR_SCALE_UP, fy=ocr.OCR_SCALE_UP, interpolation=cv2.INTER_CUBIC)
        candidates.append((name, scaled))

    if not candidates:
        return {"used_fields": [], "attempted_fields": []}, 0

    fallback_by_field, calls = ocr.run_batched_ocr(candidates)
    used = []
    for name, item in fallback_by_field.items():
        if not item.get("raw"):
            continue
        candidate = post.build_text_field_result(name, prep.FIELD_CONFIG[name], item)
        if candidate["status"] == "read" and raw_results[name].get("status") != "read":
            candidate["source"] = "paddle_fallback_tight_crop"
            raw_results[name].update(candidate)
            used.append(name)
    return {"used_fields": used, "attempted_fields": [n for n, _ in candidates]}, calls


# ============================================================================
# PRIORITY 3 -- VLM fallback (Qwen2-VL, lazy-loaded sekali di vlm.py).
# TRIGGER, bukan default: recognition (Paddle blank/review/invalid),
# mismatch_verification (Paddle mismatch confident vs referensi -- referensi
# TIDAK PERNAH dikirim ke prompt), geometry (section transform tidak
# menemukan anchor valid sama sekali, TAPI dokumen memang terlihat -- bukan
# out_of_frame).
# ============================================================================

def _text_field_mismatches_reference(field_name, raw_results, reference_record):
    if not reference_record:
        return False
    column = _REVERSE_COLUMN_MAP.get(field_name)
    if not column:
        return False
    row = raw_results.get(field_name, {})
    if row.get("status") != "read":
        return False
    data_type = FIELD_DATA_TYPES.get(field_name, "text")
    doc_n = _normalize_for_compare(row.get("value"), data_type)
    ref_n = _normalize_for_compare(reference_record.get(column), data_type)
    return bool(doc_n) and bool(ref_n) and doc_n != ref_n


def _determine_vlm_targets(raw_results, section_meta, reference_record):
    """Return dict field_name -> reason ('recognition'/'mismatch_verification'
    /'geometry'). out_of_frame field TIDAK PERNAH masuk sini -- VLM tidak
    boleh menebak konten yang memang tidak difoto."""
    targets = {}
    for field_name in DECISIVE_TEXT_FIELDS:
        row = raw_results.get(field_name, {})
        status = row.get("status")
        if status == "out_of_frame":
            continue
        if status in VLM_RECOGNITION_TRIGGER_STATUSES:
            targets[field_name] = "recognition"
            continue
        if _text_field_mismatches_reference(field_name, raw_results, reference_record):
            targets[field_name] = "mismatch_verification"
            continue
        region_name = dynamic.FIELD_REGION_MAP.get(field_name, (None, None))[0]
        section = section_meta.get(region_name, {}) if region_name else {}
        if section.get("method") == "none" and status == "read":
            targets[field_name] = "geometry"
    return targets


def _arbitrate_vlm_result(field_name, reason, row, vlm_item):
    """Terapkan aturan arbitrase konservatif Paddle vs VLM (lihat handover
    V10 'Paddle vs VLM arbitration'). Memodifikasi `row` in-place."""
    vlm_status = vlm_item.get("status", "not_visible")
    vlm_value = vlm_item.get("value")
    data_type = FIELD_DATA_TYPES.get(field_name, "text")
    row["vlm_checked"] = True
    row["vlm_status"] = vlm_status
    row["vlm_value"] = vlm_value

    if vlm_status == "not_visible":
        return  # VLM juga tidak bisa melihatnya -- biarkan hasil Paddle apa adanya

    paddle_status = row.get("status")
    paddle_value = row.get("value")

    if reason == "recognition":
        if vlm_status == "readable" and vlm_value:
            candidate = post.build_text_field_result(
                field_name, prep.FIELD_CONFIG[field_name], {"raw": vlm_value, "confidence": None}
            )
            if candidate["status"] == "read":
                candidate["source"] = "vlm"
                candidate["vlm_status"] = vlm_status
                row.update(candidate)
        return  # uncertain -> biarkan review/not_detected apa adanya

    vlm_n = _normalize_for_compare(vlm_value, data_type) if vlm_value else ""
    paddle_n = _normalize_for_compare(paddle_value, data_type) if paddle_value else ""

    if reason == "mismatch_verification":
        if vlm_status == "readable" and vlm_n:
            if vlm_n == paddle_n:
                row["vlm_confirmed_mismatch"] = True  # TOLAK tetap berlaku (comparison.py)
            else:
                row["status"] = "recognition_conflict"
                row["reason"] = f"paddle='{paddle_value}' vs vlm='{vlm_value}' (mismatch_verification)"
        return

    if reason == "geometry":
        if paddle_status == "read":
            if vlm_status == "readable" and vlm_n and paddle_n and vlm_n != paddle_n:
                row["status"] = "recognition_conflict"
                row["reason"] = f"geometry_uncertain: paddle='{paddle_value}' vs vlm='{vlm_value}'"
            return
        if vlm_status == "readable" and vlm_value:
            candidate = post.build_text_field_result(
                field_name, prep.FIELD_CONFIG[field_name], {"raw": vlm_value, "confidence": None}
            )
            if candidate["status"] == "read":
                candidate["source"] = "vlm"
                candidate["vlm_status"] = vlm_status
                row.update(candidate)


def _run_vlm_fallback(raw_results, field_errors, aligned_img, shape, section_meta, reference_record):
    vlm_meta = {"used": False, "reason": {}, "calls": 0, "fields": [], "error": None}
    if not VLM_FALLBACK_ENABLED:
        vlm_meta["reason"] = {"skipped": "vlm_fallback_disabled"}
        return vlm_meta

    targets = _determine_vlm_targets(raw_results, section_meta, reference_record)
    if not targets:
        return vlm_meta

    field_names = list(targets.keys())
    images = [aligned_img]
    max_sides = [vlm.MAX_SIDE_FULL]
    for f in field_names:
        bbox = raw_results.get(f, {}).get("roi_bbox")
        if not bbox:
            continue
        crop = _padded_crop(aligned_img, bbox, shape, pad_ratio=0.40)
        if crop.size == 0:
            continue
        images.append(crop)
        max_sides.append(vlm.MAX_SIDE_DETAIL)

    try:
        vlm_results, _raw_text = vlm.extract_fields_independent(images, field_names, max_sides=max_sides)
    except Exception as exc:
        field_errors["vlm_fallback"] = str(exc)
        vlm_meta["error"] = str(exc)
        return vlm_meta

    vlm_meta.update({"used": True, "reason": targets, "calls": 1, "fields": field_names})
    for field_name, reason in targets.items():
        row = raw_results.get(field_name)
        if row is None:
            continue
        _arbitrate_vlm_result(field_name, reason, row, vlm_results.get(field_name, {"value": None, "status": "not_visible"}))
    return vlm_meta


def run_pipeline(filled_path, template_path=TEMPLATE_PATH, progress_callback=None, reference_record=None):
    """reference_record (OPSIONAL, V10): dict record spreadsheet, dipakai
    HANYA utk MEMUTUSKAN kapan VLM mismatch-verification dipicu -- nilainya
    TIDAK PERNAH dikirim ke prompt VLM. Pemanggil lama (tanpa argumen ini)
    tetap valid (default None -> mismatch_verification trigger tidak aktif,
    perilaku setara V9.x + fallback recognition/geometry saja)."""
    _emit_progress(progress_callback, "load", 8, "Memuat template dan dokumen")
    template_img = prep.load_document(template_path)
    filled_raw = prep.load_document(filled_path)

    _emit_progress(progress_callback, "align", 20, "Menyiapkan & menyelaraskan dokumen (bandingkan asli vs rectify)")
    aligned_img, _, alignment = prep.prepare_and_align(filled_path, filled_raw, template_img)
    prep_meta = alignment.pop("input_preparation", {"rectified": False, "reason": "not_needed"})
    # V10: coverage_mask (piksel yg benar-benar berasal dari sumber, bukan
    # letterbox/border hasil warp) -- dipakai deteksi foto terpotong/parsial.
    # Dipop dari alignment SEBELUM alignment dikirim ke JSON response (app.py).
    coverage_mask = alignment.pop("coverage_mask", None)

    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    shape = aligned_img.shape

    _emit_progress(
        progress_callback, "localize", 30,
        "Menghitung transform section (identity/placement) & region OCR gabungan",
    )
    raw_results, field_errors = {}, {}
    try:
        (dynamic_ocr, dynamic_roi_boxes, dynamic_regions_debug, dynamic_calls,
         dynamic_field_statuses, dynamic_section_meta) = dynamic.extract_dynamic_fields(
            aligned_img, template_gray, aligned_gray, shape, coverage_mask=coverage_mask
        )
    except Exception as exc:
        dynamic_ocr, dynamic_roi_boxes, dynamic_regions_debug, dynamic_calls = {}, {}, {}, 0
        dynamic_field_statuses, dynamic_section_meta = {}, {}
        field_errors["dynamic_extraction"] = str(exc)

    _emit_progress(
        progress_callback, "ocr_dynamic", 42, f"OCR region gabungan selesai ({dynamic_calls}x panggilan)",
        {"ocr_calls_dynamic": dynamic_calls, "regions": list(dynamic_regions_debug.keys()),
         "section_meta": dynamic_section_meta},
    )

    for name in dynamic.FIELD_REGION_MAP:
        roi_bbox = dynamic_roi_boxes.get(name) or prep.norm_bbox_to_px(prep.FIELD_CONFIG[name]["value_bbox"], shape)
        raw_results[name] = {"roi_bbox": roi_bbox, "roi_source": "semantic_region_dynamic"}

    pending_crops = []
    name, cfg = "tempat_tanggal_surat", prep.FIELD_CONFIG["tempat_tanggal_surat"]
    try:
        template_px, target_px, roi_source, similarity = prep.resolve_roi(template_gray, aligned_gray, cfg, shape)
        if coverage_mask is not None and prep.is_out_of_frame(coverage_mask, target_px):
            raw_results[name] = {
                "roi_bbox": target_px, "roi_source": "out_of_frame", "change_ratio": 0.0,
                **post.build_out_of_frame_result(),
            }
        else:
            crop, change_ratio = prep.extract_handwriting_crop(
                template_img, aligned_img, template_gray, template_px, target_px, shape, field_name=name
            )
            raw_results[name] = {
                "roi_bbox": target_px, "roi_source": roi_source,
                "anchor_similarity": similarity, "change_ratio": change_ratio,
            }
            if crop is not None:
                scaled = cv2.resize(crop, None, fx=ocr.OCR_SCALE_UP, fy=ocr.OCR_SCALE_UP, interpolation=cv2.INTER_CUBIC)
                pending_crops.append((name, scaled))
    except Exception as exc:
        field_errors[name] = str(exc)
        raw_results[name] = {
            "roi_bbox": prep.norm_bbox_to_px(cfg["value_bbox"], shape),
            "roi_source": "error", "change_ratio": 0.0, "status": "error", "error": str(exc),
        }

    _emit_progress(
        progress_callback, "ocr_batch", 55, f"Membaca sisa field (device={ocr.OCR_DEVICE})",
        {"fields_with_content": len(pending_crops) + len(dynamic_ocr), "total_fields": len(prep.FIELD_CONFIG)},
    )
    ocr_by_field, ocr_calls = ocr.run_batched_ocr(pending_crops)
    ocr_by_field.update(dynamic_ocr)
    ocr_calls += dynamic_calls

    _emit_progress(progress_callback, "normalize", 65, "Normalisasi & validasi hasil OCR")
    for name, cfg in prep.FIELD_CONFIG.items():
        if name in field_errors:
            continue
        if raw_results.get(name, {}).get("status") == "out_of_frame":
            continue  # tempat_tanggal_surat sudah diset out_of_frame di atas -- jangan ditimpa
        if name in dynamic_field_statuses:
            raw_results[name].update(post.build_out_of_frame_result())
            continue
        raw_results[name].update(post.build_text_field_result(name, cfg, ocr_by_field.get(name)))

    # -- PRIORITY 2: Paddle tight-crop fallback (SATU batch tambahan) -------
    _emit_progress(progress_callback, "paddle_fallback", 72, "Fallback Paddle (crop rapat) utk field belum terbaca")
    paddle_fallback_meta, paddle_fallback_calls = _run_paddle_fallback(raw_results, field_errors, aligned_img, shape)
    ocr_calls += paddle_fallback_calls

    _emit_progress(progress_callback, "choices", 80, "Mendeteksi pilihan tenor dan reward")
    groups, choice_raw = post.process_choices(
        template_img, aligned_img, template_gray, aligned_gray, shape, coverage_mask=coverage_mask
    )
    raw_results.update(choice_raw)
    choice_groups = {
        "tenor_penempatan": dict(groups["tenor_penempatan"]),
        "bentuk_reward": dict(groups["bentuk_reward"]),
    }

    if groups["tenor_penempatan"].get("status") != "out_of_frame":
        groups["tenor_penempatan"] = {**groups["tenor_penempatan"], **post.resolve_tenor_source(groups, raw_results)}
    if groups["bentuk_reward"].get("status") != "out_of_frame":
        groups["bentuk_reward"] = {**groups["bentuk_reward"], **post.resolve_bentuk_reward(groups, raw_results)}

    _emit_progress(progress_callback, "signatures", 86, "Mendeteksi area tanda tangan")
    raw_results.update(post.process_signatures(
        template_img, aligned_img, template_gray, aligned_gray, shape, coverage_mask=coverage_mask
    ))

    # -- PRIORITY 3: VLM fallback (fallback dari fallback, MAKS 1 panggilan) -
    _emit_progress(progress_callback, "vlm_fallback", 90, "Mengevaluasi kebutuhan verifikasi VLM (fallback)")
    vlm_meta = _run_vlm_fallback(raw_results, field_errors, aligned_img, shape, dynamic_section_meta, reference_record)

    _emit_progress(progress_callback, "format", 93, "Menyusun hasil ekstraksi")
    final_results = dict(raw_results)
    final_results["tenor_penempatan"] = groups["tenor_penempatan"]
    final_results["bentuk_reward"] = groups["bentuk_reward"]
    fields_table = post.build_fields_table(final_results, post.FIELD_TYPES)

    _emit_progress(progress_callback, "debug", 96, "Membuat visualisasi ROI (+ region OCR & token)")
    debug_results = {n: r for n, r in raw_results.items() if isinstance(r, dict) and r.get("roi_bbox")}
    debug_template = post.draw_debug(
        template_img, debug_results, choice_raw, template_mode=True, regions=dynamic_regions_debug
    )
    debug_document = post.draw_debug(
        aligned_img, debug_results, choice_raw, template_mode=False, regions=dynamic_regions_debug
    )

    _emit_progress(progress_callback, "pipeline_done", 98, "Pipeline OCR selesai")
    return {
        "alignment": alignment,
        "input_preparation": prep_meta,
        "fields": fields_table,
        "groups": groups,
        "choice_groups": choice_groups,
        "raw_results": raw_results,
        "field_errors": field_errors,
        "debug_images": {"template": debug_template, "document": debug_document},
        "dynamic_regions_debug": dynamic_regions_debug,
        "ocr_meta": {
            "strategy": "v10_section_transform_coverage_paddle_vlm_fallback",
            "ocr_device": ocr.OCR_DEVICE,
            "ocr_calls": ocr_calls,
            "ocr_calls_dynamic": dynamic_calls,
            "ocr_calls_paddle_fallback": paddle_fallback_calls,
            "fields_ocr_ed": len(pending_crops) + len(dynamic_ocr),
            "total_fields": len(prep.FIELD_CONFIG),
            "paddle_fallback": paddle_fallback_meta,
            "vlm_fallback": vlm_meta,
            "out_of_frame_fields": [f for f, s in dynamic_field_statuses.items()] + (
                ["tempat_tanggal_surat"] if raw_results.get("tempat_tanggal_surat", {}).get("status") == "out_of_frame" else []
            ),
            "section_meta": dynamic_section_meta,
        },
    }
