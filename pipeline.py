"""
pipeline.py
===========
V12 -- mengatur URUTAN proses. Implementasi detail ada di modul lain:
  preprocessing.py     -> load, enhance, align (+coverage mask), section
                           transform, per-field ROI, ink-crop
  ocr.py                -> model OCR & batching (satu/dua kali panggilan/dok)
  postprocessing.py     -> normalisasi, choice/signature, tabel akhir, debug
  dynamic_extraction.py -> region OCR gabungan + asosiasi spasial token->field
  vlm.py                -> fallback independen (Qwen2-VL), TIDAK PERNAH default

Alur V12 (target flow, lihat juga HANDOVER):
  OCR Primary (region OCR gabungan + tempat_tanggal_surat + choice ink-diff
  tenor/reward + signature ink-diff -- SEMUA deterministik, non-VLM)
    -> Confident?  ya -> pakai hasil OCR Primary
                   tidak (not_detected/uncertain/review/low_confidence/
                          geometry_uncertain/invalid/conflict)
    -> [FALLBACK 1] Full-Page VLM (SATU panggilan, HANYA halaman penuh yg
       sudah di-align/preprocess, TIDAK PERNAH crop ROI -- lihat
       _run_vlm_fullpage_fallback)
    -> Confident?  ya -> pakai hasil VLM (source="vlm_fullpage")
                   tidak
    -> [FALLBACK 2] ROI/Template crop (KONFIGURASI ROI YANG SUDAH ADA,
       prep.FIELD_CONFIG -- lihat _run_roi_template_fallback, HANYA dipanggil
       kalau Full-Page VLM juga masih tidak yakin, BUKAN fallback pertama lagi)
    -> [FALLBACK 3] second-pass OCR/VLM di ATAS crop ROI/template yg sama
       (lihat _run_roi_second_pass_vlm) utk field yg MASIH gagal
    -> normalisasi/tenor-calculation/perbandingan Form/keputusan akhir
       (Lanjut Proses/Tolak) SELALU di Python (postprocessing.py/comparison.py)
       -- VLM HANYA membaca/menafsirkan konten dokumen, TIDAK PERNAH
       memutuskan.

Setiap field/grup menyimpan "source" (ocr_primary / vlm_fullpage /
roi_template / roi_second_pass_vlm) & "notes" (alasan fallback dipakai,
lihat _note_* di bawah) supaya bisa dilacak jalur mana yang menghasilkan
nilai final -- lihat _apply_default_sources.

Field yang boxnya jatuh di luar cakupan piksel sumber (foto terpotong)
ditandai "out_of_frame" sejak awal dan TIDAK PERNAH dikirim ke OCR/VLM atau
ditafsirkan sbg blank.
"""

import re

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

# Field TEKS decisive yang ikut alur fallback V12 penuh (Full-Page VLM ->
# ROI/template -> second-pass VLM). "unit_kerja_pengelola_rekening"
# ditambahkan (V12) supaya sesuai requirement field list Full-Page VLM.
TEXT_FALLBACK_FIELDS = (
    "nama_nasabah", "nomor_rekening", "unit_kerja_pengelola_rekening", "nominal_penempatan",
)
# Field decisive BERTIPE TEKS yg diverifikasi VLM thd REFERENSI spreadsheet
# (mismatch_verification, lihat _run_vlm_reference_check) -- fitur terpisah
# dari fallback ekstraksi V12 di atas, dipertahankan dari V10.
DECISIVE_TEXT_FIELDS = ("nama_nasabah", "nomor_rekening", "nominal_penempatan")
CHOICE_FALLBACK_GROUPS = ("tenor_penempatan", "bentuk_reward")
SIGNATURE_FALLBACK_FIELDS = ("signature_nasabah", "signature_atasan")

# Nama field internal -> nama field level-spesifikasi dipakai di prompt/
# response Full-Page VLM (lihat vlm.extract_fields_fullpage & requirement #3).
VLM_FULLPAGE_FIELD_MAP = {
    "nama_nasabah": "nama_nasabah",
    "nomor_rekening": "nomor_rekening",
    "unit_kerja_pengelola_rekening": "unit_kerja",
    "nominal_penempatan": "nominal_penempatan",
    "tenor_penempatan": "tenor",
    "bentuk_reward": "bentuk_reward",
    "signature_nasabah": "signature_nasabah",
    "signature_atasan": "signature_unit_kerja",
}
# tanggal_mulai/tanggal_selesai: BUKTI SEKUNDER SAJA (requirement #3) --
# hanya dikirim ke Full-Page VLM kalau "tenor" jadi target, dipakai utk
# derive/validate tenor lewat postprocessing.derive_tenor_from_range kalau
# pembacaan langsung "tenor" tidak yakin/tidak ada.
SECONDARY_TENOR_FIELDS = ("tanggal_mulai", "tanggal_selesai")

# V11/V12: threshold KHUSUS pemicu fallback (configurable), TERPISAH dari
# postprocessing.OCR_REVIEW_THRESHOLD (0.60, dipakai utk label "review" di
# UI/Notes) -- confidence di bawah ini dicatat sbg alasan "low_confidence"
# di notes (masih tetap memicu fallback, requirement #2).
FALLBACK_LOW_CONFIDENCE = 0.45
_REVERSE_COLUMN_MAP = {field: col for col, field in COLUMN_FIELD_MAP.items()}

# Matikan VLM fallback total lewat flag ini kalau model tidak tersedia di
# environment (mis. server tanpa GPU/torch) -- pipeline tetap jalan dgn OCR
# Primary + ROI/template saja, field yg tidak terbaca cukup jatuh ke REVIEW
# seperti sebelumnya, bukan error keras.
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


def _padded_crop(image, bbox_px, shape, field_name=None, pad_ratio=0.30):
    x1, y1, x2, y2 = bbox_px
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio) + 4, int(h * pad_ratio) + 4
    nx1, ny1 = max(0, x1 - px), max(0, y1 - py)
    nx2, ny2 = min(shape[1], x2 + px), min(shape[0], y2 + py)
    if field_name:
        # V11 #3: padding 30% simetris bisa menembus label/field tetangga --
        # kunci ulang sisi atas/bawah ke batas field tetangga (mekanisme sama
        # dgn preprocessing._clamp_field_box_y yg dipakai ROI utama), supaya
        # crop fallback tidak pernah membaca baris label/value sebelahnya.
        nx1, ny1, nx2, ny2 = prep._clamp_field_box_y((nx1, ny1, nx2, ny2), field_name, shape)
    if nx2 <= nx1 or ny2 <= ny1:
        return image[0:0, 0:0]
    return image[ny1:ny2, nx1:nx2].copy()


# ============================================================================
# NOTES HELPERS -- teks "notes" WAJIB menyatakan eksplisit fallback apa yang
# dipakai (requirement #6), dipetakan dari alasan trigger (reason) ke label
# yang mudah dibaca.
# ============================================================================

_OCR_PRIMARY_REASON_TEXT = {
    "not_detected": "OCR primary failed",
    "invalid": "OCR primary invalid",
    "low_confidence": "OCR primary low-confidence",
    "uncertain": "OCR primary uncertain",
    "review": "OCR primary uncertain",
    "geometry_uncertain": "OCR primary geometry-uncertain",
    "conflict": "OCR primary evidence conflicting",
}


def _note_ocr_primary():
    return "OCR primary used (confident, no fallback needed)"


def _note_vlm_fullpage(reason):
    prefix = _OCR_PRIMARY_REASON_TEXT.get(reason, "OCR primary uncertain")
    return f"{prefix} \u2192 Full-Page VLM fallback used"


def _note_roi_template():
    return "Full-Page VLM uncertain \u2192 ROI template fallback used"


def _note_roi_second_pass():
    return "ROI template \u2192 second-pass VLM used"


# ============================================================================
# FALLBACK-TRIGGER HELPERS (requirement #2): not_detected / uncertain /
# review / low_confidence / geometry_uncertain / invalid (teks); review /
# conflict (choice tenor/reward); uncertain (signature). out_of_frame TIDAK
# PERNAH memicu fallback apa pun (area memang tidak difoto).
# ============================================================================

def _text_fallback_reason(name, row, section_meta):
    """Return alasan (str) kalau field TEKS `name` masih butuh fallback,
    None kalau OCR Primary sudah cukup dipercaya."""
    status = row.get("status")
    if status == "not_detected":
        return "not_detected"
    if status == "review":
        if row.get("valid") is False:
            return "invalid"
        confidence = row.get("confidence")
        if confidence is None or confidence < FALLBACK_LOW_CONFIDENCE:
            return "low_confidence"
        return "uncertain"  # V12: confidence menengah TETAP diverifikasi VLM (bukan ROI langsung)
    if status == "read":
        region_name = dynamic.FIELD_REGION_MAP.get(name, (None, None))[0]
        section = section_meta.get(region_name, {}) if region_name else {}
        if section.get("method") == "none":
            return "geometry_uncertain"
    return None


def _choice_fallback_reason(group):
    status = group.get("status")
    return status if status in ("review", "conflict") else None


def _signature_fallback_reason(row):
    return "uncertain" if row.get("status") == "uncertain" else None


# ============================================================================
# FALLBACK 1 (V12) -- Full-Page VLM. SATU panggilan, HANYA halaman penuh yg
# sudah di-align/preprocess (TIDAK PERNAH crop ROI di tahap ini). Dipanggil
# SETELAH OCR Primary (teks + choice ink-diff tenor/reward + signature ink-
# diff sudah selesai dihitung) & SEBELUM ROI/template fallback (requirement
# #2/#5: ROI bukan lagi fallback pertama).
# ============================================================================

_TENOR_TEXT_RE = re.compile(r"([136])\s*bulan")


def _parse_tenor_text(value):
    if not value:
        return None
    match = _TENOR_TEXT_RE.search(str(value).lower())
    return int(match.group(1)) if match else None


def _apply_vlm_text_result(name, raw_results, vlm_results, note):
    row = raw_results.setdefault(name, {})
    spec_name = VLM_FULLPAGE_FIELD_MAP[name]
    item = vlm_results.get(spec_name, {"value": None, "status": "not_visible"})
    row["vlm_fullpage_checked"] = True
    row["vlm_fullpage_status"] = item.get("status")
    row["vlm_fullpage_value"] = item.get("value")
    if item.get("status") == "readable" and item.get("value"):
        candidate = post.build_text_field_result(
            name, prep.FIELD_CONFIG[name], {"raw": item["value"], "confidence": None}
        )
        if candidate["status"] == "read":
            candidate["source"] = "vlm_fullpage"
            candidate["notes"] = note
            row.update(candidate)
            return
    # VLM juga belum yakin/gagal -- simpan percobaan, biarkan ROI/template
    # fallback (langkah berikutnya) yang mencoba selanjutnya.
    row.setdefault("notes", note)


def _apply_vlm_choice_result(group_name, groups, vlm_results, note):
    spec_name = VLM_FULLPAGE_FIELD_MAP[group_name]
    item = vlm_results.get(spec_name, {"value": None, "status": "not_visible"})
    group = groups.setdefault(group_name, {})
    group["vlm_fullpage_checked"] = True
    group["vlm_fullpage_status"] = item.get("status")
    group["vlm_fullpage_value"] = item.get("value")

    if group_name == "tenor_penempatan":
        value = None
        if item.get("status") == "readable":
            value = _parse_tenor_text(item.get("value"))
        if value is None:
            # bukti langsung tidak yakin/tidak ada -> coba derivasi dari bukti
            # SEKUNDER tanggal_mulai/tanggal_selesai (requirement #3).
            mulai = (vlm_results.get("tanggal_mulai") or {}).get("value")
            selesai = (vlm_results.get("tanggal_selesai") or {}).get("value")
            if mulai and selesai:
                value, _reason = post.derive_tenor_from_range(f"{mulai} - {selesai}")
        if value in post.TENOR_OPTIONS:
            group.update({
                "value": value, "status": "detected", "source": "vlm_fullpage",
                "reason": "vlm_fullpage_independent_read", "notes": note,
            })
        else:
            group.setdefault("notes", note)
        return

    # bentuk_reward
    if item.get("status") == "readable" and item.get("value"):
        text = str(item["value"]).lower()
        value = "non_tunai" if "non" in text else ("tunai" if "tunai" in text else None)
        if value:
            group.update({
                "value": value, "status": "detected", "source": "vlm_fullpage",
                "reason": "vlm_fullpage_independent_read", "notes": note,
            })
            return
    group.setdefault("notes", note)


def _apply_vlm_signature_result(name, raw_results, vlm_results, note):
    spec_name = VLM_FULLPAGE_FIELD_MAP[name]
    item = vlm_results.get(spec_name, {"value": None, "status": "not_visible"})
    row = raw_results.setdefault(name, {})
    row["vlm_fullpage_checked"] = True
    row["vlm_fullpage_status"] = item.get("status")
    row["vlm_fullpage_value"] = item.get("value")
    if item.get("status") != "readable" or not item.get("value"):
        row.setdefault("notes", note)
        return
    text = str(item["value"]).lower()
    if "kosong" in text or "absent" in text or "tidak ada" in text:
        state = "absent"
    elif "ada" in text or "present" in text or "tanda" in text:
        state = "present"
    else:
        row.setdefault("notes", note)
        return
    row.update({
        "present": state == "present",
        "value": {"present": "Ada tanda tangan", "absent": "Kosong"}[state],
        "status": state,
        "source": "vlm_fullpage",
        "reason": "vlm_fullpage_independent_read",
        "notes": note,
    })


def _run_vlm_fullpage_fallback(raw_results, groups, field_errors, aligned_img, section_meta):
    """FALLBACK 1 (V12) -- lihat docstring modul & blok komentar di atas."""
    meta = {"used": False, "targets": {}, "calls": 0, "error": None}
    if not VLM_FALLBACK_ENABLED:
        meta["skipped"] = "vlm_fallback_disabled"
        return meta

    targets = {}
    for name in TEXT_FALLBACK_FIELDS:
        row = raw_results.get(name, {})
        if row.get("status") == "out_of_frame":
            continue
        reason = _text_fallback_reason(name, row, section_meta)
        if reason:
            targets[name] = reason
    for group_name in CHOICE_FALLBACK_GROUPS:
        group = groups.get(group_name, {})
        if group.get("status") == "out_of_frame":
            continue
        reason = _choice_fallback_reason(group)
        if reason:
            targets[group_name] = reason
    for name in SIGNATURE_FALLBACK_FIELDS:
        row = raw_results.get(name, {})
        if row.get("status") == "out_of_frame":
            continue
        reason = _signature_fallback_reason(row)
        if reason:
            targets[name] = reason

    if not targets:
        return meta

    spec_fields = [VLM_FULLPAGE_FIELD_MAP[n] for n in targets]
    if "tenor_penempatan" in targets:
        spec_fields += list(SECONDARY_TENOR_FIELDS)

    try:
        vlm_results, _raw_text = vlm.extract_fields_fullpage(aligned_img, spec_fields)
    except Exception as exc:
        field_errors["vlm_fullpage_fallback"] = str(exc)
        meta["error"] = str(exc)
        return meta

    meta.update({"used": True, "targets": dict(targets), "calls": 1})
    for name, reason in targets.items():
        note = _note_vlm_fullpage(reason)
        if name in TEXT_FALLBACK_FIELDS:
            _apply_vlm_text_result(name, raw_results, vlm_results, note)
        elif name in CHOICE_FALLBACK_GROUPS:
            _apply_vlm_choice_result(name, groups, vlm_results, note)
        elif name in SIGNATURE_FALLBACK_FIELDS:
            _apply_vlm_signature_result(name, raw_results, vlm_results, note)
    return meta


# ============================================================================
# FALLBACK 2 (V12) -- ROI/Template crop (Paddle tight-crop, KONFIGURASI ROI
# YANG SUDAH ADA di prep.FIELD_CONFIG -- TIDAK diubah). Dipanggil HANYA utk
# field TEKS yang MASIH butuh fallback setelah Full-Page VLM (requirement #5:
# ROI bukan lagi fallback pertama). SATU batch OCR gabungan, bukan 1
# panggilan per field.
# ============================================================================

def _run_roi_template_fallback(raw_results, field_errors, aligned_img, shape, section_meta):
    candidates, reasons = [], {}
    for name in TEXT_FALLBACK_FIELDS:
        if name in field_errors:
            continue
        row = raw_results.get(name, {})
        reason = _text_fallback_reason(name, row, section_meta)
        if not reason:
            continue
        bbox = row.get("roi_bbox")
        if not bbox:
            continue
        crop = _padded_crop(aligned_img, bbox, shape, field_name=name, pad_ratio=0.30)
        if crop.size == 0:
            continue
        scaled = cv2.resize(crop, None, fx=ocr.OCR_SCALE_UP, fy=ocr.OCR_SCALE_UP, interpolation=cv2.INTER_CUBIC)
        candidates.append((name, scaled))
        reasons[name] = reason

    if not candidates:
        return {"used_fields": [], "attempted_fields": []}, 0

    fallback_by_field, calls = ocr.run_batched_ocr(candidates)
    used = []
    for name, item in fallback_by_field.items():
        if not item.get("raw"):
            continue
        candidate = post.build_text_field_result(name, prep.FIELD_CONFIG[name], item)
        # Tidak boleh overwrite hasil yg sudah "read" (mis. dari Full-Page
        # VLM sebelumnya) kecuali candidate ini memang mencapai "read".
        if candidate["status"] == "read" and raw_results[name].get("status") != "read":
            candidate["source"] = "roi_template"
            candidate["notes"] = _note_roi_template()
            raw_results[name].update(candidate)
            used.append(name)
    return {"used_fields": used, "attempted_fields": [n for n, _ in candidates]}, calls


# ============================================================================
# FALLBACK 3 (V12) -- second-pass OCR/VLM di ATAS crop ROI/template yang SAMA
# (bukan halaman penuh lagi), utk field yang MASIH gagal setelah Full-Page
# VLM & ROI/template fallback (requirement #5, langkah terakhir sebelum
# validasi Python). Dipanggil per-field (jumlah field yg sampai ke sini
# biasanya sedikit) supaya korespondensi crop<->field selalu jelas.
# ============================================================================

def _run_roi_second_pass_vlm(raw_results, field_errors, aligned_img, shape, section_meta):
    meta = {"used": False, "fields": [], "attempted_fields": [], "calls": 0, "error": None}
    if not VLM_FALLBACK_ENABLED:
        meta["skipped"] = "vlm_fallback_disabled"
        return meta

    targets = {}
    for name in TEXT_FALLBACK_FIELDS:
        row = raw_results.get(name, {})
        if row.get("status") == "out_of_frame":
            continue
        reason = _text_fallback_reason(name, row, section_meta)
        if reason:
            targets[name] = reason
    if not targets:
        return meta

    used, attempted, calls = [], [], 0
    for name in targets:
        bbox = raw_results.get(name, {}).get("roi_bbox")
        if not bbox:
            continue
        crop = _padded_crop(aligned_img, bbox, shape, field_name=name, pad_ratio=0.40)
        if crop.size == 0:
            continue
        attempted.append(name)
        try:
            vlm_results, _raw_text = vlm.extract_fields_independent(
                [crop], [name], max_sides=[vlm.MAX_SIDE_DETAIL]
            )
            calls += 1
        except Exception as exc:
            field_errors[f"vlm_roi_second_pass_{name}"] = str(exc)
            continue
        item = vlm_results.get(name, {"value": None, "status": "not_visible"})
        if item.get("status") == "readable" and item.get("value"):
            candidate = post.build_text_field_result(
                name, prep.FIELD_CONFIG[name], {"raw": item["value"], "confidence": None}
            )
            if candidate["status"] == "read":
                candidate["source"] = "roi_second_pass_vlm"
                candidate["notes"] = _note_roi_second_pass()
                raw_results[name].update(candidate)
                used.append(name)
                continue
        raw_results[name]["notes"] = _note_roi_second_pass() + " (masih perlu review manual)"

    meta.update({"used": bool(used), "fields": used, "attempted_fields": attempted, "calls": calls})
    return meta


# ============================================================================
# Verifikasi VLM thd REFERENSI spreadsheet (dipertahankan dari V10, TERPISAH
# dari fallback ekstraksi V12 di atas -- lihat modul docstring). Hanya
# mismatch_verification: field TEKS yang sudah "read" (baik lewat OCR
# Primary MAUPUN salah satu fallback V12 di atas) tapi nilainya BERBEDA dari
# referensi -- dicek ulang independen thd halaman+crop, TANPA referensi
# pernah masuk prompt VLM.
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


def _determine_reference_check_targets(raw_results, reference_record):
    """Return dict field_name -> 'mismatch_verification'."""
    targets = {}
    if not reference_record:
        return targets
    for field_name in DECISIVE_TEXT_FIELDS:
        row = raw_results.get(field_name, {})
        if row.get("status") == "out_of_frame":
            continue
        if _text_field_mismatches_reference(field_name, raw_results, reference_record):
            targets[field_name] = "mismatch_verification"
    return targets


def _arbitrate_reference_check(field_name, row, vlm_item):
    """Terapkan aturan arbitrase konservatif: kalau VLM independen SEPAKAT
    dgn nilai dokumen (bukan referensi) -> mismatch dianggap dikonfirmasi
    (TOLAK tetap berlaku di comparison.py). Kalau VLM TIDAK sepakat dgn nilai
    dokumen -> ambigu, jadikan REVIEW (bukan auto TOLAK/OK). Memodifikasi
    `row` in-place."""
    vlm_status = vlm_item.get("status", "not_visible")
    vlm_value = vlm_item.get("value")
    data_type = FIELD_DATA_TYPES.get(field_name, "text")
    row["vlm_reference_check_status"] = vlm_status
    row["vlm_reference_check_value"] = vlm_value
    if vlm_status != "readable" or not vlm_value:
        return  # VLM juga tidak bisa memastikan -- biarkan hasil dokumen apa adanya

    doc_value = row.get("value")
    vlm_n = _normalize_for_compare(vlm_value, data_type)
    doc_n = _normalize_for_compare(doc_value, data_type)
    if vlm_n == doc_n:
        row["vlm_confirmed_mismatch"] = True  # TOLAK tetap berlaku (comparison.py)
        row["notes"] = f"{row.get('notes', '')} | VLM mengonfirmasi mismatch vs referensi".strip(" |")
    else:
        row["status"] = "recognition_conflict"
        row["reason"] = f"dokumen='{doc_value}' vs vlm='{vlm_value}' (mismatch_verification)"
        row["notes"] = "Nilai dokumen vs referensi tidak sesuai \u2192 VLM independen tidak sepakat, perlu review manual"


def _run_vlm_reference_check(raw_results, field_errors, aligned_img, shape, reference_record):
    meta = {"used": False, "targets": {}, "calls": 0, "error": None}
    if not VLM_FALLBACK_ENABLED:
        meta["skipped"] = "vlm_fallback_disabled"
        return meta

    targets = _determine_reference_check_targets(raw_results, reference_record)
    if not targets:
        return meta

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
        field_errors["vlm_reference_check"] = str(exc)
        meta["error"] = str(exc)
        return meta

    meta.update({"used": True, "targets": dict(targets), "calls": 1})
    for field_name in targets:
        row = raw_results.get(field_name)
        if row is None:
            continue
        _arbitrate_reference_check(
            field_name, row, vlm_results.get(field_name, {"value": None, "status": "not_visible"})
        )
    return meta


# ============================================================================
# Default "source"/"notes" (requirement #6): SETIAP field/grup harus
# menyimpan jalur mana yang menghasilkan nilai final -- kalau tidak ada
# fallback yang dipakai/mengubah apa pun di atas, defaultnya "ocr_primary".
# ============================================================================

def _apply_default_sources(raw_results, groups):
    for name in prep.FIELD_CONFIG:
        row = raw_results.get(name)
        if not isinstance(row, dict) or row.get("status") == "out_of_frame":
            continue
        row.setdefault("source", "ocr_primary")
        row.setdefault("notes", _note_ocr_primary())
    for name in prep.SIGNATURE_CONFIG:
        row = raw_results.get(name)
        if not isinstance(row, dict) or row.get("status") == "out_of_frame":
            continue
        row.setdefault("source", "ocr_primary")
        row.setdefault("notes", _note_ocr_primary())
    for group_name in CHOICE_FALLBACK_GROUPS:
        group = groups.get(group_name)
        if not isinstance(group, dict) or group.get("status") == "out_of_frame":
            continue
        # "source" grup choice SUDAH selalu diisi oleh
        # resolve_tenor_source/resolve_bentuk_reward (choice/rentang_tenor/
        # evidence_*/none/vlm_fullpage) -- HANYA "notes" yang perlu default.
        group.setdefault("notes", _note_ocr_primary())


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

    # -- OCR Primary (lanjutan): choice ink-diff (tenor/reward) + signature
    # ink-diff -- SEMUA deterministik/non-VLM, SAMA seperti sebelumnya. Ini
    # WAJIB selesai duluan supaya V12 tahu status "primary" LENGKAP (teks +
    # choice + signature) SEBELUM memutuskan field mana yang perlu fallback. -
    _emit_progress(progress_callback, "choices", 70, "Mendeteksi pilihan tenor dan reward (OCR Primary)")
    groups, choice_raw = post.process_choices(
        template_img, aligned_img, template_gray, aligned_gray, shape, coverage_mask=coverage_mask
    )
    raw_results.update(choice_raw)
    if groups["tenor_penempatan"].get("status") != "out_of_frame":
        groups["tenor_penempatan"] = {
            **groups["tenor_penempatan"],
            **post.resolve_tenor_source(
                groups,
                raw_results,
            ),
        }

    if groups["bentuk_reward"].get("status") != "out_of_frame":
        groups["bentuk_reward"] = {
            **groups["bentuk_reward"],
            **post.resolve_bentuk_reward(
                groups,
                raw_results,
            ),
        }

    choice_groups = {
        "tenor_penempatan": dict(groups["tenor_penempatan"]),
        "bentuk_reward": dict(groups["bentuk_reward"]),
    }
    
    _emit_progress(progress_callback, "signatures", 74, "Mendeteksi area tanda tangan (OCR Primary)")
    raw_results.update(post.process_signatures(
        template_img, aligned_img, template_gray, aligned_gray, shape, coverage_mask=coverage_mask
    ))

    # -- FALLBACK 1 (V12): Full-Page VLM -- HANYA halaman penuh, TIDAK PERNAH
    # crop ROI di tahap ini. Dipanggil kalau ADA field (teks/choice/
    # signature) yg statusnya not_detected/uncertain/review/low_confidence/
    # geometry_uncertain/invalid/conflict (requirement #2/#5). -------------
    _emit_progress(progress_callback, "vlm_fullpage_fallback", 80, "Fallback Full-Page VLM (halaman penuh) utk field belum yakin")
    vlm_fullpage_meta = _run_vlm_fullpage_fallback(raw_results, groups, field_errors, aligned_img, dynamic_section_meta)

    # -- FALLBACK 2 (V12): ROI/Template crop (Paddle tight-crop, konfigurasi
    # ROI yang sudah ada) -- HANYA utk field teks yg MASIH belum yakin
    # setelah Full-Page VLM. --------------------------------------------
    _emit_progress(progress_callback, "roi_template_fallback", 86, "Fallback ROI/template (crop rapat) utk field masih belum yakin")
    roi_template_meta, roi_template_calls = _run_roi_template_fallback(
        raw_results, field_errors, aligned_img, shape, dynamic_section_meta
    )
    ocr_calls += roi_template_calls

    # -- FALLBACK 3 (V12): second-pass OCR/VLM di atas crop ROI/template yg
    # sama, utk field yg MASIH gagal setelah fallback 1 & 2. ---------------
    _emit_progress(progress_callback, "roi_second_pass_vlm", 89, "Second-pass VLM di atas crop ROI/template utk field masih gagal")
    roi_second_pass_meta = _run_roi_second_pass_vlm(raw_results, field_errors, aligned_img, shape, dynamic_section_meta)

    # -- Verifikasi VLM thd referensi spreadsheet (fitur terpisah, lihat
    # docstring _run_vlm_reference_check) -- TIDAK PERNAH mempengaruhi
    # keputusan akhir langsung, hanya menandai status utk direview. --------
    _emit_progress(progress_callback, "vlm_reference_check", 91, "Mengevaluasi kebutuhan verifikasi VLM thd referensi")
    vlm_reference_meta = _run_vlm_reference_check(raw_results, field_errors, aligned_img, shape, reference_record)

    # requirement #6: SETIAP field/grup WAJIB catat "source"/"notes", default
    # "ocr_primary" kalau tidak ada fallback yang mengubah nilai apa pun. ---
    _apply_default_sources(raw_results, groups)

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
            "strategy": "v12_ocr_primary_vlm_fullpage_roi_template_second_pass",
            "ocr_device": ocr.OCR_DEVICE,
            "ocr_calls": ocr_calls,
            "ocr_calls_dynamic": dynamic_calls,
            "ocr_calls_roi_template_fallback": roi_template_calls,
            "fields_ocr_ed": len(pending_crops) + len(dynamic_ocr),
            "total_fields": len(prep.FIELD_CONFIG),
            # Nama lama dipertahankan sbg alias supaya pemanggil (app.py/UI)
            # lama yang masih membaca "paddle_fallback" tidak patah.
            "paddle_fallback": roi_template_meta,
            "roi_template_fallback": roi_template_meta,
            "vlm_fullpage_fallback": vlm_fullpage_meta,
            "roi_second_pass_vlm": roi_second_pass_meta,
            "vlm_reference_check": vlm_reference_meta,
            # Alias lama "vlm_fallback" -> gabungan fallback VLM V12 supaya
            # dashboard lama yang membaca key ini tetap dapat ringkasan.
            "vlm_fallback": {
                "used": vlm_fullpage_meta.get("used") or roi_second_pass_meta.get("used") or vlm_reference_meta.get("used"),
                "vlm_fullpage": vlm_fullpage_meta,
                "roi_second_pass_vlm": roi_second_pass_meta,
                "vlm_reference_check": vlm_reference_meta,
            },
            "out_of_frame_fields": [f for f, s in dynamic_field_statuses.items()] + (
                ["tempat_tanggal_surat"] if raw_results.get("tempat_tanggal_surat", {}).get("status") == "out_of_frame" else []
            ),
            "section_meta": dynamic_section_meta,
        },
    }
