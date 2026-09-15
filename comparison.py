"""
comparison.py
=============
Bandingkan nilai "Data Entry" (dari Excel/Google Spreadsheet) dengan hasil
OCR untuk field yang sama, lalu tandai match/mismatch.
"""

import re
from difflib import SequenceMatcher

from postprocessing import derive_tenor_from_range

# Kolom spreadsheet -> nama field pipeline. Kolom yang tidak ada di sini
# (Regional Office, Nama Branch Office, Surat Pernyataan) hanya ditampilkan
# sebagai info record, tidak dibandingkan ke OCR. HANYA 5 field ini yang
# dibandingkan (sesuai spesifikasi): Nama, Nomor Rekening, Nominal, Tenor,
# Bentuk Reward.
COLUMN_FIELD_MAP = {
    "Nomor Rekening": "nomor_rekening",
    "Nama Nasabah": "nama_nasabah",
    "Nominal Penempatan": "nominal_penempatan",
    "Tenor Hold": "tenor_penempatan",
    "Pilihan Hadiah": "bentuk_reward",
}

# Tipe data per field yang dibandingkan -> menentukan cara normalisasi.
FIELD_DATA_TYPES = {
    "nama_nasabah": "name",   # V11: fuzzy (bukan exact) -- lihat is_field_match
    "nomor_rekening": "numeric",
    "nominal_penempatan": "currency",
    "tenor_penempatan": "tenor",
    "bentuk_reward": "choice",
}

# V19f: nama_nasabah punya 3 state -- PASS/UNCERTAIN/MISMATCH -- bukan cuma
# 1 threshold biner lagi. similarity >= NAME_PASS_THRESHOLD -> identik/variasi
# OCR wajar (PASS). similarity < NAME_MISMATCH_THRESHOLD -> identitas jelas
# beda (MISMATCH). Di antara keduanya -> tidak bisa dipastikan sama/beda,
# TIDAK auto-lanjut (UNCERTAIN -> REVIEW), sesuai kebijakan "err toward
# review" utk nama. Threshold diverifikasi thd data eval real (lihat
# eval_runs/evaluation_results_qwen_local_v2_fixed.csv record 6/7): Ratem/
# Ratem=100 -> PASS, Ratem/Ratzen=72.7 & Mulyani/Nuryani=71.4 -> UNCERTAIN,
# Ratem/Asep=44.4 -> MISMATCH. Nomor rekening/nominal TETAP exact/
# deterministic (tidak lewat jalur ini).
NAME_PASS_THRESHOLD = 90
NAME_MISMATCH_THRESHOLD = 55


def _name_similarity(a, b):
    """Similarity dua string nama (0-100) yg SUDAH dinormalisasi (lowercase,
    trim, collapse spasi -- lihat _normalize_for_compare). Pakai
    difflib.SequenceMatcher stdlib supaya tidak menambah dependency baru."""
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio() * 100, 1)


def is_field_match(value_a, value_b, data_type):
    """SATU-SATUNYA titik keputusan match/mismatch per tipe data -- dipakai
    attach_data_entry (tabel/UI) & _validate_simple (decision backend) supaya
    keduanya TIDAK PERNAH berbeda kesimpulan utk field yang sama (nama_nasabah
    khususnya, lihat handover #2: "jangan sampai UI menunjukkan sesuai tetapi
    decision backend masih TOLAK karena exact match lama").
    Return (match, similarity) -- match adalah True/False utk semua data_type,
    KECUALI 'name' yang punya state ke-3 literal string "uncertain" (lihat
    NAME_PASS_THRESHOLD/NAME_MISMATCH_THRESHOLD di atas). similarity hanya
    terisi utk data_type 'name' (disimpan utk audit)."""
    norm_a = _normalize_for_compare(value_a, data_type)
    norm_b = _normalize_for_compare(value_b, data_type)
    if not norm_a or not norm_b:
        return False, None
    if data_type == "name":
        similarity = _name_similarity(norm_a, norm_b)
        if similarity >= NAME_PASS_THRESHOLD:
            return True, similarity
        if similarity < NAME_MISMATCH_THRESHOLD:
            return False, similarity
        return "uncertain", similarity
    return norm_a == norm_b, None


def _normalize_for_compare(value, data_type):
    """Normalisasi SESUAI TIPE DATA field (bukan satu aturan generik utk
    semua field seperti V7): angka murni utk nomor rekening, currency tanpa
    'Rp'/teks terbilang utk nominal, digit tenor, kanonik tunai/non_tunai
    utk bentuk reward, dan lowercase-trim biasa utk nama."""
    if value is None or value == "":
        return ""
    text = str(value).strip()

    if data_type == "tenor":
        return re.sub(r"\D", "", text)

    if data_type == "numeric":
        # V18 Fix 1b -- lstrip leading zero (SAMA persis spt "currency" 3
        # baris di bawah, pola yg SUDAH terbukti aman di sini): ground truth
        # nomor rekening di spreadsheet sumber kadang kehilangan leading
        # zero (koersi numerik Excel di hulu sistem ini), sehingga nomor yg
        # SAMA persis bisa salah ke-TOLAK di real decision. TIDAK digabung
        # ke branch "tenor" di atas -- tenor (jumlah bulan) tidak butuh &
        # tidak boleh ikut tersentuh oleh perubahan ini.
        digits = re.sub(r"\D", "", text)
        return digits.lstrip("0") or ("0" if digits else "")

    if data_type == "currency":
        text = re.sub(r"(?i)rp\.?", "", text).split("(")[0]
        digits = re.sub(r"\D", "", text)
        return digits.lstrip("0") or ("0" if digits else "")

    if data_type == "choice":
        t = re.sub(r"[^a-z]", "", text.lower())
        if t in ("nontunai", "barang", "nonkas", "noncash"):
            return "nontunai"
        if t in ("tunai", "cash",  "cashback",  "cash back", "kas"):
            return "tunai"
        return t

    return re.sub(r"\s+", " ", text.lower())  # text (nama, dst)


def attach_data_entry(fields_table, record):
    """Tambahkan kolom 'data_entry' + 'match' + 'display_status' ke tiap baris
    tabel hasil OCR, dan lengkapi 'reason' dengan detail data spreadsheet vs
    dokumen kalau mismatch. 'display_status' menggabungkan status field
    dengan hasil pencocokan data entry vs OCR, dipakai langsung oleh tabel
    hasil (Field / Data Entry / Data OCR / Status)."""
    reverse_map = {field: column for column, field in COLUMN_FIELD_MAP.items()}
    enriched = []
    for row in fields_table:
        row = dict(row)
        column = reverse_map.get(row["field"])
        data_type = FIELD_DATA_TYPES.get(row["field"], "text")
        entry_value = record.get(column) if column else None
        ocr_value = row.get("ocr_result")

        match, similarity = (None, None)
        if entry_value not in (None, "") and ocr_value not in (None, ""):
            match, similarity = is_field_match(entry_value, ocr_value, data_type)

        row["data_entry"] = entry_value
        row["match"] = match
        if similarity is not None:
            row["name_similarity"] = similarity  # audit trail (V11 fuzzy nama)
        row["display_status"] = _display_status(row.get("status"), match)
        if match is False or match == "uncertain":
            detail = f"data spreadsheet='{entry_value}' vs dokumen='{ocr_value}'"
            if similarity is not None:
                detail += f" (similarity {similarity:.0f}%)"
            row["reason"] = f"{row['reason']}; {detail}" if row.get("reason") else detail
        enriched.append(row)
    return enriched


def _display_status(field_status, match):
    """Satu label status final untuk tabel hasil: gabung status field OCR
    (read/review/blank/not_detected/detected/conflict/uncertain/...) dengan
    hasil match data entry. match=="uncertain" (nama, lihat is_field_match)
    -> 'perlu_review', BUKAN 'sesuai' atau 'tidak_sesuai' -- tidak pernah
    ditampilkan seolah sudah dipastikan cocok."""
    if match is True:
        return "sesuai"
    if match is False:
        return "tidak_sesuai"
    if match == "uncertain":
        return "perlu_review"
    return field_status or "-"


# ============================================================================
# FINAL STATUS PER RECORD (Tab 2): PASSED / MISMATCH / INCOMPLETE / UNCERTAIN
# / ERROR -- pengganti status generik "need_review" V7 supaya penyebabnya
# jelas dari nama status itu sendiri, dengan alasan per field di
# fields_table[i]["reason"] (lihat postprocessing.build_fields_table +
# attach_data_entry di atas).
# ============================================================================

# Field yang dibandingkan tapi hasil OCR-nya kosong/tidak terbaca -> tidak
# bisa disimpulkan sama sekali (bukan cuma "review").
INCOMPLETE_FIELD_STATUSES = {"blank", "not_detected", "out_of_frame"}
# Status yang menandakan hasil field ADA tapi tidak sepenuhnya meyakinkan
# (ambigu/bertentangan) -- dicek di SELURUH field (termasuk signature),
# bukan hanya yang dibandingkan ke spreadsheet. "recognition_conflict" (V10)
# = Paddle vs VLM tidak sepakat -- HARUS jadi REVIEW, tidak pernah TOLAK
# otomatis (lihat comparison.compute_decision_v9_2 / handover VLM arbitration).
UNCERTAIN_FIELD_STATUSES = {"review", "conflict", "uncertain", "recognition_conflict"}

COMPARED_FIELDS = set(COLUMN_FIELD_MAP.values())


def compute_final_status(fields_table):
    """Simpulkan satu status akhir per record dari seluruh baris fields_table
    (kalau ada data entry, dipanggil SETELAH attach_data_entry supaya 'match'
    tersedia). Urutan prioritas (dari yang paling pasti bermasalah):
      1. "ERROR"      : ada field dengan status 'error' (pipeline gagal).
      2. "MISMATCH"   : ada field (dari 5 yang dibandingkan) yang NILAINYA
                        beda dari data entry spreadsheet -- temuan paling
                        konkret, diprioritaskan di atas sekadar ambigu.
      3. "INCOMPLETE" : ada field yang dibandingkan tapi hasil dokumennya
                        kosong/tidak terbaca -- tidak bisa dibandingkan sama
                        sekali, bukan cuma tidak cocok.
      4. "UNCERTAIN"  : ada field (apa pun, termasuk signature) berstatus
                        ambigu/bertentangan (review/conflict/uncertain) yang
                        perlu verifikasi manual walau tidak ada mismatch pasti.
      5. "PASSED"     : selebihnya -- field yang dibandingkan terbaca & cocok.
    """
    if any(row.get("status") == "error" for row in fields_table):
        return "ERROR"

    compared_rows = [row for row in fields_table if row.get("field") in COMPARED_FIELDS]

    if any(row.get("match") is False for row in compared_rows):
        return "MISMATCH"

    if any(row.get("status") in INCOMPLETE_FIELD_STATUSES for row in compared_rows):
        return "INCOMPLETE"

    if any(row.get("status") in UNCERTAIN_FIELD_STATUSES for row in fields_table):
        return "UNCERTAIN"

    return "PASSED"


# ============================================================================
# V9.2 -- 7 VALIDASI KEPUTUSAN & DETERMINISTIC DECISION (OK/TOLAK/REVIEW)
# ============================================================================
# compute_final_status (V8, di atas) DIPERTAHANKAN APA ADANYA utk kompatibilitas
# pemanggil lama. Fungsi di bawah ini adalah IMPLEMENTASI BARU aturan V9.2:
# Tenor & Reward WAJIB dua bukti independen (bukan fallback salah satu spt
# resolve_tenor_source/resolve_bentuk_reward di postprocessing.py -- fungsi
# itu tetap dipakai sbg SUMBER evidence mentah utk ditampilkan, bukan sbg
# dasar keputusan akhir). Signature ikut jadi validasi ke-6/ke-7. AI/OCR
# TIDAK PERNAH memutuskan langsung -- hanya menyuplai evidence
# (raw_results/choice_groups), keputusan dihitung deterministik di sini.

DECISIVE_FIELDS_V9_2 = (
    "nama_nasabah", "nomor_rekening", "nominal_penempatan",
    "tenor_penempatan", "bentuk_reward",
)

# Signature checks are computed and returned for display, but per handover.md
# ("other extracted fields/signatures are informational and must not affect
# the business decision") they must NEVER drive OK/TOLAK/REVIEW -- ini
# berlaku utk mode URBAN. Utk mode RURAL (V11, lihat di bawah) signature
# JUSTRU decisive, jadi jangan dianggap "selalu informational" secara global.
INFORMATIONAL_FIELDS_V9_2 = ("signature_nasabah", "signature_atasan")

# ============================================================================
# V11 -- MODE WILAYAH (urban/rural), lihat spesifikasi:
#   record[URBAN_RURAL_COLUMN] == "urban" (case-insensitive)              -> URBAN
#   selain itu (kosong, "rural", nilai lain apa pun)                       -> RURAL
# URBAN  : logic keputusan SAMA PERSIS spt sebelumnya (DECISIVE_FIELDS_V9_2,
#          5 field: nama/nomor_rekening/nominal/tenor/reward).
# RURAL  : Final Status HANYA ditentukan oleh RURAL_DECISIVE_FIELDS (5 field
#          beda: nama/nomor_rekening/nominal/TTD Nasabah/TTD BRI). Tenor &
#          Bentuk Reward TETAP dibaca+dibandingkan (evidence sama dgn urban)
#          tapi TIDAK memengaruhi decision -- kalau bukan OK, masuk "notes".
# ============================================================================
URBAN_RURAL_COLUMN = "Urban"

RURAL_DECISIVE_FIELDS = (
    "nama_nasabah", "nomor_rekening", "nominal_penempatan",
    "signature_nasabah", "signature_atasan",
)

# Field yang tetap dibaca/dibandingkan tapi non-decisive di mode RURAL ->
# kalau TOLAK (mismatch pasti), badge tetap ini ditulis ke "notes", bukan
# "reasons". Status REVIEW (ambigu/tidak yakin) tidak dapat badge sendiri --
# sudah tercakup badge generik "OCR Kurang Yakin"/"ROI Bermasalah" di bawah,
# sesuai daftar kategori Notes yang TETAP (lihat HANDOVER_V11_NEXT.md #4).
RURAL_NOTE_FIELDS = {"tenor_penempatan": "Tenor Tidak Sesuai", "bentuk_reward": "Reward Tidak Sesuai"}

# ============================================================================
# V11 -- NOTES DIAGNOSTIC BADGES (#4). Kategori UI TETAP hanya 5 (di atas +
# 2 di sini): "ROI Bermasalah", "OCR Kurang Yakin", "Fallback OCR Digunakan".
# Detail teknis (TOO_WIDE/TOO_NARROW/WRONG_LOCATION/CUT_OFF/LABEL_BLEED, dll)
# TETAP hanya di metadata/debug (fields_table/raw_results), TIDAK PERNAH
# muncul mentah di Notes UI. Notes TIDAK PERNAH mengubah Final Status --
# murni informasi tambahan, berlaku SAMA utk mode urban & rural.
# ============================================================================
# V12: source values sesuai pipeline.py -- "ocr_primary" TIDAK termasuk
# (bukan fallback). "vlm_reference_check" = verifikasi mismatch vs referensi
# (fitur terpisah, lihat pipeline._run_vlm_reference_check).
FALLBACK_SOURCES = {"vlm_fullpage", "roi_template", "roi_second_pass_vlm", "vlm_reference_check"}


def _is_roi_issue_row(row):
    """'valid' (lihat postprocessing.build_text_field_result) True berarti
    format value OK tapi confidence rendah -- BUKAN indikasi ROI salah.
    'valid' False/None + status blank/review/not_detected -> value tidak
    terbaca sama sekali secara benar, kemungkinan besar ROI meleset."""
    status = row.get("status")
    if status == "not_detected":
        return True
    return status == "review" and row.get("valid") is not True


def _is_low_confidence_row(row):
    return row.get("status") == "review" and row.get("valid") is True


def _diagnostic_notes(fields_table):
    """Notes generik dari status/confidence/source tiap field -- dipakai
    BERSAMA oleh mode urban & rural (tidak seperti RURAL_NOTE_FIELDS yang
    rural-only)."""
    notes = []
    if any(_is_roi_issue_row(row) for row in fields_table):
        notes.append("ROI Bermasalah")
    if any(_is_low_confidence_row(row) for row in fields_table):
        notes.append("OCR Kurang Yakin")
    if any(row.get("source") in FALLBACK_SOURCES for row in fields_table):
        notes.append("Fallback OCR Digunakan")
    return notes


def _wilayah_mode(record):
    """'urban' HANYA kalau record[URBAN_RURAL_COLUMN] PERSIS 'urban'
    (case-insensitive, trim spasi) -- SELAIN itu (termasuk kosong/kolom
    tidak ada) dianggap 'rural', sesuai spesifikasi."""
    value = str((record or {}).get(URBAN_RURAL_COLUMN) or "").strip().lower()
    return "urban" if value == "urban" else "rural"


def _validate_simple(doc_value, ref_value, data_type, label, doc_status=None):
    # V10: status khusus dari geometry/VLM-arbitration MENGGANTIKAN
    # perbandingan nilai polos -- keduanya WAJIB REVIEW, tidak pernah
    # otomatis TOLAK/OK, sesuai kebijakan arbitrase Paddle vs VLM.
    if doc_status == "recognition_conflict":
        return "REVIEW", f"{label}: hasil OCR dan VLM tidak sepakat, perlu verifikasi manual"
    if doc_status == "out_of_frame":
        return "REVIEW", f"{label} berada di luar area foto (dokumen kemungkinan terpotong)"

    doc_n = _normalize_for_compare(doc_value, data_type)
    ref_n = _normalize_for_compare(ref_value, data_type)
    if not doc_n:
        return "REVIEW", f"{label} tidak terbaca dari dokumen"
    if not ref_n:
        return "REVIEW", f"{label} tidak ada data referensi"
    match, similarity = is_field_match(doc_value, ref_value, data_type)
    if match == "uncertain":
        detail = f"{label} tidak bisa dipastikan sama (kemungkinan variasi OCR): dokumen={doc_value}, referensi={ref_value}"
        if similarity is not None:
            detail += f" (similarity {similarity:.0f}%)"
        return "REVIEW", detail
    if not match:
        detail = f"{label} tidak sesuai: dokumen={doc_value}, referensi={ref_value}"
        if similarity is not None:
            detail += f" (similarity {similarity:.0f}%)"
        return "TOLAK", detail
    return "OK", None


def validate_tenor(raw_results, choice_groups, ref_tenor):
    """WAJIB 2 bukti independen (choice 1/3/6 + rentang tanggal tertulis).
    Konsisten dulu (choice == derivasi kalender dari rentang) BARU
    dibandingkan ke referensi. TIDAK PERNAH fallback ke salah satu bukti
    saja (beda dgn resolve_tenor_source di postprocessing.py, yg tetap
    dipakai sbg tampilan evidence, bukan keputusan)."""
    choice = choice_groups.get("tenor_penempatan", {})
    choice_value, choice_status = choice.get("value"), choice.get("status")
    range_raw = (raw_results.get("rentang_tenor") or {}).get("raw")
    derived_value, derive_reason = derive_tenor_from_range(range_raw)
    evidence = {
        "choice": choice_value, "date_range_derived": derived_value, "date_range_raw": range_raw,
        "date_range_derive_reason": derive_reason,
    }

    if choice_status == "out_of_frame":
        return "REVIEW", "Area pilihan tenor berada di luar cakupan foto", evidence
    if choice_status != "detected" or choice_value is None:
        return "REVIEW", "Pilihan tenor (1/3/6) tidak terbaca dgn yakin", evidence
    if derived_value is None:
        return "REVIEW", f"Rentang tanggal tenor tidak terbaca/ambigu ({derive_reason})", evidence
    if derived_value != choice_value:
        return "TOLAK", f"Tenor tidak konsisten: pilihan={choice_value} bulan, rentang={derived_value} bulan", evidence

    ref_n = _normalize_for_compare(ref_tenor, "tenor")
    if not ref_n:
        return "REVIEW", "Tenor konsisten tapi tidak ada data referensi", evidence
    if str(choice_value) != ref_n:
        return "TOLAK", f"Tenor tidak sesuai: dokumen={choice_value} bulan, referensi={ref_tenor}", evidence
    return "OK", None, evidence


def validate_reward(raw_results, choice_groups, ref_reward):
    """WAJIB 2 bukti (choice tunai/non_tunai + evidence field detail terkait
    terisi & KONSISTEN -- field pasangannya harus kosong)."""
    choice = choice_groups.get("bentuk_reward", {})
    choice_value, choice_status = choice.get("value"), choice.get("status")
    non_tunai_filled = bool((raw_results.get("reward_non_tunai") or {}).get("value"))
    tunai_filled = bool((raw_results.get("reward_tunai") or {}).get("value"))
    evidence = {"choice": choice_value, "non_tunai_filled": non_tunai_filled, "tunai_filled": tunai_filled}

    if choice_status == "out_of_frame":
        return "REVIEW", "Area pilihan bentuk reward berada di luar cakupan foto", evidence
    if choice_status != "detected" or choice_value is None:
        return "REVIEW", "Pilihan bentuk reward (tunai/non-tunai) tidak terbaca dgn yakin", evidence

    if choice_value == "tunai":
        if non_tunai_filled and not tunai_filled:
            return "TOLAK", "Bentuk reward tidak konsisten: pilihan=tunai tapi detail non-tunai terisi", evidence
        if not tunai_filled:
            return "REVIEW", "Pilihan=tunai tapi nilai reward tunai tidak terbaca", evidence
    else:  # non_tunai
        if tunai_filled and not non_tunai_filled:
            return "TOLAK", "Bentuk reward tidak konsisten: pilihan=non-tunai tapi nilai tunai terisi", evidence
        if not non_tunai_filled:
            return "REVIEW", "Pilihan=non-tunai tapi detail barang tidak terbaca", evidence

    ref_bucket = _normalize_for_compare(ref_reward, "choice")
    if not ref_bucket:
        return "REVIEW", "Bentuk reward konsisten tapi tidak ada data referensi", evidence
    ref_bucket = "non_tunai" if ref_bucket == "nontunai" else ref_bucket
    if choice_value != ref_bucket:
        return "TOLAK", f"Bentuk reward tidak sesuai: dokumen={choice_value}, referensi={ref_reward}", evidence
    return "OK", None, evidence


def validate_signature(name, sig_result):
    state = (sig_result or {}).get("status", "absent")
    label = "Nasabah" if name == "signature_nasabah" else "Unit Kerja/Atasan"
    if state == "present":
        return "OK", None
    if state == "out_of_frame":
        return "REVIEW", f"Area tanda tangan {label} berada di luar cakupan foto"
    if state == "absent":
        return "TOLAK", f"Tanda tangan {label} tidak ditemukan"
    return "REVIEW", f"Tanda tangan {label} perlu verifikasi manual"


def _compute_all_checks(fields_table, raw_results, choice_groups, record):
    """Hitung SEMUA 7 validasi (nama/nomor_rekening/nominal/tenor/reward +
    2 signature) SATU KALI -- dipakai bersama oleh compute_decision_v9_2
    (urban) dan compute_decision_rural (V11) supaya logic per-field validasi
    TIDAK diduplikasi antar mode wilayah. Return (checks, evidence) -- bentuk
    checks[field] = (status, reason) SAMA seperti sebelumnya."""
    reverse_map = {f: c for c, f in COLUMN_FIELD_MAP.items()}
    by_field = {row["field"]: row for row in fields_table}

    checks = {}
    checks["nama_nasabah"] = _validate_simple(
        by_field.get("nama_nasabah", {}).get("ocr_result"),
        record.get(reverse_map["nama_nasabah"]), "name", "Nama Nasabah",
        doc_status=by_field.get("nama_nasabah", {}).get("status"))
    checks["nomor_rekening"] = _validate_simple(
        by_field.get("nomor_rekening", {}).get("ocr_result"),
        record.get(reverse_map["nomor_rekening"]), "numeric", "Nomor Rekening",
        doc_status=by_field.get("nomor_rekening", {}).get("status"))
    checks["nominal_penempatan"] = _validate_simple(
        by_field.get("nominal_penempatan", {}).get("ocr_result"),
        record.get(reverse_map["nominal_penempatan"]), "currency", "Nominal Penempatan",
        doc_status=by_field.get("nominal_penempatan", {}).get("status"))

    tenor_status, tenor_reason, tenor_evidence = validate_tenor(
        raw_results, choice_groups, record.get(reverse_map["tenor_penempatan"]))
    checks["tenor_penempatan"] = (tenor_status, tenor_reason)

    reward_status, reward_reason, reward_evidence = validate_reward(
        raw_results, choice_groups, record.get(reverse_map["bentuk_reward"]))
    checks["bentuk_reward"] = (reward_status, reward_reason)

    checks["signature_nasabah"] = validate_signature("signature_nasabah", raw_results.get("signature_nasabah"))
    checks["signature_atasan"] = validate_signature("signature_atasan", raw_results.get("signature_atasan"))

    evidence = {"tenor_penempatan": tenor_evidence, "bentuk_reward": reward_evidence}
    return checks, evidence


def _extra_validation_notes(checks, tenor_evidence):
    """Badge tambahan (V19f) yang menjelaskan ALASAN validasi spesifik --
    dipakai BERSAMA oleh mode urban & rural, sama seperti _diagnostic_notes,
    tapi bersumber dari checks/evidence (bukan fields_table) karena butuh
    tahu field MANA yang bermasalah, bukan cuma indikator generik."""
    notes = []

    name_status, name_reason = checks.get("nama_nasabah", (None, None))
    if name_status == "TOLAK":
        notes.append("Name Mismatch")
    elif name_status == "REVIEW" and name_reason and "variasi OCR" in name_reason:
        notes.append("Name Uncertain")

    tenor_status, tenor_reason = checks.get("tenor_penempatan", (None, None))
    if (tenor_evidence or {}).get("date_range_derive_reason") == "reversed_date_range":
        notes.append("Invalid/Reversed Date Range")
    if tenor_status == "TOLAK" and tenor_reason and "tidak konsisten" in tenor_reason:
        notes.append("Tenor Conflict")

    reward_status, reward_reason = checks.get("bentuk_reward", (None, None))
    if reward_status == "REVIEW" and reward_reason and "tidak terbaca" in reward_reason:
        notes.append("Reward Uncertain")

    for sig_field in ("signature_nasabah", "signature_atasan"):
        sig_status, _sig_reason = checks.get(sig_field, (None, None))
        if sig_status == "TOLAK" and "Signature Absent" not in notes:
            notes.append("Signature Absent")

    return notes


def _decide_from_checks(checks, decisive_fields):
    """Prioritas SAMA spt sebelumnya: ada TOLAK -> TOLAK (semua alasan TOLAK
    dikumpulkan); tidak ada TOLAK tapi ada REVIEW -> REVIEW; selain itu ->
    OK. HANYA field di 'decisive_fields' yang dihitung -- inilah satu-
    satunya perbedaan antara mode urban & rural."""
    decisive_checks = {k: v for k, v in checks.items() if k in decisive_fields}
    reasons_tolak = [r for s, r in decisive_checks.values() if s == "TOLAK" and r]
    reasons_review = [r for s, r in decisive_checks.values() if s == "REVIEW" and r]
    if reasons_tolak:
        return "TOLAK", reasons_tolak
    if reasons_review:
        return "REVIEW", reasons_review
    return "OK", []


def compute_decision_v9_2(fields_table, raw_results, choice_groups, record):
    """Mode URBAN (TIDAK BERUBAH dari sebelumnya): 7 validasi deterministik
    -> {"decision": OK/TOLAK/REVIEW, "reasons": [...], "checks":
    {field: status}, "evidence": {...}}, decisive = DECISIVE_FIELDS_V9_2
    (nama/nomor_rekening/nominal/tenor/reward)."""
    checks, evidence = _compute_all_checks(fields_table, raw_results, choice_groups, record)
    decision, reasons = _decide_from_checks(checks, DECISIVE_FIELDS_V9_2)
    notes = _diagnostic_notes(fields_table)
    for badge in _extra_validation_notes(checks, evidence.get("tenor_penempatan")):
        if badge not in notes:
            notes.append(badge)
    return {
        "decision": decision,
        "reasons": reasons,
        "checks": {k: v[0] for k, v in checks.items()},
        "evidence": evidence,
        "notes": notes,
    }


def compute_decision_rural(fields_table, raw_results, choice_groups, record):
    """Mode RURAL (V11, BARU): Final Status HANYA ditentukan oleh
    RURAL_DECISIVE_FIELDS (Nama, Nomor Rekening, Nominal Penempatan, TTD
    Nasabah, TTD BRI) -- kelimanya 'passed' (OK) -> decision OK; salah satu
    TOLAK/REVIEW -> decision TOLAK/REVIEW dgn alasan field itu (frontend
    menampilkan REVIEW & TOLAK sbg badge 'TOLAK - alasan' yang sama, lihat
    static/index.html resolveHasilAkhir -- konsisten dgn mode urban).

    Tenor & Bentuk Reward TETAP dibaca & dibandingkan (validate_tenor/
    validate_reward, EVIDENCE SAMA PERSIS dgn mode urban) tapi TIDAK ikut
    menentukan decision -- kalau statusnya bukan OK, pesannya masuk ke
    'notes' (lihat RURAL_NOTE_FIELDS), bukan 'reasons'."""
    checks, evidence = _compute_all_checks(fields_table, raw_results, choice_groups, record)
    decision, reasons = _decide_from_checks(checks, RURAL_DECISIVE_FIELDS)

    notes = _diagnostic_notes(fields_table)
    for field_name, badge in RURAL_NOTE_FIELDS.items():
        status, _reason = checks.get(field_name, (None, None))
        if status == "TOLAK" and badge not in notes:
            notes.append(badge)
    for badge in _extra_validation_notes(checks, evidence.get("tenor_penempatan")):
        if badge not in notes:
            notes.append(badge)

    return {
        "decision": decision,
        "reasons": reasons,
        "checks": {k: v[0] for k, v in checks.items()},
        "evidence": evidence,
        "notes": notes,
        "wilayah": "rural",
    }


def compute_decision(fields_table, raw_results, choice_groups, record):
    """Dispatcher V11 -- SATU titik masuk yang dipanggil app.py: pilih
    compute_decision_v9_2 (urban, perilaku TIDAK BERUBAH) atau
    compute_decision_rural (rural, BARU) berdasarkan _wilayah_mode(record).
    Return dict dgn bentuk yang SAMA utk kedua mode (selalu ada key
    'notes'/'wilayah') supaya pemanggil (app.py/frontend) tidak perlu
    bercabang lagi."""
    if _wilayah_mode(record) == "urban":
        result = compute_decision_v9_2(fields_table, raw_results, choice_groups, record)
        result.setdefault("notes", [])
        result["wilayah"] = "urban"
        return result
    return compute_decision_rural(fields_table, raw_results, choice_groups, record)
