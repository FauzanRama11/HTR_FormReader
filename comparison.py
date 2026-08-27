"""
comparison.py
=============
Bandingkan nilai "Data Entry" (dari Excel/Google Spreadsheet) dengan hasil
OCR untuk field yang sama, lalu tandai match/mismatch.
"""

import re

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
    "nama_nasabah": "text",
    "nomor_rekening": "numeric",
    "nominal_penempatan": "currency",
    "tenor_penempatan": "tenor",
    "bentuk_reward": "choice",
}


def _normalize_for_compare(value, data_type):
    """Normalisasi SESUAI TIPE DATA field (bukan satu aturan generik utk
    semua field seperti V7): angka murni utk nomor rekening, currency tanpa
    'Rp'/teks terbilang utk nominal, digit tenor, kanonik tunai/non_tunai
    utk bentuk reward, dan lowercase-trim biasa utk nama."""
    if value is None or value == "":
        return ""
    text = str(value).strip()

    if data_type in ("numeric", "tenor"):
        return re.sub(r"\D", "", text)

    if data_type == "currency":
        text = re.sub(r"(?i)rp\.?", "", text).split("(")[0]
        digits = re.sub(r"\D", "", text)
        return digits.lstrip("0") or ("0" if digits else "")

    if data_type == "choice":
        t = re.sub(r"[^a-z]", "", text.lower())
        if t in ("nontunai", "barang", "nonkas", "noncash"):
            return "nontunai"
        if t in ("tunai", "cash", "kas"):
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

        match = None
        if entry_value not in (None, "") and ocr_value not in (None, ""):
            match = _normalize_for_compare(entry_value, data_type) == _normalize_for_compare(ocr_value, data_type)

        row["data_entry"] = entry_value
        row["match"] = match
        row["display_status"] = _display_status(row.get("status"), match)
        if match is False:
            detail = f"data spreadsheet='{entry_value}' vs dokumen='{ocr_value}'"
            row["reason"] = f"{row['reason']}; {detail}" if row.get("reason") else detail
        enriched.append(row)
    return enriched


def _display_status(field_status, match):
    """Satu label status final untuk tabel hasil: gabung status field OCR
    (read/review/blank/not_detected/detected/conflict/uncertain/...) dengan
    hasil match data entry."""
    if match is True:
        return "sesuai"
    if match is False:
        return "tidak_sesuai"
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
# the business decision") they must NEVER drive OK/TOLAK/REVIEW.
INFORMATIONAL_FIELDS_V9_2 = ("signature_nasabah", "signature_atasan")


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
    if doc_n != ref_n:
        return "TOLAK", f"{label} tidak sesuai: dokumen={doc_value}, referensi={ref_value}"
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
    evidence = {"choice": choice_value, "date_range_derived": derived_value, "date_range_raw": range_raw}

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


def compute_decision_v9_2(fields_table, raw_results, choice_groups, record):
    """7 validasi deterministik -> {"decision": OK/TOLAK/REVIEW, "reasons": [...],
    "checks": {field: status}, "evidence": {...}}. Prioritas: ada TOLAK ->
    TOLAK (semua alasan TOLAK dikumpulkan); tidak ada TOLAK tapi ada REVIEW
    -> REVIEW; selain itu -> OK."""
    reverse_map = {f: c for c, f in COLUMN_FIELD_MAP.items()}
    by_field = {row["field"]: row for row in fields_table}

    checks = {}
    checks["nama_nasabah"] = _validate_simple(
        by_field.get("nama_nasabah", {}).get("ocr_result"),
        record.get(reverse_map["nama_nasabah"]), "text", "Nama Nasabah",
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

    decisive_checks = {k: v for k, v in checks.items() if k in DECISIVE_FIELDS_V9_2}
    reasons_tolak = [r for s, r in decisive_checks.values() if s == "TOLAK" and r]
    reasons_review = [r for s, r in decisive_checks.values() if s == "REVIEW" and r]

    if reasons_tolak:
        decision, reasons = "TOLAK", reasons_tolak
    elif reasons_review:
        decision, reasons = "REVIEW", reasons_review
    else:
        decision, reasons = "OK", []

    return {
        "decision": decision,
        "reasons": reasons,
        "checks": {k: v[0] for k, v in checks.items()},
        "evidence": {"tenor_penempatan": tenor_evidence, "bentuk_reward": reward_evidence},
    }
