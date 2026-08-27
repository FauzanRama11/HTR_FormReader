"""
postprocessing.py
==================
Semua langkah setelah OCR (atau setelah image-diff untuk choice/signature):
normalisasi teks -> validasi -> status -> deteksi pilihan/tanda tangan
(image-diff, tidak pernah OCR) -> penyusunan tabel akhir -> debug image.
"""

import datetime
import re

import cv2
import numpy as np

from preprocessing import (
    CHOICE_GROUPS, SIGNATURE_CONFIG, FIELD_CONFIG, FIELD_ORDER,
    locate_anchor_offset, resolve_roi, ink_change_ratio, difference_mask,
    norm_bbox_to_px, _shift_bbox, _crop, _remove_line_noise,
    is_out_of_frame as _is_out_of_frame,
)

OCR_REVIEW_THRESHOLD = 0.60     # confidence di bawah ini -> status "review"
CHOICE_MIN_CHANGE = 0.010
CHOICE_MIN_GAP = 0.004

# --- Signature (lihat process_signatures) ---
SIGNATURE_MIN_COMPONENT_AREA = 12        # px^2, buang noise speck/debu scan
SIGNATURE_ABSENT_AREA_RATIO = 0.004      # di bawah ini -> "absent"
SIGNATURE_PRESENT_AREA_RATIO = 0.012     # di atas ini (+ sebaran cukup) -> "present"
SIGNATURE_MIN_SPREAD_RATIO = 0.15        # sebaran horizontal/vertikal min. relatif ROI


# ============================================================================
# NORMALISASI & VALIDASI PER TIPE FIELD
# ============================================================================


def _strip_terms(text, terms):
    result = text or ""
    for term in terms or []:
        result = re.sub(re.escape(term), " ", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


# Fragmen akhir label cetak yang paling sering ikut ter-crop di awal ROI
# kalau ROI sedikit meleset ke kiri (mis. "...asabah" dari "Nama Nasabah",
# "...ekening" dari "Nomor/Unit Kerja Pengelola Rekening"). Perbaikan utama
# ada di level ROI (preprocessing.ANCHOR_RIGHT_NORM, mencegah ROI membaca
# balik ke area label sama sekali) -- daftar ini HANYA jaring pengaman kedua
# untuk kasus sisa/residual, bukan mekanisme utama.
_LABEL_BLEED_FRAGMENTS = ("nasabah", "rekening", "penempatan", "reward", "surat")


def _strip_label_bleed(text):
    """Buang sisa teks label cetak yang ikut ter-crop di awal ROI. Prioritas:
    (1) kalau ada ':' (bentuk paling umum, mis. "asabah : Ramayana..." dari
    label "Nama Nasabah :" yang bocor) -> ambil bagian SETELAH ':' terakhir,
    karena value asli tidak pernah mengandung ':'.
    (2) kalau TIDAK ada ':' tapi kata pertama cocok salah satu fragmen label
    yang dikenal (mis. "asabah Ratem" tanpa titik dua sama sekali) -> buang
    kata pertama itu saja, sisanya (value asli) tetap utuh."""
    if not text:
        return text
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    else:
        parts = text.split(" ", 1)
        first = parts[0].strip(".,").lower() if parts else ""
        # cocokkan sbg AKHIRAN kata label (huruf awal sering hilang saat OCR
        # membaca residu terpotong, mis. "asabah" dari "Nasabah", "ekening"
        # dari "Rekening") -- bukan exact match.
        is_bleed = len(first) >= 5 and any(frag.endswith(first) for frag in _LABEL_BLEED_FRAGMENTS)
        if is_bleed and len(parts) > 1:
            text = parts[1]
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip() if value else None


def normalize_numeric(value):
    if not value:
        return None
    text = str(value)
    for source, target in {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"}.items():
        text = text.replace(source, target)
    digits = re.sub(r"\D", "", text)
    return digits or None


def normalize_currency(value):
    text = re.sub(r"(?i)rp\.?", "", str(value or ""))
    # Pada form ini, nominal SELALU diikuti teks terbilang yang diawali "(".
    # OCR kadang membaca huruf awal terbilang sebagai digit tambahan yang
    # menempel ke angka (mis. "5900000000151" dari "5.900.000.000 (lima..."),
    # jadi buang semua setelah "(" dulu SEBELUM normalisasi digit.
    text = text.split("(")[0]
    digits = normalize_numeric(text)
    return int(digits) if digits else None


def normalize_value(raw, field_type):
    if field_type == "numeric":
        return normalize_numeric(raw)
    if field_type == "currency":
        return normalize_currency(raw)
    return normalize_text(raw)


def validate_value(value, field_type):
    if value is None or value == "":
        return False
    if field_type == "numeric":
        return str(value).isdigit() and 4 <= len(str(value)) <= 30
    if field_type == "currency":
        return isinstance(value, int) and 0 < value < 10**18
    return len(str(value).strip()) >= 2


def resolve_field_status(raw_text, value, valid, confidence, field_type):
    if not raw_text:
        return "blank" if field_type == "optional_text" else "not_detected"
    if not valid:
        return "review"
    if confidence is not None and confidence < OCR_REVIEW_THRESHOLD:
        return "review"
    return "read"


def build_out_of_frame_result():
    """Hasil field yang boxnya jatuh di luar cakupan piksel sumber (foto
    terpotong/parsial) -- TIDAK PERNAH dikirim ke OCR/VLM & TIDAK boleh
    ditafsirkan sbg 'blank' (nasabah memang tidak mengisi) -- 'out_of_frame'
    adalah status TERPISAH yang eksplisit menandakan area itu tidak pernah
    benar-benar difoto."""
    return {"value": None, "raw": None, "confidence": None, "status": "out_of_frame"}


def build_text_field_result(name, cfg, ocr_item):
    """Gabungkan hasil OCR mentah + normalisasi/validasi jadi satu hasil field."""
    if ocr_item is None:
        status = "blank" if cfg["type"] == "optional_text" else "not_detected"
        return {"value": None, "raw": None, "confidence": None, "status": status}

    raw_text = _strip_terms(ocr_item.get("raw"), cfg.get("strip_terms"))
    raw_text = _strip_label_bleed(raw_text)
    value = normalize_value(raw_text, cfg["type"])
    valid = validate_value(value, cfg["type"])
    confidence = ocr_item.get("confidence")
    status = resolve_field_status(raw_text, value, valid, confidence, cfg["type"])
    return {"value": value, "raw": raw_text or None, "confidence": confidence, "status": status}


# ============================================================================
# CHOICE + SIGNATURE (image-diff murni, tidak pernah di-OCR)
# ============================================================================


def _resolve_choice_multi(scores, labels):
    """Tentukan SATU pilihan final dari skor ink-diff tiap opsi, utk grup
    dengan 3+ OPSI (mis. tenor 1/3/6). Mendukung DUA konvensi pengisian form
    sekaligus (deteksi otomatis mana yang cocok dengan pola tinta terbaca):
      1) "single_mark"  -> nasabah menandai/melingkari HANYA opsi yang dipilih
                           (tepat 1 opsi berubah -> itu langsung jawabannya).
      2) "cross_out"    -> nasabah mencoret opsi yang TIDAK dipilih (N-1 opsi
                           berubah, sisa 1 yang bersih -> itu jawabannya).
    Dengan 3+ opsi, kedua konvensi TIDAK PERNAH tumpang tindih (1 tertanda
    vs N-1 tertanda beda kondisi kalau N>=3), jadi hasil selalu tidak ambigu
    ketika salah satu polanya cocok. Kalau tidak cocok pola manapun -> None +
    status 'review'.
    """
    ordered = sorted(scores.items(), key=lambda x: x[1])
    marked = [name for name, score in scores.items() if score >= CHOICE_MIN_CHANGE]

    if len(scores) < 2:
        return None, "blank", "insufficient_options"

    # Mode 1: tepat satu opsi ditandai -> itu jawabannya (paling umum & paling jelas).
    if len(marked) == 1:
        chosen = marked[0]
        others_max = max((s for n, s in scores.items() if n != chosen), default=0)
        if scores[chosen] - others_max >= CHOICE_MIN_GAP:
            return labels[chosen], "detected", "single_mark"
        return None, "review", "single_mark_low_gap"

    # Mode 2: N-1 opsi dicoret, sisa 1 bersih -> yang bersih itu jawabannya.
    clean_name, clean_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else clean_score
    if len(marked) == len(scores) - 1 and second_score - clean_score >= CHOICE_MIN_GAP:
        return labels[clean_name], "detected", "cross_out"

    if not marked:
        return None, "blank", "belum_diisi"

    return None, "review", "ambiguous_mark"


def _resolve_choice_binary(scores, labels):
    """AUDIT V8: grup dengan TEPAT 2 OPSI (mis. bentuk reward tunai/
    non_tunai) TIDAK memakai logika yang sama dengan grup 3+ opsi
    (_resolve_choice_multi) -- keduanya berbeda secara mendasar. Dengan 2
    opsi, "tepat 1 opsi bertanda" adalah kondisi yang SAMA PERSIS baik untuk
    konvensi single_mark (opsi bertanda = dipilih) MAUPUN cross_out (opsi
    bertanda = dicoret/ditolak, jadi opsi LAIN yang dipilih) -- ink-diff saja
    tidak bisa membedakan mana yang dimaksud nasabah, beda dengan grup 3+
    opsi yang kombinasi jumlah tertandanya tidak pernah bentrok antar
    konvensi. Karena itu kasus "1 dari 2 bertanda" SENGAJA tidak diputuskan
    di sini (status 'review', ambigu) -- diserahkan ke bukti field teks di
    sekitarnya (lihat resolve_bentuk_reward), bukan ditebak dari coretan
    saja seperti sebelumnya."""
    if len(scores) != 2:
        return _resolve_choice_multi(scores, labels)

    marked = [name for name, score in scores.items() if score >= CHOICE_MIN_CHANGE]
    if not marked:
        return None, "blank", "belum_diisi"
    if len(marked) == 2:
        return None, "review", "kedua_opsi_bertanda_kemungkinan_coretan_ganda"
    return None, "review", f"satu_dari_dua_opsi_bertanda_ambigu_konvensi:{marked[0]}"


def process_choices(template_img, aligned_img, template_gray, aligned_gray, shape, coverage_mask=None):
    groups, raw = {}, {}

    for group_name, group_cfg in CHOICE_GROUPS.items():
        dx, dy, similarity, matched = locate_anchor_offset(
            template_gray, aligned_gray, group_cfg["anchor_bbox"], shape
        )
        scores, labels = {}, {}
        any_out_of_frame = False
        for option_name, option in group_cfg["options"].items():
            template_px = norm_bbox_to_px(option["bbox"], shape)
            target_px = _shift_bbox(template_px, dx, dy, shape) if matched else template_px
            if coverage_mask is not None and _is_out_of_frame(coverage_mask, target_px):
                any_out_of_frame = True
                raw[option_name] = {
                    "mark_score": 0.0, "roi_bbox": target_px,
                    "roi_source": "out_of_frame", "status": "out_of_frame",
                }
                scores[option_name] = 0.0
                labels[option_name] = option["label"]
                continue
            score = ink_change_ratio(_crop(template_img, template_px), _crop(aligned_img, target_px))
            scores[option_name] = score
            labels[option_name] = option["label"]
            raw[option_name] = {
                "mark_score": score,
                "roi_bbox": target_px,
                "roi_source": "anchor" if matched else "fallback",
                "status": "detected" if score >= CHOICE_MIN_CHANGE else "blank",
            }

        if any_out_of_frame:
            groups[group_name] = {
                "value": None, "status": "out_of_frame", "scores": scores,
                "reason": "area_opsi_di_luar_cakupan_foto",
            }
            continue

        # Grup 2 opsi (bentuk reward) & grup 3+ opsi (tenor) SENGAJA memakai
        # resolver berbeda -- lihat docstring _resolve_choice_binary.
        resolver = _resolve_choice_binary if len(group_cfg["options"]) == 2 else _resolve_choice_multi
        final_value, status, reason = resolver(scores, labels)
        # value = jawaban FINAL (1/3/6 utk tenor, "tunai"/"non_tunai" utk reward).
        groups[group_name] = {"value": final_value, "status": status, "scores": scores, "reason": reason}

    return groups, raw


# ============================================================================
# TENOR -- choice 1/3/6 tetap sumber utama; kalau kosong/ambigu, derive dari
# rentang_tenor (selisih BULAN KALENDER antar 2 tanggal). Konflik choice vs
# tanggal -> status 'conflict'. Selalu menyimpan value/source/status/reason.
# ============================================================================

_ID_MONTHS = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
}
_DATE_TEXT_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")
_DATE_NUM_RE = re.compile(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})")
TENOR_OPTIONS = (1, 3, 6)
TENOR_TOLERANCE_MONTHS = 1   # toleransi pembulatan ke opsi terdekat


def _extract_dates(text):
    dates = []
    for day, month_name, year in _DATE_TEXT_RE.findall(text or ""):
        month = _ID_MONTHS.get(month_name.strip().lower())
        if month:
            try:
                dates.append(datetime.date(int(year), month, int(day)))
            except ValueError:
                continue
    if len(dates) < 2:
        for a, b, y in _DATE_NUM_RE.findall(text or ""):
            year = int(y) if len(y) == 4 else 2000 + int(y)
            try:
                dates.append(datetime.date(year, int(b), int(a)))  # asumsi dd/mm/yyyy
            except ValueError:
                continue
    return dates


def _calendar_month_diff(d1, d2):
    return abs((d2.year - d1.year) * 12 + (d2.month - d1.month))


def derive_tenor_from_range(raw_text):
    """Ekstrak DUA tanggal dari teks rentang_tenor, hitung selisih bulan
    KALENDER (year*12+month, bukan hari/30), lalu petakan ke opsi tenor
    terdekat (1/3/6) dengan toleransi kecil. Return (value_atau_None, reason)."""
    if not raw_text:
        return None, "rentang_tenor_kosong"
    dates = _extract_dates(raw_text)
    if len(dates) < 2:
        return None, "tidak_bisa_ekstrak_2_tanggal_dari_rentang_tenor"

    months = _calendar_month_diff(dates[0], dates[1])
    if months <= 0:
        return None, "selisih_tanggal_tidak_valid"

    nearest = min(TENOR_OPTIONS, key=lambda o: abs(o - months))
    if abs(nearest - months) > TENOR_TOLERANCE_MONTHS:
        return None, f"selisih_{months}_bulan_tidak_dekat_opsi_manapun"
    return nearest, f"derived_dari_rentang_tenor_{months}_bulan_kalender"


def resolve_tenor_source(groups, raw_results):
    """value/source/status/reason final utk tenor_penempatan:
      - choice (1/3/6) terdeteksi -> sumber utama ('choice').
        Kalau rentang_tenor bisa diderivasi DAN tidak cocok choice -> 'conflict'.
      - choice kosong/ambigu -> derive dari rentang_tenor ('rentang_tenor').
      - keduanya tidak bisa dipastikan -> status 'blank'/'review' apa adanya."""
    tenor = groups.get("tenor_penempatan", {})
    choice_value, choice_status = tenor.get("value"), tenor.get("status")
    range_raw = (raw_results.get("rentang_tenor") or {}).get("raw")
    derived_value, derive_reason = derive_tenor_from_range(range_raw)

    if choice_status == "detected" and choice_value is not None:
        if derived_value is not None and derived_value != choice_value:
            return {
                "value": choice_value, "source": "choice", "status": "conflict",
                "reason": f"choice={choice_value} vs rentang_tenor={derived_value} ({derive_reason})",
            }
        return {"value": choice_value, "source": "choice", "status": "detected", "reason": tenor.get("reason")}

    if derived_value is not None:
        return {"value": derived_value, "source": "rentang_tenor", "status": "detected", "reason": derive_reason}

    status = "blank" if choice_status == "blank" and not range_raw else "review"
    return {"value": None, "source": "none", "status": status, "reason": tenor.get("reason") or derive_reason}


# ============================================================================
# BENTUK REWARD -- choice tunai/non_tunai tetap sumber utama. Kalau choice
# tidak jelas (2 opsi -> lihat _resolve_choice_binary), tentukan dari BUKTI
# isi field di sebelahnya. Nilai reward TIDAK dibandingkan ke spreadsheet di
# sini -- hanya dipakai sbg evidence utk menentukan BENTUK reward.
# ============================================================================


def resolve_bentuk_reward(groups, raw_results):
    reward = groups.get("bentuk_reward", {})
    choice_value, choice_status = reward.get("value"), reward.get("status")
    non_tunai_filled = bool((raw_results.get("reward_non_tunai") or {}).get("value"))
    tunai_filled = bool((raw_results.get("reward_tunai") or {}).get("value"))

    if choice_status == "detected" and choice_value is not None:
        # Choice jelas TAPI bertentangan dengan bukti isi field -> conflict,
        # bukan langsung dipercaya begitu saja.
        if choice_value == "tunai" and non_tunai_filled and not tunai_filled:
            return {
                "value": choice_value, "source": "choice", "status": "conflict",
                "reason": "choice=tunai tapi reward_non_tunai terisi & reward_tunai kosong",
            }
        if choice_value == "non_tunai" and tunai_filled and not non_tunai_filled:
            return {
                "value": choice_value, "source": "choice", "status": "conflict",
                "reason": "choice=non_tunai tapi reward_tunai terisi & reward_non_tunai kosong",
            }
        return {"value": choice_value, "source": "choice", "status": "detected", "reason": reward.get("reason")}

    if non_tunai_filled and tunai_filled:
        return {
            "value": None, "source": "evidence", "status": "uncertain",
            "reason": "reward_non_tunai & reward_tunai sama-sama terisi, evidence bertentangan",
        }
    if non_tunai_filled:
        return {
            "value": "non_tunai", "source": "evidence_reward_non_tunai", "status": "detected",
            "reason": "choice ambigu, reward_non_tunai terisi (barang/deskripsi spesifik)",
        }
    if tunai_filled:
        return {
            "value": "tunai", "source": "evidence_reward_tunai", "status": "detected",
            "reason": "choice ambigu, reward_tunai terisi (nominal currency)",
        }

    return {"value": None, "source": "none", "status": choice_status or "blank", "reason": reward.get("reason")}


def _connected_ink_stats(mask):
    """Analisis connected components pada mask tinta (SUDAH melalui template
    subtraction + noise-line filtering): buang komponen sangat kecil (noise/
    debu scan), lalu ukur total area tinta & SEBARAN (spread) horizontal/
    vertikal gabungan komponen yang tersisa. Tanda tangan asli biasanya
    berupa goresan yang MENYEBAR (bukan satu noda/titik tunggal di satu
    tempat) -- ini pembeda utama dari sekadar 'ink_change_ratio' polos yang
    tidak bisa membedakan 1 noda besar vs goresan tanda tangan wajar."""
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[:2]
    kept_area = 0
    xs1, ys1, xs2, ys2 = [], [], [], []
    n_components = 0
    for i in range(1, n):  # index 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < SIGNATURE_MIN_COMPONENT_AREA:
            continue
        n_components += 1
        kept_area += area
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        cw, ch = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        xs1.append(x); ys1.append(y); xs2.append(x + cw); ys2.append(y + ch)

    if n_components == 0:
        return {"components": 0, "area_ratio": 0.0, "spread_x": 0.0, "spread_y": 0.0}

    spread_x = (max(xs2) - min(xs1)) / w if w else 0.0
    spread_y = (max(ys2) - min(ys1)) / h if h else 0.0
    return {
        "components": n_components,
        "area_ratio": (kept_area / mask.size) if mask.size else 0.0,
        "spread_x": float(spread_x),
        "spread_y": float(spread_y),
    }


def _classify_signature(stats):
    """present / absent / uncertain berdasarkan gabungan area tinta +
    sebaran (bukan cuma satu ambang change-ratio seperti V7)."""
    area_ratio = stats["area_ratio"]
    spread = max(stats["spread_x"], stats["spread_y"])

    if stats["components"] == 0 or area_ratio < SIGNATURE_ABSENT_AREA_RATIO:
        return "absent", "tidak_ada_tinta_signifikan_setelah_noise_filtering"
    if area_ratio >= SIGNATURE_PRESENT_AREA_RATIO and spread >= SIGNATURE_MIN_SPREAD_RATIO:
        return "present", "area_dan_sebaran_tinta_konsisten_dgn_tanda_tangan"
    if area_ratio >= SIGNATURE_PRESENT_AREA_RATIO and spread < SIGNATURE_MIN_SPREAD_RATIO:
        return "uncertain", "area_tinta_cukup_tapi_sebaran_sempit_kemungkinan_noda_bukan_goresan"
    return "uncertain", "area_tinta_di_sekitar_ambang_batas"


def process_signatures(template_img, aligned_img, template_gray, aligned_gray, shape, coverage_mask=None):
    """Deteksi tanda tangan menggabungkan: template subtraction (difference_mask,
    sama dgn field lain) + local alignment ROI (resolve_roi) + connected
    components (buang noise, ukur jumlah goresan) + ink area & spread + noise
    filtering (_remove_line_noise) -- BUKAN cuma 1 angka ink_change_ratio.
    Output tiap signature: present/absent/uncertain + score & reason."""
    results = {}
    for name, cfg in SIGNATURE_CONFIG.items():
        template_px, target_px, roi_source, similarity = resolve_roi(
            template_gray, aligned_gray, cfg, shape
        )
        if coverage_mask is not None and _is_out_of_frame(coverage_mask, target_px):
            results[name] = {
                "present": False, "value": "Di luar area foto (dokumen terpotong)",
                "status": "out_of_frame", "reason": "area_tanda_tangan_di_luar_cakupan_foto",
                "change_ratio": 0.0, "component_count": 0, "ink_area_ratio": 0.0, "ink_spread": 0.0,
                "roi_source": "out_of_frame", "roi_bbox": target_px, "anchor_similarity": similarity,
            }
            continue
        t_roi = _crop(template_img, template_px)
        f_roi = _crop(aligned_img, target_px)

        mask = _remove_line_noise(difference_mask(t_roi, f_roi))
        change_ratio = float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0
        stats = _connected_ink_stats(mask)
        state, reason = _classify_signature(stats)

        results[name] = {
            "present": state == "present",
            "value": {
                "present": "Ada tanda tangan",
                "uncertain": "Perlu verifikasi manual",
                "absent": "Kosong",
            }[state],
            "status": state,
            "reason": reason,
            "change_ratio": change_ratio,
            "component_count": stats["components"],
            "ink_area_ratio": stats["area_ratio"],
            "ink_spread": max(stats["spread_x"], stats["spread_y"]),
            "roi_source": roi_source,
            "roi_bbox": target_px,
            "anchor_similarity": similarity,
        }
    return results


# ============================================================================
# TABEL AKHIR
# ============================================================================


def build_fields_table(final_results, field_types):
    """Susun hasil per field jadi list rapi untuk ditampilkan di UI. Termasuk
    "source" (mis. choice / rentang_tenor / evidence_reward_tunai) dan
    "reason" per field, dipakai final status (Tab 2) utk menjelaskan ALASAN
    di balik tiap nilai -- bukan cuma status generik."""
    fields_table = []
    for name in FIELD_ORDER:
        result = final_results.get(name, {})
        field_type = field_types[name]

        if field_type == "signature":
            ocr_result = result.get("value")
            raw_value = (
                f"skor={result.get('change_ratio', 0.0):.4f} "
                f"komponen={result.get('component_count', 0)} "
                f"area={result.get('ink_area_ratio', 0.0):.4f} "
                f"sebaran={result.get('ink_spread', 0.0):.3f}"
            )
        elif field_type == "choice":
            ocr_result = result.get("value")
            raw_value = ", ".join(f"{k}={v:.4f}" for k, v in result.get("scores", {}).items())
        else:
            ocr_result = result.get("value")
            raw_value = result.get("raw")

        fields_table.append({
            "field": name,
            "type": field_type,
            "ocr_result": ocr_result,
            "raw_result": raw_value,
            "status": result.get("status", "-"),
            "confidence": result.get("confidence"),
            "source": result.get("source"),
            "reason": result.get("reason"),
        })
    return fields_table


FIELD_TYPES = {name: cfg["type"] for name, cfg in FIELD_CONFIG.items()}
FIELD_TYPES.update({
    "tenor_penempatan": "choice", "bentuk_reward": "choice",
    "signature_nasabah": "signature", "signature_atasan": "signature",
})


# ============================================================================
# DEBUG IMAGE
# ============================================================================

STATUS_COLORS = {
    "read": (80, 175, 76), "detected": (80, 175, 76), "present": (80, 175, 76),
    "review": (0, 165, 255), "uncertain": (0, 165, 255), "conflict": (0, 140, 255),
    "recognition_conflict": (0, 140, 255),
    "blank": (150, 150, 150), "absent": (150, 150, 150),
    "not_detected": (60, 60, 220), "error": (60, 60, 220),
    "out_of_frame": (0, 0, 160),
}


REGION_DEBUG_COLOR = (200, 0, 200)     # magenta -- kotak region OCR gabungan
REGION_SKIPPED_COLOR = (0, 0, 160)     # merah tua -- region out_of_frame (tidak di-OCR)
TOKEN_DEBUG_COLOR = (0, 220, 220)      # cyan -- bbox token OCR individual


def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash=6):
    x1, y1 = pt1
    x2, y2 = pt2
    for (a, b) in [((x1, y1), (x2, y1)), ((x1, y2), (x2, y2))]:
        x_start, x_end = sorted([a[0], b[0]])
        for x in range(x_start, x_end, dash * 2):
            cv2.line(img, (x, a[1]), (min(x + dash, x_end), a[1]), color, thickness)
    for (a, b) in [((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))]:
        y_start, y_end = sorted([a[1], b[1]])
        for y in range(y_start, y_end, dash * 2):
            cv2.line(img, (a[0], y), (a[0], min(y + dash, y_end)), color, thickness)


def draw_debug(image, results, choice_raw=None, template_mode=False, regions=None):
    """regions (opsional): dict region_name -> {"bbox_px": (x1,y1,x2,y2),
    "tokens": [{"bbox": (x1,y1,x2,y2), ...}, ...]} -- koordinat SAMA dgn
    roi_bbox field (full-image / aligned-document space), dari
    dynamic_extraction.extract_dynamic_fields(). Menampilkan kotak region OCR
    gabungan (garis putus-putus magenta) + bbox tiap token OCR individual
    (kotak cyan tipis) supaya region & token yang benar-benar dipakai model
    terlihat, bukan cuma ROI final per field (perbaikan visibilitas #4)."""
    out = image.copy()
    all_results = dict(results)
    all_results.update(choice_raw or {})

    for region_name, info in (regions or {}).items():
        rbbox = info.get("bbox_px")
        if rbbox:
            skipped = info.get("out_of_frame")
            color = REGION_SKIPPED_COLOR if skipped else REGION_DEBUG_COLOR
            rx1, ry1, rx2, ry2 = rbbox
            _draw_dashed_rect(out, (rx1, ry1), (rx2, ry2), color, 1)
            label = f"{region_name} (out_of_frame)" if skipped else region_name
            cv2.putText(out, label, (rx1, max(12, ry1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        for tok in info.get("tokens", []):
            tbbox = tok.get("bbox")
            if not tbbox:
                continue
            tx1, ty1, tx2, ty2 = [int(round(v)) for v in tbbox]
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), TOKEN_DEBUG_COLOR, 1)

    for name, result in all_results.items():
        bbox = result.get("roi_bbox")
        if not bbox:
            continue
        color = (200, 0, 0) if template_mode else STATUS_COLORS.get(result.get("status"), (200, 0, 0))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, name, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)
    return out
