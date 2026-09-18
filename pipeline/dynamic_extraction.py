"""
dynamic_extraction.py
======================
V9.3 (fix atas 2 root-cause regresi V9.2 rev.2 yang dilaporkan user):

1. TOKEN BBOX HILANG DI BATCHING. `ocr.assign_items_to_fields()` (dipakai
   lewat `ocr.run_batched_ocr`) langsung menggabung semua token jadi SATU
   STRING per crop SEBELUM sempat dikembalikan ke pemanggil -- padahal
   `_get_tokens()` versi lama mengharapkan list token individual dgn bbox.
   Kondisi `isinstance(raw, list)` di situ TIDAK PERNAH terpenuhi -> setiap
   region SELALU jatuh ke mode degradasi 1-token, jadi "asosiasi spasial
   token->field" yang jadi inti V9.2 sebenarnya tidak pernah berjalan.
   FIX: `ocr.run_batched_ocr_tokens()` (baru) mengembalikan token individual
   (text/confidence/bbox, bbox sudah region-local) -- dipakai di sini
   langsung, tanpa fallback "degraded_single_token" lagi.

2. REGION DIBANGUN DARI KOORDINAT TEMPLATE, SEBELUM resolve_roi(). Coarse
   region (identity_area/placement_area) sebelumnya berasal dari envelope
   statis `preprocessing.SEMANTIC_REGIONS_NORM` (posisi template apa
   adanya + padding 1%) -- pada dokumen yang skew/goyang, tulisan tangan
   aktual bisa jatuh DI LUAR crop itu sehingga tidak pernah ter-OCR sama
   sekali, walau resolve_roi() PER FIELD (dipanggil belakangan hanya utk
   window koreksi) sebenarnya sudah tahu posisi yang benar.
   FIX: region sekarang dibangun SETELAH & DARI hasil resolve_roi() semua
   field terkait (envelope box target_px+anchor_target_px yang sudah
   dikoreksi per field), bukan sebaliknya.

Sebagai bonus dari fix #1, identity_area & placement_area sekarang cukup
di-OCR dalam SATU panggilan batch gabungan (bukan 1 panggilan per region),
menurunkan jumlah inferensi OCR dinamis dari 2x jadi (ideal) 1x per dokumen.

Alur baru:
  resolve_roi per field (identity+placement fields)
  -> envelope box per region dari hasil TERKOREKSI itu
  -> crop region dari aligned_img (bukan dari static template coords)
  -> upscale + SATU panggilan run_batched_ocr_tokens() utk semua region
  -> bbox token: kanvas-lokal -> region-lokal (unscale) -> FULL-IMAGE
     (koordinat sama dgn target_px/roi_bbox field & dgn aligned_img asli)
  -> asosiasi spasial token->field, SEMUA dalam satu sistem koordinat
     (full-image), pakai window = target_px field (bukan window_local
     campuran skala spt sebelumnya)
  -> fallback: field tanpa token dari batch utama tetap "not_detected"
     (ditangani postprocessing.build_text_field_result spt biasa).
"""

import re

import cv2
from pipeline import ocr
from pipeline import preprocessing as prep
from pipeline.preprocessing import FIELD_CONFIG, CHOICE_GROUPS

# V16 -- bentuk alnum token cetak baris PILIHAN choice (mis. "1/3/6" utk
# tenor, "tunainontunai" utk bentuk reward), DITURUNKAN LANGSUNG dari
# preprocessing.CHOICE_GROUPS (bukan hardcode "1/3/6"). Baris pilihan ini
# printed template & vertikal berdekatan dgn field teks di region yg sama
# (mis. nominal_penempatan bertetangga dgn baris tenor 1/3/6) -- token OCR
# darinya bisa ikut ke window field lain & lolos filter template-text biasa
# krn PENDEK (mis. "136" cuma 3 karakter, di bawah ambang panjang minimum
# _is_static_template_text). Exact-match SAJA (bukan substring) supaya tidak
# menolak angka asli yg kebetulan mengandung sub-string sama.
_CHOICE_OPTION_ALNUM = {
    re.sub(r"[^a-z0-9]", "", "".join(str(opt["label"]) for opt in group["options"].values()).lower())
    for group in CHOICE_GROUPS.values()
}

SEARCH_EXPAND_X = 0.025  # fraksi lebar dokumen -- pelebaran window pencarian value
SEARCH_EXPAND_Y = 0.02   # fraksi tinggi dokumen

REGION_PAD_X_RATIO = 0.03   # padding tambahan sekeliling envelope region (fraksi lebar region)
REGION_PAD_Y_RATIO = 0.03   # (fraksi tinggi region)

# field -> (nama semantic region, label anchor cetak utk menyaring token label)
FIELD_REGION_MAP = {
    "nama_nasabah": ("identity_area", "nama nasabah"),
    "nomor_rekening": ("identity_area", "nomor rekening"),
    "unit_kerja_pengelola_rekening": ("identity_area", "unit kerja"),
    "nominal_penempatan": ("placement_area", "nominal penempatan"),
    "rentang_tenor": ("placement_area", "tenor penempatan"),
    "reward_non_tunai": ("placement_area", "non tunai"),
    "reward_tunai": ("placement_area", "bentuk reward"),
}

# V15: field yg window asosiasinya boleh tumbuh ke bawah (lihat
# _grow_window_downward) -- HANYA reward_tunai, satu-satunya yg terkonfirmasi
# overflow ke baris ke-2 (terbilang nilai reward sering panjang). JANGAN
# tambah field lain ke sini tanpa bukti overflow spesifik (nominal_penempatan
# sempat dicoba, terbukti regresi -- lihat handover.md).
GROWABLE_FIELDS = {"reward_tunai"}

# V17 Finding 1 -- kalau hasil _clamp_field_box_y() (identity_area, SETELAH
# dy-aware clamp -- lihat _resolve_field_boxes) MASIH mengecil jadi kurang
# dari rasio ini thd tinggi template field itu sendiri (norm_bbox_to_px(
# cfg["value_bbox"], shape), TANPA shift), box dianggap COLLAPSED &
# di-rescue lewat prep.resolve_roi() per-field sbg jaring pengaman TAMBAHAN
# (dy-aware clamp sudah menyelesaikan SEMUA 5 kasus collapse yg diuji
# isolated -- record 9/14/18/6/20 -- lihat handover.md; cabang ini defensive
# utk kasus di luar sampel itu, mis. dy antar baris genuinely tidak seragam
# dlm 1 dokumen). Ambang DIUKUR dari baseline 25-dokumen sebelum fix
# (identity_roi_baseline.csv): rasio TERBURUK yg masih genuinely broken
# (dikonfirmasi visual, record 9/14/18/6/20) = 0.731 (record 6 nama_nasabah,
# 19/26px); rasio TERKECIL yg TIDAK pernah dilaporkan bermasalah = 0.782
# (record 4 nomor_rekening, 18/23px). 0.75 duduk tepat di celah itu (margin
# ~0.05 di kedua sisi).
FIELD_BOX_COLLAPSE_RATIO = 0.75


# ============================================================================
# BBOX HELPERS (koordinat konsisten: semua fungsi di bawah bekerja di ATAS
# koordinat full-image/aligned-document, kecuali disebutkan region-local)
# ============================================================================

def _normalize_bbox(bbox):
    """Terima bbox axis-aligned (x1,y1,x2,y2) ATAU polygon 4-titik -> selalu
    kembalikan axis-aligned (x1,y1,x2,y2). Defensif krn bentuk PERSIS bbox
    dari ocr.py tidak dijamin stabil lintas versi PaddleOCR."""
    pts = list(bbox)
    if len(pts) == 8 and all(isinstance(v, (int, float)) for v in pts):
        xs, ys = pts[0::2], pts[1::2]
        return (min(xs), min(ys), max(xs), max(ys))
    if len(pts) == 4 and all(hasattr(p, "__len__") and len(p) == 2 for p in pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    return tuple(pts)


def _clip(bbox, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def _envelope(boxes):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def _pad_bbox(bbox, shape, pad_x_ratio, pad_y_ratio):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px, py = w * pad_x_ratio, h * pad_y_ratio
    return _clip((x1 - px, y1 - py, x2 + px, y2 + py), shape)


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _expand(bbox, scale_w, scale_h, bounds):
    """Perlebar window pencarian value SEBANDING dgn ukuran REGION (scale_w/
    scale_h -- spt V9.1/rev.2, BUKAN ukuran halaman penuh: field-field dlm
    satu region berjarak rapat, jadi window harus tetap sempit relatif thd
    region, bukan relatif thd halaman -- kalau tidak, window antar field yg
    bertetangga bisa saling tumpang tindih & token 1 field ikut "bocor" ke
    field lain). Diklip ke `bounds` (region bbox, full-image coords) supaya
    window tidak pernah menembus keluar region OCR yang sebenarnya di-OCR."""
    x1, y1, x2, y2 = bbox
    dx, dy = scale_w * SEARCH_EXPAND_X, scale_h * SEARCH_EXPAND_Y
    bx1, by1, bx2, by2 = bounds
    return (max(bx1, x1 - dx), max(by1, y1 - dy), min(bx2, x2 + dx), min(by2, y2 + dy))


def _grow_window_downward(target_px, field_name, shape):
    """V15: perluas sisi BAWAH window asosiasi token field ke bawah (bukan
    target_px/debug box asli, bukan sisi atas) supaya menangkap baris
    lanjutan tulisan tangan yg overflow (mis. terbilang panjang "tiga puluh
    ...tujuh juta lima ratus ribu rupiah" yg ditulis lanjut ke baris ke-2,
    field aslinya cuma setinggi 1 baris cetak) -- dikonfirmasi via inspeksi
    debug image: token baris ke-2 sebelumnya SAMA SEKALI tidak ter-assign ke
    field manapun (di luar window manapun, bukan salah field).

    Pertumbuhan di-cap KERAS ke prep.NEXT_FIELD_TOP_NORM (batas field/choice/
    tanda tangan LAIN terdekat di bawahnya, sudah dihitung utk SEMUA field --
    lihat preprocessing._build_next_field_top_map) -- utk identity_area yg
    gap antar barisnya NOL (lihat V14), ini otomatis membatasi pertumbuhan ke
    ~0px (aman by construction, tidak perlu exclude identity_area secara
    eksplisit). Tidak pernah mengubah sisi atas -- growth SEARAH (turun saja),
    tidak membuka kembali risiko field "naik" ke baris sebelumnya."""
    x1, y1, x2, y2 = target_px
    base_h = max(1.0, y2 - y1)
    extra = base_h * prep.ROI_EXTRA_HEIGHT_RATIO
    ceiling_norm = prep.NEXT_FIELD_TOP_NORM.get(field_name, 1.0)
    ceiling_px = ceiling_norm * shape[0] - 2  # margin kecil, jangan sampai sentuh field tetangga
    grown_y2 = min(y2 + extra, ceiling_px)
    return (x1, y1, x2, int(round(max(y2, grown_y2))))


_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def _is_noise_token(text):
    """Token OCR yg cuma tanda baca/simbol murni (mis. artefak pemisah baris
    yg ke-baca sbg karakter sendiri) -- tidak pernah bagian isian asli."""
    t = (text or "").strip()
    return not t or bool(_PUNCT_ONLY_RE.match(t))


def _is_static_template_text(text):
    """V16 (Section 1) -- tolak token yg sebenarnya teks CETAK STATIS template
    (judul/paragraf, BUKAN label field individual -- itu ditangani terpisah
    lewat neighbor_labels di pemanggil) sebelum ikut jadi kandidat value.
    Diturunkan LANGSUNG dari template.pdf (prep.extract_template_static_text,
    di-cache), TIDAK di-hardcode dari daftar contoh manapun. Token pendek
    (<4 karakter alnum) dilewati dari cek ini supaya angka/kata pendek yg
    memang bagian isian asli (mis. potongan nominal) tidak ikut tertolak
    krn kebetulan cocok potongan kalimat statis."""
    alnum = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    if len(alnum) < 4:
        return False
    return any(_fuzzy_contains(text, fragment) for fragment in prep.extract_template_static_text())


def _cluster_token_rows(tokens):
    """Kelompokkan token jadi baris berdasarkan overlap vertikal ANTAR TOKEN
    (bukan kuantisasi kasar relatif tinggi region) -- dua token dianggap SATU
    baris kalau pusat-y salah satu berada dlm rentang tinggi gabungan baris
    itu (+toleransi), supaya sedikit miring/naik-turun tulisan tangan tetap
    tergabung, tapi baris judul/label yg jelas beda tinggi tidak ikut
    tercampur ke baris value yg sebenarnya."""
    ordered = sorted(tokens, key=lambda t: t["bbox"][1])
    rows = []
    for tok in ordered:
        y1, y2 = tok["bbox"][1], tok["bbox"][3]
        cy = (y1 + y2) / 2.0
        for row in rows:
            ry1, ry2 = row["y_range"]
            tol = 0.4 * max(1.0, ry2 - ry1)
            if ry1 - tol <= cy <= ry2 + tol:
                row["tokens"].append(tok)
                row["y_range"] = (min(ry1, y1), max(ry2, y2))
                break
        else:
            rows.append({"tokens": [tok], "y_range": (y1, y2)})
    return rows


def _score_row_group(row, target_px, anchor_target_px, window_px, field_type):
    """Rangking kandidat baris (Section 1): Y-overlap dgn target_px (same-row),
    arah yg benar dari label (harus di KANAN anchor cetak), jarak ke pusat
    target_px, overlap dgn window ROI akhir yg dipakai, kecocokan tipe field
    (digit-dominan utk numeric/currency, huruf-dominan utk teks), dan
    confidence rata-rata sbg SATU sinyal tambahan (bukan penentu tunggal --
    lihat requirement #3, tidak pernah dipakai utk MENOLAK kandidat)."""
    tx1, ty1, tx2, ty2 = target_px
    boxes = [t["bbox"] for t in row["tokens"]]
    gx1 = min(b[0] for b in boxes); gy1 = min(b[1] for b in boxes)
    gx2 = max(b[2] for b in boxes); gy2 = max(b[3] for b in boxes)

    row_overlap = max(0.0, min(gy2, ty2) - max(gy1, ty1)) / max(1.0, ty2 - ty1)

    ax2 = anchor_target_px[2]
    direction_ok = 1.0 if gx1 >= ax2 - 4 else 0.0

    tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
    gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
    win_h = max(1.0, window_px[3] - window_px[1])
    dist = ((gcx - tcx) ** 2 + (gcy - tcy) ** 2) ** 0.5 / win_h

    wx1, wy1, wx2, wy2 = window_px
    inter_area = max(0.0, min(gx2, wx2) - max(gx1, wx1)) * max(0.0, min(gy2, wy2) - max(gy1, wy1))
    group_area = max(1.0, (gx2 - gx1) * (gy2 - gy1))
    roi_overlap = inter_area / group_area

    text = "".join(t.get("text", "") for t in row["tokens"])
    digits = sum(c.isdigit() for c in text)
    alpha = sum(c.isalpha() for c in text)
    total = max(1, digits + alpha)
    type_score = (digits / total) if field_type in ("numeric", "currency") else (alpha / total)

    confs = [t["confidence"] for t in row["tokens"] if t.get("confidence") is not None]
    conf_score = (sum(confs) / len(confs)) if confs else 0.5

    score = (
        3.0 * row_overlap + 1.5 * direction_ok + 2.0 * roi_overlap
        + 1.0 * type_score + 0.3 * conf_score - 1.0 * dist
    )
    return score


def _fuzzy_contains(token_text, label):
    t = re.sub(r"[^a-z]", "", (token_text or "").lower())
    l = re.sub(r"[^a-z]", "", label.lower())
    if not t or not l:
        return False
    return t in l or l[: max(4, len(l) - 2)] in t


def _label_pattern(label):
    words = re.split(r"\s+", label.strip())
    return re.compile(r"\s*".join(re.escape(w) for w in words), re.IGNORECASE)


def _strip_label_remainder(text, label):
    """Kalau satu token OCR menggabungkan label+value (baris pendek yg
    ke-merge, sering terjadi pd foto miring/goyang), potong bagian label &
    kembalikan sisa teks setelah label sbg kandidat value -- bukan dibuang
    sepenuhnya."""
    m = _label_pattern(label).search(text or "")
    if not m:
        return None
    remainder = (text or "")[m.end():].lstrip(" :.\u2013\u2014-")
    return remainder.strip() or None


# ============================================================================
# STEP 1: resolve_roi per field (SUMBER KOREKSI UTAMA, dipanggil SEKALI di
# awal) -> dipakai baik utk membangun region (fix root-cause #2) maupun utk
# window asosiasi spasial per field (fix konsistensi koordinat).
# ============================================================================

def _resolve_section_transforms(template_gray, aligned_gray, shape):
    """V10: satu transform robust per section (identity/placement), dibangun
    dari BEBERAPA anchor cetak sekaligus (lihat preprocessing.
    estimate_section_transform) -- menggantikan koreksi dx/dy independen per
    field sebagai sumber koreksi UTAMA. Return dict region_name -> transform
    diagnostik (termasuk 'M' atau None).

    V15.1: anchor judul (prep.TITLE_ANCHOR_ENTRY) SELALU dicocokkan sbg
    cross-check independen (bukan digabung ke anchor field) -- section_bbox_norm
    (SEMANTIC_REGIONS_NORM) diteruskan supaya cross-check-nya pakai overlap
    geometris (V15.1-b), bukan jarak translasi polos (terbukti tidak cukup
    diskriminatif, lihat update.md). Lihat preprocessing.estimate_section_transform."""
    return {
        region_name: prep.estimate_section_transform(
            template_gray, aligned_gray, shape, anchor_bboxes, title_anchor=prep.TITLE_ANCHOR_ENTRY,
            section_bbox_norm=prep.SEMANTIC_REGIONS_NORM.get(region_name)
        )
        for region_name, anchor_bboxes in prep.SECTION_ANCHOR_BBOXES.items()
    }


def _resolve_field_boxes(template_gray, aligned_gray, shape, section_transforms):
    boxes = {}
    for field_name, (region_name, _label) in FIELD_REGION_MAP.items():
        cfg = FIELD_CONFIG[field_name]
        template_px, target_px, source, evidence = prep.resolve_roi_section(
            template_gray, aligned_gray, cfg, shape, section_transforms.get(region_name)
        )
        # V14: kunci sisi atas/bawah ke batas field tetangga (lihat
        # prep._clamp_field_box_y docstring) -- section transform berbasis
        # translasi-saja (fallback <3 anchor) tidak bisa mengoreksi rotasi
        # sisa, jadi row yg jauh dari anchor bisa drift ke row tetangga yg
        # gap-nya NOL di template. Safety net ini SUDAH ADA di codebase tapi
        # sebelumnya cuma dipakai utk tempat_tanggal_surat.
        #
        # SENGAJA DIBATASI ke identity_area saja (bukan semua region): dicoba
        # universal dulu, tapi evaluasi 25-dokumen (evaluation_summary_
        # roi_clamp_fix.json vs _fixed_v13.json) menunjukkan regresi nyata di
        # placement_area (record 15/18, rentang_tenor benar->N/A) -- di sana
        # PREV/NEXT_FIELD_*_NORM ikut dibangun dari box CHOICE_GROUPS
        # (opsi tenor/reward) yang SECARA VERTIKAL TUMPANG TINDIH dgn baris
        # teks di sekitarnya (bukan struktur 3-baris rapi spt identity_area),
        # jadi batas tetangga yang dihitung bisa salah/kontradiktif di sana.
        # identity_area TIDAK punya masalah itu (3 field teks murni, tanpa
        # choice box yang overlap) -- root cause yang terverifikasi (lihat
        # handover.md) memang spesifik di sini.
        if region_name == "identity_area":
            # V17 Finding 1 -- batas tetangga (PREV_FIELD_BOTTOM_NORM/
            # NEXT_FIELD_TOP_NORM, koordinat TEMPLATE tanpa shift) HARUS
            # digeser sejauh dy YANG SAMA dgn box field ini sendiri SEBELUM
            # di-clamp -- lihat prep._clamp_field_box_y docstring utk root
            # cause lengkap (TERVERIFIKASI via debug empirik, BUKAN teori):
            # section transform (consensus_translation/section_affine) BISA
            # menggeser satu baris beberapa piksel scr SAH, tapi batas
            # statis lama TIDAK ikut bergeser -> box yg SUDAH BENAR malah
            # ke-clip. RENCANA AWAL (rescue via prep.resolve_roi per-field,
            # TANPA dy-aware clamp) TERBUKTI TIDAK CUKUP saat diuji isolated
            # (record 9/14/18): anchor lokal field itu SENDIRI menghitung
            # dx/dy PERSIS SAMA dgn section transform pada 4/5 kasus yg
            # diuji -- BUKAN evidence independen spt diasumsikan, keduanya
            # menemukan pergeseran GENUINE yg sama, jadi rescue lama
            # ke-clamp dgn cara & hasil yg SAMA (collapse tidak hilang).
            raw_dy = target_px[1] - template_px[1]
            target_px = prep._clamp_field_box_y(target_px, field_name, shape, dy=raw_dy)
            # Defensive fallback (jarang aktif stlh dy-aware clamp di atas,
            # tapi tetap dipertahankan sbg jaring pengaman tambahan): kalau
            # MASIH collapse (mis. dy antar baris genuinely tidak seragam
            # dlm 1 dokumen), rescue lewat resolve_roi per-field (V9.x,
            # anchor lokal sendiri), clamp ULANG pakai dy hasil rescue-nya
            # sendiri (BUKAN dy lama yg sudah terbukti tidak relevan lagi
            # kalau resolve_roi menemukan offset berbeda).
            template_h = max(1.0, template_px[3] - template_px[1])
            clamped_h = target_px[3] - target_px[1]
            if clamped_h / template_h < FIELD_BOX_COLLAPSE_RATIO:
                _, rescued_px, rescue_source, rescue_evidence = prep.resolve_roi(
                    template_gray, aligned_gray, cfg, shape
                )
                rescued_dy = rescued_px[1] - template_px[1]
                target_px = prep._clamp_field_box_y(rescued_px, field_name, shape, dy=rescued_dy)
                source = f"identity_collapse_rescue_{rescue_source}"
                evidence = rescue_evidence
        dx = target_px[0] - template_px[0]
        dy = target_px[1] - template_px[1]
        anchor_template_px = prep.norm_bbox_to_px(cfg["anchor_bbox"], shape)
        anchor_target_px = _clip(
            (anchor_template_px[0] + dx, anchor_template_px[1] + dy,
             anchor_template_px[2] + dx, anchor_template_px[3] + dy),
            shape,
        )
        boxes[field_name] = {
            "target_px": target_px,
            "anchor_target_px": anchor_target_px,
            "source": source,
            "evidence": evidence,
        }
    return boxes


def _build_regions(field_boxes, shape):
    """Region OCR gabungan = envelope dari box field yang SUDAH TERKOREKSI
    (target_px + anchor_target_px, hasil resolve_roi per field), BUKAN
    envelope statis koordinat template. Ini memastikan crop yang benar-benar
    di-OCR selalu mengikuti posisi aktual dokumen (skew/goyang), bukan
    posisi nominal template."""
    region_fields = {}
    for field_name, (region_name, _label) in FIELD_REGION_MAP.items():
        region_fields.setdefault(region_name, []).append(field_name)

    regions = {}
    for region_name, field_names in region_fields.items():
        boxes = []
        for f in field_names:
            boxes.append(field_boxes[f]["target_px"])
            boxes.append(field_boxes[f]["anchor_target_px"])
        env = _envelope(boxes)
        regions[region_name] = _pad_bbox(env, shape, REGION_PAD_X_RATIO, REGION_PAD_Y_RATIO)
    return regions


# ============================================================================
# STEP 2-4: crop region terkoreksi -> SATU batch OCR token-preserving ->
# konversi bbox token ke koordinat full-image -> asosiasi spasial per field.
# ============================================================================

def extract_dynamic_fields(aligned_img, template_gray, aligned_gray, shape, coverage_mask=None):
    """Return (result, roi_boxes, regions_debug, ocr_calls, field_statuses, section_meta):
      - result: field_name -> {"raw": text, "confidence": float} -- bentuk
        SAMA seperti sebelumnya, supaya postprocessing.build_text_field_result
        tidak perlu diubah.
      - roi_boxes: field_name -> bbox piksel full-image (utk debug overlay
        ROI final per field, spt sebelumnya).
      - regions_debug: region_name -> {"bbox_px": ..., "tokens": [...]}
        (koordinat full-image) -- utk visibilitas debug region OCR gabungan
        & bbox token individual.
      - ocr_calls: jumlah panggilan model OCR aktual (ideal: 1 utk dokumen
        normal, krn identity_area+placement_area digabung dlm satu batch).
      - field_statuses: field_name -> "out_of_frame" utk field yang boxnya
        jatuh di luar cakupan piksel sumber (foto terpotong/parsial) -- TIDAK
        dikirim ke OCR sama sekali & TIDAK boleh ditafsirkan sbg "blank".
      - section_meta: region_name -> diagnostik section transform (method/
        anchors_used/anchors_total/outliers/rotation/scale) utk debug/eval.
    """
    section_transforms = _resolve_section_transforms(template_gray, aligned_gray, shape)
    field_boxes = _resolve_field_boxes(template_gray, aligned_gray, shape, section_transforms)
    region_bbox_px = _build_regions(field_boxes, shape)

    field_statuses = {}
    active_region_bbox = {}
    for region_name, bbox in region_bbox_px.items():
        if coverage_mask is not None and prep.is_out_of_frame(coverage_mask, bbox):
            for f, (rn, _l) in FIELD_REGION_MAP.items():
                if rn == region_name:
                    field_statuses[f] = "out_of_frame"
            continue
        active_region_bbox[region_name] = bbox

    scale = getattr(ocr, "OCR_SCALE_UP", 1.0) or 1.0
    crops = []
    for region_name, bbox in active_region_bbox.items():
        x1, y1, x2, y2 = bbox
        crop = aligned_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        scaled_crop = (
            cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if scale != 1.0 else crop
        )
        crops.append((region_name, scaled_crop))

    # SATU panggilan batch (internal bisa terpecah jadi >1 kanvas HANYA
    # kalau melebihi batas VRAM device, lihat ocr.MAX_CANVAS_HEIGHT_*) utk
    # identity_area + placement_area sekaligus -- bukan 1 panggilan per
    # region seperti sebelumnya.
    tokens_by_region, calls = ocr.run_batched_ocr_tokens(crops)

    # bbox token: kanvas-lokal (SUDAH di-upscale) -> region-lokal asli
    # (unscale) -> FULL-IMAGE (tambah origin region). Satu sistem koordinat
    # konsisten dgn target_px/roi_bbox field mulai titik ini.
    full_tokens_by_region = {}
    for region_name, tokens in tokens_by_region.items():
        if region_name not in active_region_bbox:
            continue
        rx1, ry1, _, _ = active_region_bbox[region_name]
        converted = []
        for t in tokens:
            bx1, by1, bx2, by2 = _normalize_bbox(t["bbox"])
            if scale != 1.0:
                bx1, by1, bx2, by2 = bx1 / scale, by1 / scale, bx2 / scale, by2 / scale
            full_bbox = (bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1)
            converted.append({**t, "bbox": full_bbox})
        full_tokens_by_region[region_name] = converted

    result, roi_boxes = {}, {}
    for field_name, (region_name, label) in FIELD_REGION_MAP.items():
        fb = field_boxes[field_name]
        roi_boxes[field_name] = fb["target_px"]

        if field_name in field_statuses:
            continue  # out_of_frame -- tidak pernah di-OCR, jangan dianggap blank

        tokens = full_tokens_by_region.get(region_name)
        if not tokens:
            continue

        rbbox = active_region_bbox[region_name]
        rw, rh = rbbox[2] - rbbox[0], rbbox[3] - rbbox[1]
        # V15: growth dibatasi ke field yg TERKONFIRMASI overflow ke baris ke-2
        # (reward_tunai -- terbilang nilai reward sering panjang, lihat
        # docstring _grow_window_downward). SEMPAT dicoba ke semua 7 field,
        # TERBUKTI regresi nominal_penempatan (field angka 1 baris, tidak
        # overflow -- growth-nya cuma menambah risiko menangkap token yg
        # bukan miliknya tanpa manfaat nyata). Jangan diperluas lagi ke field
        # lain tanpa bukti overflow spesifik spt ini.
        if field_name in GROWABLE_FIELDS:
            target_for_window = _grow_window_downward(fb["target_px"], field_name, shape)
            roi_boxes[field_name] = target_for_window  # debug box jujur menampilkan window yg sebenarnya dipakai
        else:
            target_for_window = fb["target_px"]
        wx1, wy1, wx2, wy2 = _expand(target_for_window, rw, rh, rbbox)

        # V16 (Section 1) -- best-candidate spatial assignment, bukan lagi
        # "kumpulkan semua token dlm window & gabung apa adanya". Tiga
        # penolakan dulu (label sendiri -- SAMA seperti sebelumnya; label
        # field TETANGGA di region yg sama; teks cetak statis template dari
        # template.pdf), baru sisanya dikelompokkan jadi baris & baris
        # TERBAIK yg dipilih (bukan seluruh window) -- lihat _score_row_group.
        neighbor_labels = [
            other_label for other_name, (other_region, other_label) in FIELD_REGION_MAP.items()
            if other_region == region_name and other_name != field_name
        ]

        candidates, raw_token_texts = [], []
        for tok in tokens:
            text = tok.get("text") or ""
            if _fuzzy_contains(text, label):
                remainder = _strip_label_remainder(text, label)
                if remainder:
                    candidates.append({**tok, "text": remainder})
                continue  # token label cetak murni (tanpa sisa) -- lewati, SAMA spt sebelumnya
            tcx, tcy = _bbox_center(tok["bbox"])
            if not (wx1 <= tcx <= wx2 and wy1 <= tcy <= wy2):
                continue
            raw_token_texts.append(text)  # bukti mentah SEMUA token dlm window, SEBELUM filter di bawah
            if _is_noise_token(text):
                continue
            if any(_fuzzy_contains(text, nb) for nb in neighbor_labels):
                continue  # label field TETANGGA yg ikut ke-window (leakage antar field) -- lewati
            if _is_static_template_text(text):
                continue  # teks cetak statis (judul/paragraf template) -- lewati
            if re.sub(r"[^a-z0-9]", "", text.lower()) in _CHOICE_OPTION_ALNUM:
                continue  # baris pilihan choice cetak (mis. "1/3/6") -- lewati, lihat _CHOICE_OPTION_ALNUM
            candidates.append(tok)

        if not candidates:
            continue

        if field_name in GROWABLE_FIELDS:
            # Field ini SUDAH terkonfirmasi overflow ke baris ke-2 (lihat
            # docstring GROWABLE_FIELDS) -- SEMUA baris dlm window (setelah
            # filtering di atas) tetap digabung apa adanya (bukan single-
            # best-row) supaya baris lanjutan terbilang tidak hilang.
            ordered = sorted(
                candidates,
                key=lambda t: (round((t["bbox"][1] - wy1) / max(1.0, wy2 - wy1) * 10), t["bbox"][0]),
            )
        else:
            rows = _cluster_token_rows(candidates)
            best_row = max(
                rows,
                key=lambda r: _score_row_group(
                    r, fb["target_px"], fb["anchor_target_px"], (wx1, wy1, wx2, wy2),
                    FIELD_CONFIG[field_name]["type"],
                ),
            )
            ordered = sorted(best_row["tokens"], key=lambda t: t["bbox"][0])

        text = " ".join(t.get("text", "") for t in ordered if t.get("text")).strip()
        confs = [t["confidence"] for t in ordered if t.get("confidence") is not None]
        confidence = min(confs) if confs else None
        if text:
            result[field_name] = {"raw": text, "confidence": confidence, "raw_tokens": raw_token_texts}

    regions_debug = {
        region_name: {
            "bbox_px": bbox,
            "tokens": full_tokens_by_region.get(region_name, []),
            "out_of_frame": region_name not in active_region_bbox,
        }
        for region_name, bbox in region_bbox_px.items()
    }
    section_meta = {
        region_name: {k: v for k, v in t.items() if k != "M"}
        for region_name, t in section_transforms.items()
    }

    return result, roi_boxes, regions_debug, calls, field_statuses, section_meta
