"""
preprocessing.py
================
Semua langkah dari "file mentah" -> "crop tulisan tangan siap OCR":
  load dokumen -> enhance scan (foto) -> align ke template ->
  localized anchor detection (matchTemplate, TANPA OCR) -> dynamic ROI ->
  template subtraction -> buang garis -> crop tinta rapat.

Juga berisi definisi ROI (FIELD_CONFIG/CHOICE_GROUPS/SIGNATURE_CONFIG) karena
koordinat ini adalah bagian dari "persiapan area OCR", bukan logika OCR itu
sendiri. Koordinat diambil presisi dari template.pdf via pdfplumber
(lihat handover.md kalau template berubah dan koordinat perlu diekstrak ulang).
"""

from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from core.paths import ASSETS_DIR, PROJECT_ROOT


# ============================================================================
# CONFIG
# ============================================================================

PDF_DPI = 180
PHOTO_RECTIFY = True

# V13 -- lihat _find_document_quad(): minimal fraksi area foto yg harus
# ditempati kertas (kontur terluar) supaya dianggap "kertas ketemu" & layak
# di-rectify. Sama seperti batas lama (rectify_photo V-awal), dipertahankan
# supaya tidak asal menerima kontur kecil sbg kertas -- yang berubah adalah
# ROBUSTNESS deteksi kontur itu sendiri (adaptif thd pencahayaan + tepi
# terputus), bukan seberapa longgar syarat ukurannya.
DOCUMENT_QUAD_MIN_AREA_RATIO = 0.30

ALIGN_MAX_SIDE = 1800
ALIGN_FEATURES = 6000
ALIGN_RATIO = 0.78
ALIGN_RANSAC = 4.0
ALIGN_MIN_MATCHES = 18

# V9.1 -- kebijakan alignment konservatif: similarity/affine (translasi +
# rotasi + skala seragam, TANPA shear/perspektif) didahulukan drpd
# homography, karena homography bisa "meng-shear/meng-perspective-warp"
# dokumen yang sebenarnya sudah lurus. Homography hanya dipakai kalau
# affine gagal sanity-check ATAU homography objectively better (inlier
# ratio jelas lebih tinggi).
AFFINE_MAX_ROTATION_DEG = 8.0
AFFINE_SCALE_RANGE = (0.85, 1.15)
HOMOGRAPHY_MIN_RELATIVE_GAIN = 0.15

# V10 -- geometry sanity for GLOBAL alignment quality (beyond good_matches/
# inlier_ratio alone): reject transforms whose inliers are clustered in a
# small corner of the page (mis-scoped "good" alignment) or whose
# reprojection error is high (numerically unstable fit), even if inlier
# COUNT looked fine. Measured in the RESIZED/small ORB matching space.
ALIGN_MIN_COVERAGE_RATIO = 0.12       # inlier bbox area / small-canvas area
ALIGN_MAX_REPROJ_ERROR_PX = 4.0       # mean reprojection error, small-canvas px
# Exclude the top of the page (letterhead/branch/address -- highly variable
# across scans, unlike the printed form body) from ORB keypoint detection so
# matches are not dominated by header text.
HEADER_EXCLUDE_RATIO = 0.10

# V10 -- section-level local transform (robust multi-anchor affine per
# coarse section) sanity, tighter than global alignment since it only needs
# to explain a small local patch, not the whole page.
SECTION_MAX_ROTATION_DEG = 6.0
SECTION_SCALE_RANGE = (0.85, 1.18)
SECTION_ANCHOR_MIN_SIMILARITY = 0.32
SECTION_OUTLIER_DIST_RATIO = 0.02     # fraction of page diagonal

# V10 -- source-pixel coverage: a region/field whose corrected box falls
# mostly outside the actual photographed content (cropped/partial photo)
# must not silently become "blank" -- it is flagged "out_of_frame" instead.
COVERAGE_MIN_RATIO = 0.85

ANCHOR_SEARCH_TOL_X = 0.020     # toleransi pencarian anchor (fraksi lebar)
ANCHOR_SEARCH_TOL_Y = 0.015     # toleransi pencarian anchor (fraksi tinggi)
ANCHOR_MATCH_MIN = 0.30         # skor matchTemplate minimum agar dianggap valid

# V15 -- root cause "block-shift": cabang mencetak letterhead dgn TINGGI
# BERBEDA-BEDA (sebagian pakai blok alamat/telp/fax tambahan spt template
# referensi, sebagian tidak) -- konten identity_area/placement_area di bawah
# letterhead ikut bergeser sejumlah TETAP (bukan proporsional/scale), yg
# TIDAK bisa direpresentasikan oleh affine/homography global (itu hanya
# scale+rotasi+translasi SERAGAM utk seluruh halaman). Terverifikasi visual
# 25-dokumen: 6/25 (24%) identity_area/placement_area jatuh di teks judul
# cetak "FORMULIR KEIKUTSERTAAN"/paragraf, BUKAN field asli -- krn toleransi
# pencarian anchor per-field (ANCHOR_SEARCH_TOL_Y di atas, ~31px pd kanvas
# 2105px) jauh lebih kecil drpd selisih tinggi letterhead (~60-100px),
# sehingga anchor field itu sendiri gagal/salah ketemu SEBELUM sempat
# mengoreksi apa pun.
#
# Fix: judul form ("FORMULIR KEIKUTSERTAAN" + "Program BRI CUAN HADIAH 2026")
# SELALU ada persis di posisi yg sama relatif thd identity_area/placement_area
# di SEMUA sampel yg diperiksa, besar & khas (2 baris tebal terpusat) --
# kandidat anchor yg jauh lebih andal drpd label field kecil. Dipakai sbg
# anchor TAMBAHAN (bukan pengganti) di SECTION_ANCHOR_BBOXES dgn toleransi
# pencarian JAUH LEBIH LEBAR (lihat TITLE_ANCHOR_SEARCH_TOL_Y) khusus utk
# anchor ini saja, supaya bisa menjangkau letterhead setinggi apa pun --
# tetap masuk lewat estimate_section_transform's outlier-rejection +
# affine-fit yg SAMA (bukan jalur terpisah), jadi kalau ternyata tidak
# ketemu/tidak konsisten dgn anchor lain, otomatis tidak dipakai (tidak
# pernah memperburuk dokumen yg sudah normal).
TITLE_ANCHOR_BBOX_NORM = (0.38, 0.111, 0.63, 0.136)
TITLE_ANCHOR_SEARCH_TOL_X = 0.030
TITLE_ANCHOR_SEARCH_TOL_Y = 0.070   # ~147px pd kanvas 2105px -- jangkau letterhead jauh lebih tinggi drpd template

DIFF_THRESHOLD = 38
BLANK_THRESHOLD = 0.008         # rasio piksel tinta minimum agar dianggap terisi
CROP_PAD_RATIO = 0.20           # padding vertikal/horizontal utk ascender/descender

# ROI diperluas dari value_bbox template (BUKAN kembali ke fixed ROI absolut):
# masih anchor-relative, hanya batasnya dilonggarkan supaya tulisan yang
# bergeser/lebih panjang/turun ke baris berikut tetap tertangkap.
ROI_EXPAND_X_RATIO = 0.35        # tambahan lebar kanan-kiri, fraksi dari lebar ROI asli
ROI_EXTRA_HEIGHT_RATIO = 1.0     # tambahan tinggi ke bawah, fraksi dari tinggi ROI asli (~1 baris)
LINE_GAP_MAX_RATIO = 0.9         # jarak maksimum antar baris tinta agar dianggap "baris lanjutan"
MAX_INK_LINES = 2                # maksimum baris yang diambil (baris utama + 1 baris bawah)
INK_BAND_MERGE_GAP = 3           # px kosong yang masih dianggap satu goresan (bukan baris baru)

_ASSET_TEMPLATE = ASSETS_DIR / "template.pdf"
_LOCAL_TEMPLATE = PROJECT_ROOT / "template.pdf"
TEMPLATE_PATH = str(_ASSET_TEMPLATE if _ASSET_TEMPLATE.exists() else _LOCAL_TEMPLATE)

_template_static_text_cache = None


def extract_template_static_text(template_path=None):
    """V15.2 -- ekstrak fragmen teks CETAK STATIS (judul/label/paragraf) dari
    template.pdf langsung (pymupdf, SUDAH dependency -- tidak perlu
    pdfplumber baru) -- dipakai field quality gate (pipeline.py) utk deteksi
    "template leakage": field value yg ternyata berisi teks cetak (label
    sendiri/tetangga/judul/paragraf), bukan isian tulisan tangan/user -- bukti
    kuat ROI salah tempat. TIDAK di-hardcode dari luar, SELALU diturunkan
    ulang dari template.pdf per baris teks (per requirement eksplisit).
    Di-cache in-memory (template.pdf statis, tidak berubah selama proses
    jalan) -- panggil sekali per proses, bukan per field/per dokumen."""
    global _template_static_text_cache
    if _template_static_text_cache is not None:
        return _template_static_text_cache
    path = template_path or TEMPLATE_PATH
    doc = fitz.open(str(path))
    text = doc[0].get_text("text")
    doc.close()
    fragments = []
    for line in text.split("\n"):
        line = line.replace("​", "").strip()
        alnum = sum(1 for c in line if c.isalnum())
        if len(line) < 3 or alnum < 3:
            continue  # baris kosong/placeholder isian (titik-titik/underscore), bukan teks statis sungguhan
        fragments.append(line)
    _template_static_text_cache = fragments
    return fragments


# Koordinat dinormalisasi (0-1), diekstrak presisi dari template.pdf asli.
# anchor_bbox : potongan cetakan yang TIDAK PERNAH tertimpa tulisan/coretan
#               pengguna -> dipakai template matching lokal.
# value_bbox  : lokasi nilai yang harus diisi (posisi dasar tanpa offset;
#               juga dipakai sebagai fixed-ROI fallback).
FIELD_CONFIG = {
    "nama_nasabah": {
        "type": "text",
        "anchor_bbox": (0.1208, 0.1369, 0.2102, 0.1464),
        "value_bbox": (0.3691, 0.1354, 0.9396, 0.1478),
    },
    "nomor_rekening": {
        "type": "numeric",
        "anchor_bbox": (0.1208, 0.1484, 0.2188, 0.1579),
        "value_bbox": (0.3691, 0.1478, 0.9396, 0.1585),
    },
    "unit_kerja_pengelola_rekening": {
        "type": "text",
        "anchor_bbox": (0.1208, 0.1598, 0.2982, 0.1693),
        "value_bbox": (0.3691, 0.1585, 0.9396, 0.1704),
    },
    "nominal_penempatan": {
        "type": "currency",
        "anchor_bbox": (0.1208, 0.2457, 0.2465, 0.2552),
        "value_bbox": (0.3691, 0.2441, 0.8893, 0.2568),
        "strip_terms": ["rp"],
    },
    "rentang_tenor": {
        "type": "text",
        "anchor_bbox": (0.1208, 0.2571, 0.2315, 0.2666),   # "Tenor Penempatan"
        "value_bbox": (0.3520, 0.2673, 0.7718, 0.2798),
    },
    "reward_non_tunai": {
        "type": "optional_text",
        "anchor_bbox": (0.3708, 0.2915, 0.6142, 0.3010),
        "value_bbox": (0.6184, 0.2900, 0.9200, 0.3025),
    },
    "reward_tunai": {
        "type": "currency",
        "anchor_bbox": (0.1208, 0.3145, 0.2998, 0.3240),
        "value_bbox": (0.3691, 0.3130, 0.7718, 0.3246),
        "strip_terms": ["rp"],
    },
    "tempat_tanggal_surat": {
        "type": "text",
        "anchor_bbox": (0.1208, 0.7437, 0.2459, 0.7532),   # "Demikian pernyataan ini"
        "value_bbox": (0.1510, 0.7756, 0.5201, 0.7900),
    },
}

# Choice & signature: HANYA image-diff, tidak pernah di-OCR.
# CATATAN anchor: SEBELUMNYA anchor group memakai label baris ("Tenor
# Penempatan"/"Bentuk Reward") yang berjarak JAUH (~0.25 lebar halaman) dari
# kotak opsi 1/3/6 & tunai/non-tunai yang SANGAT SEMPIT (~0.02 lebar). Skew/
# distorsi kecil pada anchor yang jauh itu ikut membesar (amplified) di posisi
# opsi yang jauh darinya, membuat ROI opsi meleset dari karakter yang mau
# dibaca -> hasil "tidak bisa terpilih". Anchor sekarang dipindah ke teks
# "(*coret salah satu)" yang PERSIS ada di baris & sisi kanan yang sama
# dengan opsi (jauh lebih dekat), sehingga koreksi offset jauh lebih presisi
# di posisi opsi itu sendiri.
CHOICE_GROUPS = {
    "tenor_penempatan": {
        "anchor_bbox": (0.4617, 0.2571, 0.5040, 0.2666),   # "(*coret" baris Tenor
        "options": {
            "tenor_opsi_1": {"label": 1, "bbox": (0.3650, 0.2540, 0.3840, 0.2700)},
            "tenor_opsi_3": {"label": 3, "bbox": (0.3860, 0.2540, 0.4060, 0.2700)},
            "tenor_opsi_6": {"label": 6, "bbox": (0.4060, 0.2540, 0.4270, 0.2700)},
        },
    },
    "bentuk_reward": {
        "anchor_bbox": (0.4746, 0.2801, 0.5168, 0.2896),   # "(*coret" baris Bentuk Reward
        "options": {
            "pilihan_reward_tunai": {"label": "tunai", "bbox": (0.3650, 0.2770, 0.4070, 0.2925)},
            "pilihan_reward_non_tunai": {"label": "non_tunai", "bbox": (0.4070, 0.2770, 0.4790, 0.2925)},
        },
    },
}

SIGNATURE_CONFIG = {
    "signature_nasabah": {
        "anchor_bbox": (0.2260, 0.7886, 0.4234, 0.7981),
        "value_bbox": (0.1678, 0.7981, 0.4530, 0.8622),
    },
    "signature_atasan": {
        "anchor_bbox": (0.6720, 0.7886, 0.7449, 0.7981),
        "value_bbox": (0.5872, 0.7981, 0.8305, 0.8622),
    },
}

# V17 Finding 3 (Phase A) -- signature_nasabah's value_bbox, unlike
# signature_atasan's, sits almost exactly on top of a PRINTED graphic in
# template.pdf ("Opsional (Tidak Wajib) materai Rp10.000" rounded-rectangle
# border + 4 lines of text, confirmed visually by cropping the template at
# this exact bbox) -- template_subtraction against a box that itself has
# printed content is inherently noisy (registration misalignment of that
# printed border/text alone can register as "ink"). Threshold at 230
# (measured on the template crop itself: pixel intensities are either ~255
# background or <220 border/text, gap confirmed empirically -- see
# handover.md) isolates just those printed pixels; NOT hardcoded from a
# guess. Dilated 3x3/1 iter for registration tolerance (alignment isn't
# pixel-perfect). For signature_atasan the template crop is pure white, so
# this naturally returns an all-zero (no-op) mask -- safe to call for BOTH
# fields unconditionally, no field-specific branching needed.
SIGNATURE_STATIC_MASK_THRESHOLD = 230
_signature_static_mask_cache = {}


def signature_static_content_mask(template_gray, name, shape):
    """Mask (uint8, same size as the field's template/target crop) of pixels
    that are PRINTED CONTENT in template.pdf at `name`'s value_bbox -- derived
    directly from the template crop's own intensity (same "derive from
    template, don't hardcode" pattern as extract_template_static_text, just
    at pixel level instead of text level). Cached per (name, shape) since
    template_gray/shape don't change within one process run."""
    cache_key = (name, shape[0], shape[1])
    cached = _signature_static_mask_cache.get(cache_key)
    if cached is not None:
        return cached
    template_px = norm_bbox_to_px(SIGNATURE_CONFIG[name]["value_bbox"], shape)
    x1, y1, x2, y2 = template_px
    crop = template_gray[y1:y2, x1:x2]
    _, mask = cv2.threshold(crop, SIGNATURE_STATIC_MASK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    _signature_static_mask_cache[cache_key] = mask
    return mask

# V10 -- anchors used for SECTION-level (identity/placement) robust local
# transform estimation. Reuses existing FIELD_CONFIG/CHOICE_GROUPS anchors
# (no new hardcoded coordinates) -- multiple anchors per section so a single
# bad/outlier anchor cannot drag the whole section's fields off target the
# way a lone per-field dx/dy could.
_IDENTITY_ANCHOR_FIELDS = ["nama_nasabah", "nomor_rekening", "unit_kerja_pengelola_rekening"]
_PLACEMENT_ANCHOR_FIELDS = ["nominal_penempatan", "rentang_tenor", "reward_non_tunai", "reward_tunai"]

# V15: judul form dipakai sbg anchor FALLBACK terpisah (lihat
# estimate_section_transform's `title_anchor` param) utk KEDUA section --
# letterhead cabang yg lebih tinggi drpd template menggeser identity_area DAN
# placement_area dgn jumlah yg SAMA (satu blok, di bawah header yg sama).
# SENGAJA TIDAK digabung ke SECTION_ANCHOR_BBOXES (dicoba, terbukti merusak
# dokumen yg field-anchor-nya sendiri sudah benar -- lihat
# _fit_transform_from_anchors docstring) -- dipakai lewat parameter terpisah
# di _resolve_section_transforms (dynamic_extraction.py).
TITLE_ANCHOR_ENTRY = (TITLE_ANCHOR_BBOX_NORM, TITLE_ANCHOR_SEARCH_TOL_X, TITLE_ANCHOR_SEARCH_TOL_Y)

SECTION_ANCHOR_BBOXES = {
    "identity_area": [FIELD_CONFIG[f]["anchor_bbox"] for f in _IDENTITY_ANCHOR_FIELDS],
    "placement_area": (
        [FIELD_CONFIG[f]["anchor_bbox"] for f in _PLACEMENT_ANCHOR_FIELDS]
        + [g["anchor_bbox"] for g in CHOICE_GROUPS.values()]
    ),
}


def _build_next_field_top_map():
    """Untuk tiap field: batas atas (y normalized) ROI field/opsi-choice/tanda
    tangan LAIN terdekat yang tumpang tindih horizontal & ada di bawahnya.
    Dipakai sebagai batas keras saat ROI diperluas ke bawah, supaya tidak
    pernah menembus area field lain -- termasuk saat area itu kosong di
    template (kasus yang tidak tertangkap kalau hanya mengukur blank space)."""
    boxes = [(name, cfg["value_bbox"]) for name, cfg in FIELD_CONFIG.items()]
    for group in CHOICE_GROUPS.values():
        boxes += [(opt_name, opt["bbox"]) for opt_name, opt in group["options"].items()]
    boxes += [(name, cfg["value_bbox"]) for name, cfg in SIGNATURE_CONFIG.items()]

    limits = {}
    for name, cfg in FIELD_CONFIG.items():
        x1, y1, x2, y2 = cfg["value_bbox"]
        nearest = 1.0
        for other_name, (ox1, oy1, ox2, oy2) in boxes:
            if other_name == name or oy1 < y2:
                continue
            if min(x2, ox2) - max(x1, ox1) > 0:  # tumpang tindih horizontal
                nearest = min(nearest, oy1)
        limits[name] = nearest
    return limits


NEXT_FIELD_TOP_NORM = _build_next_field_top_map()


def _build_prev_field_bottom_map():
    """Kebalikan dari NEXT_FIELD_TOP_NORM: batas BAWAH (y normalized) field
    lain terdekat yang ada DI ATAS field ini & tumpang tindih horizontal.
    Dipakai sebagai batas keras di sisi ATAS, supaya ROI tidak pernah
    "naik" menembus baris/field sebelumnya kalau offset anchor (dx,dy)
    meleset -- ini realisasi dari keluhan ROI "terlalu ke atas"."""
    boxes = [(name, cfg["value_bbox"]) for name, cfg in FIELD_CONFIG.items()]
    limits = {}
    for name, cfg in FIELD_CONFIG.items():
        x1, y1, x2, y2 = cfg["value_bbox"]
        nearest = 0.0
        for other_name, (ox1, oy1, ox2, oy2) in boxes:
            if other_name == name or oy2 > y1:
                continue
            if min(x2, ox2) - max(x1, ox1) > 0:
                nearest = max(nearest, oy2)
        limits[name] = nearest
    return limits


PREV_FIELD_BOTTOM_NORM = _build_prev_field_bottom_map()

# Batas kiri keras per field = tepi kanan label cetak (anchor_bbox) + margin
# kecil. SEBELUMNYA perluasan ROI ke kiri (ROI_EXPAND_X_RATIO) pada field
# yang lebar (mis. nama_nasabah) bisa menembus balik ke area label cetak
# ("Nama Nasabah :") dan ikut ke-diff/ke-crop sebagai "tinta" -- inilah
# sumber bug hasil OCR kebocoran label mis. "asabah Ratem". Batas ini
# dipakai untuk MENGUNCI x1 ROI supaya tidak pernah melewati tepi label.
# HANYA dipakai kalau anchor_bbox benar-benar SEBARIS dengan value_bbox
# (label yang letaknya persis di kiri value) -- beberapa field (rentang_tenor,
# tempat_tanggal_surat) memakai anchor_bbox dari baris LAIN utk keperluan
# lokalisasi skew, bukan label-di-kiri-value, jadi harus dikecualikan.
def _same_row(a_bbox, v_bbox):
    return min(a_bbox[3], v_bbox[3]) - max(a_bbox[1], v_bbox[1]) > 0

ANCHOR_RIGHT_NORM = {
    name: cfg["anchor_bbox"][2] for name, cfg in FIELD_CONFIG.items()
    if _same_row(cfg["anchor_bbox"], cfg["value_bbox"])
}
LABEL_SAFE_MARGIN_NORM = 0.006


# tempat_tanggal_surat SENGAJA tidak dimasukkan (tidak ditampilkan di tabel
# perbandingan sesuai permintaan) -- tetap diekstrak lewat FIELD_CONFIG dan
# ada di raw_results/raw_json untuk debugging, hanya disembunyikan dari tabel.
FIELD_ORDER = [
    "nama_nasabah", "nomor_rekening", "unit_kerja_pengelola_rekening",
    "nominal_penempatan", "tenor_penempatan", "rentang_tenor", "bentuk_reward",
    "reward_non_tunai", "reward_tunai",
    "signature_nasabah", "signature_atasan",
]


# ============================================================================
# LOAD DOKUMEN + SCAN PREPROCESSING
# ============================================================================


def _looks_like_pdf(path):
    """Deteksi PDF dari isi file (magic header '%PDF-'), BUKAN cuma dari
    ekstensi -- dibutuhkan utk dokumen hasil download Google Drive yang
    disimpan tanpa ekstensi (lihat data_input.download_drive_document, yang
    menyimpan file sebagai <file_id> tanpa suffix). Baca cepat (5 byte),
    dilakukan untuk semua path yang bukan ber-suffix .pdf."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def is_pdf_document(path):
    """Deteksi PDF dari EKSTENSI ATAU ISI FILE -- satu-satunya cara yang
    benar utk membedakan PDF vs foto (JPEG/PNG) pada dokumen hasil download
    Google Drive, yang disimpan tanpa ekstensi (`<file_id>`, lihat
    data_input.download_drive_document). Sebelum V13, beberapa pemanggil
    (prepare_input/prepare_and_align) memakai `Path(path).suffix.lower() ==
    ".pdf"` saja -- SELALU False utk file tanpa ekstensi, sehingga dokumen
    PDF asli salah dikira foto dan masuk ke jalur enhance_scan/rectify_photo
    (didesain utk foto kamera dgn latar belakang, bukan render PDF penuh
    bingkai) secara sia-sia/keliru. Dipakai di semua tempat yang sebelumnya
    mengecek `suffix == ".pdf"` supaya konsisten dgn load_document()."""
    path = str(path)
    return Path(path).suffix.lower() == ".pdf" or _looks_like_pdf(path)


def load_document(path, dpi=PDF_DPI):
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".pdf" or _looks_like_pdf(path):
        try:
            doc = fitz.open(path)
        except Exception as exc:
            raise ValueError(f"Gagal membuka PDF: {exc}")
        if len(doc) == 0:
            doc.close()
            raise ValueError("PDF tidak memiliki halaman")
        page = doc[0]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
        doc.close()
        return image

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Gagal membaca file (format tidak didukung/file rusak): {Path(path).suffix or 'tanpa ekstensi'}")
    return image


def _needs_enhancement(gray):
    """Skip enhancement kalau gambar sudah cukup terang & kontras -- enhancement
    yang dipaksakan ke gambar yang sudah bagus justru merusak stroke tulisan."""
    return gray.std() < 40 or gray.mean() < 120


def enhance_scan(image):
    """
    Koreksi ringan untuk foto yang gelap/kontras rendah: normalisasi
    pencahayaan (background division) di-blend dengan gambar asli (bukan
    menggantikan total) supaya stroke tulisan asli tetap dominan, lalu CLAHE
    dengan clip rendah. Kalau gambar sudah cukup baik, tidak diapa-apakan.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not _needs_enhancement(gray):
        return image

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=31)
    divided = cv2.divide(gray, background, scale=255)
    # blend 65:35 supaya stroke asli tidak "dilindas" hasil normalisasi
    blended = cv2.addWeighted(divided, 0.65, gray, 0.35, 0)
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    enhanced = clahe.apply(blended)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _order_points(points):
    p = np.asarray(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).reshape(-1)
    rect[0], rect[2] = p[np.argmin(s)], p[np.argmax(s)]
    rect[1], rect[3] = p[np.argmin(d)], p[np.argmax(d)]
    return rect


DOCUMENT_QUAD_MAX_AREA_RATIO = 0.985  # tolak "kontur = nyaris seluruh frame"
                                       # (segmentasi gagal membedakan latar
                                       # sama sekali, bukan kertas ketemu)


def _quad_from_contour(contour):
    """approxPolyDP 4-titik kalau bisa (mengikuti bentuk kertas apa adanya,
    termasuk trapesium akibat perspektif); fallback minAreaRect(convexHull)
    kalau tepinya tidak lurus sempurna (umum pd foto kamera nyata)."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype(np.float32)
    hull = cv2.convexHull(contour)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _largest_valid_contour(contours, img_area, min_area_ratio):
    """Kontur terbesar yang lolos rentang area [min_area_ratio,
    DOCUMENT_QUAD_MAX_AREA_RATIO], atau None. Dipisah dari pemanggil supaya
    tiap strategi deteksi (segmentasi kecerahan / tepi) bisa dibandingkan
    dgn kriteria yang sama."""
    if not contours or img_area <= 0:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    ratio = float(cv2.contourArea(largest)) / img_area
    if min_area_ratio <= ratio <= DOCUMENT_QUAD_MAX_AREA_RATIO:
        return largest, ratio
    return None, ratio


def _paper_mask(gray, invert):
    """Segmentasi kertas via Otsu (kecerahan), BUKAN tepi -- jauh lebih tahan
    thd tepi kertas yang bergradasi lembut/redup krn bayangan atau fokus
    kamera (lihat _find_document_quad utk root-cause). invert=False utk
    kasus umum (kertas TERANG di latar lebih gelap); invert=True utk kasus
    kertas GELAP di latar TERANG. Morphology close+open (kernel skala
    resolusi) mengisi lubang teks & buang noise kecil di dalam mask supaya
    kontur luarnya bersih, satu blok."""
    sigma = max(3.0, min(gray.shape) * 0.01)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)
    k = max(3, int(min(gray.shape) * 0.02)) | 1
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _find_document_quad(image, min_area_ratio=DOCUMENT_QUAD_MIN_AREA_RATIO):
    """Deteksi kuadrilateral kertas dlm foto. Dipakai OLEH rectify_photo
    (produksi) DAN oleh diagnostik evaluasi (stage_evaluation.py) supaya
    keduanya konsisten "melihat" tepi kertas dgn cara yang SAMA.

    V13 -- root-cause fix: evaluasi 25-record menunjukkan pendekatan lama
    (Canny tepi tetap + syarat approxPolyDP PERSIS 4 titik) GAGAL menemukan
    page_quad pada SEMUA foto (kontur terbesar yg ketemu cuma ~2-6% frame),
    padahal kertas jelas dominan secara VISUAL. Root cause TERVERIFIKASI via
    inspeksi manual: tepi kertas di foto kamera nyata sering punya GRADIEN
    SANGAT LEMBUT (blur fokus/pencahayaan rata), jauh lebih lemah drpd
    kontras internal (teks/tabel) -- Canny (berbasis gradien tepi) SELALU
    menangkap teks internal duluan & TIDAK PERNAH menutup satu loop penuh
    sekeliling kertas walau sudah pakai threshold adaptif + dilasi.

    Perbaikan UTAMA: deteksi lewat SEGMENTASI KECERAHAN (Otsu threshold
    kertas-vs-latar, region-based) sbg strategi pertama -- jauh lebih tahan
    thd tepi lembut krn tidak bergantung pada gradien tajam, hanya beda
    terang/gelap kertas vs latar (terverifikasi pada sampel dataset: rasio
    area naik dari ~0.03 jadi 0.65-0.99, sesuai proporsi visual sebenarnya).
    Coba 2 arah (kertas terang DAN kertas gelap), lalu deteksi tepi (Canny
    adaptif+dilasi, versi sebelumnya) sbg fallback TERAKHIR kalau segmentasi
    kecerahan gagal (mis. latar & kertas sama-sama terang/tekstur kompleks).
    Kandidat dgn rasio area TERBESAR (dlm rentang valid) yang menang.

    Tetap AMAN dipakai agresif krn prepare_and_align() hanya memakai hasil
    rectify kalau skor alignment-nya OBJEKTIF LEBIH BAIK drpd dokumen asli
    (lihat _alignment_quality_score) -- quad yang meleset tidak akan lolos,
    otomatis kembali ke dokumen asli.

    Return (quad_4x2_float32_atau_None, area_ratio_kandidat_terbaik_atau_None)
    -- area_ratio dilaporkan walau di bawah min_area_ratio (utk diagnostik)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_area = image.shape[0] * image.shape[1]

    candidates = []  # list of (ratio, contour)
    for invert in (False, True):
        mask = _paper_mask(gray, invert)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour, ratio = _largest_valid_contour(contours, img_area, min_area_ratio)
        if contour is not None:
            candidates.append((ratio, contour))

    if not candidates:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = float(np.median(blurred))
        lo, hi = int(max(0, 0.66 * median)), int(min(255, 1.33 * median))
        edges = cv2.dilate(cv2.Canny(blurred, lo, hi), np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour, ratio = _largest_valid_contour(contours, img_area, min_area_ratio)
        if contour is not None:
            candidates.append((ratio, contour))
        elif ratio is not None:
            return None, ratio  # laporkan rasio kandidat tepi utk diagnostik

    if not candidates:
        return None, None

    best_ratio, best_contour = max(candidates, key=lambda c: c[0])
    return _quad_from_contour(best_contour), best_ratio


def rectify_photo(image, template_shape):
    """Deteksi tepi kertas + perspective warp agar sejajar dengan template."""
    page_quad, _ratio = _find_document_quad(image)
    if page_quad is None:
        return image, {"rectified": False, "reason": "page_not_found"}

    th, tw = template_shape[:2]
    src = _order_points(page_quad)
    dst = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(image, matrix, (tw, th), borderValue=(255, 255, 255))
    return out, {"rectified": True}


def prepare_input(path, image, template_shape):
    """Untuk foto (bukan PDF): enhance pencahayaan/kontras lalu rectify perspektif."""
    if is_pdf_document(path) or not PHOTO_RECTIFY:
        return image, {"rectified": False, "reason": "not_needed"}
    enhanced = enhance_scan(image)
    rectified, meta = rectify_photo(enhanced, template_shape)
    meta["scan_enhanced"] = True
    return rectified, meta


# ============================================================================
# GLOBAL ALIGNMENT KE TEMPLATE (ORB + homography, sekali per dokumen)
# ============================================================================


def _resize_for_match(image):
    h, w = image.shape[:2]
    scale = min(1.0, ALIGN_MAX_SIDE / float(max(h, w)))
    if scale == 1.0:
        return image, np.eye(3, dtype=np.float64)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return resized, S


def _decompose_linear_part(A2x2):
    """Urai bagian linear 2x2 (dari matrix affine 2x3 atau homography 3x3)
    jadi rotasi (derajat), skala x/y, dan shear (derajat) -- dipakai sbg
    diagnostik kompak utk audit seberapa 'destruktif' sebuah transform
    sebelum dipakai (item wajib evaluasi V9.1)."""
    a, b = float(A2x2[0, 0]), float(A2x2[0, 1])
    c, d = float(A2x2[1, 0]), float(A2x2[1, 1])
    scale_x = float(np.hypot(a, c))
    rotation = float(np.degrees(np.arctan2(c, a)))
    shear = 0.0
    if scale_x > 1e-9:
        shear = float(np.degrees(np.arctan2(a * b + c * d, scale_x * scale_x)))
    scale_y = float(np.hypot(b, d))
    return {
        "rotation_deg": round(rotation, 3),
        "scale_x": round(scale_x, 4),
        "scale_y": round(scale_y, 4),
        "shear_deg": round(shear, 3),
    }


def _homography_is_sane(H, w, h):
    """Tolak homography yang degenerate (warp 'meledak'/menyebar dari satu
    titik) meskipun cv2.findHomography tidak me-return None.
    AUDIT V8 -- pengecekan sebelumnya (area ratio + convex-hull 4 titik) lolos
    utk kasus SHEAR/FLIP ringan yang tetap convex 4-titik tapi sudah rusak
    (mis. sudut kanan-atas ketuker turun ke bawah kiri). Ditambah 2 lapis:
      1. Urutan sudut (TL harus tetap kira2 kiri-atas dst, dgn toleransi
         margin kecil) -- menolak flip/rotasi ekstrem.
      2. Aspect ratio hasil warp dibandingkan aspect ratio template -- menolak
         shear/skew berlebihan yang membuat dokumen "terentang" tidak wajar.
    """
    try:
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(warped)):
        return False

    area = cv2.contourArea(warped.astype(np.float32))
    if area <= 0:
        return False
    ratio = area / float(w * h)
    if ratio < 0.25 or ratio > 4.0:
        return False

    hull = cv2.convexHull(warped.astype(np.float32))
    if len(hull) != 4:
        return False

    tl, tr, br, bl = warped
    margin = 0.06 * max(w, h)
    if not (tl[0] < tr[0] + margin and bl[0] < br[0] + margin
            and tl[1] < bl[1] + margin and tr[1] < br[1] + margin):
        return False  # sudut sudah tertukar/flip -> bukan quad wajar

    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    avg_w, avg_h = (top_w + bottom_w) / 2, (left_h + right_h) / 2
    if avg_w <= 0 or avg_h <= 0:
        return False
    warped_aspect = avg_w / avg_h
    template_aspect = w / float(h)
    if not (0.5 * template_aspect <= warped_aspect <= 1.8 * template_aspect):
        return False

    return True


def _resize_keep_aspect(image, target_w, target_h, border_value=(255, 255, 255)):
    """Resize mempertahankan aspect ratio asli (letterbox ke kanvas
    target_w x target_h), BUKAN stretch/distort seperti resize langsung ke
    (tw, th). Dipakai sebagai fallback saat alignment gagal total -- gambar
    tetap boleh lebih kecil dari kanvas & dipusatkan dgn border putih,
    daripada dipepetkan paksa dan merusak proporsi tulisan/field."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w, 3), border_value, dtype=np.uint8)
    scale = min(target_w / float(w), target_h / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), border_value, dtype=np.uint8)
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _estimate_affine_candidate(src, dst, S_t, S_f):
    """Kandidat PALING KONSERVATIF: similarity/affine (translasi + rotasi +
    skala seragam, TANPA shear/perspektif) via estimateAffinePartial2D.
    Ditolak (return None) kalau rotasi/skala di luar rentang wajar --
    itu tanda match ORB yang nyasar, bukan skew dokumen yang sebenarnya."""
    A_small, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ALIGN_RANSAC
    )
    if A_small is None or inlier_mask is None:
        return None
    A = np.linalg.inv(S_t) @ np.vstack([A_small, [0, 0, 1]]) @ S_f
    decomposed = _decompose_linear_part(A[:2, :2])
    if abs(decomposed["rotation_deg"]) > AFFINE_MAX_ROTATION_DEG:
        return None
    lo, hi = AFFINE_SCALE_RANGE
    if not (lo <= decomposed["scale_x"] <= hi and lo <= decomposed["scale_y"] <= hi):
        return None
    return {
        "M": A, "inliers": int(inlier_mask.ravel().sum()), "decomposed": decomposed,
        "A_small": A_small, "inlier_mask": inlier_mask,
    }


def _match_geometry_diagnostics(src_small, dst_small, inlier_mask, M_small, tw_small, th_small):
    """Diagnostik geometri TAMBAHAN di atas good_matches/inlier_ratio:
    (1) coverage_ratio -- seberapa luas SEBARAN titik inlier menutupi kanvas
        (bukan cuma jumlahnya) -- banyak match yang menggerombol di satu
        sudut kecil TIDAK dianggap alignment halaman penuh yang baik.
    (2) reproj_error_px -- error reprojeksi rata-rata titik inlier memakai
        transform yang dipilih -- fit yang numerically tidak stabil tetap
        bisa punya inlier_ratio tinggi tapi error besar.
    Dihitung di ruang ORB (resized/small), M_small bisa 2x3 (affine) atau
    3x3 (homography)."""
    if inlier_mask is None:
        return {"coverage_ratio": 0.0, "reproj_error_px": None}
    mask_bool = inlier_mask.ravel().astype(bool)
    dst_in = dst_small.reshape(-1, 2)[mask_bool]
    src_in = src_small.reshape(-1, 2)[mask_bool]
    if len(dst_in) < 2:
        return {"coverage_ratio": 0.0, "reproj_error_px": None}

    x1, y1 = dst_in[:, 0].min(), dst_in[:, 1].min()
    x2, y2 = dst_in[:, 0].max(), dst_in[:, 1].max()
    coverage_ratio = float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, tw_small * th_small))

    ones = np.ones((len(src_in), 1))
    src_h = np.hstack([src_in, ones])
    if M_small.shape[0] == 2:
        proj = (M_small @ src_h.T).T
    else:
        proj_h = (M_small @ src_h.T).T
        proj = proj_h[:, :2] / np.clip(proj_h[:, 2:3], 1e-9, None)
    err = float(np.mean(np.linalg.norm(proj - dst_in, axis=1)))
    return {"coverage_ratio": coverage_ratio, "reproj_error_px": err}


def _form_body_mask(shape_hw, header_ratio=HEADER_EXCLUDE_RATIO):
    """Mask ORB keypoint detection ke 'form body' -- kecualikan strip atas
    (letterhead/alamat/cabang, sangat variatif antar scan) supaya alignment
    tidak didominasi match dari area yang bukan bagian struktur form yang
    stabil."""
    h, w = shape_hw
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[: int(h * header_ratio), :] = 0
    return mask


def _fallback_coverage_mask(filled_shape, target_w, target_h):
    """Coverage mask utk jalur fallback (_resize_keep_aspect): area kanvas
    yang BENAR-BENAR berisi piksel sumber (bukan letterbox putih)."""
    h, w = filled_shape[:2]
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return mask
    scale = min(target_w / float(w), target_h / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    mask[y_off:y_off + new_h, x_off:x_off + new_w] = 255
    return mask


def align_to_template(template, filled):
    """Alignment KONSERVATIF (V9.1): coba similarity/affine dulu (paling
    tidak merusak -- tidak bisa shear/perspective-warp dokumen yang sudah
    lurus), homography hanya dipakai sbg fallback kalau affine gagal
    sanity-check ATAU homography-nya objectively better (inlier ratio jelas
    lebih tinggi, lihat HOMOGRAPHY_MIN_RELATIVE_GAIN). Diagnostik kompak
    (method/rotation_deg/scale_x/scale_y/shear_deg) selalu disertakan utk
    kebutuhan evaluasi, di atas key lama (status/good_matches/inliers/
    inlier_ratio) yang dipertahankan agar kompatibel dgn pemanggil lama."""
    th, tw = template.shape[:2]
    fh, fw = filled.shape[:2]
    fallback = _resize_keep_aspect(filled, tw, th)
    fallback_coverage = _fallback_coverage_mask(filled.shape, tw, th)

    # Audit dimensi: tolak lebih awal kalau template/dokumen terlalu kecil
    # utk ORB+transform bermakna (mis. thumbnail rusak/file kosong) --
    # daripada memaksakan matching yang pasti gagal/nyasar.
    if th < 20 or tw < 20 or fh < 20 or fw < 20:
        return fallback, np.eye(3), {
            "status": "failed", "reason": "invalid_dimensions", "method": "none",
            "coverage_mask": fallback_coverage,
        }

    t_small, S_t = _resize_for_match(template)
    f_small, S_f = _resize_for_match(filled)

    t_gray = cv2.cvtColor(t_small, cv2.COLOR_BGR2GRAY)
    f_gray = cv2.cvtColor(f_small, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=ALIGN_FEATURES)
    mask_t = _form_body_mask(t_gray.shape)
    mask_f = _form_body_mask(f_gray.shape)
    kp_t, des_t = orb.detectAndCompute(t_gray, mask_t)
    kp_f, des_f = orb.detectAndCompute(f_gray, mask_f)

    if des_t is None or des_f is None:
        return fallback, np.eye(3), {
            "status": "failed", "reason": "features_not_found", "method": "none",
            "coverage_mask": fallback_coverage,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_f, des_t, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < ALIGN_RATIO * n.distance]

    if len(good) < 4:
        return fallback, np.eye(3), {
            "status": "failed", "reason": "insufficient_matches", "good_matches": len(good), "method": "none",
            "coverage_mask": fallback_coverage,
        }

    src = np.float32([kp_f[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    tw_small, th_small = t_small.shape[1], t_small.shape[0]

    def _gate_status(base_ok, diag):
        if not base_ok:
            return "warning"
        if diag["coverage_ratio"] < ALIGN_MIN_COVERAGE_RATIO:
            return "warning"
        if diag["reproj_error_px"] is not None and diag["reproj_error_px"] > ALIGN_MAX_REPROJ_ERROR_PX:
            return "warning"
        return "ok"

    # --- Kandidat 1 (didahulukan): similarity/affine ----------------------
    affine = _estimate_affine_candidate(src, dst, S_t, S_f)
    affine_meta = None
    if affine is not None:
        ratio = affine["inliers"] / max(1, len(good))
        diag = _match_geometry_diagnostics(src, dst, affine["inlier_mask"], affine["A_small"], tw_small, th_small)
        base_ok = len(good) >= ALIGN_MIN_MATCHES and ratio >= 0.30
        affine_meta = {
            "status": _gate_status(base_ok, diag),
            "good_matches": len(good), "inliers": affine["inliers"], "inlier_ratio": float(ratio),
            "method": "similarity", **affine["decomposed"], **diag,
        }

    # --- Kandidat 2 (fallback lebih destruktif): homography ---------------
    H_small, mask = cv2.findHomography(src, dst, cv2.RANSAC, ALIGN_RANSAC)
    homography_meta, H = None, None
    if H_small is not None and mask is not None:
        H_candidate = np.linalg.inv(S_t) @ H_small @ S_f
        if _homography_is_sane(H_candidate, tw, th):
            inliers_h = int(mask.ravel().sum())
            ratio_h = inliers_h / max(1, len(good))
            diag_h = _match_geometry_diagnostics(src, dst, mask, H_small, tw_small, th_small)
            base_ok_h = len(good) >= ALIGN_MIN_MATCHES and ratio_h >= 0.30
            H = H_candidate
            homography_meta = {
                "status": _gate_status(base_ok_h, diag_h),
                "good_matches": len(good), "inliers": inliers_h, "inlier_ratio": float(ratio_h),
                "method": "homography", **_decompose_linear_part(H_candidate[:2, :2]), **diag_h,
            }

    # --- Pilih transform paling tidak merusak yang cukup baik --------------
    if affine_meta is not None:
        use_homography = (
            homography_meta is not None
            and homography_meta["status"] == "ok"
            and (homography_meta["inlier_ratio"] - affine_meta["inlier_ratio"]) >= HOMOGRAPHY_MIN_RELATIVE_GAIN
        )
        if use_homography:
            aligned = cv2.warpPerspective(filled, H, (tw, th), borderValue=(255, 255, 255))
            coverage = cv2.warpPerspective(
                np.full((fh, fw), 255, dtype=np.uint8), H, (tw, th), borderValue=0
            )
            homography_meta["escalated_from_affine"] = True
            homography_meta["affine_candidate"] = {
                k: affine_meta[k] for k in ("inlier_ratio", "good_matches", "rotation_deg")
            }
            homography_meta["coverage_mask"] = coverage
            return aligned, H, homography_meta

        M_2x3 = affine["M"][:2, :].astype(np.float32)
        aligned = cv2.warpAffine(filled, M_2x3, (tw, th), borderValue=(255, 255, 255))
        coverage = cv2.warpAffine(
            np.full((fh, fw), 255, dtype=np.uint8), M_2x3, (tw, th), borderValue=0
        )
        if homography_meta is not None:
            affine_meta["homography_candidate"] = {
                k: homography_meta[k] for k in ("inlier_ratio", "good_matches", "rotation_deg")
            }
        affine_meta["coverage_mask"] = coverage
        return aligned, affine["M"], affine_meta

    if homography_meta is not None:
        aligned = cv2.warpPerspective(filled, H, (tw, th), borderValue=(255, 255, 255))
        coverage = cv2.warpPerspective(
            np.full((fh, fw), 255, dtype=np.uint8), H, (tw, th), borderValue=0
        )
        homography_meta["escalated_from_affine"] = False
        homography_meta["reason"] = "affine_failed_sanity_or_unavailable"
        homography_meta["coverage_mask"] = coverage
        return aligned, H, homography_meta

    return fallback, np.eye(3), {
        "status": "failed", "reason": "no_valid_transform", "good_matches": len(good), "method": "none",
        "coverage_mask": fallback_coverage,
    }


# ============================================================================
# ORKESTRASI: bandingkan kualitas ALIGNMENT dokumen asli vs hasil rectify,
# pakai transform (enhance+rectify) HANYA kalau benar-benar memperbaiki hasil
# alignment. Jangan pernah rectify/warp dokumen yang sudah bagus.
# ============================================================================

GOOD_ALIGNMENT_INLIER_RATIO = 0.55
GOOD_ALIGNMENT_MIN_MATCHES = 30


def _alignment_quality_score(meta):
    """Skor terurut (status, inlier_ratio, good_matches) utk membandingkan
    dua hasil align_to_template -- dipakai sbg key perbandingan langsung."""
    status_rank = {"ok": 2, "warning": 1, "failed": 0}.get(meta.get("status"), 0)
    return (status_rank, meta.get("inlier_ratio", 0.0), meta.get("good_matches", 0))


def _alignment_is_good(meta):
    return (
        meta.get("status") == "ok"
        and meta.get("inlier_ratio", 0.0) >= GOOD_ALIGNMENT_INLIER_RATIO
        and meta.get("good_matches", 0) >= GOOD_ALIGNMENT_MIN_MATCHES
    )


# ============================================================================
# COARSE ORIENTATION CORRECTION (V19f) -- ditambahkan setelah kasus nyata
# ditemukan (Reska Nurdianti, downloaded_documents/
# 1NLnzLKXVlI-1lrn7WF0T1OgKWSYKHFkK): foto formulir diambil dgn tablet dalam
# posisi LANDSCAPE penuh (dokumen portrait miring 90 derajat di dalam frame),
# EXIF Orientation=0 (nilai TIDAK valid/tidak actionable, bukan salah satu
# dari 8 nilai standar) sehingga tidak ada koreksi EXIF apa pun yang bisa
# dipakai. align_to_template's affine/homography SENGAJA hanya mentoleransi
# rotasi KECIL (skew, lihat AFFINE_MAX_ROTATION_DEG=8 derajat) -- ORB
# matching-nya sendiri tidak dirancang menembus rotasi 90/180/270 derajat
# PENUH, jadi alignment gagal total pada kasus ini, BUKAN karena affine/
# homography kurang toleran (keputusan V9.1 itu tetap benar utk skew kecil,
# TIDAK diubah di sini -- lihat align_to_template's docstring).
#
# Pendekatan (V19f REVISI -- lihat catatan di bawah): reuse align_to_template
# YANG SUDAH ADA (TIDAK menambah algoritma deteksi orientasi baru/Hough/dsb),
# tp kandidat rotasi yg DIBANDINGKAN ditentukan dulu dari ASPECT RATIO
# (bukan langsung membandingkan skor ORB ke-4 rotasi sekaligus). HANYA
# mencoba rotasi tambahan kalau percobaan 0 derajat (orientasi apa adanya)
# belum "good" (_alignment_is_good) -- dokumen yang sudah portrait/lurus
# (mayoritas kasus) TIDAK PERNAH kena overhead tambahan ini.
#
# CATATAN PENTING (jangan ulangi tanpa fix ini): percobaan pertama versi ini
# langsung membandingkan skor _alignment_quality_score ke-4 rotasi (0/90/
# 180/270) sekaligus -- GAGAL pd dokumen nyata (Reska Nurdianti): homography
# `align_to_template` PADA ORIENTASI SALAH (0 derajat, padahal dokumen
# sebenarnya miring 90 derajat) tetap menghasilkan status "warning" dgn
# inlier_ratio (0.248) yg SEDIKIT LEBIH TINGGI drpd orientasi yg BENAR (90
# derajat, inlier_ratio 0.237) -- homography cukup ekspresif utk "menyerap"
# rotasi 90 derajat ke dalam transform-nya sendiri shg skornya TIDAK
# reliable membedakan orientasi benar vs salah scr langsung. Diverifikasi
# lgs: membandingkan HANYA dua kandidat yg aspect ratio-nya SUDAH cocok dgn
# template (mis. 90 vs 270 kalau raw image landscape & template portrait)
# jauh lebih diskriminatif (90 dpt status "warning", 270 "failed" tegas).
# ============================================================================


def _aspect_ratio_orientation_mismatch(template_shape, image_shape, tolerance=0.35):
    """True kalau image_shape's aspect ratio (w/h) adalah KEBALIKAN dari
    template_shape's (mis. template portrait, image landscape) -- sinyal
    KASAR tp jauh lebih robust drpd skor ORB/homography utk membedakan
    rotasi 90/270 derajat PENUH dari 0/180 (lihat catatan di atas). Dokumen
    yang hampir persegi (rasio < 1+tolerance) tidak dicurigai sama sekali --
    ambigu scr geometris, biarkan skor ORB yang memutuskan lewat jalur 180."""
    th, tw = template_shape[:2]
    ih, iw = image_shape[:2]
    template_is_portrait = th >= tw
    image_is_portrait = ih >= iw
    if template_is_portrait == image_is_portrait:
        return False
    template_ratio = max(tw, th) / max(1, min(tw, th))
    return template_ratio >= (1 + tolerance)


def align_with_orientation_correction(template_img, raw_image):
    """Return (aligned_img, H, meta, corrected_raw_image, rotation_deg).
    `corrected_raw_image` adalah `raw_image` itu sendiri (rotation_deg=0)
    kalau orientasi awal sudah oke/terbaik, atau versi ter-rotasi 90/180/270
    derajat kalau salah satu dari itu terbukti jauh lebih baik -- dipakai
    pemanggil (prepare_and_align) SEBAGAI PENGGANTI raw_image asli utk
    langkah selanjutnya (enhance/rectify), supaya tidak mengoreksi orientasi
    dua kali atau balik ke versi yang salah."""
    aligned, H, meta = align_to_template(template_img, raw_image)
    if _alignment_is_good(meta):
        meta["orientation_correction_deg"] = 0
        return aligned, H, meta, raw_image, 0

    if _aspect_ratio_orientation_mismatch(template_img.shape, raw_image.shape):
        # 0 derajat MUSTAHIL benar scr geometris di sini (aspect ratio
        # terbalik dari template) -- skornya JANGAN dipakai sbg baseline
        # pembanding sama sekali (itu justru penyebab kegagalan versi
        # sebelumnya, lihat catatan panjang di atas: skor "0 derajat salah"
        # bisa numerically MENGALAHKAN skor "90 derajat benar"). Pilih
        # LANGSUNG yang terbaik di antara 90 vs 270 saja, satu sama lain.
        best_aligned, best_H, best_meta = None, None, None
        best_score, best_raw, best_deg = (-1, 0.0, 0), raw_image, 0
        for deg, rotate_code in ((90, cv2.ROTATE_90_CLOCKWISE), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)):
            candidate_raw = cv2.rotate(raw_image, rotate_code)
            cand_aligned, cand_H, cand_meta = align_to_template(template_img, candidate_raw)
            score = _alignment_quality_score(cand_meta)
            if score > best_score:
                best_score = score
                best_aligned, best_H, best_meta = cand_aligned, cand_H, cand_meta
                best_raw, best_deg = candidate_raw, deg
        if best_meta is not None and best_meta.get("status") != "failed":
            best_meta["orientation_correction_deg"] = best_deg
            return best_aligned, best_H, best_meta, best_raw, best_deg
        # 90 & 270 sama-sama gagal total (no_valid_transform) -- fallback ke
        # 0 derajat drpd tidak menghasilkan apa-apa.
        meta["orientation_correction_deg"] = 0
        return aligned, H, meta, raw_image, 0

    # Aspect ratio SUDAH cocok dgn template -> 0 derajat MASIH mungkin benar
    # (baseline yg valid di sini, beda dari kasus di atas), HANYA 180 yg
    # masuk akal scr geometris sbg pembanding tambahan (90/270 SENGAJA tidak
    # dicoba -- aspect ratio-nya sendiri sudah menyingkirkan kemungkinan itu).
    candidate_raw = cv2.rotate(raw_image, cv2.ROTATE_180)
    cand_aligned, cand_H, cand_meta = align_to_template(template_img, candidate_raw)
    if _alignment_quality_score(cand_meta) > _alignment_quality_score(meta):
        cand_meta["orientation_correction_deg"] = 180
        return cand_aligned, cand_H, cand_meta, candidate_raw, 180

    meta["orientation_correction_deg"] = 0
    return aligned, H, meta, raw_image, 0


def prepare_and_align(path, raw_image, template_img):
    """Satu pintu masuk utk 'siapkan input lalu align ke template', menjaga
    prinsip "jangan apa-apakan dokumen yang sudah bagus":

    1. Coba align dokumen ASLI dulu (tanpa enhance/rectify apa pun). Kalau
       hasilnya SUDAH BAIK (inlier ratio & jumlah match tinggi, status "ok"),
       pakai langsung -- rectify/enhance tidak pernah dihitung sama sekali.
    2. Kalau alignment dokumen asli kurang baik (dan filenya foto, bukan PDF,
       serta PHOTO_RECTIFY aktif), BARU coba enhance+rectify, lalu align lagi
       hasilnya.
    3. Bandingkan skor alignment (raw vs rectified) -- pakai versi yang
       hasil alignment-nya LEBIH BAIK. Kalau rectify justru tidak membantu
       (atau memperburuk), dokumen ASLI yang tetap dipakai, bukan versi
       ter-warp.

    0 (sebelum langkah 1 di atas). Deteksi/koreksi orientasi KASAR (0/90/180/
    270 derajat) -- lihat align_with_orientation_correction -- HANYA
    dijalankan kalau percobaan alignment pada orientasi asli belum "good";
    dokumen yang sudah portrait/lurus tidak kena overhead ini sama sekali.
    Kalau salah satu rotasi kanonik terbukti jauh lebih baik, `raw_image`
    yang dipakai utk SISA fungsi ini (termasuk enhance/rectify di langkah 2)
    diganti ke versi ter-rotasi itu -- bukan orientasi asli yang salah.

    Return: (aligned_img, H, alignment_meta) -- alignment_meta memuat juga
    "candidate" (raw / rectified / raw_preferred_over_rectified),
    "orientation_correction_deg" (0/90/180/270), dan "input_preparation"
    (dict rectified/reason, kompatibel dgn field prep_meta versi
    sebelumnya)."""
    aligned_raw, H_raw, meta_raw, raw_image, orientation_deg = align_with_orientation_correction(
        template_img, raw_image
    )

    is_pdf = is_pdf_document(path)
    if is_pdf or not PHOTO_RECTIFY:
        meta_raw["candidate"] = "raw"
        meta_raw["input_preparation"] = {"rectified": False, "reason": "not_needed"}
        return aligned_raw, H_raw, meta_raw

    if _alignment_is_good(meta_raw):
        meta_raw["candidate"] = "raw"
        meta_raw["input_preparation"] = {"rectified": False, "reason": "already_good_no_rectify_needed"}
        return aligned_raw, H_raw, meta_raw

    enhanced = enhance_scan(raw_image)
    rectified, rect_meta = rectify_photo(enhanced, template_img.shape)
    rect_meta["scan_enhanced"] = True
    aligned_rect, H_rect, meta_rect = align_to_template(template_img, rectified)

    compact = lambda m: {k: m.get(k) for k in ("status", "inlier_ratio", "good_matches")}  # noqa: E731

    if _alignment_quality_score(meta_rect) > _alignment_quality_score(meta_raw):
        meta_rect["candidate"] = "rectified"
        meta_rect["input_preparation"] = rect_meta
        meta_rect["compared_raw"] = compact(meta_raw)
        return aligned_rect, H_rect, meta_rect

    meta_raw["candidate"] = "raw_preferred_over_rectified"
    meta_raw["input_preparation"] = {"rectified": False, "reason": "rectify_did_not_improve_alignment"}
    meta_raw["compared_rectified"] = compact(meta_rect)
    return aligned_raw, H_raw, meta_raw


# ============================================================================
# BBOX HELPERS
# ============================================================================


def _clip_bbox(bbox, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def norm_bbox_to_px(bbox, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    return _clip_bbox((x1 * w, y1 * h, x2 * w, y2 * h), shape)


def _shift_bbox(bbox_px, dx, dy, shape):
    x1, y1, x2, y2 = bbox_px
    return _clip_bbox((x1 + dx, y1 + dy, x2 + dx, y2 + dy), shape)


def _crop(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


# ============================================================================
# LOCALIZED ANCHOR DETECTION (template matching, TANPA OCR)
# ============================================================================


def locate_anchor_offset(template_gray, target_gray, anchor_bbox_norm, shape,
                          tol_x=ANCHOR_SEARCH_TOL_X, tol_y=ANCHOR_SEARCH_TOL_Y):
    """
    Cari posisi aktual sepotong cetakan (anchor) pada dokumen yang sudah
    di-align, kembalikan pergeseran (dx, dy) terhadap posisi template.
    Dipakai untuk mengoreksi ROI nilai secara lokal tanpa OCR sama sekali.
    """
    h, w = shape[:2]
    ax1, ay1, ax2, ay2 = norm_bbox_to_px(anchor_bbox_norm, shape)
    patch = template_gray[ay1:ay2, ax1:ax2]
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return 0, 0, 0.0, False

    tol_px_x, tol_px_y = int(tol_x * w), int(tol_y * h)
    sx1, sy1, sx2, sy2 = _clip_bbox(
        (ax1 - tol_px_x, ay1 - tol_px_y, ax2 + tol_px_x, ay2 + tol_px_y), shape
    )
    window = target_gray[sy1:sy2, sx1:sx2]
    if window.shape[0] < patch.shape[0] or window.shape[1] < patch.shape[1]:
        return 0, 0, 0.0, False

    res = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    dx = (sx1 + max_loc[0]) - ax1
    dy = (sy1 + max_loc[1]) - ay1
    return dx, dy, float(max_val), max_val >= ANCHOR_MATCH_MIN


def resolve_roi(template_gray, target_gray, cfg, shape):
    """ROI = value_bbox template + offset anchor lokal. Fixed ROI hanya fallback."""
    template_px = norm_bbox_to_px(cfg["value_bbox"], shape)
    dx, dy, similarity, matched = locate_anchor_offset(
        template_gray, target_gray, cfg["anchor_bbox"], shape
    )
    if matched:
        return template_px, _shift_bbox(template_px, dx, dy, shape), "anchor", similarity
    return template_px, template_px, "fallback", similarity


# ============================================================================
# V10 -- SECTION-LEVEL LOCALIZATION
# Beberapa anchor cetak per section (identity/placement) -> tolak anchor
# berkualitas rendah/outlier -> satu transform (affine kalau >=3 anchor
# valid, konsensus translasi kalau 1-2) -> transform itu dipakai KONSISTEN
# ke semua field dalam section, bukan tiap field mengoreksi dirinya sendiri
# dengan SATU dx/dy independen (V9.x). Fallback hierarchy: section affine ->
# konsensus translasi -> resolve_roi per-field (V9.x) -> template ROI polos.
# ============================================================================


def _fit_transform_from_anchors(pts_template, pts_target, n_total, w, h):
    """Inti estimate_section_transform (TIDAK BERUBAH dari pre-V15): median
    offset -> outlier rejection -> affine (>=3 anchor bersih) -> konsensus
    translasi (1-2) -> none. Dipisah jadi fungsi sendiri supaya bisa dipakai
    utk 2 pool anchor yang BERBEDA (field vs judul, lihat
    estimate_section_transform) TANPA mencampur keduanya dalam satu vote --
    V15 sempat mencoba menggabung keduanya dlm satu pool (median/similarity-
    weighted), TERBUKTI merusak dokumen yang sebelumnya sudah benar (anchor
    field & anchor judul bisa sama-sama valid tapi mengukur offset lokal
    berbeda pada posisi halaman berbeda -- merata-ratakan dua kebenaran yang
    berbeda menghasilkan kesalahan ketiga, bukan kebenaran ketiga)."""
    if not pts_template:
        return {"M": None, "method": "none", "anchors_used": 0, "anchors_total": n_total, "outliers": 0}

    pts_t = np.array(pts_template, dtype=np.float64)
    pts_f = np.array(pts_target, dtype=np.float64)
    offsets = pts_f - pts_t

    # Reject offset outliers relative to the median offset (robust to a
    # single anchor that matched the wrong local pattern).
    median_off = np.median(offsets, axis=0)
    dists = np.linalg.norm(offsets - median_off, axis=1)
    diag = float(np.hypot(w, h))
    keep = dists <= SECTION_OUTLIER_DIST_RATIO * diag
    n_outliers = int((~keep).sum())
    pts_t_k, pts_f_k = pts_t[keep], pts_f[keep]

    if len(pts_t_k) >= 3:
        M, _inliers = cv2.estimateAffinePartial2D(
            pts_t_k.astype(np.float32).reshape(-1, 1, 2),
            pts_f_k.astype(np.float32).reshape(-1, 1, 2),
            method=cv2.LMEDS,
        )
        if M is not None:
            decomposed = _decompose_linear_part(M[:, :2])
            lo, hi = SECTION_SCALE_RANGE
            if (abs(decomposed["rotation_deg"]) <= SECTION_MAX_ROTATION_DEG
                    and lo <= decomposed["scale_x"] <= hi and lo <= decomposed["scale_y"] <= hi):
                return {
                    "M": M.astype(np.float64), "method": "section_affine",
                    "anchors_used": int(len(pts_t_k)), "anchors_total": n_total,
                    "outliers": n_outliers, **decomposed,
                }

    if len(pts_t_k) >= 1:
        mean_off = pts_f_k.mean(axis=0) - pts_t_k.mean(axis=0)
        M = np.array([[1.0, 0.0, mean_off[0]], [0.0, 1.0, mean_off[1]]], dtype=np.float64)
        return {
            "M": M, "method": "consensus_translation", "anchors_used": int(len(pts_t_k)),
            "anchors_total": n_total, "outliers": n_outliers,
        }

    return {"M": None, "method": "none", "anchors_used": 0, "anchors_total": n_total, "outliers": n_outliers}


def _match_anchor_pool(template_gray, aligned_gray, shape, anchor_bboxes_norm):
    """Cocokkan satu pool anchor (list of bbox 4-float, ATAU (bbox,tol_x,tol_y)
    utk toleransi pencarian custom) -> (pts_template, pts_target, sims,
    max_sim). Dipisah dari _fit_transform_from_anchors supaya field-pool dan
    title-pool bisa dicocokkan independen (lihat estimate_section_transform)."""
    pts_template, pts_target, sims = [], [], []
    for entry in anchor_bboxes_norm:
        if len(entry) == 4:
            bbox_norm, tol_x, tol_y = entry, ANCHOR_SEARCH_TOL_X, ANCHOR_SEARCH_TOL_Y
        else:
            bbox_norm, tol_x, tol_y = entry
        ax1, ay1, ax2, ay2 = norm_bbox_to_px(bbox_norm, shape)
        cx, cy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        dx, dy, sim, matched = locate_anchor_offset(
            template_gray, aligned_gray, bbox_norm, shape, tol_x=tol_x, tol_y=tol_y
        )
        if not matched or sim < SECTION_ANCHOR_MIN_SIMILARITY:
            continue
        pts_template.append((cx, cy))
        pts_target.append((cx + dx, cy + dy))
        sims.append(sim)
    max_sim = max(sims) if sims else 0.0
    return pts_template, pts_target, sims, max_sim


# V15 -- letterhead cabang yg lebih tinggi drpd template membuat anchor field
# gagal/tidak akurat, lihat TITLE_ANCHOR_* di atas. Ambang MIN_SIMILARITY judul
# SEMULA 0.5 (dipilih dari 1 kasus, blm divalidasi luas) -- TERBUKTI terlalu
# ketat, membuang 2 match judul yg SEBENARNYA genuine & konsisten dgn pola
# pergeseran nyata (record 15 sim=0.448 dy=59px, record 19 sim=0.496 dy=54px
# -- keduanya sejalan dgn record 10/13/22 yg dy~52-78px). Diturunkan ke 0.40,
# masih di atas record 20 (sim=0.310, DI BAWAH SECTION_ANCHOR_MIN_SIMILARITY
# 0.32 -- genuinely tidak ketemu, tetap tidak dapat rujukan & itu wajar).
TITLE_ANCHOR_FALLBACK_MIN_SIMILARITY = 0.40

# V15.1 -- ROOT CAUSE FIX dari V15: trigger fallback SEBELUMNYA cuma
# "similarity anchor field tertinggi < 0.36" (FIELD_ANCHORS_WEAK_MAX_SIMILARITY,
# dihapus) -- TERBUKTI tidak cukup. Kasus nyata (record block-shift, section
# placement_area): 4 anchor dgn similarity individual biasa-biasa saja
# (0.33-0.45, SEMUANYA di atas 0.36) tapi CUKUP utk lolos ke jalur
# `estimateAffinePartial2D` (>=3 anchor) & menghasilkan fit yg LULUS semua
# sanity check (rotasi 4.5 derajat, scale 0.96 -- keduanya wajar) TAPI SALAH
# scr faktual (translasi hasil fit dy=-13, padahal judul yg sama-sama diukur
# di dokumen yg sama menunjukkan dy=+61 -- confidently wrong, bukan terlihat
# lemah sama sekali). Similarity anchor individual TIDAK cukup jadi sinyal
# "apakah HASIL fit-nya benar" -- fit yg salah bisa tetap lolos semua
# pengecekan sanity yg sudah ada (rotasi/scale/outlier-distance).
#
# Fix: anchor judul SELALU dihitung (kalau tersedia & similarity-nya sendiri
# cukup tinggi -- lihat TITLE_ANCHOR_FALLBACK_MIN_SIMILARITY), lalu translasi
# HASIL field-only (kolom translasi M, berlaku sama utk section_affine maupun
# consensus_translation) dibandingkan LANGSUNG ke translasi judul. Disagree
# lebih dari ambang ini -> title dipakai SENDIRIAN (bukan dicampur, prinsip yg
# sama spt sebelumnya), TERLEPAS dari seberapa "percaya diri" fit field-only
# terlihat. Ambang dipakai ULANG dari SECTION_OUTLIER_DIST_RATIO (constant yg
# SAMA dipakai utk outlier-rejection antar anchor -- konsisten, bukan angka
# baru) -- diverifikasi cocok utk 2 kasus nyata: record block-shift
# placement_area (jarak translasi field-vs-judul ~94px, MELEBIHI ambang -> di-
# override, BENAR) dan record normal identity_area (jarak ~36px, DI BAWAH
# ambang -> TIDAK di-override, mempertahankan hasil yg SUDAH benar).
# V15.1-b -- ROOT CAUSE FIX dari V15.1: perbandingan JARAK translasi penuh
# (2D) TERBUKTI bukan diskriminator yg cukup (record 2 [identity_area, HARUS
# dipertahankan] jarak ~36px vs record 13 [identity_area, HARUS di-override]
# jarak ~30px -- lebih kecil drpd record 2, satu ambang jarak tidak bisa
# memisahkan keduanya). Percobaan kedua (overlap kotak kandidat vs kotak
# judul sungguhan) benar utk identity_area TAPI gagal total utk
# placement_area (record 10): box placement_area yg SALAH "nangkring" di
# PARAGRAF LAIN (bukan judul itu sendiri), jadi tidak pernah tumpang tindih
# dgn kotak judul sama sekali walau translasinya sama-sama salah.
#
# Fix final: pakai SUMBU-Y SAJA (fenomena block-shift murni pergeseran
# vertikal krn tinggi letterhead, bukan horizontal), bandingkan translasi-Y
# field-only thd DUA rujukan: 0 (posisi template asli/tanpa koreksi) DAN
# translasi-Y judul (pergeseran halaman yg sungguhan terukur, independen
# dari section mana pun). Field-only dianggap "nyangkut di posisi asli"
# (anchor-nya gagal menjangkau konten yg SUDAH bergeser, bukan koreksi lokal
# yg genuinely berbeda) kalau translasi-Y-nya LEBIH DEKAT ke 0 drpd ke
# translasi-Y judul -- sinyal ini brlaku SAMA baik field-only itu nyangkut di
# judul (identity_area) MAUPUN di paragraf lain (placement_area), krn tidak
# bergantung pd APA yg ada di lokasi yg salah itu, hanya pd SEBERAPA JAUH dari
# posisi asli field-only berhasil bergerak dibanding yg SEHARUSNYA (per
# judul). Judul sendiri harus menunjukkan pergeseran yg BERARTI (>=
# MIN_TITLE_SHIFT_PX) sebelum dipakai sbg rujukan -- dokumen normal (judul
# sendiri dy~0-1px) tidak pernah masuk perbandingan ini sama sekali, jadi
# TIDAK PERNAH override field-only yg sudah genuinely benar (record 2/4/12,
# semua dgn |title_dy| dibawah ambang) terlepas dari translasi-Y field-only-
# nya sendiri berapa.
MIN_TITLE_SHIFT_PX = 15.0


def estimate_section_transform(template_gray, aligned_gray, shape, anchor_bboxes_norm, title_anchor=None,
                                section_bbox_norm=None):
    """Return dict diagnostik + 'M' (2x3 float64, template->aligned) atau
    None kalau tidak ada anchor valid sama sekali.

    `title_anchor`: entri (bbox, tol_x, tol_y) opsional (lihat
    TITLE_ANCHOR_BBOX_NORM). V15.1: SELALU dicocokkan (bukan cuma saat anchor
    field terlihat lemah) dan dipakai sbg CROSS-CHECK thd hasil field-only --
    TIDAK PERNAH dicampur ke pool anchor field yang sama (lihat
    _fit_transform_from_anchors docstring kenapa).
    `section_bbox_norm`: TIDAK dipakai lagi oleh cross-check (lihat
    MIN_TITLE_SHIFT_PX docstring -- perbandingan sumbu-Y polos terbukti lebih
    andal drpd overlap kotak), dipertahankan sbg parameter utk kompatibilitas
    pemanggil, boleh None."""
    h, w = shape[:2]
    pts_template, pts_target, sims, max_sim = _match_anchor_pool(
        template_gray, aligned_gray, shape, anchor_bboxes_norm
    )
    field_result = _fit_transform_from_anchors(pts_template, pts_target, len(anchor_bboxes_norm), w, h)

    if title_anchor is None:
        return field_result

    t_pts_template, t_pts_target, t_sims, t_max_sim = _match_anchor_pool(
        template_gray, aligned_gray, shape, [title_anchor]
    )
    if t_max_sim < TITLE_ANCHOR_FALLBACK_MIN_SIMILARITY:
        return field_result  # judul sendiri tidak ketemu dgn cukup yakin -- tidak ada rujukan yg valid

    title_result = _fit_transform_from_anchors(t_pts_template, t_pts_target, 1, w, h)
    if title_result["M"] is None:
        return field_result

    title_dy = float(title_result["M"][1, 2])
    if field_result["M"] is None:
        disagree = abs(title_dy) >= MIN_TITLE_SHIFT_PX  # field-only tidak punya hasil sama sekali
    elif abs(title_dy) < MIN_TITLE_SHIFT_PX:
        disagree = False  # judul sendiri tidak menunjukkan pergeseran berarti -- tidak ada dasar utk override
    else:
        field_dy = float(field_result["M"][1, 2])
        disagree = abs(field_dy) < abs(field_dy - title_dy)  # field-only nyangkut lebih dekat ke posisi asli (0) drpd ke pergeseran sungguhan

    if disagree:
        title_result["method"] = "title_fallback_translation"
        title_result["field_anchors_max_similarity"] = max_sim
        title_result["field_result_method"] = field_result["method"]
        return title_result

    return field_result


def apply_section_transform(bbox_px, M):
    """Terapkan transform 2x3 section ke bbox axis-aligned template ->
    bbox axis-aligned baru (envelope dari 4 sudut yang ditransform, supaya
    tetap benar walau ada sedikit rotasi)."""
    x1, y1, x2, y2 = bbox_px
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    ones = np.ones((4, 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    transformed = (M @ pts_h.T).T
    return (
        float(transformed[:, 0].min()), float(transformed[:, 1].min()),
        float(transformed[:, 0].max()), float(transformed[:, 1].max()),
    )


def resolve_roi_section(template_gray, aligned_gray, cfg, shape, section_transform):
    """Fallback hierarchy per field: section transform (kalau tersedia & valid)
    -> resolve_roi per-field (V9.x, anchor lokal tunggal) -> template ROI polos
    (sudah ditangani apa adanya di dalam resolve_roi). Return 4-tuple sama
    bentuknya dgn resolve_roi: (template_px, target_px, source, evidence)."""
    template_px = norm_bbox_to_px(cfg["value_bbox"], shape)
    M = (section_transform or {}).get("M")
    if M is not None:
        target_px = _clip_bbox(apply_section_transform(template_px, M), shape)
        source = f"section_{section_transform['method']}"
        evidence = section_transform.get("anchors_used", 0)
        return template_px, target_px, source, evidence
    return resolve_roi(template_gray, aligned_gray, cfg, shape)


# ============================================================================
# V10 -- SOURCE-PIXEL COVERAGE (partial/cropped document detection)
# ============================================================================


def region_coverage(coverage_mask, bbox_px):
    """Fraksi piksel dalam bbox yang benar-benar berasal dari piksel sumber
    (bukan area letterbox/border hasil warp). coverage_mask=None -> anggap
    sepenuhnya tercakup (kompatibel dgn pemanggil yang belum menyediakan
    mask, mis. jalur lama)."""
    if coverage_mask is None:
        return 1.0
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_px]
    h, w = coverage_mask.shape[:2]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    patch = coverage_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch) / patch.size)


def is_out_of_frame(coverage_mask, bbox_px, min_ratio=COVERAGE_MIN_RATIO):
    return region_coverage(coverage_mask, bbox_px) < min_ratio


# ============================================================================
# IMAGE DIFFERENCE / HANDWRITING REGION DETECTION
# ============================================================================


def difference_mask(template_roi, filled_roi, threshold=DIFF_THRESHOLD):
    if template_roi.shape[:2] != filled_roi.shape[:2]:
        filled_roi = cv2.resize(filled_roi, (template_roi.shape[1], template_roi.shape[0]))

    t = cv2.cvtColor(template_roi, cv2.COLOR_BGR2GRAY)
    f = cv2.cvtColor(filled_roi, cv2.COLOR_BGR2GRAY)
    t = cv2.GaussianBlur(t, (3, 3), 0)
    f = cv2.GaussianBlur(f, (3, 3), 0)
    diff = cv2.absdiff(t, f)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def ink_change_ratio(template_roi, filled_roi):
    mask = difference_mask(template_roi, filled_roi)
    return float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0


def _remove_line_noise(mask):
    """Buang garis titik-titik/garis bawah cetakan agar tidak dianggap tulisan."""
    h, w = mask.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 2), 1))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(mask, lines)


def _expand_bbox(bbox_px, shape, expand_x_ratio, extra_height_px, min_x1_px=None):
    """Perluas ROI relatif terhadap ukurannya sendiri (bukan fixed absolut):
    kanan-kiri untuk tulisan yang lebih panjang/bergeser, bawah untuk baris
    lanjutan. Tetap anchor-relative karena bbox awal sudah hasil resolve_roi.
    min_x1_px (opsional): batas keras kiri (mis. tepi label cetak) -- expand
    kiri tidak akan pernah melewati ini walau expand_x_ratio menghitung lebih."""
    x1, y1, x2, y2 = bbox_px
    dx = int((x2 - x1) * expand_x_ratio)
    new_x1 = x1 - dx
    if min_x1_px is not None:
        new_x1 = max(new_x1, min_x1_px)
    return _clip_bbox((new_x1, y1, x2 + dx, y2 + extra_height_px), shape)


def _clamp_field_box_y(bbox_px, field_name, shape, dy=0.0):
    """Kunci sisi ATAS & BAWAH bbox field (SEBELUM diperluas) ke batas field
    tetangga (PREV_FIELD_BOTTOM_NORM / NEXT_FIELD_TOP_NORM). Ini jaring
    pengaman independen dari akurasi offset anchor (dx,dy): walau anchor
    meleset (mis. karena skew foto), ROI tidak akan pernah "naik" ke field
    sebelumnya (keluhan "terlalu ke atas") atau "turun" membaca awal field
    berikutnya (keluhan hasil field tertukar, mis. nama ketimpa nomor
    rekening).

    `dy` (V17 Finding 1, default 0.0 -- SEMUA pemanggil lama TIDAK berubah
    perilakunya): geser batas tetangga (yg dihitung dari koordinat TEMPLATE,
    tanpa shift) sejauh dy YANG SAMA dgn pergeseran bbox_px ini sendiri
    SEBELUM di-clamp. Root cause TERVERIFIKASI via debug empirik (lihat
    handover.md): section transform (consensus_translation/section_affine)
    BISA menggeser satu baris identity_area turun/naik beberapa piksel scr
    SAH (bukan drift acak, terkonfirmasi konsisten dgn baris lain di record
    yg sama) -- tapi batas tetangga statis (posisi template, dy=0) TIDAK
    ikut bergeser, jadi box yg SUDAH BENAR malah ke-clip oleh batas yg belum
    ikut pindah. Memanggil dgn dy=0 (default) menghasilkan hasil PERSIS SAMA
    dgn versi lama (batas statis) -- HANYA pemanggil yg eksplisit memberikan
    dy yg perilakunya berubah."""
    x1, y1, x2, y2 = bbox_px
    h = shape[0]
    if field_name in PREV_FIELD_BOTTOM_NORM:
        y1 = max(y1, int(PREV_FIELD_BOTTOM_NORM[field_name] * h + dy))
    if field_name in NEXT_FIELD_TOP_NORM:
        y2 = min(y2, int(NEXT_FIELD_TOP_NORM[field_name] * h + dy))
    if y2 <= y1:  # jaga-jaga kalau batas tabrakan -> jangan sampai bbox invalid
        y1, y2 = bbox_px[1], bbox_px[3]
    return _clip_bbox((x1, y1, x2, y2), shape)


def _measure_safe_extra_height(template_gray, bbox_px, max_extra, shape):
    """
    Ukur berapa px ruang KOSONG di template di bawah value_bbox sebelum
    konten cetakan berikutnya (label field lain) muncul. Perluasan ROI ke
    bawah dibatasi seukuran ini, sehingga field yang barisnya rapat (gap 0,
    mis. Nominal Penempatan -> Tenor Penempatan) otomatis tidak diperluas
    dan tidak pernah menyerempet field tetangga -- tanpa perlu hardcode
    jarak antar field secara manual.
    """
    if max_extra <= 0:
        return 0
    x1, y1, x2, y2 = bbox_px
    y_end = min(shape[0], y2 + max_extra)
    strip = template_gray[y2:y_end, x1:x2]
    if strip.size == 0:
        return 0
    row_ink_ratio = (strip < 200).mean(axis=1)
    content_rows = np.where(row_ink_ratio > 0.05)[0]
    if content_rows.size == 0:
        return y_end - y2
    return max(0, min(max_extra, int(content_rows[0]) - 2))


def _find_ink_bands(mask, merge_gap=INK_BAND_MERGE_GAP):
    """Kelompokkan baris-baris bertinta jadi 'band' (calon baris tulisan).
    Gap kecil antar baris (goresan huruf wajar) digabung; gap besar = baris
    terpisah/tidak terkait."""
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return []
    bands, start, prev = [], rows[0], rows[0]
    for y in rows[1:]:
        if y - prev > merge_gap:
            bands.append((int(start), int(prev) + 1))
            start = y
        prev = y
    bands.append((int(start), int(prev) + 1))
    return bands


def extract_handwriting_crop(template_img, aligned_img, template_gray, template_px, target_px, shape,
                              field_name=None, pad_ratio=CROP_PAD_RATIO):
    """
    ROI diperluas dulu (kanan-kiri + baris lanjutan ke bawah) dari value_bbox
    ber-anchor, dengan perluasan bawah DIBATASI GANDA: (1) ruang kosong asli
    di template, dan (2) posisi field/choice/signature lain terdekat di
    bawahnya (NEXT_FIELD_TOP_NORM) -- supaya tidak pernah menembus area field
    lain walau area itu kosong di template. Lalu template subtraction -> buang
    garis -> lacak band tinta: ambil baris utama + baris lanjutan HANYA kalau
    jaraknya dekat, maksimal MAX_INK_LINES baris. Fixed ROI (posisi dasar)
    tetap acuan awal -- yang diperluas cuma batasnya.
    Return (crop_atau_None, change_ratio).
    """
    # Jaring pengaman posisi: kunci base box (SEBELUM ekspansi) ke batas
    # field tetangga -- independen dari akurasi offset anchor (lihat
    # docstring _clamp_field_box_y). Base ROI adalah acuan awal, jadi ini
    # dilakukan SEBELUM langkah lain.
    if field_name is not None:
        template_px = _clamp_field_box_y(template_px, field_name, shape)
        target_px = _clamp_field_box_y(target_px, field_name, shape)

    base_h = template_px[3] - template_px[1]
    max_extra_h = int(base_h * ROI_EXTRA_HEIGHT_RATIO)

    if field_name in NEXT_FIELD_TOP_NORM:
        next_top_px = NEXT_FIELD_TOP_NORM[field_name] * shape[0]
        layout_cap = max(0, int(next_top_px - template_px[3]) - 3)
        max_extra_h = min(max_extra_h, layout_cap)

    extra_h = _measure_safe_extra_height(template_gray, template_px, max_extra_h, shape)

    # Batas kiri keras = tepi kanan label cetak field ini (+ margin), supaya
    # perluasan ke kiri tidak pernah membaca balik ke teks label (lihat
    # docstring ANCHOR_RIGHT_NORM) -- sumber bug "asabah Ratem" dkk.
    min_x1_px = None
    if field_name in ANCHOR_RIGHT_NORM:
        min_x1_px = int((ANCHOR_RIGHT_NORM[field_name] + LABEL_SAFE_MARGIN_NORM) * shape[1])

    t_px = _expand_bbox(template_px, shape, ROI_EXPAND_X_RATIO, extra_h, min_x1_px)
    f_px = _expand_bbox(target_px, shape, ROI_EXPAND_X_RATIO, extra_h, min_x1_px)

    t_roi = _crop(template_img, t_px)
    f_roi = _crop(aligned_img, f_px)
    if f_roi.shape[:2] != t_roi.shape[:2]:
        f_roi = cv2.resize(f_roi, (t_roi.shape[1], t_roi.shape[0]))

    mask = _remove_line_noise(difference_mask(t_roi, f_roi))
    change_ratio = float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0
    if change_ratio < BLANK_THRESHOLD:
        return None, change_ratio

    bands = _find_ink_bands(mask)
    if not bands:
        return None, change_ratio

    kept = [bands[0]]
    for band in bands[1:]:
        if len(kept) >= MAX_INK_LINES:
            break
        gap = band[0] - kept[-1][1]
        if gap <= max(6, int(base_h * LINE_GAP_MAX_RATIO)):
            kept.append(band)
        else:
            break  # jarak jauh -> bukan baris lanjutan, berhenti melacak

    y1, y2 = kept[0][0], kept[-1][1]
    row_mask = mask[y1:y2, :]
    xs = np.where(row_mask.any(axis=0))[0]
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    pad_x = max(3, int(f_roi.shape[1] * 0.02))
    pad_y = max(3, int((y2 - y1) * pad_ratio))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(f_roi.shape[1], x2 + pad_x), min(f_roi.shape[0], y2 + pad_y)

    return f_roi[y1:y2, x1:x2].copy(), change_ratio


# ============================================================================
# SEMANTIC (COARSE) REGIONS -- V9.1 tambahan, fondasi V9.2
# ============================================================================
# Tiga area kasar (identity/placement/signature) yang mengelilingi
# field-field terkait, dibangun dari envelope FIELD_CONFIG/CHOICE_GROUPS/
# SIGNATURE_CONFIG yang SUDAH ADA (bukan koordinat baru yang di-hardcode).
# Dipakai untuk observability/evaluasi & fondasi rewrite V9.2 (anchor+bbox
# dynamic association). Jalur ekstraksi produksi (resolve_roi +
# extract_handwriting_crop per FIELD_CONFIG, dipanggil pipeline.py) TIDAK
# diubah/diganti -- fungsi di bawah ini murni tambahan, tidak dipanggil dari
# alur produksi manapun.

SEMANTIC_REGION_PAD_NORM = 0.01


def _envelope_bbox(bboxes):
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    pad = SEMANTIC_REGION_PAD_NORM
    return (max(0.0, x1 - pad), max(0.0, y1 - pad), min(1.0, x2 + pad), min(1.0, y2 + pad))


_IDENTITY_FIELDS = ["nama_nasabah", "nomor_rekening", "unit_kerja_pengelola_rekening"]
_PLACEMENT_FIELDS = ["nominal_penempatan", "rentang_tenor", "reward_non_tunai", "reward_tunai"]

SEMANTIC_REGIONS_NORM = {
    "identity_area": _envelope_bbox(
        [FIELD_CONFIG[f]["anchor_bbox"] for f in _IDENTITY_FIELDS]
        + [FIELD_CONFIG[f]["value_bbox"] for f in _IDENTITY_FIELDS]
    ),
    "placement_area": _envelope_bbox(
        [FIELD_CONFIG[f]["anchor_bbox"] for f in _PLACEMENT_FIELDS]
        + [FIELD_CONFIG[f]["value_bbox"] for f in _PLACEMENT_FIELDS]
        + [CHOICE_GROUPS[g]["anchor_bbox"] for g in CHOICE_GROUPS]
        + [opt["bbox"] for g in CHOICE_GROUPS.values() for opt in g["options"].values()]
    ),
    "signature_area": _envelope_bbox(
        [cfg["anchor_bbox"] for cfg in SIGNATURE_CONFIG.values()]
        + [cfg["value_bbox"] for cfg in SIGNATURE_CONFIG.values()]
    ),
}


def extract_semantic_regions(image, shape):
    """Ekstrak 3 area kasar (identity_area/placement_area/signature_area)
    dari dokumen yang SUDAH di-align ke template. Coarse region utk
    evaluasi & fondasi V9.2 -- BUKAN pengganti ROI per-field produksi
    (FIELD_CONFIG/resolve_roi tetap dipakai apa adanya oleh pipeline.py).
    Return dict: name -> {bbox_norm, bbox_px, crop}."""
    regions = {}
    for name, bbox_norm in SEMANTIC_REGIONS_NORM.items():
        bbox_px = norm_bbox_to_px(bbox_norm, shape)
        regions[name] = {
            "bbox_norm": bbox_norm,
            "bbox_px": bbox_px,
            "crop": _crop(image, bbox_px),
        }
    return regions
