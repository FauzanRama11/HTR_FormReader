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
import ocr
import preprocessing as prep
from preprocessing import FIELD_CONFIG

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
    diagnostik (termasuk 'M' atau None)."""
    return {
        region_name: prep.estimate_section_transform(template_gray, aligned_gray, shape, anchor_bboxes)
        for region_name, anchor_bboxes in prep.SECTION_ANCHOR_BBOXES.items()
    }


def _resolve_field_boxes(template_gray, aligned_gray, shape, section_transforms):
    boxes = {}
    for field_name, (region_name, _label) in FIELD_REGION_MAP.items():
        cfg = FIELD_CONFIG[field_name]
        template_px, target_px, source, evidence = prep.resolve_roi_section(
            template_gray, aligned_gray, cfg, shape, section_transforms.get(region_name)
        )
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
        wx1, wy1, wx2, wy2 = _expand(fb["target_px"], rw, rh, rbbox)

        picked = []
        for tok in tokens:
            text = tok.get("text") or ""
            if _fuzzy_contains(text, label):
                remainder = _strip_label_remainder(text, label)
                if remainder:
                    picked.append({**tok, "text": remainder})
                continue  # token label cetak murni (tanpa sisa) -- lewati
            tcx, tcy = _bbox_center(tok["bbox"])
            if wx1 <= tcx <= wx2 and wy1 <= tcy <= wy2:
                picked.append(tok)

        if not picked:
            continue

        region_y0 = active_region_bbox[region_name][1]
        region_h = max(1, active_region_bbox[region_name][3] - region_y0)
        # reading order: baris (dikuantisasi relatif thd region) dulu, lalu kiri->kanan
        picked.sort(key=lambda t: (round((t["bbox"][1] - region_y0) / region_h * 40), t["bbox"][0]))
        text = " ".join(t.get("text", "") for t in picked if t.get("text")).strip()
        confs = [t["confidence"] for t in picked if t.get("confidence") is not None]
        confidence = min(confs) if confs else None
        if text:
            result[field_name] = {"raw": text, "confidence": confidence}

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
