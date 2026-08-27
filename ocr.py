"""
ocr.py
======
Satu-satunya tempat model OCR diinisialisasi & dipanggil. Ganti model/engine
cukup di sini (lihat OCR_MODEL) -- modul lain tidak perlu tahu detailnya.

Juga berisi batching: semua crop tulisan tangan digabung jadi satu kanvas
supaya PaddleOCR hanya dipanggil SATU kali per dokumen (bukan per field).

run_batched_ocr_tokens() adalah kontrak token-preserving (text+confidence+
bbox per token, bukan satu string tergabung) yang dipakai dynamic_extraction.py
utk asosiasi spasial; run_batched_ocr() (kontrak lama, 1 string per field)
dibangun DI ATAS fungsi itu supaya cuma ada SATU jalur agregasi OCR-batch.
"""

import numpy as np
from paddleocr import PaddleOCR

# ============================================================================
# MIXING MODEL -- pilih engine OCR yang dipakai lewat OCR_ENGINE.
# "paddle" adalah default AKTIF (satu-satunya yang punya dependency terpasang
# di requirements.txt). 3 engine lain disediakan sebagai KERANGKA (kode +
# instruksi) yang tinggal di-uncomment kalau mau dicoba -- tiap engine WAJIB
# mengembalikan bentuk yang sama: list[{"text": str, "confidence": float|None,
# "bbox": (x1,y1,x2,y2)|None}], supaya assign_items_to_fields() di bawah tidak
# perlu diubah sama sekali (kontrak output disamakan lintas model).
# ============================================================================
OCR_ENGINE = "paddle"   # ganti ke "easyocr" / "trocr" / "tesseract" utk coba model lain

# --- Konfigurasi model PaddleOCR (AKTIF) ---
# CATATAN GPU: sebelumnya "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')"
# di sini MEMAKSA CPU dengan menyembunyikan semua GPU dari proses -- baris itu
# DIHAPUS. Device sekarang dipilih otomatis lewat _select_device() (GPU
# pertama/gpu:0 kalau paddle dibuild dengan CUDA & GPU terdeteksi, fallback ke
# CPU kalau tidak) -- cocok utk device dgn 1 GPU kecil mis. NVIDIA MX230 2GB.
OCR_MODEL = dict(
    ocr_version="PP-OCRv5",
    lang="id",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def _select_device():
    """Deteksi GPU CUDA yang tersedia lewat paddle sendiri (bukan env var
    yang menyembunyikan device). gpu:0 kalau paddle dibuild dgn CUDA DAN ada
    minimal 1 device CUDA terlihat; selain itu fallback ke cpu -- tidak
    pernah gagal keras hanya karena deteksi GPU error."""
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:
        pass
    return "cpu"


OCR_DEVICE = _select_device()

# Upscale crop tulisan tangan sebelum OCR (kualitas) -- dipertahankan sama di
# GPU/CPU supaya akurasi tidak turun; kontrol VRAM dilakukan lewat UKURAN
# KANVAS BATCH (lihat MAX_CANVAS_HEIGHT_*), bukan menurunkan upscale ini.
OCR_SCALE_UP = 2.0
BATCH_ROW_GAP = 10     # jarak antar crop pada kanvas OCR gabungan (px)

# Batas tinggi kanvas gabungan PER PANGGILAN OCR (adaptive scaling VRAM):
# GPU kecil (mis. MX230 2GB) dapat kanvas lebih pendek -> beberapa panggilan
# batch-kecil berurutan, bukan satu kanvas raksasa yang bisa OOM. CPU (RAM
# jauh lebih longgar) dapat batas lebih tinggi supaya tetap sesedikit mungkin
# panggilan model (idealnya tetap 1x per dokumen). Dokumen ini TIDAK PERNAH
# di-OCR penuh satu halaman -- hanya kanvas gabungan crop tulisan tangan per
# field, jadi tingginya sudah kecil secara alami; batas ini hanya jaring
# pengaman untuk dokumen dengan banyak field terisi sekaligus.
MAX_CANVAS_HEIGHT_GPU = 2200
MAX_CANVAS_HEIGHT_CPU = 6000

# Kwargs tambahan yang HANYA dicoba di GPU kecil untuk menekan pemakaian
# VRAM (batch rekognisi teks internal PaddleOCR sekecil mungkin). Beberapa
# versi PaddleOCR tidak mengenal parameter ini -- get_ocr_engine() mencoba
# beberapa kombinasi kwargs dan otomatis mundur (fallback) kalau ditolak.
_GPU_LOW_VRAM_EXTRA = dict(text_recognition_batch_size=1)

_ocr_engine = None


def get_ocr_engine():
    """Inisialisasi PaddleOCR sekali (lazy), dengan device gpu:0/cpu otomatis.
    Mencoba beberapa kombinasi kwargs dari yang paling lengkap (device +
    hemat VRAM) sampai paling minimal, supaya tetap jalan di versi PaddleOCR
    manapun yang terpasang di requirements.txt tanpa perlu tahu versi pasti."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    base = dict(OCR_MODEL)
    extra = dict(_GPU_LOW_VRAM_EXTRA) if OCR_DEVICE.startswith("gpu") else {}
    candidates = [
        {**base, **extra, "device": OCR_DEVICE},
        {**base, "device": OCR_DEVICE},
        {**base, **extra, "use_gpu": OCR_DEVICE.startswith("gpu")},
        {**base, "use_gpu": OCR_DEVICE.startswith("gpu")},
        base,
    ]
    last_exc = None
    for kwargs in candidates:
        try:
            _ocr_engine = PaddleOCR(**kwargs)
            return _ocr_engine
        except TypeError as exc:
            last_exc = exc
            continue
    # Semua kombinasi kwargs ditolak (versi PaddleOCR sangat berbeda) -- coba
    # paling minimal tanpa penanganan, biar error aslinya kelihatan di log.
    if last_exc is not None:
        _ocr_engine = PaddleOCR(**base)
    return _ocr_engine


# ----------------------------------------------------------------------------
# ALTERNATIF 1: EasyOCR
# Install : pip install easyocr
# Cocok utk  : perbandingan cepat, mendukung banyak bahasa sekaligus.
# ----------------------------------------------------------------------------
# import easyocr
# _easyocr_engine = None
#
# def get_easyocr_engine():
#     global _easyocr_engine
#     if _easyocr_engine is None:
#         _easyocr_engine = easyocr.Reader(["id", "en"], gpu=False)
#     return _easyocr_engine
#
# def run_ocr_easyocr(image):
#     items = []
#     for bbox, text, conf in get_easyocr_engine().readtext(image):
#         text = str(text).strip()
#         if not text:
#             continue
#         xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
#         items.append({
#             "text": text, "confidence": float(conf),
#             "bbox": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
#         })
#     return items


# ----------------------------------------------------------------------------
# ALTERNATIF 2: TrOCR (HuggingFace transformers) -- khusus tulisan tangan
# Install : pip install transformers torch pillow
# Cocok utk  : tulisan tangan (handwritten), tapi TIDAK punya deteksi bbox
#              sendiri -- dipakai per-CROP FIELD (bukan satu kanvas gabungan
#              seperti paddle), jadi assign_items_to_fields() dilewati kalau
#              engine ini dipakai (lihat catatan di run_ocr()).
# ----------------------------------------------------------------------------
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image as PILImage
# _trocr_processor, _trocr_model = None, None
#
# def get_trocr_engine():
#     global _trocr_processor, _trocr_model
#     if _trocr_model is None:
#         _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#         _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     return _trocr_processor, _trocr_model
#
# def run_ocr_trocr_single_crop(image_bgr):
#     """Dipanggil PER FIELD CROP (bukan kanvas gabungan) -> return 1 string."""
#     processor, model = get_trocr_engine()
#     pil_img = PILImage.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB
#     pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values)
#     return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# ----------------------------------------------------------------------------
# ALTERNATIF 3: Tesseract (pytesseract)
# Install : pip install pytesseract  (+ install binary tesseract-ocr di OS,
#           dan paket bahasa: tesseract-ocr-ind utk Bahasa Indonesia)
# Cocok utk  : baseline ringan/cepat, cocok utk teks cetak; kurang akurat
#              utk tulisan tangan dibanding paddle/trocr.
# ----------------------------------------------------------------------------
# import pytesseract
#
# def run_ocr_tesseract(image):
#     data = pytesseract.image_to_data(
#         image, lang="ind", output_type=pytesseract.Output.DICT,
#         config="--psm 6",
#     )
#     items = []
#     for i, text in enumerate(data["text"]):
#         text = str(text).strip()
#         if not text:
#             continue
#         conf = float(data["conf"][i]) / 100.0 if str(data["conf"][i]) not in ("-1", "") else None
#         x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
#         items.append({"text": text, "confidence": conf, "bbox": (x, y, x + w, y + h)})
#     return items


def _payload(result):
    data = result.json
    if callable(data):
        data = data()
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    return data if isinstance(data, dict) else {}


def _payload_items(data):
    texts = data.get("rec_texts", []) or []
    scores = data.get("rec_scores", []) or []
    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    polys = data.get("dt_polys") or data.get("rec_polys")
    boxes = data.get("rec_boxes")

    items = []
    for i, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue

        score = float(scores[i]) if i < len(scores) else None
        bbox = None
        try:
            if polys is not None and i < len(polys):
                p = np.asarray(polys[i], dtype=np.float32).reshape(-1, 2)
                bbox = (
                    int(np.floor(p[:, 0].min())), int(np.floor(p[:, 1].min())),
                    int(np.ceil(p[:, 0].max())), int(np.ceil(p[:, 1].max())),
                )
            elif boxes is not None and i < len(boxes):
                b = np.asarray(boxes[i], dtype=np.float32).reshape(-1)
                if b.size >= 4:
                    bbox = tuple(int(round(v)) for v in b[:4])
        except Exception:
            bbox = None

        items.append({"text": text, "confidence": score, "bbox": bbox})
    return items


def _run_ocr_paddle(image):
    prediction = get_ocr_engine().predict(image)
    items = []
    for result in prediction:
        items.extend(_payload_items(_payload(result)))
    return items


def run_ocr(image):
    """Satu pemanggilan model OCR -> list item {text, confidence, bbox}.
    Dispatcher berdasarkan OCR_ENGINE -- ganti konstanta itu (bukan fungsi
    ini) untuk pindah model. Engine selain 'paddle' butuh uncomment kode +
    install dependency di atas terlebih dahulu."""
    if OCR_ENGINE == "paddle":
        return _run_ocr_paddle(image)
    # if OCR_ENGINE == "easyocr":
    #     return run_ocr_easyocr(image)
    # if OCR_ENGINE == "tesseract":
    #     return run_ocr_tesseract(image)
    # if OCR_ENGINE == "trocr":
    #     raise RuntimeError(
    #         "trocr bekerja per-crop field, bukan kanvas gabungan -- pipeline.py "
    #         "perlu memanggil run_ocr_trocr_single_crop() per field, bukan run_ocr() "
    #         "sekali untuk kanvas. Lihat komentar ALTERNATIF 2 di atas."
    #     )
    raise ValueError(f"OCR_ENGINE tidak dikenal: {OCR_ENGINE!r}")


def compose_batch_canvas(crops):
    """Gabung semua crop tulisan tangan jadi satu kanvas vertikal, supaya
    satu dokumen hanya butuh SATU kali panggilan OCR. crops: list[(name, image)]."""
    if not crops:
        return None, {}

    max_w = max(im.shape[1] for _, im in crops)
    rows, layout = [], {}
    y_cursor = 0
    gap_row = np.full((BATCH_ROW_GAP, max_w, 3), 255, dtype=np.uint8)

    for name, im in crops:
        h, w = im.shape[:2]
        if w < max_w:
            im = np.hstack([im, np.full((h, max_w - w, 3), 255, dtype=np.uint8)])
        rows.append(im)
        layout[name] = (y_cursor, y_cursor + h)
        y_cursor += h
        rows.append(gap_row)
        y_cursor += BATCH_ROW_GAP

    return np.vstack(rows), layout


def _group_tokens_by_layout(items, layout):
    """Petakan token kanvas gabungan -> nama crop asal (spatial grouping
    berdasarkan posisi baris/y kanvas), bbox DIKEMBALIKAN ke koordinat LOKAL
    crop asal (origin (0,0) = pojok kiri-atas crop tsb SEBELUM digabung ke
    kanvas -- offset y0 dari layout dikurangi; x tidak perlu digeser karena
    penggabungan hanya VERTIKAL, semua crop mulai dari x=0).

    ROOT CAUSE FIX (V9.2 rev.2 bug): assign_items_to_fields() versi lama
    langsung menggabung token jadi SATU STRING per field DI TITIK INI --
    bbox token individual hilang total sebelum sempat sampai ke pemanggil
    (dynamic_extraction._get_tokens mengharapkan list token dgn bbox, tapi
    yang diterima selalu dict {"raw","confidence"} sudah rata -- kondisi
    `isinstance(raw, list)` di sana TIDAK PERNAH terpenuhi, jadi asosiasi
    spasial token->field selalu jatuh ke mode degradasi 1-token). Fungsi ini
    memisahkan "pengelompokan spasial" dari "penggabungan jadi string",
    supaya token individual bisa dipakai ulang oleh run_batched_ocr_tokens()."""
    grouped = {name: [] for name in layout}
    for item in items:
        bbox = item.get("bbox")
        if not bbox:
            continue
        cy = (bbox[1] + bbox[3]) / 2
        for name, (y0, y1) in layout.items():
            if y0 - 2 <= cy <= y1 + 2:
                local_bbox = (bbox[0], bbox[1] - y0, bbox[2], bbox[3] - y0)
                grouped[name].append({**item, "bbox": local_bbox})
                break
    return grouped


def assign_items_to_fields(items, layout):
    """DIPERTAHANKAN utk kompatibilitas pemanggil lama: petakan hasil OCR
    kanvas gabungan kembali ke field asal, lalu susun teks dalam urutan
    baca jadi satu string per field. Sekarang dibangun di atas
    _group_tokens_by_layout() (lihat docstringnya utk root-cause bug yang
    diperbaiki) supaya logika pengelompokan spasial tidak dobel-implementasi."""
    grouped = _group_tokens_by_layout(items, layout)
    results = {}
    for name, group_items in grouped.items():
        group_items = sorted(group_items, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        texts = [x["text"] for x in group_items if x.get("text")]
        scores = [x["confidence"] for x in group_items if x.get("confidence") is not None]
        results[name] = {
            "raw": " ".join(texts).strip() or None,
            "confidence": float(np.mean(scores)) if scores else None,
        }
    return results


# ============================================================================
# ADAPTIVE BATCHING (kontrol VRAM) -- pecah kanvas gabungan jadi beberapa
# kanvas kecil kalau totalnya melebihi batas tinggi per device, supaya GPU
# kecil (mis. MX230 2GB) tidak coba menge-OCR satu kanvas raksasa sekaligus.
# Tetap JAUH lebih hemat panggilan dibanding OCR per-field (biasanya 1-3
# panggilan per dokumen, bukan 1 per field), dan tidak pernah menge-OCR
# halaman penuh -- hanya crop tulisan tangan yang sudah dipersempit per field.
# ============================================================================


def _chunk_crops(crops, max_height):
    chunks, current, height = [], [], 0
    for item in crops:
        h = item[1].shape[0] + BATCH_ROW_GAP
        if current and height + h > max_height:
            chunks.append(current)
            current, height = [], 0
        current.append(item)
        height += h
    if current:
        chunks.append(current)
    return chunks


def run_batched_ocr_tokens(crops):
    """Jalankan OCR utk semua crop sekaligus (dipecah jadi beberapa kanvas
    kecil sesuai batas VRAM device aktif, spt run_batched_ocr), TAPI
    mengembalikan TOKEN INDIVIDUAL (text+confidence+bbox, bbox SUDAH
    region/crop-local -- lihat _group_tokens_by_layout) per nama crop,
    bukan satu string tergabung. Ini kontrak yang dibutuhkan
    dynamic_extraction.py utk asosiasi spasial token->field per-field di
    dalam satu coarse region OCR (mis. identity_area berisi 3 field).
    Return (tokens_by_name: dict[name -> list[token]], jumlah_panggilan_ocr)."""
    if not crops:
        return {}, 0

    max_height = MAX_CANVAS_HEIGHT_GPU if OCR_DEVICE.startswith("gpu") else MAX_CANVAS_HEIGHT_CPU
    chunks = _chunk_crops(crops, max_height)

    tokens_by_name = {name: [] for name, _ in crops}
    calls = 0
    for chunk in chunks:
        canvas, layout = compose_batch_canvas(chunk)
        if canvas is None:
            continue
        calls += 1
        grouped = _group_tokens_by_layout(run_ocr(canvas), layout)
        for name, toks in grouped.items():
            tokens_by_name.setdefault(name, []).extend(toks)
    return tokens_by_name, calls


def run_batched_ocr(crops):
    """Jalankan OCR utk semua field sekaligus, dipecah jadi beberapa kanvas
    kecil (adaptive scaling) sesuai device aktif. Return (ocr_by_field dict,
    jumlah_panggilan_ocr).

    DIPERTAHANKAN utk kompatibilitas pemanggil lama (mis. pipeline.py utk
    tempat_tanggal_surat) -- sekarang dibangun DI ATAS run_batched_ocr_tokens()
    (satu-satunya jalur agregasi OCR-batch), supaya hasil "satu string per
    field" di sini dan "token individual" di dynamic_extraction.py TIDAK
    PERNAH bisa berbeda/divergen krn diambil dari sumber yang sama."""
    tokens_by_name, calls = run_batched_ocr_tokens(crops)
    merged = {}
    for name, toks in tokens_by_name.items():
        toks_sorted = sorted(toks, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        texts = [t["text"] for t in toks_sorted if t.get("text")]
        scores = [t["confidence"] for t in toks_sorted if t.get("confidence") is not None]
        merged[name] = {
            "raw": " ".join(texts).strip() or None,
            "confidence": float(np.mean(scores)) if scores else None,
        }
    return merged, calls
