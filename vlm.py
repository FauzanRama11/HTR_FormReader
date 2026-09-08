"""VLM lokal (Qwen2-VL, lazy-loaded) -- dipakai HANYA sbg fallback V10 lewat
extract_fields_independent() (lihat pipeline._run_vlm_fallback): membaca
field yang diminta secara independen dari halaman/crop dokumen, TANPA nilai
referensi apa pun di prompt. Tidak menggunakan PaddleOCR/Tesseract/OCR engine.
"""
import json
import os
import re
from pathlib import Path

import cv2
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_CANDIDATES = [
    BASE_DIR / "models" / "Qwen2-VL-2B-Instruct",
    Path("C:/models/Qwen2-VL-2B-Instruct"),
    Path("/models/Qwen2-VL-2B-Instruct"),
]

MODEL_PATH_ENV = os.environ.get("VLM_MODEL_PATH", "").strip()
OFFLINE = os.environ.get("VLM_OFFLINE", "1").lower() not in {"0", "false", "no"}
DEVICE_ENV = os.environ.get("VLM_DEVICE", "auto").strip().lower()

# ----------------------------------------------------------------------------
# HEMAT TOKEN: gambar yang dikirim ke VLM diubah jadi token gambar (jumlahnya
# naik seiring resolusi). MAX_SIDE_FULL (halaman penuh) & MAX_SIDE_DETAIL
# (crop per-field) dipakai oleh extract_fields_independent() -- satu-satunya
# jalur VLM yang aktif (fallback V10, lihat pipeline._run_vlm_fallback).
# Tetap bisa dioverride lewat environment variable bila perlu resolusi lebih
# tinggi untuk dokumen yang sulit dibaca.
# ----------------------------------------------------------------------------
MAX_SIDE_FULL = int(os.environ.get("VLM_MAX_SIDE_FULL", "1600"))
MAX_SIDE_DETAIL = int(os.environ.get("VLM_MAX_SIDE_DETAIL", "1500"))

if OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_model = None
_processor = None
_device = None
_model_path = None

def _resolve_model_path():
    if MODEL_PATH_ENV:
        p = Path(MODEL_PATH_ENV)
        if p.exists():
            return str(p)
        if not OFFLINE:
            return MODEL_PATH_ENV  # boleh repo id HF jika online sengaja diaktifkan
        raise RuntimeError("Folder model VLM dari VLM_MODEL_PATH tidak ditemukan.")
    for p in DEFAULT_LOCAL_CANDIDATES:
        if p.exists():
            return str(p)
    if not OFFLINE:
        return "Qwen/Qwen2-VL-2B-Instruct"
    raise RuntimeError(
        "Model VLM lokal tidak ditemukan. Set VLM_MODEL_PATH ke folder Qwen2-VL/Qwen2.5-VL lokal."
    )


def _choose_device(torch):
    if DEVICE_ENV not in {"", "auto"}:
        if DEVICE_ENV.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("VLM_DEVICE meminta CUDA tetapi CUDA tidak tersedia pada PyTorch saat ini.")
        return DEVICE_ENV
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _model, _processor, _device, _model_path
    if _model is not None:
        return _model, _processor, _device

    try:
        import torch
        from transformers import AutoConfig, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Dependensi VLM belum terpasang. Jalankan: pip install -r requirements.txt") from exc

    _device = _choose_device(torch)
    _model_path = _resolve_model_path()

    # Batasi oversubscription CPU. Nilai dapat diubah lewat VLM_CPU_THREADS.
    if _device == "cpu":
        threads = int(os.environ.get("VLM_CPU_THREADS", str(max(1, min(os.cpu_count() or 4, 8)))))
        torch.set_num_threads(max(1, threads))

    config = AutoConfig.from_pretrained(_model_path, local_files_only=OFFLINE, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as ModelClass
    elif model_type in {"qwen2_5_vl", "qwen2_5_omni"}:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError as exc:
            raise RuntimeError("Versi transformers terlalu lama untuk Qwen2.5-VL.") from exc
    else:
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    _processor = AutoProcessor.from_pretrained(
        _model_path, local_files_only=OFFLINE, trust_remote_code=True
    )

    # 'auto' mempertahankan dtype bawaan weight sehingga CPU tidak dipaksa
    # float32 8+ GB. Pada CUDA, float16 biasanya paling hemat VRAM.
    dtype = torch.float16 if _device.startswith("cuda") else "auto"
    try:
        _model = ModelClass.from_pretrained(
            _model_path,
            torch_dtype=dtype,
            local_files_only=OFFLINE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        _model = ModelClass.from_pretrained(
            _model_path,
            local_files_only=OFFLINE,
            trust_remote_code=True,
        )

    _model = _model.to(_device).eval()
    return _model, _processor, _device


def _resize_bgr(image, max_side):
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 0.999:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _to_pil(image_bgr, max_side):
    img = _resize_bgr(image_bgr, max_side)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _try_parse(candidate):
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _repair_truncated_json(candidate):
    """Perbaikan darurat bila output VLM terpotong pas di batas max_new_tokens
    (mis. saat menghemat token), supaya field yang SUDAH lengkap tidak ikut
    hilang hanya gara-gara field terakhir belum selesai ditulis model.
    Prioritas: buang dulu key:value terakhir yang tanggung (lebih aman
    daripada menyimpan potongan nilai yang belum tentu benar); tutup tanda
    kutip yang masih terbuka hanya sebagai upaya terakhir."""
    text = candidate.rstrip()
    text = re.sub(r",\s*$", "", text)  # koma gantung di akhir

    for _ in range(4):
        obj = _try_parse(text + "}")
        if obj is not None:
            return obj
        last_comma = text.rfind(",")
        if last_comma == -1:
            break
        text = text[:last_comma]

    # Semua field tampak tanggung (tidak ada koma tersisa) -- coba tutup
    # tanda kutip yang masih terbuka sebagai upaya terakhir.
    if text.count('"') % 2 == 1:
        obj = _try_parse(text + '"}')
        if obj is not None:
            return obj
    return None


def _extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    obj = _try_parse(text)
    if obj is not None:
        return obj

    start = text.find("{")
    if start < 0:
        raise ValueError("VLM tidak menghasilkan JSON yang dapat dibaca.")
    end = text.rfind("}")
    if end > start:
        candidate = text[start:end + 1]
        obj = _try_parse(candidate)
        if obj is not None:
            return obj
        # Perbaikan ringan untuk trailing comma yang kadang muncul.
        obj = _try_parse(re.sub(r",\s*([}\]])", r"\1", candidate))
        if obj is not None:
            return obj

    # Tidak ada "}" penutup sama sekali (atau tetap gagal) -> kemungkinan
    # output terpotong oleh batas token. Coba perbaikan darurat sebelum
    # menyerah, supaya field yang sudah lengkap tidak ikut hilang.
    repaired = _repair_truncated_json(text[start:])
    if repaired is not None:
        return repaired
    raise ValueError("VLM tidak menghasilkan JSON yang dapat dibaca.")


def _infer(images_bgr, prompt, max_new_tokens, max_sides):
    import torch
    model, processor, device = _load()
    pil_images = [_to_pil(img, side) for img, side in zip(images_bgr, max_sides)]

    content = [{"type": "image"} for _ in pil_images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=pil_images, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated[:, input_len:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _extract_json(text), text


# ============================================================================
# V10 -- FALLBACK KONTRAK "INDEPENDENT READ" (dipakai pipeline.py sbg fallback
# Paddle, BUKAN default). Prompt ini SENGAJA TIDAK PERNAH menyertakan nilai
# referensi spreadsheet apa pun -- model membaca dokumen 100% independen;
# reference hanya dipakai pemanggil (pipeline.py) utk MEMUTUSKAN kapan
# fallback ini dipanggil, tidak pernah utk membentuk prompt.
# Status kontrak: "readable" (terbaca yakin) / "uncertain" (terlihat tapi
# tidak yakin) / "not_visible" (tidak terlihat/di luar frame yang dikirim).
# ============================================================================

_ALLOWED_INDEPENDENT_STATUSES = {"readable", "uncertain", "not_visible"}

INDEPENDENT_READ_PROMPT_TEMPLATE = r"""
Anda membaca sebuah formulir Indonesia yang sudah diisi tangan. Gambar yang
diberikan adalah dokumen tersebut (halaman penuh dan/atau potongan area
tertentu untuk memperjelas detail).

Baca HANYA field berikut, PERSIS seperti yang tertulis secara visual:
{fields}

Aturan WAJIB:
- Baca apa adanya. JANGAN menebak, melengkapi, menormalisasi, atau
  mengoreksi ejaan/format.
- Anda TIDAK diberi nilai pembanding apa pun -- jangan mengasumsikan nilai
  tertentu "seharusnya" benar; laporkan hanya apa yang benar-benar terlihat.
- Kalau tulisan tidak bisa dipastikan (buram/tertutup/terlalu ambigu), set
  status "uncertain" dan value berisi bacaan terbaik Anda (boleh null kalau
  benar-benar tidak ada bacaan).
- Kalau area field tidak terlihat sama sekali pada gambar yang diberikan
  (di luar frame/terpotong), set status "not_visible" dan value null.
- Kalau terbaca jelas & yakin, set status "readable".

Balas HANYA satu objek JSON valid, tanpa markdown, dengan key persis nama
field yang diminta, masing-masing berisi objek {{"value": string|null,
"status": "readable"|"uncertain"|"not_visible"}}.
""".strip()


# ============================================================================
# V12 -- FULL-PAGE FALLBACK CONTRACT. Dipanggil oleh pipeline.py sbg PRIORITAS
# KEDUA (setelah OCR Primary, SEBELUM ROI/template crop fallback -- lihat
# pipeline._run_vlm_fullpage_fallback). SATU panggilan, HANYA halaman penuh
# yang sudah di-align/preprocess (`aligned_img_bgr`) -- TIDAK PERNAH menerima
# crop ROI di tahap ini (crop baru dipakai di tahap ROI/template & second-
# pass VLM sesudahnya). Nama field di prompt/response memakai nama level-
# spesifikasi (nama_nasabah, nomor_rekening, unit_kerja, nominal_penempatan,
# tenor, bentuk_reward, signature_nasabah, signature_unit_kerja,
# tanggal_mulai, tanggal_selesai) -- pemanggil (pipeline.py) yang memetakan
# balik ke nama field internal (lihat pipeline.VLM_FULLPAGE_FIELD_MAP).
# ============================================================================

FULLPAGE_PROMPT_TEMPLATE = r"""
Anda membaca SATU HALAMAN PENUH formulir Indonesia yang sudah diisi tangan
(foto/scan sudah diluruskan/dikoreksi). Baca HANYA field berikut, PERSIS
seperti yang tertulis/tertanda secara visual pada halaman ini:
{fields}

Aturan WAJIB:
- Baca apa adanya. JANGAN menebak, melengkapi, menormalisasi, atau
  mengoreksi ejaan/format.
- Anda TIDAK diberi nilai pembanding apa pun -- laporkan hanya apa yang
  benar-benar terlihat di halaman ini.
- Kalau tulisan tidak bisa dipastikan (buram/tertutup/ambigu), set status
  "uncertain" dan value berisi bacaan terbaik Anda (boleh null kalau
  benar-benar tidak ada bacaan).
- Kalau field tidak terlihat sama sekali di halaman ini, set status
  "not_visible" dan value null.
- Kalau terbaca jelas & yakin, set status "readable".
{tenor_rule}{signature_rule}
Balas HANYA satu objek JSON valid, tanpa markdown, dengan key persis nama
field yang diminta, masing-masing berisi objek {{"value": string|null,
"status": "readable"|"uncertain"|"not_visible"}}.
""".strip()

_TENOR_RULE = """
- Field "tenor": pilihan yang DITANDAI/DICORET nasabah di antara opsi
  "1 bulan" / "3 bulan" / "6 bulan" -- ini adalah bukti UTAMA. Laporkan
  HANYA salah satu dari ketiga nilai persis itu (atau null kalau tidak
  yakin opsi mana yang ditandai) -- jangan menuliskan nilai lain.
- Field "tanggal_mulai" & "tanggal_selesai" HANYA bukti SEKUNDER (dipakai
  program utk mem-validasi/menurunkan tenor kalau pilihan di atas tidak
  ada/tidak yakin) -- baca apa adanya, JANGAN dipakai utk mengoreksi nilai
  field "tenor".
"""

_SIGNATURE_RULE = """
- Field "signature_nasabah"/"signature_unit_kerja": laporkan value "ada"
  kalau tampak ada goresan tanda tangan berarti di area itu, "kosong" kalau
  area tampak bersih/kosong. Status tetap ikuti aturan umum di atas (set
  "uncertain" kalau area tampak ada tinta tapi tidak yakin itu tanda tangan
  atau sekadar noda/coretan lain).
"""


def extract_fields_fullpage(aligned_img_bgr, spec_fields, max_new_tokens=None):
    """Full-Page VLM (PRIORITAS ke-2 di pipeline V12, lihat modul docstring
    di atas). `aligned_img_bgr`: satu gambar, halaman PENUH yang sudah
    di-align/preprocess -- TIDAK PERNAH crop ROI. `spec_fields`: list nama
    field level-spesifikasi (boleh termasuk "tenor"/"tanggal_mulai"/
    "tanggal_selesai"/"signature_nasabah"/"signature_unit_kerja"/dst).
    Return (results: dict[spec_field -> {"value":.., "status":..}], raw_text).
    SATU panggilan model walau field yang diminta >1 (sama seperti
    extract_fields_independent, kontrak status readable/uncertain/
    not_visible identik)."""
    if not spec_fields:
        return {}, ""
    tenor_rule = _TENOR_RULE if "tenor" in spec_fields else ""
    signature_rule = _SIGNATURE_RULE if any(f.startswith("signature_") for f in spec_fields) else ""
    prompt = FULLPAGE_PROMPT_TEMPLATE.format(
        fields=", ".join(spec_fields), tenor_rule=tenor_rule, signature_rule=signature_rule,
    )
    tokens = max_new_tokens or max(160, 90 * len(spec_fields))
    data, raw_text = _infer([aligned_img_bgr], prompt, tokens, [MAX_SIDE_FULL])

    clean = {}
    for f in spec_fields:
        item = data.get(f)
        if isinstance(item, dict):
            value = item.get("value")
            status = item.get("status")
            if status not in _ALLOWED_INDEPENDENT_STATUSES:
                status = "readable" if value else "not_visible"
        elif item:
            value, status = str(item), "readable"
        else:
            value, status = None, "not_visible"
        clean[f] = {"value": value, "status": status}
    return clean, raw_text


def extract_fields_independent(images_bgr, fields, max_new_tokens=None, max_sides=None):
    """Baca `fields` (list nama field) dari `images_bgr` (list gambar --
    biasa: [halaman penuh terkoreksi, *crop area/field beresolusi tinggi])
    secara independen, TANPA nilai referensi apa pun di prompt. Return
    (results: dict[field -> {"value":.., "status":..}], raw_text).
    SATU panggilan model, walau field yang diminta >1 & gambar >1 (Qwen2-VL
    mendukung banyak image dalam satu pesan). `max_sides`: list panjang sama
    dgn images_bgr (opsional) utk resolusi per-gambar berbeda (mis. halaman
    penuh vs crop detail); default MAX_SIDE_DETAIL utk semua."""
    if not fields:
        return {}, ""
    prompt = INDEPENDENT_READ_PROMPT_TEMPLATE.format(fields=", ".join(fields))
    tokens = max_new_tokens or max(160, 90 * len(fields))
    sides = list(max_sides) if max_sides else [MAX_SIDE_DETAIL] * len(images_bgr)
    data, raw_text = _infer(images_bgr, prompt, tokens, sides)

    clean = {}
    for f in fields:
        item = data.get(f)
        if isinstance(item, dict):
            value = item.get("value")
            status = item.get("status")
            if status not in _ALLOWED_INDEPENDENT_STATUSES:
                status = "readable" if value else "not_visible"
        elif item:
            value, status = str(item), "readable"
        else:
            value, status = None, "not_visible"
        clean[f] = {"value": value, "status": status}
    return clean, raw_text
