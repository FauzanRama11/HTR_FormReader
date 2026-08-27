"""VLM lokal untuk ekstraksi formulir full-page + second-pass area detail.

Tidak menggunakan PaddleOCR/Tesseract/OCR engine. Model melihat template kosong
sebagai referensi dan dokumen terisi sebagai target.
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
# naik seiring resolusi). Ada 3 titik hemat yang TIDAK mengurangi kebutuhan:
# 1) MAX_SIDE_TEMPLATE diturunkan (bukan MAX_SIDE_FULL/dokumen) karena template
#    hanya berisi label cetak -- tidak perlu se-detail dokumen tulisan tangan.
#    Template ini juga ikut dikirim pada SETIAP panggilan (full + tiap detail
#    zone), jadi penurunan di sini paling terasa hematnya.
# 2) MAX_NEW_TOKENS_* diturunkan tipis (bukan dipepetkan) -- masih cukup untuk
#    JSON terpanjang yang mungkin muncul, hanya membuang buffer berlebih.
# 3) DETAIL_MODE default diubah ke "auto" di pipeline.py (lihat file itu):
#    zone yang bukan field visual (coretan/tanda tangan) dilewati bila full
#    page sudah yakin, sementara zone visual-critical TETAP selalu diverifikasi
#    dua kali supaya akurasi tidak turun.
# Semua nilai ini tetap bisa dioverride lewat environment variable bila perlu
# resolusi/token lebih tinggi untuk dokumen yang sulit dibaca.
# ----------------------------------------------------------------------------
MAX_SIDE_FULL = int(os.environ.get("VLM_MAX_SIDE_FULL", "1600"))
MAX_SIDE_TEMPLATE = int(os.environ.get("VLM_MAX_SIDE_TEMPLATE", "1100"))
MAX_SIDE_DETAIL = int(os.environ.get("VLM_MAX_SIDE_DETAIL", "1500"))
MAX_NEW_TOKENS_FULL = int(os.environ.get("VLM_MAX_NEW_TOKENS_FULL", "360"))
MAX_NEW_TOKENS_DETAIL = int(os.environ.get("VLM_MAX_NEW_TOKENS_DETAIL", "220"))

if OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_model = None
_processor = None
_device = None
_model_path = None

FULL_KEYS = [
    "nama_nasabah", "nomor_rekening", "unit_kerja_pengelola_rekening",
    "nominal_penempatan", "tenor_penempatan", "rentang_tenor",
    "bentuk_reward", "reward_non_tunai", "reward_tunai",
    "tempat_tanggal_surat", "signature_nasabah", "signature_atasan",
]

FULL_PROMPT = r"""
Anda membaca formulir Indonesia yang memiliki TEMPLATE KOSONG dan DOKUMEN TERISI.
Gambar pertama adalah TEMPLATE KOSONG. Gambar kedua adalah DOKUMEN TERISI.
Bandingkan keduanya dan ambil HANYA nilai yang diisi/ditulis/ditandai pada dokumen terisi.
Jangan menyalin label cetak seperti "Nama Nasabah", "Nomor Rekening", atau potongan kata label.

Aturan pilihan visual:
- Tenor Penempatan menyediakan 1 / 3 / 6 Bulan dengan instruksi *coret salah satu*.
  Jika dua pilihan dicoret, nilai yang TIDAK dicoret adalah jawaban. Jika hanya satu yang diberi tanda/lingkaran, nilai bertanda adalah jawaban.
- Bentuk Reward menyediakan tunai / non tunai dengan instruksi *coret salah satu*.
  Gunakan logika yang sama. Kembalikan hanya "tunai" atau "non_tunai".
- signature_nasabah dan signature_atasan hanya menilai ADA/TIDAK ADA goresan tanda tangan pada area masing-masing; jangan mengidentifikasi orangnya.
- Jika tidak yakin, isi null. Jangan menebak.
- Pertahankan tulisan nama/unit/rentang tanggal apa adanya sebisa mungkin.

Balas HANYA satu objek JSON valid, tanpa markdown, dengan key persis berikut:
{
  "nama_nasabah": string|null,
  "nomor_rekening": string|null,
  "unit_kerja_pengelola_rekening": string|null,
  "nominal_penempatan": string|null,
  "tenor_penempatan": 1|3|6|null,
  "rentang_tenor": string|null,
  "bentuk_reward": "tunai"|"non_tunai"|null,
  "reward_non_tunai": string|null,
  "reward_tunai": string|null,
  "tempat_tanggal_surat": string|null,
  "signature_nasabah": true|false|null,
  "signature_atasan": true|false|null
}
""".strip()

DETAIL_PROMPT_TEMPLATE = r"""
Gambar pertama adalah potongan TEMPLATE KOSONG. Gambar kedua adalah area yang sama dari DOKUMEN TERISI.
Baca hanya field berikut: {fields}.
Bandingkan template vs dokumen agar teks label cetak tidak ikut dianggap sebagai nilai.
Untuk pilihan 1/3/6 atau tunai/non tunai, pahami coretan/lingkaran secara visual sesuai instruksi *coret salah satu*.
Untuk signature, kembalikan true bila ada tanda tangan/goretan nyata pada area tanda tangan, false bila kosong, null bila tidak yakin.
Jangan menebak. Balas HANYA JSON valid yang berisi key yang diminta, tanpa markdown.
""".strip()


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


def runtime_info():
    return {
        "model_path": _model_path or MODEL_PATH_ENV or "auto-local",
        "device": _device or DEVICE_ENV,
        "offline": OFFLINE,
    }


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


def extract_full_page(template_bgr, document_bgr):
    data, raw_text = _infer(
        [template_bgr, document_bgr],
        FULL_PROMPT,
        MAX_NEW_TOKENS_FULL,
        [MAX_SIDE_TEMPLATE, MAX_SIDE_FULL],
    )
    # Hanya key kontrak yang diterima; key liar dibuang.
    clean = {k: data.get(k) for k in FULL_KEYS}
    return clean, raw_text


def extract_detail(template_crop_bgr, document_crop_bgr, fields):
    prompt = DETAIL_PROMPT_TEMPLATE.format(fields=", ".join(fields))
    data, raw_text = _infer(
        [template_crop_bgr, document_crop_bgr],
        prompt,
        MAX_NEW_TOKENS_DETAIL,
        [MAX_SIDE_DETAIL, MAX_SIDE_DETAIL],
    )
    return {k: data.get(k) for k in fields}, raw_text


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
