"""VLM lokal (Qwen2-VL, lazy-loaded) -- dipakai HANYA sbg fallback V10 lewat
extract_fields_independent() (lihat pipeline._run_vlm_fallback): membaca
field yang diminta secara independen dari halaman/crop dokumen, TANPA nilai
referensi apa pun di prompt. Tidak menggunakan PaddleOCR/Tesseract/OCR engine.
"""
import base64
import json
import os
import re
from pathlib import Path

import cv2
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_CANDIDATES = [
    BASE_DIR / "models" / "Qwen3-VL-2B-Instruct",
    BASE_DIR / "models" / "Qwen2-VL-2B-Instruct",
    Path("C:/models/Qwen3-VL-2B-Instruct"),
    Path("C:/models/Qwen2-VL-2B-Instruct"),
    Path("/models/Qwen2-VL-2B-Instruct"),
]

MODEL_PATH_ENV = os.environ.get("VLM_MODEL_PATH", "").strip()
OFFLINE = os.environ.get("VLM_OFFLINE", "1").lower() not in {"0", "false", "no"}
DEVICE_ENV = os.environ.get("VLM_DEVICE", "auto").strip().lower()

# 4-bit (bitsandbytes nf4) quantized loading for VRAM-constrained GPUs (e.g.
# a 2GB MX230 -- a 2B-param model needs ~4GB in fp16, ~1-1.3GB in 4-bit).
# Defaults ON whenever CUDA is actually selected (VLM_LOAD_IN_4BIT=0 to force
# full precision on GPU instead) and is a no-op on CPU regardless of this
# setting -- bitsandbytes 4-bit is CUDA-only.
LOAD_IN_4BIT_ENV = os.environ.get("VLM_LOAD_IN_4BIT", "auto").strip().lower()

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

# Separate constant for the HOSTED qwen3_vl path only (extract_direct_
# semantic_hosted) -- NOT shared with MAX_SIDE_FULL above, which stays
# untouched for the local V10/V12 fallback. Tried bumping this to 2600 to see
# if it helped the model judge small signature boxes -- it didn't (signature
# answers kept flip-flopping across runs at temperature=0, and it introduced
# a NEW misread on nominal_penempatan on the same test document), so reverted
# to match the local default. Kept as its own env var/constant rather than
# just reusing MAX_SIDE_FULL in case a future experiment wants to change it
# again without touching the local fallback's resolution.
HOSTED_MAX_SIDE_FULL = int(os.environ.get("HF_MAX_SIDE_FULL", "1600"))

# Repeated, independent calls to check the two signature fields ONLY (see
# extract_direct_semantic_hosted_majority below) -- added after observing the
# hosted model give a DIFFERENT true/false verdict for the SAME signature
# field on the IDENTICAL document across 5 separate single-call tests at
# temperature=0, while every other field (name/account/amount/tenor/reward)
# stayed reliably consistent across all of them. Aggregation is LENIENT
# (ANY run saying present wins, see extract_direct_semantic_hosted_majority),
# not a majority vote, so this is really "how many independent chances to
# spot a real signature" rather than a tie-breaking count.
HOSTED_SIGNATURE_VOTE_COUNT = int(os.environ.get("HF_SIGNATURE_VOTE_COUNT", "3"))

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
    elif model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError as exc:
            raise RuntimeError("Versi transformers terlalu lama untuk Qwen3-VL.") from exc
    elif model_type == "qwen3_vl_moe":
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration as ModelClass
        except ImportError as exc:
            raise RuntimeError("Versi transformers terlalu lama untuk Qwen3-VL-MoE.") from exc
    else:
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    _processor = AutoProcessor.from_pretrained(
        _model_path, local_files_only=OFFLINE, trust_remote_code=True
    )

    # 4-bit quantized load -- CUDA only, bitsandbytes nf4. "auto" enables it
    # whenever a CUDA device was actually selected (see LOAD_IN_4BIT_ENV
    # above); quantized models load pre-placed via device_map instead of a
    # manual .to(device) call afterward (transformers restriction).
    use_4bit = _device.startswith("cuda") and LOAD_IN_4BIT_ENV not in {"0", "false", "no"}
    quantization_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            use_4bit = False  # bitsandbytes not installed -- fall back to full precision below

    if use_4bit:
        _model = ModelClass.from_pretrained(
            _model_path,
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=OFFLINE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _model = _model.eval()
        return _model, _processor, _device

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


# ============================================================================
# DIRECT SEMANTIC CONTRACT for extract_direct_semantic_hosted() below
# (extractors.run_qwen3_vl_hosted, engine id "qwen3_vl"). PRIMARY
# EXPERIMENTAL engine (not a fallback like the two contracts above) -- the
# full untouched document image (no ROI/crop, see extractors.
# _qwen_prepare_image) plus a blank template reference image, asked to read
# the document SEMANTICALLY (e.g. distinguish the customer's handwritten
# name from a printed program title) rather than by fixed label/coordinate
# position. Confidence is the model's own self-reported number -- NOT
# derived from any geometry/OCR signal, and unrelated to the EXISTING V18
# OCR_REVIEW_THRESHOLD confidence semantics (postprocessing.py), which this
# contract never touches.
# ============================================================================

DIRECT_SEMANTIC_FIELDS = (
    "nama", "nomor_rekening", "unit_kerja", "nominal_penempatan", "tenor_penempatan",
    "tanggal_mulai", "tanggal_selesai", "bentuk_reward",
    "signature_nasabah", "signature_bri",
)

# English (the model follows English instructions more reliably than the
# Indonesian prompts elsewhere in this file -- those belong to a SEPARATE,
# still-active local VLM fallback used by pipeline.py, see the note above
# this constant's section header, and are intentionally NOT touched here).
# Exact required JSON shape is spelled out at the end so a model with no
# native structured-output/constrained-decoding support (unlike Gemini's
# response_json_schema) reliably produces parseable, correctly-keyed JSON.
# Layout/content hints below are grounded in actually rendering
# assets/template.pdf and a real filled sample side by side (not guessed):
# box positions match preprocessing.SIGNATURE_CONFIG's measured coordinates
# (signature_nasabah's value_bbox x-range 0.17-0.45 = left half of the page,
# signature_atasan's/"signature_bri"'s 0.59-0.83 = right half, same vertical
# band); the exact printed tenor/reward text and the "coret salah satu"
# (strike-through) marking convention, and the printed stamp-duty graphic
# inside the nasabah signature box, were read directly off the rendered
# images.
DIRECT_SEMANTIC_PROMPT = """You are given TWO images, in this order:
1. The BLANK template of this Indonesian bank form (empty, no customer data -- reference only, for layout/labels. Do not extract any value from it.)
2. The FILLED document -- extract every field below from THIS image only, using the blank template to understand where each field/box is and what its printed label says.

Fields to extract:
nama - the customer's own handwritten name
nomor_rekening - account number
unit_kerja - the BRI unit/branch office that manages the account, on the line printed "Unit Kerja Pengelola Rekening" (the third of three stacked identity lines: Nama Nasabah, then Nomor Rekening directly above this one, then this one, then Nominal Penempatan below)
nominal_penempatan - placement amount
tenor_penempatan - placement tenor
tanggal_mulai - start date of the placement period (ISO format)
tanggal_selesai - end date of the placement period (ISO format)
bentuk_reward - reward type
signature_nasabah - customer's signature, in the LEFT signature box near the bottom of the page
signature_bri - bank/branch representative's signature, in the RIGHT signature box at the same height

Rules:
- nama / nomor_rekening / nominal_penempatan: read the customer's own handwriting only. Ignore printed labels, headers, and instructions.
- nama: some customers are businesses, not individuals. Common Indonesian legal-entity prefixes like "CV", "PT", "UD", or "Koperasi" are FIXED, well-known abbreviations -- read them as those exact known abbreviations rather than sounding out unfamiliar letters, then read the rest of the business name normally.
- unit_kerja: this line sits directly below nomor_rekening, and the two are easy to visually merge (e.g. trailing digits of the account number bleeding into this line, or vice versa) -- read ONLY the handwritten unit/branch name written on the "Unit Kerja Pengelola Rekening" line itself, never any digits, and never the printed label text as if it were the value. If you cannot cleanly separate this line's handwriting from the account number above it, prefer null over guessing a merged value.
- tenor_penempatan: the template prints the literal text "1 / 3 / 6 Bulan (*coret salah satu)" -- three digits separated by slashes. Customers mark their chosen digit in any of three ways, sometimes combined: (a) a strike-through/cross-out pen line drawn over the digit(s) NOT chosen, leaving the chosen one as the only one with no line through it; (b) a circle/oval drawn around the chosen digit; (c) the chosen digit re-traced/written over in bold, visibly thicker/darker ink than the printed digit and the other two options (not circled, not involving a strike on the others -- just heavier ink on the one digit). Check for all three. Report the ONE digit indicated by whichever of these signals is present (if more than one signal appears, they will agree on the same digit -- use that). Never report more than one digit, and never report the raw printed text like "1/3/6" or "3/6 Bulan". If you cannot clearly identify a single chosen digit this way, answer null.
- tanggal_mulai / tanggal_selesai: the line directly BELOW the tenor choice line prints "(tanggal........... s/d. tanggal...........)" -- a start date and an end date the customer may write by hand in the two blanks. Read them and convert each to ISO format YYYY-MM-DD. If this line is blank/not filled in (very common -- the customer already indicated tenor via the mark above), answer null for both. Never guess a date from the tenor_penempatan digit.
- bentuk_reward: same marking convention as tenor_penempatan (strike-through the word NOT chosen, OR circle the chosen word, OR the chosen word re-traced in bold/darker ink -- check all three, they will agree if more than one appears), printed as "tunai / non tunai (*coret salah satu)". Report ONLY "tunai" or "non_tunai" (exactly one word, underscore not space, no other text); null if unclear. Ignore anything printed or written further below this line (there is a separate instructional line about "non-tunai" details below it, printed on the form for every document regardless of choice -- it is NOT evidence of which choice was made).
- signature_nasabah (customer, LEFT box): this specific box ALSO contains PRINTED (not handwritten) gray text reading "Opsional (Tidak Wajib) materai Rp10.000" inside a rounded rectangle -- that printed graphic is NOT a signature, ignore it completely even if it's the only thing in the box. Also ignore table borders, underlines, stamps, and scan noise/smudges -- none of these count as a signature. Answer true only for an actual handwritten signature-like mark (pen strokes, a name written in cursive/script, an initial) clearly beyond the printed graphic. Answer false if the box is genuinely empty (nothing beyond the printed graphic).
- signature_bri (bank representative, RIGHT box): this box is otherwise blank, with only a printed dotted line "(.......)". The printed dotted line itself, table borders, stamps, and scan noise/smudges do NOT count as a signature. Answer true only for an actual handwritten signature-like mark on or near that line. Answer false if the box is truly empty -- nothing but the printed dotted line.
- For both signature fields, your `confidence` self-rating (see the JSON shape below) matters as much as the true/false value: give a LOW confidence (below 0.5) whenever the mark is faint, partial, ambiguous, could plausibly be a print artifact/smudge/stray pen touch rather than a deliberate signature, or you are otherwise not sure -- regardless of which way you lean on true/false. Reserve a HIGH confidence (0.8+) for a clear, unambiguous read (either a plainly visible signature, or a box that is plainly, cleanly empty). Do not force a confident true/false when you are actually unsure -- an honest low confidence is far more useful than a guessed high one.
- Use null for any text/number field you cannot read with confidence. Do not guess.
- For every field below EXCEPT the two signature fields, also report your own status: "detected" if you read it clearly and confidently, "uncertain" if you can see something there but aren't sure it's correct/complete, "not_detected" if the field is blank or not visible at all. Use null for value whenever status is "not_detected" (uncertain may still carry your best-guess value).
- Return JSON only -- no markdown, no explanation.

Reply with EXACTLY this JSON shape (status is your own detected/uncertain/not_detected judgment for that field; the two signature fields keep a separate 0.0-1.0 confidence self-rating instead):
{"nama":{"value":null,"status":"not_detected"},"nomor_rekening":{"value":null,"status":"not_detected"},"unit_kerja":{"value":null,"status":"not_detected"},"nominal_penempatan":{"value":null,"status":"not_detected"},"tenor_penempatan":{"value":null,"status":"not_detected"},"tanggal_mulai":{"value":null,"status":"not_detected"},"tanggal_selesai":{"value":null,"status":"not_detected"},"bentuk_reward":{"value":null,"status":"not_detected"},"signature_nasabah":{"value":false,"confidence":0.0},"signature_bri":{"value":false,"confidence":0.0}}"""


def _clean_confidence(raw):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, value))


_SIGNATURE_FALSY_STRINGS = {"false", "no", "absent", "tidak", "tidak ada", "kosong", "0", ""}


def _coerce_signature_bool(raw_value):
    """Coerce a signature field's raw value to bool -- deliberately NOT a
    blind `bool(raw_value)` cast. Qwen has no structured-output/schema
    enforcement (unlike Gemini's response_json_schema), so nothing guarantees
    the model always emits a literal JSON true/false; if it ever quotes the
    boolean as a string (`"value": "false"`), Python's bool("false") is True
    -- silently flipping a negative answer to positive. Any non-empty string
    matching a known "falsy" word (case-insensitive) is treated as False;
    everything else falls back to a normal truthiness check."""
    if isinstance(raw_value, str):
        return raw_value.strip().lower() not in _SIGNATURE_FALSY_STRINGS
    return bool(raw_value)


_ALLOWED_DIRECT_SEMANTIC_STATUSES = {"detected", "uncertain", "not_detected"}
_DIRECT_SEMANTIC_SIGNATURE_FIELDS = ("signature_nasabah", "signature_bri")


def _parse_direct_semantic_json(data):
    """Parsing for the DIRECT_SEMANTIC contract's response JSON, used by
    extract_direct_semantic_hosted() below. The two signature fields keep
    their own {value, confidence} contract (feeds extract_direct_semantic_
    hosted_majority's boolean lenient voting, untouched here). Every other
    field now reports a detected/uncertain/not_detected status directly
    (the SAME enum Gemini's response_json_schema uses, see
    extractors._response_json_schema/_field_entry) instead of a raw
    self-reported 0.0-1.0 confidence -- a small VLM making a categorical
    judgment call tends to be more reliable than it self-rating a
    continuous score, and this now matches the shape every other engine's
    canonical dict already provides."""
    clean = {}
    for f in DIRECT_SEMANTIC_FIELDS:
        item = data.get(f)
        raw_value = item.get("value") if isinstance(item, dict) else item

        if f in _DIRECT_SEMANTIC_SIGNATURE_FIELDS:
            confidence = _clean_confidence(item.get("confidence", 0.0)) if isinstance(item, dict) else 0.0
            clean[f] = {"value": _coerce_signature_bool(raw_value), "confidence": confidence}
            continue

        value = raw_value if raw_value not in ("", None) else None
        status = item.get("status") if isinstance(item, dict) else None
        if status not in _ALLOWED_DIRECT_SEMANTIC_STATUSES:
            # Model answered with a flat scalar, or an invalid/missing
            # status -- tolerate it (same defensive pattern as
            # extract_fields_independent/extract_fields_fullpage above)
            # rather than silently discarding a real answer, deriving the
            # status from whether a value is actually present instead.
            status = "not_detected" if value is None else "detected"
        clean[f] = {"value": value, "status": status}
    return clean


# ============================================================================
# LOCAL -- runs DIRECT_SEMANTIC_PROMPT against a LOCAL Qwen model (default
# models/Qwen3-VL-4B-Instruct, see VLM_MODEL_PATH/DEFAULT_LOCAL_CANDIDATES
# above; extractors.run_qwen3_vl_local, engine id "qwen3_vl_local"), reusing
# _load()/_infer() (SAME loader as the V10/V12 local fallback contracts --
# see _load()'s "qwen3_vl" model_type branch, and its 4-bit bitsandbytes
# quantized-load path for VRAM-constrained GPUs). Sends the SAME two images
# (template + document, at the SAME resolution -- see the reverted-
# optimization note below for why they must match) and SAME prompt/schema
# as the hosted path, so results are directly comparable and
# extractors._adapt_qwen_to_common needs no changes to consume either.
#
# Re-verified on real hardware (RTX 4060 Laptop, 8.6GB VRAM) with the
# current 4-bit-quantized 4B model: ~15s one-time model load, ~2.9GB VRAM
# allocated, ~22-26s per document at MAX_SIDE_FULL=1600, decoding is fully
# deterministic (do_sample=False -- repeated calls on the same image
# produce byte-identical output, confirmed by direct test), and field
# values matched the hosted 30B-A3B engine on real documents. This
# supersedes an EARLIER finding in handover.md ("~0% accuracy on name/
# account number, forced down to VLM_MAX_SIDE_FULL=640") -- that was
# measured on a 2GB-VRAM MX230 with the smaller 2B model, a fundamentally
# different hardware/model combination; it does NOT describe this engine's
# behavior on adequate VRAM and should not be assumed to still apply
# without re-measuring on whatever hardware is actually being deployed to.
# Bumping MAX_SIDE_FULL further was ALSO re-tested directly against this
# combination (not assumed from the hosted-engine signature-only finding
# already noted near HOSTED_MAX_SIDE_FULL above) on a genuinely hard
# document (dense repeated-digit account number, ambiguous business-name
# handwriting): 1600 -> 2200 -> 2800 did not monotonically improve, or even
# reliably improve, accuracy on either field, while costing meaningfully
# more time (+38% at 2800) -- the bottleneck on hard cases is the
# handwriting itself, not this resolution range, so MAX_SIDE_FULL's default
# is deliberately left unchanged here rather than raised on a guess.
# ============================================================================

# TRIED AND REVERTED (do not repeat without a fix to the root cause): sending
# the template reference image at a LOWER resolution than the document image
# (e.g. template=900 vs document=1600) looked like a free ~10% speed win on
# an isolated single-document test (byte-identical extracted values), but a
# full 10-document A/B run caught a real, reproducible bug the single-doc
# test missed -- on 2 of 10 real documents, the ASYMMETRIC resolution pair
# made the model return a blanket null/not_detected for EVERY field
# (byte-identical output across two genuinely different document images,
# confirmed by direct test), even though the document image was still at
# full MAX_SIDE_FULL resolution. Isolated follow-up test on one of those
# documents confirmed the cause directly: template=1600/doc=1600 ->
# correct answer; template=900/doc=1600 (asymmetric) -> blanket null;
# template=900/doc=900 (symmetric, just low-res) -> null too (document
# genuinely needs 1600 to be legible) but does NOT show the SAME
# byte-identical-across-documents signature as the asymmetric case. This
# points to a real bug in how this Qwen3-VL build merges multiple images of
# different pixel dimensions/vision-token counts, not a legitimate model
# judgment call -- both images sent to _infer() below are kept at the SAME
# resolution (MAX_SIDE_FULL) until/unless that root cause is understood and
# fixed, matching what the hosted path already does for the same reason.


def extract_direct_semantic_local(image_bgr, max_new_tokens=None):
    """Local counterpart of extract_direct_semantic_hosted() -- same two
    images (template + document, SAME resolution -- see comment above for
    why they must match) and same DIRECT_SEMANTIC_PROMPT, but run through
    the local model via _infer()/_load() instead of an HF API call. No
    provider/timeout/retry concerns (nothing goes over the network) -- the
    only real constraint is local VRAM/RAM, controlled via the existing
    MAX_SIDE_FULL (VLM_MAX_SIDE_FULL env var) and 4-bit quantized loading in
    _load(). Returns (clean_dict, raw_text), same shape as the hosted
    function minus the meta_dict (no model/provider/elapsed_s to report --
    the caller already knows which local model is configured)."""
    template_img = _get_template_image()
    tokens = max_new_tokens or 400
    data, raw_text = _infer(
        [template_img, image_bgr], DIRECT_SEMANTIC_PROMPT, tokens,
        [MAX_SIDE_FULL, MAX_SIDE_FULL],
    )
    clean = _parse_direct_semantic_json(data)

    # V19f: conditional reward-detail follow-up (see section comments above
    # REWARD_DETAIL_PROMPT/REWARD_TUNAI_DETAIL_PROMPT) -- previously ONLY
    # wired into extract_direct_semantic_hosted_majority, so local Qwen's
    # reward_tunai/reward_non_tunai were always the mirrored choice-word
    # placeholder, never a real amount/description. One extra local-model
    # call, fired only when the main call already resolved a reward choice.
    reward_choice = clean.get("bentuk_reward", {}).get("value")
    non_tunai_detail, tunai_detail = None, None
    if reward_choice == "non_tunai":
        non_tunai_detail = _extract_reward_detail_local(image_bgr)
    elif reward_choice == "tunai":
        tunai_detail = _extract_reward_tunai_detail_local(image_bgr)
    clean["reward_non_tunai_detail"] = {"value": non_tunai_detail, "confidence": 0.0}
    clean["reward_tunai_detail"] = {"value": tunai_detail, "confidence": 0.0}

    return clean, raw_text


# ============================================================================
# HOSTED -- Hugging Face Inference API test for Qwen3-VL-2B-Instruct
# (extractors.run_qwen3_vl_hosted, engine id "qwen3_vl"). NO local weights,
# NO torch/transformers -- one chat-completion call to HF's hosted
# infrastructure via huggingface_hub.InferenceClient. Reuses the SAME
# DIRECT_SEMANTIC_PROMPT/DIRECT_SEMANTIC_FIELDS contract + _extract_json/
# _parse_direct_semantic_json parsing as the local path above, so
# extractors._adapt_qwen_to_common needs no changes to consume either.
# ============================================================================


class HostedInferenceError(RuntimeError):
    """Raised on any Hugging Face hosted-inference failure (missing token,
    auth, timeout, bad/unparseable response). Never caught inside this
    module -- extractors.run_qwen3_vl_hosted lets it propagate as a real
    error instead of silently falling back to the OCR/ROI pipeline (this
    test's explicit requirement: no silent fallback)."""


def _encode_jpeg_data_uri(image_bgr, max_side):
    img = _resize_bgr(image_bgr, max_side)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HostedInferenceError("Failed to JPEG-encode the document image for hosted inference.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


_TEMPLATE_IMAGE = None
_TEMPLATE_DATA_URI = None


def _get_template_image():
    """Blank assets/template.pdf rendered once and cached in memory (static
    file, never changes mid-process -- same caching approach already used
    for other template reads, e.g. preprocessing.
    extract_template_static_text()). Raw BGR array -- shared by both the
    hosted path (_get_template_data_uri, JPEG-encodes this) and the local
    path (extract_direct_semantic_local, passes this straight to _infer)."""
    global _TEMPLATE_IMAGE
    if _TEMPLATE_IMAGE is None:
        import preprocessing as prep
        _TEMPLATE_IMAGE = prep.load_document(prep.TEMPLATE_PATH)
    return _TEMPLATE_IMAGE


def _get_template_data_uri():
    """Sent as a SECOND reference image alongside the filled document (see
    extract_direct_semantic_hosted) so the model can ground field/box
    locations against the actual printed layout instead of guessing from
    the filled page alone."""
    global _TEMPLATE_DATA_URI
    if _TEMPLATE_DATA_URI is None:
        _TEMPLATE_DATA_URI = _encode_jpeg_data_uri(_get_template_image(), HOSTED_MAX_SIDE_FULL)
    return _TEMPLATE_DATA_URI


def _resolve_live_provider(model, token):
    """Look up which Hugging Face Inference Provider(s) actually serve
    `model` right now (some models, e.g. Qwen/Qwen3-VL-2B-Instruct at the
    time this was written, are served by exactly ONE non-default provider --
    InferenceClient's default provider="auto" does not always route to it,
    producing a "not supported by any provider" error even though the model
    IS servable). Returns the first provider with status "live", or None if
    the lookup itself fails/finds nothing (caller then surfaces the
    ORIGINAL error instead of this lookup's)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=token).model_info(model, expand="inferenceProviderMapping")
        for mapping in getattr(info, "inference_provider_mapping", None) or []:
            if getattr(mapping, "status", None) == "live":
                return mapping.provider
    except Exception:
        return None
    return None


def extract_direct_semantic_hosted(image_bgr, model=None, hf_token=None, timeout=None, max_new_tokens=None, provider=None):
    """Sends the blank template image + the full filled-page image to
    Qwen3-VL running on Hugging Face's hosted inference infrastructure (no
    local model). `model`/`hf_token`/`timeout` fall back to the QWEN_MODEL/
    HF_TOKEN/HF_REQUEST_TIMEOUT_S env vars (see config.py for the non-secret
    defaults) when not passed explicitly. `provider` defaults to HF's "auto"
    routing (env var HF_PROVIDER overrides); if that fails with a
    "not supported by any provider" error, ONE retry is made against the
    model's actual live provider (see _resolve_live_provider) -- still
    Hugging Face hosted Qwen, NOT the OCR/ROI fallback this test forbids.
    Logs model/status/elapsed time/raw response/parsed JSON to stdout (task
    requirement) and returns (clean_dict, raw_text, meta_dict) where
    meta_dict = {"model", "provider", "status", "elapsed_s"} for the caller
    to also surface via ocr_meta. Raises HostedInferenceError (wrapping the
    REAL underlying error) on any failure -- never falls back to anything
    itself."""
    import time as _time
    from huggingface_hub import InferenceClient

    token = (hf_token or os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        raise HostedInferenceError("HF_TOKEN environment variable is not set")
    model = (model or os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-2B-Instruct")).strip()
    timeout = timeout if timeout is not None else int(os.environ.get("HF_REQUEST_TIMEOUT_S", "60"))
    provider = provider or os.environ.get("HF_PROVIDER", "").strip() or None

    template_uri = _get_template_data_uri()
    data_uri = _encode_jpeg_data_uri(image_bgr, HOSTED_MAX_SIDE_FULL)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": template_uri}},
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": DIRECT_SEMANTIC_PROMPT},
        ],
    }]

    def _call(active_provider):
        client = InferenceClient(token=token, timeout=timeout, provider=active_provider)
        return client.chat_completion(
            messages=messages, model=model,
            max_tokens=max_new_tokens or 360, temperature=0,
        )

    started = _time.monotonic()
    print(f"[qwen3_vl_hosted] model={model} provider={provider or 'auto'} request=start timeout={timeout}s")
    try:
        response = _call(provider)
    except Exception as exc:
        if provider is None and "not supported by any provider" in str(exc):
            fallback_provider = _resolve_live_provider(model, token)
            if fallback_provider:
                print(f"[qwen3_vl_hosted] auto-provider routing failed, retrying with provider={fallback_provider}")
                try:
                    response = _call(fallback_provider)
                    provider = fallback_provider
                except Exception as exc2:
                    elapsed = _time.monotonic() - started
                    print(f"[qwen3_vl_hosted] model={model} request=error elapsed={elapsed:.2f}s error={exc2}")
                    raise HostedInferenceError(f"Hugging Face hosted inference failed ({type(exc2).__name__}): {exc2}") from exc2
            else:
                elapsed = _time.monotonic() - started
                print(f"[qwen3_vl_hosted] model={model} request=error elapsed={elapsed:.2f}s error={exc}")
                raise HostedInferenceError(f"Hugging Face hosted inference failed ({type(exc).__name__}): {exc}") from exc
        else:
            elapsed = _time.monotonic() - started
            print(f"[qwen3_vl_hosted] model={model} request=error elapsed={elapsed:.2f}s error={exc}")
            raise HostedInferenceError(f"Hugging Face hosted inference failed ({type(exc).__name__}): {exc}") from exc
    elapsed = _time.monotonic() - started

    raw_text = (response.choices[0].message.content or "") if response.choices else ""
    print(f"[qwen3_vl_hosted] model={model} request=ok elapsed={elapsed:.2f}s")
    print(f"[qwen3_vl_hosted] raw_response={raw_text!r}")

    try:
        data = _extract_json(raw_text)
    except Exception as exc:
        raise HostedInferenceError(f"invalid JSON from hosted Qwen3-VL: {exc}") from exc

    clean = _parse_direct_semantic_json(data)
    print(f"[qwen3_vl_hosted] parsed={clean!r}")
    meta = {"model": model, "provider": provider or "auto", "status": "ok", "elapsed_s": round(elapsed, 3)}
    return clean, raw_text, meta


# ============================================================================
# REWARD NON-TUNAI DETAIL -- separate, CONDITIONAL follow-up call, deliberately
# NOT part of DIRECT_SEMANTIC_PROMPT above. Real test evidence (downloaded_
# documents/1a7X78Si8DTJCpf6Czq33IdnokoXoFRuA, "tunai" circled, ZERO
# strike-through anywhere on that line): asking for this detail in the SAME
# call as bentuk_reward biased the model into answering bentuk_reward=
# "non_tunai" (wrong -- ground truth is tunai) AND copying the form's own
# PRINTED example text ("iPhone 17 Pro Max 128 GB Warna Silver") as if it
# were the customer's answer -- reproduced twice, including after adding an
# explicit "ignore the printed example" instruction to that same prompt.
# Splitting this into its own call that ONLY fires when the (now unbiased)
# main call already said bentuk_reward=="non_tunai" removes the priming
# effect at the source instead of wording around it, and costs nothing extra
# for the common case (this dataset's ground truth is 100% tunai -- this
# follow-up is not expected to fire at all against it).
# ============================================================================

REWARD_DETAIL_PROMPT = """You are given TWO images: 1) the BLANK template of this Indonesian bank form, 2) the FILLED document.

The document's "Bentuk Reward" choice was already read as "non_tunai" (non-cash reward). Below that choice line, the template prints "(Jika non-tunai, sebutkan barang dengan spesifik:.....)" -- a handwritten blank -- followed by a SEPARATE printed line "Contoh Pengisian: iPhone 17 Pro Max 128 GB Warna Silver". That "Contoh Pengisian" line is a PRINTED EXAMPLE of how to fill the blank, on every copy of this form -- it is never the customer's actual answer.

Read ONLY the customer's own handwritten text in the blank after "spesifik:". If your answer would be "iPhone 17 Pro Max 128 GB Warna Silver" or a close variant of it, you have read the printed example by mistake -- answer null instead. If the blank looks empty/unfilled, answer null.

Reply with EXACTLY this JSON shape, no markdown, no explanation:
{"reward_non_tunai_detail":{"value":null,"confidence":0.0}}"""

_REWARD_DETAIL_PRINTED_EXAMPLE = "iphone 17 pro max 128 gb warna silver"


def _extract_reward_detail_hosted(image_bgr, model, hf_token, timeout, provider=None):
    """Fires the REWARD_DETAIL_PROMPT follow-up call (see section comment
    above). Returns the detail string or None -- never raises on failure
    (this is a best-effort enrichment of an already-complete result; a
    failure here should not turn a working extraction into a hard error),
    logging the failure instead."""
    from huggingface_hub import InferenceClient

    template_uri = _get_template_data_uri()
    data_uri = _encode_jpeg_data_uri(image_bgr, HOSTED_MAX_SIDE_FULL)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": template_uri}},
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": REWARD_DETAIL_PROMPT},
        ],
    }]
    try:
        client = InferenceClient(token=hf_token, timeout=timeout, provider=provider)
        response = client.chat_completion(messages=messages, model=model, max_tokens=80, temperature=0)
        raw_text = (response.choices[0].message.content or "") if response.choices else ""
        data = _extract_json(raw_text)
        item = data.get("reward_non_tunai_detail")
        value = item.get("value") if isinstance(item, dict) else item
    except Exception as exc:
        print(f"[qwen3_vl_hosted] reward_non_tunai_detail follow-up failed (non-fatal): {exc}")
        return None

    if value and _REWARD_DETAIL_PRINTED_EXAMPLE in str(value).strip().lower():
        print(f"[qwen3_vl_hosted] reward_non_tunai_detail follow-up returned the printed example verbatim -- discarding: {value!r}")
        return None
    print(f"[qwen3_vl_hosted] reward_non_tunai_detail follow-up result: {value!r}")
    return value if value not in ("", None) else None


def _extract_reward_detail_local(image_bgr):
    """LOCAL counterpart of _extract_reward_detail_hosted -- same
    REWARD_DETAIL_PROMPT/printed-example guard/conditional-isolation policy
    (see the section comment above REWARD_DETAIL_PROMPT), but run through
    the on-device model via _infer() instead of an HF API call. Added V19f:
    extract_direct_semantic_local() previously had NO equivalent of this
    follow-up at all, so reward_non_tunai was always the mirrored choice-word
    placeholder for local Qwen even when the hosted engine could already get
    a real value. Never raises on failure (best-effort enrichment), same
    contract as the hosted version."""
    template_img = _get_template_image()
    try:
        data, _raw = _infer(
            [template_img, image_bgr], REWARD_DETAIL_PROMPT, 80, [MAX_SIDE_FULL, MAX_SIDE_FULL],
        )
        item = data.get("reward_non_tunai_detail")
        value = item.get("value") if isinstance(item, dict) else item
    except Exception as exc:
        print(f"[qwen3_vl_local] reward_non_tunai_detail follow-up failed (non-fatal): {exc}")
        return None

    if value and _REWARD_DETAIL_PRINTED_EXAMPLE in str(value).strip().lower():
        print(f"[qwen3_vl_local] reward_non_tunai_detail follow-up returned the printed example verbatim -- discarding: {value!r}")
        return None
    print(f"[qwen3_vl_local] reward_non_tunai_detail follow-up result: {value!r}")
    return value if value not in ("", None) else None


# ============================================================================
# REWARD TUNAI DETAIL -- V19f, symmetric counterpart to REWARD_NON_TUNAI
# DETAIL above. Same gap existed on the "tunai" side but for BOTH engines
# (hosted and local): DIRECT_SEMANTIC_FIELDS has no field for the actual cash
# amount, so extractors.adapt_common_to_pipeline_shape always fell back to
# the literal string "tunai" for reward_tunai, never a real value. Added as
# its own conditional call (NOT merged into DIRECT_SEMANTIC_PROMPT, same
# reasoning as the non-tunai case above) -- fires only when the main call's
# bentuk_reward=="tunai". Real eval data (eval_runs/evaluation_results_
# qwen_local_v2_fixed.csv) shows this dataset's ground truth is ~100% tunai,
# so this is the MORE commonly-hit gap of the two, not the rarer one.
#
# V19f FIX: the first version of this prompt guessed the amount was "on the
# same row/area as the reward choice" -- WRONG, confirmed by reading
# assets/template.pdf directly (real user report: reward_tunai was still
# coming back as the literal placeholder "tunai"). The template's actual
# printed line is its OWN separate line, several lines BELOW both the
# Bentuk Reward choice and the non-tunai "sebutkan barang" blank:
#   "Nilai Reward (termasuk pajak) : Rp…………………………... (……………………………………)"
# -- same two-blank format (digits + spelled-out terbilang in parentheses)
# as "Nominal Penempatan" above it, which the model already reads reliably
# via DIRECT_SEMANTIC_PROMPT's nominal_penempatan field. Prompt rewritten
# below to point at this exact line instead of guessing a position.
# ============================================================================

REWARD_TUNAI_DETAIL_PROMPT = """You are given TWO images: 1) the BLANK template of this Indonesian bank form, 2) the FILLED document.

The document's "Bentuk Reward" choice was already read as "tunai" (cash reward). Several lines BELOW that choice line (and below the separate "(Jika non-tunai, sebutkan barang dengan spesifik:.....)" blank, which is not relevant here), the template prints its own line:
"Nilai Reward (termasuk pajak) : Rp…………………………... (……………………………………)"
This is a DIFFERENT line from "Nominal Penempatan" further up the page -- do not confuse the two. It has the same two-blank format: a Rupiah amount after "Rp", followed by that amount spelled out in words inside the parentheses.

Read ONLY the customer's/bank staff's own handwritten amount written after "Nilai Reward (termasuk pajak) : Rp" on that specific line. Report it as DIGITS ONLY -- no "Rp", no thousands separators (dots/commas), and never the spelled-out words in parentheses -- e.g. answer "500000" for a handwritten "Rp500.000 (lima ratus ribu rupiah)". If that blank looks empty/unfilled, answer null. Do not invent or guess a plausible-looking number, and do not answer with the Nominal Penempatan amount instead.

Reply with EXACTLY this JSON shape, no markdown, no explanation:
{"reward_tunai_detail":{"value":null,"confidence":0.0}}"""


def _clean_reward_tunai_digits(value):
    """Digits-only cleanup for REWARD_TUNAI_DETAIL_PROMPT's answer -- the
    model is already asked for digits-only, but this is a cheap safety net
    against it echoing "Rp"/separators/spelled-out words anyway."""
    if value in (None, ""):
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits or None


def _extract_reward_tunai_detail_hosted(image_bgr, model, hf_token, timeout, provider=None):
    """Hosted counterpart of _extract_reward_detail_hosted, for the tunai
    amount instead of the non-tunai description (see section comment above
    REWARD_TUNAI_DETAIL_PROMPT). Never raises on failure."""
    from huggingface_hub import InferenceClient

    template_uri = _get_template_data_uri()
    data_uri = _encode_jpeg_data_uri(image_bgr, HOSTED_MAX_SIDE_FULL)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": template_uri}},
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": REWARD_TUNAI_DETAIL_PROMPT},
        ],
    }]
    try:
        client = InferenceClient(token=hf_token, timeout=timeout, provider=provider)
        response = client.chat_completion(messages=messages, model=model, max_tokens=40, temperature=0)
        raw_text = (response.choices[0].message.content or "") if response.choices else ""
        data = _extract_json(raw_text)
        item = data.get("reward_tunai_detail")
        value = item.get("value") if isinstance(item, dict) else item
    except Exception as exc:
        print(f"[qwen3_vl_hosted] reward_tunai_detail follow-up failed (non-fatal): {exc}")
        return None

    value = _clean_reward_tunai_digits(value)
    print(f"[qwen3_vl_hosted] reward_tunai_detail follow-up result: {value!r}")
    return value


def _extract_reward_tunai_detail_local(image_bgr):
    """LOCAL counterpart of _extract_reward_tunai_detail_hosted, run through
    the on-device model via _infer() instead of an HF API call. Never raises
    on failure."""
    template_img = _get_template_image()
    try:
        data, _raw = _infer(
            [template_img, image_bgr], REWARD_TUNAI_DETAIL_PROMPT, 40, [MAX_SIDE_FULL, MAX_SIDE_FULL],
        )
        item = data.get("reward_tunai_detail")
        value = item.get("value") if isinstance(item, dict) else item
    except Exception as exc:
        print(f"[qwen3_vl_local] reward_tunai_detail follow-up failed (non-fatal): {exc}")
        return None

    value = _clean_reward_tunai_digits(value)
    print(f"[qwen3_vl_local] reward_tunai_detail follow-up result: {value!r}")
    return value


_VOTED_SIGNATURE_FIELDS = ("signature_nasabah", "signature_bri")


def extract_direct_semantic_hosted_majority(image_bgr, model=None, hf_token=None, timeout=None,
                                             max_new_tokens=None, provider=None, votes=None):
    """Calls extract_direct_semantic_hosted() `votes` times (default
    HOSTED_SIGNATURE_VOTE_COUNT) and combines runs for ONLY the two
    signature fields -- added specifically because those two fields (and
    only those two) gave a DIFFERENT true/false verdict on the IDENTICAL
    document across repeated single-call tests at temperature=0 (documented
    in handover.md). Aggregation is LENIENT, not a strict majority (explicit
    user policy): ANY run saying "present" makes the final answer present;
    "absent" only survives if EVERY run agrees nothing is there -- false
    negatives (rejecting a real signature) are treated as worse than false
    positives here. Text/number fields are deliberately NOT re-voted: their
    formatting varies harmlessly between runs (e.g. "5000000" vs "Rp
    5.000.000" for the same correct amount), which would break a naive
    exact-match vote, and they were already reliably correct from a single
    call -- so they're taken as-is from the FIRST run.

    Costs `votes`x the API calls/latency/token spend of a single call (each
    full run still re-extracts every field, simplest correct
    implementation, not the cheapest possible one). Returns (clean_dict,
    first_run_raw_text, meta_dict) -- meta_dict adds "votes" (int) and
    "signature_votes" ({field: [bool, ...]}) on top of the usual
    model/provider/status/elapsed_s keys, for full transparency into what
    each individual run answered."""
    votes = votes or HOSTED_SIGNATURE_VOTE_COUNT
    runs = []
    for i in range(votes):
        clean, raw_text, meta = extract_direct_semantic_hosted(
            image_bgr, model=model, hf_token=hf_token, timeout=timeout,
            max_new_tokens=max_new_tokens, provider=provider,
        )
        runs.append((clean, raw_text, meta))
        print(f"[qwen3_vl_hosted] vote {i + 1}/{votes}: "
              f"signature_nasabah={clean['signature_nasabah']['value']} "
              f"signature_bri={clean['signature_bri']['value']}")

    final_clean, first_raw_text, first_meta = runs[0]
    signature_votes = {}
    for field in _VOTED_SIGNATURE_FIELDS:
        values = [r[0][field]["value"] for r in runs]
        confidences = [r[0][field]["confidence"] for r in runs]
        signature_votes[field] = values
        # LENIENT aggregation (explicit user policy): absence should be the
        # harder conclusion to reach -- ANY vote saying "present" is enough
        # to call it present; "absent" requires EVERY vote to agree nothing
        # is there. Deliberately NOT a strict majority (that would still let
        # 2-out-of-3 "absent" votes override a single correct "present").
        final_clean[field] = {
            "value": any(values),
            "confidence": sum(confidences) / len(confidences),
        }

    meta = dict(first_meta)
    meta["votes"] = votes
    meta["signature_votes"] = signature_votes
    print(f"[qwen3_vl_hosted] lenient vote result: "
          f"signature_nasabah={final_clean['signature_nasabah']['value']} "
          f"signature_bri={final_clean['signature_bri']['value']} "
          f"(raw votes={signature_votes})")

    # Conditional follow-up (see section comment above _extract_reward_
    # detail_hosted) -- ONLY fires when bentuk_reward is non_tunai, so it
    # costs nothing extra for the (in this dataset, universal) tunai case.
    detail_value = None
    tunai_detail_value = None
    reward_choice = final_clean.get("bentuk_reward", {}).get("value")
    if reward_choice == "non_tunai":
        detail_value = _extract_reward_detail_hosted(
            image_bgr, model=first_meta["model"], hf_token=hf_token,
            timeout=timeout, provider=first_meta.get("provider"),
        )
    elif reward_choice == "tunai":
        # V19f: symmetric follow-up for the cash amount (see section comment
        # above REWARD_TUNAI_DETAIL_PROMPT) -- this dataset's ground truth is
        # ~100% tunai, so THIS is the branch that actually fires in practice.
        tunai_detail_value = _extract_reward_tunai_detail_hosted(
            image_bgr, model=first_meta["model"], hf_token=hf_token,
            timeout=timeout, provider=first_meta.get("provider"),
        )
    final_clean["reward_non_tunai_detail"] = {"value": detail_value, "confidence": 0.0}
    final_clean["reward_tunai_detail"] = {"value": tunai_detail_value, "confidence": 0.0}

    return final_clean, first_raw_text, meta


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
