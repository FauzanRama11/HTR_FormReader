"""
app.py
======
Backend FastAPI. Hanya penghubung: terima input (file/spreadsheet) -> panggil
pipeline -> kembalikan hasil. Semua logika OCR/preprocessing ada di modul lain.

Tab 1 (PDF/Image)
    POST /api/upload            -> simpan file, kembalikan preview
    POST /api/process/{id}      -> jalankan OCR penuh

Tab 2 (Excel/Google Spreadsheet) -- trigger & fetch dipisah:
    POST /api/sheet/upload            -> baca Excel -> daftar record (belum OCR)
    POST /api/sheet/load-url          -> baca Google Sheet -> daftar record
    POST /api/sheet/submit/{session}  -> TRIGGER: proses index/range terpilih
                                          di background thread (tidak blocking)
    GET  /api/sheet/batch-status/{session}     -> status tiap record (polling)
    GET  /api/sheet/result/{session}/{record}  -> FETCH hasil yang sudah jadi
                                                   (tidak memproses ulang)

Bersama: /api/progress/{request_id}, /api/template-preview

Jalankan:
    pip install -r requirements.txt
    uvicorn app:app --reload
"""

import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# V19 -- optional: auto-load GEMINI_API_KEY/HF_TOKEN/VLM_MODEL_PATH/etc from a
# local .env file if python-dotenv is installed (see .env.example). Guarded so
# a missing .env or missing package is a silent no-op -- values can also just
# be exported in the shell environment directly, this is convenience only,
# never required. MUST run BEFORE importing any project module below --
# vlm.py reads several env vars (VLM_MODEL_PATH chief among them) as
# MODULE-LEVEL constants, evaluated once at first import; loading .env any
# later leaves them frozen at their empty/default values for the life of the
# process (confirmed: this silently made the local VLM engine fall back to
# the wrong/older model in models/Qwen2-VL-2B-Instruct instead of
# VLM_MODEL_PATH's Qwen3-VL-4B-Instruct on every real server run).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline import comparison
from core import data_input
import extractors.extractors as extractors
from eval_tools import live_evaluation
import pipeline.pipeline as pipeline
from pipeline import preprocessing as prep
from pipeline import postprocessing as post
from core.paths import UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

# Progress & sesi spreadsheet disimpan di memory (server single-worker/lokal).
_PROGRESS = {}
_PROGRESS_LOCK = threading.Lock()
_SESSIONS = {}  # session_id -> {"records": [...], "status": {no: str}, "results": {no: dict}, "errors": {no: str}}
_SESSIONS_LOCK = threading.Lock()


# ============================================================================
# SANITASI PESAN ERROR -- pesan mentah exception (str(exc)) bisa memuat path
# file server (mis. "E:\data\uploads\xxx.pdf") atau detail internal lain yang
# tidak boleh tampil ke user. sanitize_error() membuang path & memetakan
# penyebab UMUM ke pesan yang cukup spesifik (masih bisa dilacak lewat nama
# exception) tanpa data sensitif ataupun terlalu generik.
# ============================================================================
_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']*|(?<![\w.])/[^\s\"']*|\\\\[^\s\"']*")

_ERROR_HINTS = [
    # V19 -- extractor API errors (extractors.ExtractorError), diperiksa
    # DULUAN (lebih spesifik) sebelum hint generik connection/timeout di
    # bawah supaya pesan yg ditampilkan tepat menyebut extractor, bukan
    # "gagal mengunduh dokumen" yg membingungkan utk kegagalan API model.
    (r"(?i)api[_ ]?key.*(not set|tidak|missing)|environment variable is not set", "API key extractor belum diisi (lihat .env)."),
    (r"(?i)unauthorized|invalid api key|authentication|permission_denied|api key not valid", "API key extractor tidak valid/ditolak."),
    (r"(?i)rate limit|quota|resource_exhausted|too many requests", "Batas rate/quota API extractor tercapai, coba lagi nanti."),
    (r"(?i)google-genai package not installed", "Package SDK extractor belum terpasang (lihat requirements.txt)."),
    (r"(?i)invalid structured response", "Respons extractor tidak sesuai skema yang diharapkan."),
    (r"(?i)pdf tidak memiliki halaman", "Dokumen PDF kosong (tidak memiliki halaman)."),
    (r"(?i)gagal membaca file|imread|cannot identify image", "Berkas dokumen tidak dapat dibaca (format tidak didukung/file rusak)."),
    (r"(?i)memoryerror|out of memory", "Server kehabisan memori saat memproses dokumen."),
    (r"(?i)permission|access is denied|errno 13", "Server tidak memiliki izin mengakses berkas."),
    (r"(?i)no such file|errno 2|filenotfound", "Berkas dokumen tidak ditemukan di server."),
    (r"(?i)timeout|timed out", "Proses memakan waktu terlalu lama (timeout)."),
    (r"(?i)connection|network|resolve|urlopen|http error|ssl", "Gagal mengunduh dokumen (masalah koneksi/link)."),
    (r"(?i)homography|findhomography", "Gagal menyelaraskan dokumen dengan template (kualitas scan/foto kurang baik)."),
    (r"(?i)paddle|ocr", "Mesin OCR gagal membaca dokumen."),
]


def sanitize_error(exc):
    """Pesan aman utk ditampilkan ke user: tanpa path server, cukup spesifik
    (disertai nama exception sbg kode pelacakan), tidak generik total."""
    code = type(exc).__name__
    cleaned = _PATH_RE.sub("<berkas>", str(exc) or "")
    for pattern, hint in _ERROR_HINTS:
        if re.search(pattern, cleaned):
            return f"{hint} [{code}]"
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 140:
        cleaned = cleaned[:140] + "…"
    return f"Gagal memproses dokumen [{code}]" + (f": {cleaned}" if cleaned else "")


def _decision_to_export_text(decision):
    """Render decision_v9_2 (urban ATAU rural, lihat comparison.compute_decision)
    jadi teks "Final Status"/"Notes" siap Excel -- format SAMA dgn contoh
    pada spesifikasi ('OK Lanjut Proses' / 'Tolak – <alasan>'). Tidak
    menjalankan OCR/decision baru, murni format ulang hasil yang SUDAH ada
    di sesi (session['results'][no]['decision_v9_2'])."""
    if not decision:
        return "Belum Diproses", ""
    notes_text = "; ".join(n for n in (decision.get("notes") or []) if n)
    if str(decision.get("decision", "")).upper() == "OK":
        return "OK Lanjut Proses", notes_text
    reasons = "; ".join(r for r in (decision.get("reasons") or []) if r) or "Perlu verifikasi manual"
    return f"Tolak – {reasons}", notes_text


def _set_progress(request_id, **data):
    with _PROGRESS_LOCK:
        current = _PROGRESS.get(request_id, {})
        current.update(data)
        current["updated_at"] = time.time()
        _PROGRESS[request_id] = current


def _get_progress(request_id):
    with _PROGRESS_LOCK:
        value = _PROGRESS.get(request_id)
        return dict(value) if value else None


app = FastAPI(title="HTR Form OCR")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _save_preview_image(image, filename):
    path = OUTPUT_DIR / filename
    cv2.imwrite(str(path), image)
    return f"/outputs/{filename}"


def _ensure_template_roi_image():
    """V15.2: SATU file `template_roi.jpg` yg dipakai bersama utk SEMUA
    record (bukan per-record spt `_roi_template.jpg` versi lama, sudah
    dihapus V15) -- layout referensi (posisi+nama field/choice/tanda tangan
    dari template.pdf, TANPA status/warna per-dokumen krn ini bukan hasil
    ekstraksi dokumen manapun). Cache-checked spt `template_preview.jpg` yg
    sudah ada -- generate sekali, regenerate hanya kalau file-nya dihapus."""
    cache_path = OUTPUT_DIR / "template_roi.jpg"
    if cache_path.exists():
        return f"/outputs/{cache_path.name}"
    template_img = pipeline.load_document(pipeline.TEMPLATE_PATH)
    shape = template_img.shape
    synthetic = {name: {"roi_bbox": prep.norm_bbox_to_px(cfg["value_bbox"], shape)}
                 for name, cfg in prep.FIELD_CONFIG.items()}
    for group in prep.CHOICE_GROUPS.values():
        for opt_name, opt in group["options"].items():
            synthetic[opt_name] = {"roi_bbox": prep.norm_bbox_to_px(opt["bbox"], shape)}
    for name, cfg in prep.SIGNATURE_CONFIG.items():
        synthetic[name] = {"roi_bbox": prep.norm_bbox_to_px(cfg["value_bbox"], shape)}
    img = post.draw_debug(template_img, synthetic, template_mode=True)
    return _save_preview_image(img, cache_path.name)


def _run_ocr_and_format(request_id, document_path, data_entry_record=None, engine="v18", cancel_event=None):
    """Jalankan extractor penuh untuk satu dokumen -> payload response siap kirim.
    PENTING: pemanggil WAJIB sudah men-set progress awal untuk request_id ini
    SEBELUM memanggil fungsi ini (mis. sebelum proses download dokumen yang
    bisa makan waktu), supaya GET /api/progress/{id} tidak 404 di fase awal.

    V19: `engine` menentukan SATU extractor yang dipakai (lihat extractors.py
    -- "v18" [default, pipeline lama TIDAK berubah], "gemini-3.8-flash",
    "qwen3_vl"). extractors.run() adalah dispatcher
    TUNGGAL -- TIDAK PERNAH menjalankan >1 extractor utk satu dokumen, TIDAK
    PERNAH fallback diam-diam ke extractor lain kalau yang dipilih gagal
    (kegagalan API tetap kegagalan, ditangkap oleh try/except pemanggil yang
    SUDAH ADA -- lihat process_document/_process_one_record).

    V20: `t0`/`processing_time_seconds` below wrap the ENTIRE function body
    (extraction + debug-image save + comparison/decision) with a monotonic
    timer -- purely additive instrumentation for the Live Evaluation feature
    (live_evaluation.py), no change to what's computed or returned otherwise.
    """
    t0 = time.monotonic()

    def progress_callback(payload):
        _set_progress(request_id, status="processing", **payload)

    # V10: reference_record dipakai HANYA di dalam pipeline V18 utk memutuskan
    # kapan VLM mismatch-verification dipicu -- nilainya TIDAK PERNAH
    # dikirim ke prompt VLM (lihat pipeline._run_vlm_fallback). Argumen ini
    # OPSIONAL (default None) supaya panggilan lama tetap valid. V19: engine
    # Gemini TIDAK PERNAH menerima reference_record sama sekali (lihat
    # extractors.run_gemini -- parameter ini bahkan tidak ada di
    # signature-nya) -- kebocoran data referensi scr struktural tidak
    # mungkin terjadi utk engine itu.
    result = extractors.run(
        engine, str(document_path), progress_callback=progress_callback, reference_record=data_entry_record,
        cancel_event=cancel_event,
    )

    _set_progress(request_id, status="processing", step="save_debug", percent=98, message="Menyimpan gambar debug ROI")
    # V15.2: struktur output final -- template_roi.jpg SATU file bersama
    # (bukan per-record), _original/_preprocessed utk audit kualitas
    # preprocessing (row 1), _roi_document utk ROI final row 2). TIDAK ADA
    # lagi per-record template overlay (`_roi_template.jpg`, dihapus V15) &
    # TIDAK ADA gambar debug tambahan lain.
    template_roi_url = _ensure_template_roi_image()
    original_url = _save_preview_image(result["debug_images"]["original"], f"{request_id}_original.jpg")
    preprocessed_url = _save_preview_image(result["debug_images"]["preprocessed"], f"{request_id}_preprocessed.jpg")
    roi_document_url = _save_preview_image(result["debug_images"]["document"], f"{request_id}_roi_document.jpg")

    fields = result["fields"]
    ocr_meta = result.get("ocr_meta") or {}
    # fallback_source: "Qwen3-VL Hosted" for engine="qwen3_vl" (see
    # extractors.run_qwen3_vl_hosted -- this engine never silently falls back
    # to OCR/ROI, so this is the only value it ever reports). Absent (None)
    # for every other engine.
    fallback_source = ocr_meta.get("fallback_source")
    decision_v9_2 = None
    if data_entry_record:
        fields = comparison.attach_data_entry(fields, data_entry_record)
        # V9.2 deterministic OK/TOLAK/REVIEW (5 decisive fields, dual
        # evidence for tenor/reward) utk mode urban, ATAU V11 rural (5 field
        # beda: nama/nomor_rekening/nominal/TTD Nasabah/TTD BRI, tenor/reward
        # jadi 'notes' non-decisive) -- dipilih otomatis dari kolom
        # Urban/Rural di data_entry_record (lihat comparison.compute_decision).
        decision_v9_2 = comparison.compute_decision(
            fields, result["raw_results"], result["choice_groups"], data_entry_record)
        # V19f: SATU badge "Extractor: <label>" per record (dari
        # extractors.ENGINE_LABELS -- SATU-SATUNYA sumber label, tidak ada
        # mapping baru di sini/frontend), menggantikan note lama "Engine
        # Fallback: X" -- framing "fallback" salah utk qwen3_vl_local
        # (bukan tahap fallback pipeline v18, itu MEMANG extractor yang
        # dipilih utk seluruh record). "ROI Bermasalah"/"OCR Kurang Yakin"/
        # "Fallback OCR Digunakan" (_diagnostic_notes) HANYA relevan utk
        # pipeline OCR/ROI v18 -- disembunyikan utk engine lain supaya tidak
        # menyesatkan (mis. status "review" dari VLM disalahartikan sbg
        # "ROI Bermasalah").
        decision_v9_2 = dict(decision_v9_2)
        notes = list(decision_v9_2.get("notes") or [])
        if engine != "v18":
            notes = [n for n in notes if n not in
                     ("ROI Bermasalah", "OCR Kurang Yakin", "Fallback OCR Digunakan")]
        notes.append(f"Extractor: {extractors.ENGINE_LABELS.get(engine, engine)}")
        # V20: qwen3_vl/qwen3_vl_local flag a distinct "collapsed" state
        # (extractors._is_extraction_collapsed) when EVERY core field came
        # back not_detected in one otherwise well-formed response -- a
        # reproducible model failure mode found on ~24% of a real 25-doc
        # sample, NOT the same thing as a genuinely blank/unreadable
        # document. Surfaced as its own badge (purely informational, no
        # auto-retry/escalation per explicit user decision) so it isn't
        # silently indistinguishable from an ordinary "field unreadable"
        # result in the notes/decision reasons above.
        if ocr_meta.get("extraction_collapsed"):
            notes.append("Extraction Collapsed")
        decision_v9_2["notes"] = notes
    # final_status: dipakai sbg fallback utk Tab 1 (upload manual tanpa
    # pembanding), tetap dihitung selalu.
    final_status = comparison.compute_final_status(fields)

    _set_progress(request_id, status="done", step="done", percent=100, message="Selesai diproses")
    processing_time_seconds = round(time.monotonic() - t0, 3)
    return {
        "request_id": request_id,
        "processing_time_seconds": processing_time_seconds,
        "alignment": result["alignment"],
        "fields": fields,
        "groups": result["groups"],
        "field_errors": result["field_errors"],
        "raw_json": result["raw_results"],
        "template_roi_url": template_roi_url,
        "original_url": original_url,
        "preprocessed_url": preprocessed_url,
        "roi_document_url": roi_document_url,
        "final_status": final_status,
        "decision_v9_2": decision_v9_2,
        "engine": engine,
        "engine_label": extractors.ENGINE_LABELS.get(engine, engine),
        "ocr_meta": ocr_meta,
        "fallback_source": fallback_source,
        # V19 sec 7-9 -- present ONLY for records that actually called
        # Gemini (extractors.run_gemini sets this; every other engine's
        # result dict has no "gemini_usage" key at all, so this is None).
        "gemini_usage": result.get("gemini_usage"),
    }


@app.get("/api/template-preview")
def template_preview():
    cache_filename = "template_preview.jpg"
    cache_path = OUTPUT_DIR / cache_filename
    try:
        if not cache_path.exists():
            template_img = pipeline.load_document(pipeline.TEMPLATE_PATH)
            _save_preview_image(template_img, cache_filename)
        return {"preview_url": f"/outputs/{cache_filename}"}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": "Gagal memuat template.", "detail": str(exc)})


@app.get("/api/progress/{request_id}")
def process_progress(request_id: str):
    progress = _get_progress(request_id)
    if progress is None:
        return JSONResponse(status_code=404, content={"error": "Progress belum tersedia."})
    return progress


# ============================================================================
# TAB 1: PDF / Image
# ============================================================================


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """STEP 1: simpan file, langsung kembalikan preview (belum OCR)."""
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=400, content={"error": f"Tipe file tidak didukung: {extension}."})

    request_id = uuid.uuid4().hex[:12]
    saved_path = UPLOAD_DIR / f"{request_id}{extension}"
    _set_progress(request_id, status="uploading", step="upload", percent=2, message="Menyimpan file upload")

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        preview_img = pipeline.load_document(str(saved_path))
        preview_url = _save_preview_image(preview_img, f"{request_id}_preview.jpg")
        _set_progress(request_id, status="ready", step="uploaded", percent=7, message="Upload selesai")
        return {"request_id": request_id, "preview_url": preview_url}
    except Exception as exc:
        detail = sanitize_error(exc)
        _set_progress(request_id, status="error", step="upload_error", percent=0, message="Gagal membaca file", detail=detail)
        return JSONResponse(status_code=500, content={"error": "Gagal membaca file.", "detail": detail})


class ProcessPayload(BaseModel):
    engine: str = ""  # V19 -- WAJIB salah satu extractors.ENGINES, lihat validasi di bawah


@app.post("/api/process/{request_id}")
def process_document(request_id: str, payload: ProcessPayload):
    """STEP 2: jalankan extractor penuh untuk file yang sudah diupload.
    Progress untuk request_id ini sudah di-set sejak /api/upload, jadi aman.
    V19: `payload.engine` WAJIB diisi salah satu extractors.ENGINES -- frontend
    HARUS mencegah tombol Run aktif sebelum extractor dipilih (lihat
    static/index.html), tapi divalidasi lagi di sini sbg jaring pengaman
    server-side (mis. panggilan API langsung tanpa lewat UI)."""
    if payload.engine not in extractors.ENGINES:
        return JSONResponse(status_code=400, content={"error": "Pilih extractor terlebih dahulu."})
    matches = list(UPLOAD_DIR.glob(f"{request_id}.*"))
    if not matches:
        return JSONResponse(status_code=404, content={"error": "File tidak ditemukan. Upload ulang."})
    try:
        _set_progress(request_id, status="processing", step="start", percent=8, message="Memulai pipeline OCR")
        return _run_ocr_and_format(request_id, matches[0], engine=payload.engine)
    except Exception as exc:
        traceback.print_exc()  # detail lengkap cukup di log server, TIDAK dikirim ke client
        detail = sanitize_error(exc)
        _set_progress(request_id, status="error", step="error", message="Pipeline gagal", detail=detail)
        return JSONResponse(status_code=500, content={"error": "Gagal memproses dokumen.", "detail": detail})


# ============================================================================
# TAB 2: Excel / Google Spreadsheet
# ============================================================================


class SheetUrlPayload(BaseModel):
    url: str


class SheetSubmitPayload(BaseModel):
    selector: str = ""  # mis. "1,3,5-8". Kosong = semua record.
    engine: str = ""    # V19 -- WAJIB salah satu extractors.ENGINES, dipakai utk SELURUH batch


def _new_session(records):
    session_id = uuid.uuid4().hex[:12]
    with _SESSIONS_LOCK:
        _SESSIONS[session_id] = {
            "records": records,
            "status": {r["Record"]: "pending" for r in records},
            "results": {},
            "errors": {},
            "final_status": {},
            "cancel_requested": False,  # V15: lihat /api/sheet/cancel + _process_batch
            # V19f: threading.Event, TERPISAH dari cancel_requested (bool) --
            # cancel_requested dicek HANYA di antar-record (_process_batch),
            # cancel_event diteruskan ke pipeline.run_pipeline/extractors.run
            # supaya record yang SEDANG berjalan bisa diinterupsi di antara
            # tahap-tahap mahal (lihat _process_one_record/cancel_sheet_batch).
            "cancel_event": threading.Event(),
            "engine": None,  # V19 -- di-set saat /api/sheet/submit, SATU extractor utk seluruh batch
        }
    return session_id


@app.post("/api/sheet/upload")
async def upload_sheet(file: UploadFile = File(...)):
    """Upload file Excel -> daftar record (belum di-OCR)."""
    saved_path = UPLOAD_DIR / f"sheet_{uuid.uuid4().hex[:8]}_{file.filename}"
    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        records = data_input.read_excel_records(str(saved_path))
        session_id = _new_session(records)
        return {"session_id": session_id, "records": records}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": "Gagal membaca Excel.", "detail": str(exc)})


@app.post("/api/sheet/load-url")
def load_sheet_url(payload: SheetUrlPayload):
    """Baca Google Spreadsheet publik lewat link -> daftar record."""
    try:
        records = data_input.read_google_sheet_records(payload.url)
        session_id = _new_session(records)
        return {"session_id": session_id, "records": records}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": "Gagal membaca Google Spreadsheet.", "detail": str(exc)})


def _process_one_record(session_id, record):
    """Proses satu record: download -> OCR -> simpan hasil ke sesi. Exception
    di sini TIDAK menghentikan record lain (dipanggil di dalam loop try/except)."""
    record_no = record["Record"]
    request_id = f"{session_id}_{record_no}"
    session = _SESSIONS[session_id]

    # Set progress SEBELUM proses download (termasuk yang lama) supaya
    # /api/progress/{request_id} tidak pernah 404 sejak record mulai diproses.
    _set_progress(request_id, status="processing", step="queued", percent=1, message="Menunggu diproses")
    session["status"][record_no] = "processing"

    try:
        document_path = data_input.get_document_path(record)
        if not document_path:
            session["status"][record_no] = "no_document"
            session["final_status"][record_no] = "ERROR"
            session["errors"][record_no] = "Dokumen tidak tersedia (link Drive kosong/gagal diunduh)."
            _set_progress(request_id, status="error", step="no_document", percent=0, message=session["errors"][record_no])
            return

        result = _run_ocr_and_format(
            request_id, document_path, data_entry_record=record, engine=session.get("engine") or "v18",
            cancel_event=session.get("cancel_event"),
        )
        # V15.2: preview_url dulu di-load+simpan TERPISAH (`_preview.jpg`) di
        # sini, SEBELUM _run_ocr_and_format -- duplikat kerja krn pipeline.
        # run_pipeline() di dalamnya SUDAH memuat & mengembalikan dokumen
        # mentah yg SAMA (`original_url`, lihat Section A output structure).
        # Alias saja, jangan load+simpan dua kali.
        result["preview_url"] = result.get("original_url")
        result["record"] = record

        session["results"][record_no] = result
        session["status"][record_no] = "done"
        session["final_status"][record_no] = result.get("final_status") or "PASSED"
    except pipeline.PipelineCancelled:
        # V19f: interupsi di TENGAH record (bukan error) -- lihat
        # cancel_sheet_batch/_new_session's cancel_event. TIDAK dicatat ke
        # session["errors"] (bukan kegagalan), progress ditandai final di sini
        # supaya polling frontend berhenti menunggu record ini.
        session["status"][record_no] = "cancelled"
        _set_progress(request_id, status="cancelled", step="cancelled", percent=0, message="Dibatalkan")
    except Exception as exc:
        traceback.print_exc()  # detail lengkap cukup di log server, TIDAK dikirim ke client
        detail = sanitize_error(exc)
        session["status"][record_no] = "error"
        session["final_status"][record_no] = "ERROR"
        session["errors"][record_no] = detail
        _set_progress(request_id, status="error", step="error", message="Pipeline gagal", detail=detail)


def _process_batch(session_id, indices):
    session = _SESSIONS[session_id]
    records_by_no = {r["Record"]: r for r in session["records"]}
    for record_no in indices:
        # Cek SEBELUM memulai record berikutnya -- record yg BELUM dimulai
        # dibatalkan di sini. Record yg SUDAH berjalan (dipanggil di iterasi
        # sebelumnya) diinterupsi SENDIRI oleh cancel_event di dalam
        # _process_one_record/extractors.run/pipeline.run_pipeline (V19f --
        # lihat /api/sheet/cancel), TIDAK menunggu sampai selesai wajar lagi.
        if session.get("cancel_requested"):
            for remaining_no in indices[indices.index(record_no):]:
                if session["status"].get(remaining_no) == "queued":
                    session["status"][remaining_no] = "cancelled"
            break
        record = records_by_no.get(record_no)
        if record is None:
            continue
        _process_one_record(session_id, record)  # error 1 record tidak menghentikan loop
    session["cancel_requested"] = False  # reset -- batch ini selesai (habis atau dibatalkan)
    cancel_event = session.get("cancel_event")
    if cancel_event is not None:
        cancel_event.clear()  # V19f: reset supaya submit berikutnya tidak langsung ter-cancel


@app.post("/api/sheet/submit/{session_id}")
def submit_sheet_batch(session_id: str, payload: SheetSubmitPayload):
    """TRIGGER: proses index/range record terpilih di background thread.
    Response langsung balik tanpa menunggu OCR selesai (async)."""
    session = _SESSIONS.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Sesi spreadsheet tidak ditemukan. Muat ulang."})

    # V19 -- SATU extractor utk SELURUH batch (individual/index-range/bulk
    # SEMUA lewat jalur ini, lihat runRecords() di static/index.html) --
    # TIDAK PERNAH per-record berbeda dalam satu submit.
    if payload.engine not in extractors.ENGINES:
        return JSONResponse(status_code=400, content={"error": "Pilih extractor terlebih dahulu."})

    valid_indices = [r["Record"] for r in session["records"]]
    indices = data_input.parse_record_selector(payload.selector, valid_indices)
    if not indices:
        return JSONResponse(status_code=400, content={"error": "Tidak ada record valid pada index/range yang diberikan."})

    session["engine"] = payload.engine
    session["cancel_requested"] = False  # run baru -- pastikan tidak langsung kebawa cancel dari run sebelumnya
    cancel_event = session.get("cancel_event")
    if cancel_event is not None:
        cancel_event.clear()  # V19f: run baru -- pastikan tidak langsung kebawa cancel_event dari run sebelumnya
    for record_no in indices:
        session["status"][record_no] = "queued"
        session["final_status"].pop(record_no, None)  # rerun -> hapus final_status lama

    threading.Thread(target=_process_batch, args=(session_id, indices), daemon=True).start()
    return {"session_id": session_id, "queued": indices, "total": len(indices)}


@app.post("/api/sheet/cancel/{session_id}")
def cancel_sheet_batch(session_id: str):
    """Hentikan SEGERA: record yang MASIH queued (belum mulai diproses) DAN
    record yang SEDANG berjalan saat ini (V19f -- lihat cancel_event, dicek
    di pipeline.run_pipeline/extractors.py's engine functions di antara
    tahap-tahap mahal, TIDAK menunggu record itu selesai wajar). Hasil yang
    sudah selesai TETAP tersimpan."""
    session = _SESSIONS.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Sesi spreadsheet tidak ditemukan. Muat ulang."})
    session["cancel_requested"] = True
    cancel_event = session.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    return {"session_id": session_id, "cancel_requested": True}


def _compute_session_gemini_summary(session):
    """V19 sec 9 -- session/batch totals, counted ONLY over records that
    ACTUALLY called Gemini (result["gemini_usage"] present -- see
    extractors.run_gemini). v18/Mistral/Qwen records (including Qwen's own
    fallback to v18) never carry "gemini_usage", so they never inflate this
    -- "if Gemini is only used as fallback, count only the actual Gemini
    call" falls out naturally since there IS no Gemini fallback path here."""
    documents = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
              "total_tokens": 0, "estimated_cost_usd": 0.0, "estimated_cost_idr": 0.0}
    for result in session["results"].values():
        usage = (result or {}).get("gemini_usage")
        if not usage:
            continue
        documents += 1
        for key in totals:
            totals[key] += usage.get(key) or 0
    return {
        "documents": documents,
        **totals,
        "average_cost_usd_per_document": round(totals["estimated_cost_usd"] / documents, 6) if documents else 0.0,
        "average_cost_idr_per_document": round(totals["estimated_cost_idr"] / documents, 2) if documents else 0.0,
    }


@app.get("/api/sheet/batch-status/{session_id}")
def sheet_batch_status(session_id: str):
    """Polling ringan: status + progress (persen & keterangan) + final_status
    tiap record, dipakai frontend utk MENGISI SATU kolom Status (progress bar
    + persen + teks) tanpa perlu polling /api/progress terpisah per baris."""
    session = _SESSIONS.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Sesi spreadsheet tidak ditemukan."})

    # PENTING: sebelumnya progress HANYA diambil kalau status masih
    # processing/queued -- begitu status berubah jadi "done" tepat sebelum
    # polling berikutnya, update progress TERAKHIR (percent=100) tidak
    # pernah ikut terkirim, sehingga UI nyangkut di persen sebelumnya
    # (mis. 91%/98%) walau proses SUDAH selesai. Ambil progress utk semua
    # status selain "pending" (belum pernah diproses) supaya nilai akhir
    # (100% utk done, atau detail gagal) selalu ikut terbaca.
    progress = {}
    for record_no, status in session["status"].items():
        if status == "pending":
            continue
        p = _get_progress(f"{session_id}_{record_no}")
        if p:
            progress[record_no] = {"percent": p.get("percent", 0), "message": p.get("message", "")}

    # V15: "active" = masih ada record queued/processing -- dipakai frontend
    # utk tahu kapan tombol Stop harus tampil/aktif & kapan harus reset lagi
    # (lihat cancel_sheet_batch). cancel_requested dilaporkan juga supaya
    # frontend bisa menonaktifkan tombol Stop segera setelah diklik, sebelum
    # loop background sempat memproses pembatalan pada iterasi berikutnya.
    active = any(s in ("queued", "processing") for s in session["status"].values())
    return {
        "session_id": session_id,
        "status": session["status"],
        "errors": session["errors"],
        "final_status": session["final_status"],
        "progress": progress,
        "done_count": sum(1 for s in session["status"].values() if s == "done"),
        "total": len(session["records"]),
        "active": active,
        "cancel_requested": session.get("cancel_requested", False),
        "engine": session.get("engine"),  # V19 -- extractor sedang/terakhir dipakai utk sesi ini
        "engine_label": extractors.ENGINE_LABELS.get(session.get("engine"), session.get("engine")),
        "gemini_usage_summary": _compute_session_gemini_summary(session),  # V19 sec 9
    }


@app.get("/api/sheet/result/{session_id}/{record_no}")
def sheet_result(session_id: str, record_no: int):
    """FETCH hasil yang SUDAH diproses (tidak memproses ulang). 404 kalau
    belum ada hasil -- frontend harus submit dulu lewat /api/sheet/submit."""
    session = _SESSIONS.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Sesi spreadsheet tidak ditemukan."})

    result = session["results"].get(record_no)
    if result is not None:
        return result

    status = session["status"].get(record_no, "unknown")
    error = session["errors"].get(record_no)
    return JSONResponse(
        status_code=404,
        content={"error": f"Hasil belum tersedia (status: {status}).", "status": status, "detail": error},
    )


@app.get("/api/sheet/evaluation/{session_id}")
def sheet_evaluation(session_id: str):
    """V20 -- Live Evaluation modal. Pure read/aggregation over ALREADY-
    processed results (session["results"], the same latest-per-record store
    'result' fetches read from) -- NEVER triggers OCR/processing, safe to
    call repeatedly. 404 shape matches sheet_batch_status/sheet_result above
    so the frontend's existing error-message handling works unmodified."""
    session = _SESSIONS.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Sesi spreadsheet tidak ditemukan."})
    return live_evaluation.build_session_evaluation(session)


@app.get("/api/sheet/export/{session_id}")
def export_sheet_excel(session_id: str):
    """Export tabel record (kolom asli + Status Proses + Final Status +
    Notes) ke .xlsx -- HANYA membaca hasil yang SUDAH tersimpan di sesi
    (session['results']/['status']), TIDAK menjalankan OCR/decision ulang."""
    session = _SESSIONS.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Sesi spreadsheet tidak ditemukan."})

    rows = []
    for record in session["records"]:
        record_no = record["Record"]
        result = session["results"].get(record_no) or {}
        decision = result.get("decision_v9_2")
        final_status_text, notes_text = _decision_to_export_text(decision)
        row = dict(record)
        row["Status Proses"] = session["status"].get(record_no, "pending")
        row["Final Status"] = final_status_text
        row["Notes"] = notes_text
        # V19 sec 10 -- engine/model/token usage/cost/fallback columns.
        # "model" = the engine/model that ACTUALLY produced the data (from
        # ocr_meta, e.g. "v18" when Qwen fell back), which can differ from
        # "engine" = the engine the user SELECTED for this batch. Blank/zero
        # Gemini usage for non-Gemini records (nothing to show).
        ocr_meta = result.get("ocr_meta") or {}
        usage = result.get("gemini_usage") or {}
        row["engine"] = result.get("engine") or session.get("engine") or ""
        row["model"] = ocr_meta.get("engine") or ""
        row["input_tokens"] = usage.get("input_tokens", 0)
        row["output_tokens"] = usage.get("output_tokens", 0)
        row["thinking_tokens"] = usage.get("thinking_tokens", 0)
        row["total_tokens"] = usage.get("total_tokens", 0)
        row["cost_usd"] = usage.get("estimated_cost_usd", 0.0)
        row["cost_idr"] = usage.get("estimated_cost_idr", 0.0)
        row["fallback_source"] = result.get("fallback_source") or ""
        rows.append(row)

    try:
        df = pd.DataFrame(rows)
        out_path = OUTPUT_DIR / f"export_{session_id}.xlsx"
        df.to_excel(out_path, index=False)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": "Gagal membuat file Excel.", "detail": str(exc)})

    return FileResponse(
        out_path,
        filename=f"hasil_verifikasi_{session_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
