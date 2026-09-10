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

import comparison
import data_input
import pipeline
import preprocessing as prep
import postprocessing as post

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
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


def _run_ocr_and_format(request_id, document_path, data_entry_record=None):
    """Jalankan pipeline penuh untuk satu dokumen -> payload response siap kirim.
    PENTING: pemanggil WAJIB sudah men-set progress awal untuk request_id ini
    SEBELUM memanggil fungsi ini (mis. sebelum proses download dokumen yang
    bisa makan waktu), supaya GET /api/progress/{id} tidak 404 di fase awal."""

    def progress_callback(payload):
        _set_progress(request_id, status="processing", **payload)

    # V10: reference_record dipakai HANYA di dalam pipeline utk memutuskan
    # kapan VLM mismatch-verification dipicu -- nilainya TIDAK PERNAH
    # dikirim ke prompt VLM (lihat pipeline._run_vlm_fallback). Argumen ini
    # OPSIONAL (default None) supaya panggilan lama tetap valid.
    result = pipeline.run_pipeline(
        str(document_path), progress_callback=progress_callback, reference_record=data_entry_record
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
    # final_status: dipakai sbg fallback utk Tab 1 (upload manual tanpa
    # pembanding), tetap dihitung selalu.
    final_status = comparison.compute_final_status(fields)

    _set_progress(request_id, status="done", step="done", percent=100, message="Selesai diproses")
    return {
        "request_id": request_id,
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
        "ocr_meta": result.get("ocr_meta"),
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


@app.post("/api/process/{request_id}")
def process_document(request_id: str):
    """STEP 2: jalankan pipeline OCR penuh untuk file yang sudah diupload.
    Progress untuk request_id ini sudah di-set sejak /api/upload, jadi aman."""
    matches = list(UPLOAD_DIR.glob(f"{request_id}.*"))
    if not matches:
        return JSONResponse(status_code=404, content={"error": "File tidak ditemukan. Upload ulang."})
    try:
        _set_progress(request_id, status="processing", step="start", percent=8, message="Memulai pipeline OCR")
        return _run_ocr_and_format(request_id, matches[0])
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

        result = _run_ocr_and_format(request_id, document_path, data_entry_record=record)
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
        # V15: cek SEBELUM memulai record berikutnya -- record yg SUDAH
        # berjalan (dipanggil di iterasi sebelumnya) selalu selesai wajar,
        # TIDAK diinterupsi di tengah proses. Hanya record yg BELUM dimulai
        # yang dibatalkan (lihat /api/sheet/cancel).
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


@app.post("/api/sheet/submit/{session_id}")
def submit_sheet_batch(session_id: str, payload: SheetSubmitPayload):
    """TRIGGER: proses index/range record terpilih di background thread.
    Response langsung balik tanpa menunggu OCR selesai (async)."""
    session = _SESSIONS.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Sesi spreadsheet tidak ditemukan. Muat ulang."})

    valid_indices = [r["Record"] for r in session["records"]]
    indices = data_input.parse_record_selector(payload.selector, valid_indices)
    if not indices:
        return JSONResponse(status_code=400, content={"error": "Tidak ada record valid pada index/range yang diberikan."})

    session["cancel_requested"] = False  # run baru -- pastikan tidak langsung kebawa cancel dari run sebelumnya
    for record_no in indices:
        session["status"][record_no] = "queued"
        session["final_status"].pop(record_no, None)  # rerun -> hapus final_status lama

    threading.Thread(target=_process_batch, args=(session_id, indices), daemon=True).start()
    return {"session_id": session_id, "queued": indices, "total": len(indices)}


@app.post("/api/sheet/cancel/{session_id}")
def cancel_sheet_batch(session_id: str):
    """Hentikan record yang MASIH queued (belum mulai diproses) pada batch
    yang sedang berjalan. Record yang sudah mulai diproses tetap diselesaikan
    -- lihat _process_batch. Hasil yang sudah selesai TETAP tersimpan."""
    session = _SESSIONS.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Sesi spreadsheet tidak ditemukan. Muat ulang."})
    session["cancel_requested"] = True
    return {"session_id": session_id, "cancel_requested": True}


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
        result = session["results"].get(record_no)
        decision = (result or {}).get("decision_v9_2")
        final_status_text, notes_text = _decision_to_export_text(decision)
        row = dict(record)
        row["Status Proses"] = session["status"].get(record_no, "pending")
        row["Final Status"] = final_status_text
        row["Notes"] = notes_text
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
