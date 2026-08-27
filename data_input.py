"""
data_input.py
=============
Sumber data untuk Tab "Excel / Google Spreadsheet":
  - baca Excel upload / baca Google Spreadsheet publik (CSV export, tanpa auth)
  - bersihkan record kosong/missing
  - ekstrak file_id dari link Google Drive & download (di-cache, tidak dobel)
"""

import os
import re
from pathlib import Path

import pandas as pd

# Kolom yang dipakai, sesuai array kolom pada spreadsheet_extract.ipynb.
SPREADSHEET_COLUMNS = [
    "Regional Office",
    "Nama Branch Office",
    "Nomor Rekening",
    "Nama Nasabah",
    "Nominal Penempatan",
    "Tenor Hold",
    "Pilihan Hadiah",
    "Surat Pernyataan",
    # V9.2: label keputusan ground truth (evaluation.py saja -- utk Decision
    # Accuracy/FAR/FRR). Kolom opsional: diabaikan otomatis kalau tidak ada
    # di sumber data (lihat _clean_dataframe: `columns = [c for c in
    # SPREADSHEET_COLUMNS if c in df.columns]`), tidak memengaruhi record
    # produksi yang sudah ada.
    "Status Verifikasi Form Pendaftaran Nasabah",
]

MISSING_VALUES = {"", "none", "nan", "null", "n/a", "na", "-", "--"}

# Nama kolom index yang mungkin sudah ada di Excel/Google Sheet asli. Kalau
# salah satu ketemu (case-insensitive), PAKAI nilainya sebagai "Record" --
# TIDAK membuat index baru (1..N) supaya nomor tetap sama seperti di sumber.
INDEX_COLUMN_CANDIDATES = ["no", "no.", "nomor", "no urut", "index", "id", "record"]

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloaded_documents"
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Cache di memori: file_id -> path lokal. Mencegah download dokumen Drive
# yang sama lebih dari sekali dalam satu proses server.
_download_cache = {}


def clean_value(value):
    """Kosongkan value yang secara efektif "kosong" (NaN/none/-/dst)."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING_VALUES else text


def _find_existing_index_column(df):
    """Cari kolom index yang SUDAH ADA di Excel/Sheet asli (mis. 'No', 'Nomor').
    Dikembalikan nama kolom aslinya, atau None kalau tidak ada."""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in INDEX_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _clean_dataframe(df):
    index_col = _find_existing_index_column(df)
    index_values = None
    if index_col is not None:
        # Simpan nilai index ASLI sebelum kolom lain difilter/dibuang barisnya.
        index_values = df[index_col].apply(clean_value)

    columns = [c for c in SPREADSHEET_COLUMNS if c in df.columns]
    df = df[columns].copy()
    if index_values is not None:
        df.insert(0, "__record_index_raw", index_values)

    mapper = df.map if hasattr(df, "map") else df.applymap  # pandas >=2.1 vs lama
    df = mapper(clean_value)
    df = df.dropna(how="all", subset=columns).reset_index(drop=True)

    if "__record_index_raw" in df.columns:
        record_no = df["__record_index_raw"]
        df = df.drop(columns=["__record_index_raw"])

        def _to_int(v, fallback):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return fallback
        df.insert(0, "Record", [
            _to_int(v, i + 1) for i, v in enumerate(record_no)
        ])
    else:
        # Tidak ada kolom index di sumber -> baru dibuat sekuensial (fallback).
        df.insert(0, "Record", range(1, len(df) + 1))
    return df


def read_excel_records(file_path):
    """Baca file Excel upload -> list of dict per record."""
    df = pd.read_excel(file_path, dtype=str).fillna("")
    return _clean_dataframe(df).to_dict(orient="records")


def read_google_sheet_records(sheet_url_or_id, gid="0"):
    """Baca Google Spreadsheet publik lewat CSV export (tanpa autentikasi)."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", sheet_url_or_id)
    sheet_id = match.group(1) if match else sheet_url_or_id.strip()

    gid_match = re.search(r"[?&#]gid=(\d+)", sheet_url_or_id)
    if gid_match:
        gid = gid_match.group(1)

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(url, dtype=str, keep_default_na=False)
    return _clean_dataframe(df).to_dict(orient="records")


def extract_drive_file_id(url):
    """Ambil file_id dari link Google Drive (format /d/<id>/ atau ?id=<id>)."""
    if not url:
        return None
    url = str(url).strip()
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def download_drive_document(file_id):
    """Download dokumen Drive by file_id. Di-cache (memori + disk) supaya
    dokumen yang sama tidak pernah didownload dua kali."""
    if not file_id:
        return None
    if file_id in _download_cache:
        return _download_cache[file_id]

    output_path = DOWNLOAD_DIR / file_id
    if output_path.exists():
        _download_cache[file_id] = str(output_path)
        return str(output_path)

    import gdown
    try:
        path = gdown.download(id=file_id, output=str(output_path), quiet=True)
        if path and os.path.exists(path):
            _download_cache[file_id] = path
            return path
    except Exception:
        return None
    return None


def get_document_path(record):
    """Dari satu record (dict hasil read_*_records), ambil path dokumen
    lokal (download dulu kalau perlu). None kalau tidak ada link/gagal."""
    file_id = extract_drive_file_id(record.get("Surat Pernyataan"))
    return download_drive_document(file_id)


def parse_record_selector(text, valid_indices):
    """Parse input index/range dari user, mis. '1,3,5-8' -> [1,3,5,6,7,8].
    Kosong -> semua record valid. Index di luar valid_indices diabaikan."""
    text = (text or "").strip()
    valid = set(valid_indices)
    if not text:
        return sorted(valid)

    result = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            try:
                a, b = int(start), int(end)
                result.update(range(min(a, b), max(a, b) + 1))
            except ValueError:
                continue
        else:
            try:
                result.add(int(part))
            except ValueError:
                continue
    return sorted(result & valid)
