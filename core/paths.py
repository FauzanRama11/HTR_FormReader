"""
paths.py
========
Satu sumber kebenaran untuk lokasi folder project. Sebelum restructure (lihat
handover.md, bagian restructure di akhir file), tiap modul (app.py,
data_input.py, preprocessing.py, vlm.py, signature_detector.py, evaluation.py,
stage_evaluation.py, compare_runs.py, signature_diagnostics.py) punya baris
sendiri-sendiri `BASE_DIR = Path(__file__).resolve().parent` untuk cari folder
root project -- benar SELAMA file-nya ada di root. Setelah file dipindah ke
folder (core/, pipeline/, extractors/, eval_tools/), `Path(__file__).parent`
sudah tidak lagi menunjuk ke root, jadi tiap modul harus import PROJECT_ROOT
dari sini, bukan hitung sendiri.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
EVAL_RUNS_DIR = PROJECT_ROOT / "eval_runs"
MODELS_DIR = PROJECT_ROOT / "models"
DOWNLOAD_DIR = PROJECT_ROOT / "downloaded_documents"
