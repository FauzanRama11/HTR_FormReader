"""Standalone, one-off local-Qwen3-VL-2B (4-bit, GPU) evaluation across all 25
records in assets/ocr_evaluation.xlsx. NOT wired into extractors.py/evaluation.py
-- this is purely a quality check for the local GPU path set up this session
(see handover.md), reusing vlm.extract_fields_fullpage (the existing V12 local
contract) as-is. Writes one row per record to local_qwen_eval_results.csv as it
goes (so a crash/OOM partway doesn't lose prior progress)."""
import csv
import time

import pandas as pd

import data_input as di
import preprocessing as prep
import vlm

FIELDS = ["nama_nasabah", "nomor_rekening", "nominal_penempatan", "tenor", "bentuk_reward",
          "signature_nasabah", "signature_unit_kerja"]
OUT_CSV = "eval_runs/local_qwen_eval_results.csv"

df = pd.read_excel("assets/ocr_evaluation.xlsx").dropna(subset=["Surat Pernyataan"])
print(f"{len(df)} records to process. Loading model once...")

t0 = time.time()
vlm._load()
print(f"model loaded in {time.time() - t0:.1f}s")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["record", "file_id", "elapsed_s", "error",
                      "gt_nama", "gt_rekening", "gt_nominal", "gt_tenor", "gt_reward",
                      "ex_nama", "ex_rekening", "ex_nominal", "ex_tenor", "ex_reward",
                      "ex_sig_nasabah", "ex_sig_unit_kerja", "raw_text"])
    f.flush()

    for i, row in df.iterrows():
        record_num = i + 1
        try:
            doc_path = di.get_document_path(row)
        except Exception as exc:
            writer.writerow([record_num, None, 0, f"download_failed: {exc}"] + [""] * 12)
            f.flush()
            print(f"[{record_num}/{len(df)}] download failed: {exc}")
            continue

        try:
            image_bgr = prep.load_document(doc_path)  # handles PDF rendering too, unlike bare cv2.imread
        except Exception:
            image_bgr = None
        if image_bgr is None:
            writer.writerow([record_num, doc_path, 0, "imread_failed"] + [""] * 12)
            f.flush()
            print(f"[{record_num}/{len(df)}] imread failed: {doc_path}")
            continue

        t1 = time.time()
        try:
            clean, raw_text = vlm.extract_fields_fullpage(image_bgr, FIELDS)
            elapsed = time.time() - t1
            row_out = [
                record_num, doc_path, round(elapsed, 1), "",
                row.get("Nama Nasabah"), row.get("Nomor Rekening"),
                row.get("Nominal Penempatan"), row.get("Tenor Hold"), row.get("Pilihan Hadiah"),
                clean.get("nama_nasabah", {}).get("value"),
                clean.get("nomor_rekening", {}).get("value"),
                clean.get("nominal_penempatan", {}).get("value"),
                clean.get("tenor", {}).get("value"),
                clean.get("bentuk_reward", {}).get("value"),
                clean.get("signature_nasabah", {}).get("value"),
                clean.get("signature_unit_kerja", {}).get("value"),
                raw_text.replace("\n", " ")[:400],
            ]
            print(f"[{record_num}/{len(df)}] OK in {elapsed:.1f}s -- "
                  f"nama={clean.get('nama_nasabah', {}).get('value')!r} "
                  f"(GT={row.get('Nama Nasabah')!r})")
        except Exception as exc:
            elapsed = time.time() - t1
            row_out = [record_num, doc_path, round(elapsed, 1), str(exc)] + [
                row.get("Nama Nasabah"), row.get("Nomor Rekening"),
                row.get("Nominal Penempatan"), row.get("Tenor Hold"), row.get("Pilihan Hadiah"),
            ] + [""] * 7
            print(f"[{record_num}/{len(df)}] FAILED after {elapsed:.1f}s: {exc}")
        writer.writerow(row_out)
        f.flush()

print("DONE. Results in", OUT_CSV)
