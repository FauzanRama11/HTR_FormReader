# HANDOVER — OCR Pipeline Preprocessing & Evaluation

**Last update:** 8 September 2026 (V14)
**Baca urutan:** 1 (deskripsi) → 2 (state) → 3 (next steps) → sisanya kalau perlu detail.
**Prinsip kerja:** MEASURE → FIND ROOT CAUSE → FIX → RE-EVALUATE. **Satu perubahan
kecil, satu pengukuran full 25-dokumen, baru lanjut** — V14 sengaja hanya 1 fix
(bukan sapuan banyak fix sekaligus) atas permintaan eksplisit user, supaya efek tiap
perubahan bisa diverifikasi dgn jelas & token/waktu sesi tidak boros.

## 1. Deskripsi Project

Membaca **Surat Pernyataan nasabah** (foto HP atau PDF hasil scan), mengekstrak field
(nama, no. rekening, nominal, tenor, reward, TTD), membandingkan dengan data referensi
spreadsheet, lalu menghasilkan `OK Lanjut Proses` / `Tolak` (+ alasan) / `Review`.

```
Document -> Preprocessing (align ke template) -> ROI per field -> OCR (PaddleOCR)
         -> [fallback VLM Qwen2-VL kalau OCR tidak yakin] -> Field Comparison -> Final Status
```

Dataset ground truth: `assets/ocr_evaluation.xlsx` (25 record) + `downloaded_documents/`
(25 dokumen ter-cache, JANGAN dihapus — re-download butuh akses Google Drive).

## 2. State Saat Ini (V13)

Root cause 3 bug (2 di produksi, 1 di evaluator) yang membuat evaluasi sebelumnya TIDAK
BISA DIPERCAYA (100% preprocessing gagal, 0% field accuracy terukur — bukan pipeline
buruk, tapi evaluatornya sendiri salah baca). Sudah diperbaiki + diverifikasi dgn
evaluation.py di 25 dokumen:

| Metrik                     | Sebelum (bug)     | Sesudah (fix)      |
|-----------------------------|-------------------|---------------------|
| alignment_success_rate      | 0.96 (1 gagal)    | **1.00** (0 gagal)  |
| pipeline_success_rate       | 0.96              | **1.00**            |
| field_accuracy (5 field)    | N/A semua (bug evaluator, BUKAN 0 sungguhan) | nama_nasabah 20%, nomor_rekening 33%, nominal_penempatan 24%, tenor_penempatan 83%, bentuk_reward 100% |
| decision_accuracy           | 0.08              | **0.12**            |
| avg processing time/doc     | 12.0s             | 10.8s               |

File referensi: `evaluation_summary_baseline_full.json` (sebelum) vs
`evaluation_summary_fixed_v13.json` (sesudah) — hasil mentah per-record ada di
`evaluation_results_*.csv`. JANGAN dihapus, ini bukti before/after.

### Perubahan yang dibuat

1. **`preprocessing.py` — `is_pdf_document()` (baru).** Dokumen hasil download Google
   Drive disimpan TANPA ekstensi (`data_input.download_drive_document`, file = `<file_id>`).
   Kode lama (`prepare_input`, `prepare_and_align`) cek `Path(path).suffix == ".pdf"` —
   SELALU False utk file tanpa ekstensi, jadi 7/25 dokumen yg sebenarnya PDF salah
   dikira foto & diproses lewat jalur rectify/enhance foto yg tidak relevan. Fix: sniff
   isi file (magic header `%PDF-`), reuse `_looks_like_pdf` yg sudah ada. Dipakai jg di
   `stage_evaluation.py` (`evaluate_preprocessing_gate`).

2. **`preprocessing.py` — `_find_document_quad()` (rewrite total, dipakai `rectify_photo`).**
   ROOT CAUSE UTAMA: deteksi kertas versi lama (Canny edge + `approxPolyDP` PERSIS 4
   titik) gagal menemukan boundary kertas pada **0 dari 25** foto — diverifikasi visual
   (lihat bukti di bawah). Tepi kertas di foto kamera nyata punya gradien SANGAT LEMBUT
   (fokus/bayangan/pencahayaan), jauh lebih lemah drpd kontras teks internal — Canny
   selalu menangkap teks dulu, tidak pernah menutup satu loop penuh sekeliling kertas.
   Fix: segmentasi KECERAHAN (Otsu threshold kertas-vs-latar, region-based, bukan
   gradien) sbg strategi utama, Canny adaptif sbg fallback terakhir. Terverifikasi:
   rasio area kertas terdeteksi naik dari ~0.02–0.06 (salah) jadi ~0.65–0.92 (sesuai
   visual). AMAN dipakai agresif krn `prepare_and_align()` cuma pakai hasil rectify
   kalau skor alignment-nya OBJEKTIF LEBIH BAIK drpd dokumen asli — quad yang meleset
   otomatis tidak dipakai.

3. **`evaluation.py` — `_extract_value()` key salah (bug evaluator, BUKAN pipeline).**
   `post.build_fields_table()` (production, `postprocessing.py`) menaruh nilai di key
   `"ocr_result"`. Evaluator lama menebak key `"normalized_value"/"value"/"text"/
   "label"/"final_value"` — TIDAK ADA yang cocok, jadi field_accuracy SELALU 0
   measurable utk SEMUA field/dokumen, terlepas dari akurasi pipeline sungguhan. Fix:
   tambah `"ocr_result"` sbg key prioritas pertama. Dipakai jg oleh `stage_evaluation.py`
   (reuse `ev9._extract_value`).

4. **`evaluation.py` + `stage_evaluation.py` — flag `--pipeline` / `--vlm-fallback`
   (baru).** Supaya bisa jalankan dataset yg SAMA lewat modul pipeline lain
   (`pipeline1.py` = versi V10/11 lama) atau toggle `VLM_FALLBACK_ENABLED` tanpa edit
   kode, lalu dibandingkan lewat `compare_runs.py` (baru) — gabungkan beberapa
   `evaluation_summary_<label>.json` / `stage_evaluation_summary_<label>.json` jadi satu
   tabel sisi-berdampingan (kolom pertama = baseline, kolom lain = delta).
   ```
   python evaluation.py --dataset assets/ocr_evaluation.xlsx --label paddle_only --vlm-fallback off
   python evaluation.py --dataset assets/ocr_evaluation.xlsx --label paddle_vlm  --vlm-fallback on
   python compare_runs.py evaluation_summary_paddle_only.json evaluation_summary_paddle_vlm.json
   ```

## 3. V14 — Fix ROI Bleeding Identity Area (SELESAI, terverifikasi)

**Masalah** (ditemukan dari inspeksi visual debug image `outputs/*_roi_document.jpg`
yg di-generate user via app.py setelah V13): box ROI `nama_nasabah`/`nomor_rekening`/
`unit_kerja_pengelola_rekening` sering overlap/tertukar antar baris. Root cause:
`FIELD_CONFIG` (`preprocessing.py:107-121`) mendefinisikan 3 baris identity_area
dgn **gap NOL** (baris N bawah == baris N+1 atas), dan `estimate_section_transform`
(`preprocessing.py:886-945`, fallback ~1038-1063) jatuh ke `consensus_translation`
(TIDAK bisa koreksi rotasi) kalau anchor section yg lolos outlier-rejection <3 —
baris yg jauh dari anchor drift ke wilayah baris tetangga yg tanpa margin. Safety net
`prep._clamp_field_box_y()` sudah ADA di codebase tapi sebelumnya cuma dipakai utk
`tempat_tanggal_surat`, TIDAK dipakai di jalur `dynamic_extraction.py`.

**Fix**: satu baris di `dynamic_extraction.py` `_resolve_field_boxes()` (~line 190) —
`target_px = prep._clamp_field_box_y(target_px, field_name, shape)` — **HANYA utk
`region_name == "identity_area"`**. Sempat dicoba universal (semua field/region)
tapi terbukti regresi di `placement_area` (lihat §4, dilarang diulang) krn box
`CHOICE_GROUPS` (opsi tenor/reward) overlap scr vertikal dgn baris teks di sana,
beda dari identity_area yg murni 3 baris teks tanpa choice box.

**Hasil terverifikasi** (`evaluation.py`, full 25 dokumen, 1 evaluator per run —
lihat §5 soal RAM):

| Metrik | V13 (`fixed_v13`) | V14 (`roi_clamp_identity_only`) |
|---|---|---|
| pipeline_success_rate | 1.00 | 1.00 |
| alignment_success_rate | 1.00 | 1.00 |
| nama_nasabah accuracy | 0.20 (4/20) | **0.25 (4/16)** |
| nomor_rekening accuracy | 0.3333 (7/21) | **0.4444 (8/18)** |
| nominal_penempatan accuracy | 0.2381 (5/21) | 0.2381 (5/21) — **unchanged, sesuai rencana** (scoped away dari placement_area) |
| tenor_penempatan accuracy | 0.8333 (5/6) | 0.80 (4/5) — noise n-kecil, bukan regresi (versi universal yg regresi ke 0.75 sudah dibuang) |
| bentuk_reward accuracy | 1.00 (10/10) | 1.00 (10/10) |
| decision_accuracy | 0.12 | **0.16** |

File bukti: `evaluation_summary_fixed_v13.json` (before) vs
`evaluation_summary_roi_clamp_identity_only.json` (after, FINAL — dipakai). File
`evaluation_summary_roi_clamp_fix.json` (versi universal yg DIBUANG) disimpan sbg
bukti kenapa scoping ke identity_area diperlukan — JANGAN dipakai sbg baseline.

**Keterbatasan yang masih tersisa (diverifikasi via debug image, BUKAN tebakan):**
Fix ini menghilangkan OVERLAP antar box (record 5: sekarang 3 baris terpisah bersih,
lihat sebelumnya 1 baris tanpa box sama sekali). TAPI untuk dokumen dgn drift lebih
besar (record 1: seluruh blok 3-baris bergeser ~1 baris penuh ke bawah relatif thd
window template), box yg sudah bersih/non-overlap TETAP bisa "kosong" (window benar
scr geometri, tapi kontennya sendiri sudah bukan di situ lagi) — `nomor_rekening`
record 1 jadi `None` (aman, masuk review, BUKAN salah-tapi-percaya-diri spt
sebelumnya), sementara `unit_kerja_pengelola_rekening` malah membaca isi
`nomor_rekening` (field ini TIDAK termasuk 5 field yg dibandingkan evaluation.py,
jadi tidak terlihat di metrik, tapi tercatat di sini utk next step). Ini BUKAN bug
dari fix V14 — ini kasus drift SATU ARAH SELURUH BLOK yg tidak bisa diperbaiki cuma
dgn clamp batas tetangga (clamp mencegah overlap, tidak memperbaiki posisi absolut).
Perbaikan sesungguhnya perlu localisasi PER-BARIS independen (bukan 1 transform
section utk semua baris) — opsi ini sudah dipertimbangkan sblm V14 tapi ditolak sbg
scope-lebih-besar; sekarang ada bukti konkret (record 1) knp itu next step yg valid.

## 4. Prioritas Berikutnya (berdasar data V14 di atas, BUKAN tebakan)

1. **Per-baris localisasi independen utk identity_area** (bukan 1 shared section
   transform) — utk kasus drift-satu-blok spt record 1 di atas. Pendekatan: setelah
   section transform diterapkan, cari ink band lokal di sekitar window tiap baris
   (mirip `_find_ink_bands` yg sudah ada di `extract_handwriting_crop`) utk snap
   posisi tiap baris ke kontennya sendiri, bukan asumsi window template selalu tepat.
2. **field_accuracy nama_nasabah/nomor_rekening masih di bawah 50%** meski sudah naik
   — jalankan `stage_evaluation.py --label v14 --dataset assets/ocr_evaluation.xlsx`
   (belum sempat di-run penuh sesi ini) utk funnel ROI-fail vs OCR-fail per field
   sebelum menebak apakah sisa error ada di ROI atau di model OCR itu sendiri.
3. **decision_accuracy naik ke 0.16 tapi masih rendah**, `review_rate = 0.0` (semua
   record jatuh ke OK/TOLAK tegas) — cek `decision_reasons` di
   `evaluation_results_roi_clamp_identity_only.csv` utk lihat check mana yg paling
   sering gagal (7-check jauh lebih ketat drpd akurasi per-field individual).
4. **Record 20 (`outputs/b392caa0633a_20_roi_document.jpg`)**: seluruh blok
   identity_area landing ~4 baris di bawah field asli, di atas teks paragraf. Beda
   root cause dari #1 (bukan soal row-gap, lebih ke alignment/kemungkinan header
   cabang yg tingginya beda dari template.pdf referensi). BELUM diinvestigasi lebih
   lanjut — jangan diasumsikan sama dgn masalah row-drift di atas.
5. Mesin evaluasi ini hanya ~8GB RAM — **JANGAN** jalankan 2 evaluator penuh (25
   dokumen) bersamaan (sudah 2x trigger OOM/MemoryError session ini). Jalankan
   satu-satu, atau pakai `--limit` utk uji cepat dulu.

## 5. Sudah Dicoba / Jangan Diulang (hemat waktu+token sesi berikutnya)

- Canny edge (`Canny(50,150)` tetap atau adaptif) utk deteksi boundary kertas → SUDAH
  dikonfirmasi gagal total di dataset ini (visual + rasio area), JANGAN kembalikan jadi
  strategi utama. Otsu brightness segmentation (§2, poin 2) adalah fix yg terbukti bekerja.
- Cek suffix file (`Path(path).suffix == ".pdf"`) utk deteksi PDF pada dokumen hasil
  download → SELALU salah utk file tanpa ekstensi. Selalu pakai `prep.is_pdf_document()`.
- Key `"value"/"normalized_value"/"text"/"label"/"final_value"` sbg SUMBER UTAMA baca
  hasil field di evaluator → key asli production adalah `"ocr_result"` (tetap ada sbg
  fallback di `_extract_value`, tapi jangan andalkan itu duluan).
- **Menerapkan `_clamp_field_box_y` scr UNIVERSAL ke semua region** (bukan hanya
  identity_area) → SUDAH dicoba (`evaluation_summary_roi_clamp_fix.json`), TERBUKTI
  regresi di `placement_area` (`rentang_tenor` record 15/18: MATCH→N/A) krn
  `CHOICE_GROUPS` box (opsi tenor/reward) overlap vertikal dgn baris teks di sana.
  JANGAN diulang tanpa dulu memperbaiki bagaimana `PREV_FIELD_BOTTOM_NORM`/
  `NEXT_FIELD_TOP_NORM` menghitung tetangga saat ada choice-box yg overlap.

## 6. Cara Re-run Evaluasi (perintah persis)

```bash
# end-to-end (field accuracy + decision accuracy) -- paling relevan utk "akurasi asli"
python evaluation.py --dataset assets/ocr_evaluation.xlsx --label <nama_run> --debug-dir none

# 3-gate funnel (PREPROCESSING -> ROI -> OCR), CER, dst -- utk isolasi root cause per stage
python stage_evaluation.py --dataset assets/ocr_evaluation.xlsx --label <nama_run> --no-xlsx --debug-dir none

# uji cepat sebelum full run (hemat waktu ~12s/dokumen)
python evaluation.py --dataset assets/ocr_evaluation.xlsx --label smoke --limit 5 --debug-dir none

# bandingkan beberapa run (lihat §2, poin 4)
python compare_runs.py evaluation_summary_<a>.json evaluation_summary_<b>.json
```

## 7. Peta File

**Aktif (produksi, di-import `app.py`/`pipeline.py`):** `preprocessing.py`, `ocr.py`,
`postprocessing.py`, `dynamic_extraction.py`, `comparison.py`, `vlm.py`, `pipeline.py`,
`data_input.py`, `app.py`, `static/index.html`.

**Evaluasi (standalone CLI, tidak di-import app.py):** `evaluation.py`,
`stage_evaluation.py`, `compare_runs.py` (V13).

**Eksperimen/versi lama, TIDAK di-import mana pun secara aktif (verifikasi ke user
sebelum dihapus — mungkin masih dipakai manual utk banding V10/11 vs V12 lewat
`--pipeline pipeline1`):** `pipeline1.py`/`pipeline2.py`, `comparison1.py`/`comparison2.py`,
`postprocessing1.py`, `vlm1.py`, `index1.html`, `static/index1.html`,
`static/serep2.html`/`serep3.html`/`serep4.html`/`serep6.html`.

## 8. Field & Rules (tidak berubah dari versi sebelumnya)

Field: Nama (fuzzy match 80%), Nomor Rekening, Unit Kerja, Nominal Penempatan, Tenor
(dari rentang tanggal, <1 bulan = tidak sesuai), Bentuk Reward (tunai/non-tunai +
fallback), TTD Nasabah & TTD Unit Kerja/BRI (presence only, bukan baca isi).

Rule rural (`Urban` != "urban" di data): cukup validasi Nama, Nomor Rekening, Nominal
Penempatan, TTD Nasabah, TTD BRI — kalau semua lolos, `OK Lanjut Proses`.

Environment: Windows, FastAPI `127.0.0.1:8001`, GPU NVIDIA MX230 2GB (PaddleOCR
mayoritas jalan CPU di environment evaluasi), `MAX_CANVAS_HEIGHT_GPU=2200` /
`_CPU=6000`, `text_recognition_batch_size=1` (low VRAM GPU kecil).

## 9. Instruksi utk Claude (sesi berikutnya)

1. Baca §1-4 dulu — itu cukup utk lanjut kerja tanpa re-explore dari nol.
2. Jangan ulangi §5 (sudah dicoba & terbukti gagal/terbukti benar).
3. Sebelum ubah kode: jalankan evaluator (§6), baca angkanya, baru putuskan.
4. **Satu perubahan kecil per sesi, langsung diverifikasi full 25-dokumen** (bukan
   sapuan banyak fix sekaligus) — ini permintaan eksplisit user, bukan preferensi.
5. Setelah ubah kode: update §2/§3 (state tiap versi) di file ini dgn angka baru
   — bukan cuma cerita naratif, supaya sesi berikutnya bisa langsung baca tabel.
6. Jangan jalankan 2 evaluator penuh (25 dokumen) bersamaan (§4 poin 5, resource mesin
   cuma ~8GB RAM, sudah 2x OOM/MemoryError).
