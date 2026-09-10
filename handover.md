# HANDOVER — OCR Pipeline Preprocessing & Evaluation

**Last update:** 10 September 2026 (V18)
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
lihat §9 poin 7 soal RAM):

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

## 4. V15 — Title Anchor + ROI Height Growth + Stop Button + Debug Cleanup (SELESAI)

Dipicu oleh permintaan user setelah audit visual 25-dokumen fresh
(`outputs/aa93e09523da_*_roi_document.jpg`, batch YANG SAMA dipakai utk hitung
kategori di bawah): 11/25 (44%) bersih, **6/25 (24%) "block-shift"** (records
10/13/15/19/20/22 — letterhead cabang lebih TINGGI drpd template, identity_area
DAN placement_area jatuh di teks judul/paragraf, field asli TANPA box sama
sekali), 5/25 (20%) residual row-drift (sisa dari V14, lihat §3), 1/25 (4%)
rotasi tak terkoreksi (record 6). User minta 4 hal sekaligus (bukan 1 spt V14 —
lihat instruksi eksplisit di §14):

**4a. Title anchor (root cause block-shift).** Judul cetak "FORMULIR
KEIKUTSERTAAN"/"Program BRI CUAN HADIAH 2026" (bbox normalized diukur langsung
dari `template.pdf`: `(0.38, 0.111, 0.63, 0.136)`) SELALU ada & posisinya
konsisten thd identity_area/placement_area — dipakai sbg anchor tambahan
dgn toleransi pencarian jauh lebih lebar (`TITLE_ANCHOR_SEARCH_TOL_X/Y`,
`preprocessing.py`), krn toleransi anchor field biasa (`ANCHOR_SEARCH_TOL_Y`
±31px) TERBUKTI tidak menjangkau selisih tinggi letterhead nyata (~60-100px).

**PENTING — jangan diulang:** percobaan PERTAMA menggabungkan anchor judul ke
POOL yang sama dgn anchor field (median lalu similarity-weighted average) —
DIBUANG. Anchor judul & anchor field bisa SAMA-SAMA valid tapi mengukur offset
lokal berbeda di posisi halaman berbeda; merata-ratakan 2 kebenaran berbeda
menghasilkan kesalahan ketiga.

**Desain V15 AWAL (INCOMPLETE — diganti V15.1, lihat di bawah, JANGAN
kembalikan):** `estimate_section_transform()` dipecah jadi
`_fit_transform_from_anchors()` + `_match_anchor_pool()`, dipanggil 2x
terpisah, anchor judul dipakai sbg fallback translasi-sendirian HANYA kalau
similarity TERTINGGI anchor field < `FIELD_ANCHORS_WEAK_MAX_SIMILARITY`
(0.36). **User menemukan (via live-app screenshot, BUKAN dari klaim kode)
bahwa fix ini SETENGAH JADI**: `identity_area` benar diperbaiki tapi
`placement_area` TIDAK, krn placement_area punya 4 anchor bersimilarity
biasa-biasa saja (0.33-0.45, semua > 0.36) yg lolos ke `estimateAffinePartial2D`
& menghasilkan fit yg LULUS semua sanity check (rotasi 4.5°, scale 0.96)
TAPI SALAH scr faktual (dy=-13, padahal judul menunjukkan dy=+61 di dokumen
yg SAMA). Trigger "similarity anchor individual rendah" TIDAK CUKUP jadi
sinyal "apakah HASIL fit-nya benar" — fit yg salah bisa tetap lulus semua
pengecekan yg ada.

**V15.1 (desain final, TERVERIFIKASI):** anchor judul SELALU dicocokkan
(bukan cuma saat field terlihat lemah) dan dipakai sbg CROSS-CHECK thd
translasi-Y hasil field-only. Dua pendekatan cross-check DICOBA & DIBUANG
sebelum ketemu yg benar — **jangan diulang**:
1. *Jarak translasi 2D penuh* (`field_result.M[:,2]` vs `title_result.M[:,2]`,
   ambang `SECTION_OUTLIER_DIST_RATIO*diagonal`) — DIBUANG. Record 2
   (`identity_area`, HARUS dipertahankan) berjarak ~36px dari judul; record 13
   (`identity_area`, HARUS di-override) berjarak ~30px — LEBIH KECIL drpd
   record 2. Satu ambang jarak scr matematis tidak bisa memisahkan 2 kasus ini.
2. *Overlap kotak kandidat vs kotak judul* (geometris, cek tumpang tindih
   area) — DIBUANG. Benar utk `identity_area` (record 13 overlap ~95% vs
   record 4/bersih overlap ~36% hanya krn kedekatan template nominal — bukan
   leakage sungguhan), TAPI gagal total utk `placement_area` (record 10):
   box placement_area yg salah nangkring di PARAGRAF LAIN, bukan judul itu
   sendiri, jadi tidak pernah tumpang tindih dgn kotak judul sama sekali.

**Fix yang benar-benar bekerja**: bandingkan translasi-Y SAJA (fenomena
block-shift murni pergeseran vertikal) thd DUA rujukan — 0 (posisi template
asli) dan translasi-Y judul (pergeseran sungguhan, independen section mana
pun). Field-only dianggap "nyangkut di posisi asli" (anchornya gagal
menjangkau konten yg sudah bergeser) kalau `|field_dy| < |field_dy -
title_dy|` — sinyal ini berlaku SAMA baik yg nyangkut itu ada di judul
(identity_area) MAUPUN di paragraf lain (placement_area), krn tidak
bergantung pd APA yg ada di lokasi salah itu. Judul HANYA dipakai sbg rujukan
kalau `|title_dy| >= MIN_TITLE_SHIFT_PX` (15px) — dokumen normal (judul
sendiri dy~0-1px) TIDAK PERNAH masuk perbandingan ini sama sekali, jadi tidak
pernah override hasil field-only yg sudah genuinely benar, terlepas translasi
field-only-nya sendiri berapa. `TITLE_ANCHOR_FALLBACK_MIN_SIMILARITY`
diturunkan dari 0.5 ke **0.40** (0.5 terlalu ketat, membuang 2 match judul yg
genuine: record 15 sim=0.448, record 19 sim=0.496 — keduanya konsisten dgn
pola pergeseran nyata ~52-78px).

**Hasil per-record (dari 6 record block-shift yg diketahui), diverifikasi
VISUAL (box digambar di dokumen asli, bukan cuma baca angka M):**
- Record 10, 13, 15: **identity_area DAN placement_area keduanya benar**,
  dikonfirmasi visual langsung (record 15: nama/rekening/unit kerja DAN
  nominal/tenor/reward semua correctly boxed, cocok GT).
- Record 22: kedua section konsisten via title fallback (belum di-screenshot
  ulang, tapi angka M konsisten dgn pola yg sudah diverifikasi visual di 3
  record lain).
- Record 19: `identity_area` diperbaiki (title fallback); `placement_area`
  TIDAK di-override tapi itu BENAR — fit `section_affine`-nya sendiri
  (dy=50.6) sudah dekat dgn nilai judul (dy=54), tidak perlu override.
- Record 20: **MASIH TIDAK TERSELESAIKAN** — anchor judul sendiri cuma
  similarity 0.310 (di bawah bahkan `SECTION_ANCHOR_MIN_SIMILARITY`=0.32),
  genuinely tidak ketemu dgn cukup yakin di dokumen ini. Root cause berbeda
  (kenapa judul sendiri gagal match di sini — blur? layout beda? belum
  diinvestigasi), BUKAN sesuatu yg diperbaiki fix ini. Known limitation.
- Record 2, 4, 12 (bersih): TIDAK berubah sama sekali (diverifikasi angka M
  identik sebelum/sesudah) — fix ini tidak menyentuh dokumen yg sudah benar.

**4b. ROI height growth (overflow baris ke-2).** Ditemukan terpisah dari
inspeksi zoom record 2: teks terbilang nilai reward yang panjang overflow ke
baris ke-2 SAMA SEKALI tidak ter-assign ke field manapun (window fixed-height,
tidak ada mekanisme growth spt `extract_handwriting_crop` yg sudah ada utk
`tempat_tanggal_surat`). Fix: `_grow_window_downward()` (`dynamic_extraction.py`)
— perluas SISI BAWAH window asosiasi (bukan target_px/posisi asli, bukan sisi
atas) sejauh `prep.ROI_EXTRA_HEIGHT_RATIO` (1x tinggi field), di-cap KERAS ke
`prep.NEXT_FIELD_TOP_NORM` (batas field tetangga, sudah ada, dihitung utk
SEMUA field) supaya tidak pernah menembus baris berikutnya. **PENTING — jangan
diulang:** sempat diterapkan ke SEMUA 7 field, TERBUKTI regresi
`nominal_penempatan` (field angka 1-baris, tidak overflow, growth-nya cuma
nambah risiko tanpa manfaat). Di-scope ULANG ke `GROWABLE_FIELDS = {"reward_tunai"}`
saja (`dynamic_extraction.py`) — SATU-SATUNYA field yg terkonfirmasi overflow.
`reward_tunai` sendiri TIDAK termasuk 5 field yang dibandingkan `evaluation.py`
(lihat `FIELD_GT_MAP`), jadi perbaikan ini tidak terlihat di metrik agregat
tapi tetap benar utk `raw_json`/export Excel.

**4c. Stop button** (`app.py` + `static/index.html`): endpoint baru
`POST /api/sheet/cancel/{session_id}` set flag `cancel_requested` di session
dict; `_process_batch()` cek flag itu di ATAS tiap iterasi SEBELUM mulai
record berikutnya (record yg SEDANG jalan tetap selesai wajar, TIDAK
diinterupsi) lalu tandai sisa record `queued` jadi `cancelled`. Tombol
`sheetStopBtn` (di sebelah `sheetBulkRunBtn`, TIDAK menggantikannya) tampil
hanya selagi `batch-status` melaporkan `active`, nonaktif lagi setelah
`cancel_requested` true atau proses benar2 berhenti.

**4d. Debug output cleanup**: `app.py` berhenti menyimpan
`{request_id}_roi_template.jpg` per record (25 file nyaris-duplikat/batch,
cuma beda warna kotak overlay — base template-nya statis). Field
`roi_template_url` dihapus dari response; panel "Template (ROI)" di UI
(`roiTemplateBox`/`sheetRoiTemplateBox`, kedua tab) juga dihapus krn sumber
gambarnya sudah tidak ada. `_roi_document.jpg` (foto asli + kotak, sinyal
diagnostik sebenarnya) & `template_preview.jpg` (cache singleton, sudah ada
sejak sebelumnya) TIDAK berubah.

**Hasil terverifikasi FINAL (V15.1 dy-relative fix, `evaluation.py`, full 25
dokumen) — ini angka yang BENAR, abaikan angka V15-awal di commit/draft
sebelumnya kalau ketemu, sudah digantikan:**

| Metrik | V14 (`roi_clamp_identity_only`) | V15 awal (`v15_final`, INCOMPLETE) | **V15.1 (`v15_1_dy_fix`, FINAL)** |
|---|---|---|---|
| pipeline_success_rate | 1.00 | 1.00 | 1.00 |
| alignment_success_rate | 1.00 | 1.00 | 1.00 |
| nama_nasabah accuracy | 0.25 (4/16) | 0.2632 (5/19) | **0.3333 (6/18)** |
| nomor_rekening accuracy | 0.4444 (8/18) | 0.45 (9/20) | **0.5789 (11/19)** |
| nominal_penempatan accuracy | 0.2381 (5/21) | 0.2381 (5/21) | **0.3043 (7/23)** |
| tenor_penempatan accuracy | 0.80 (4/5) | 0.80 (4/5) | 0.80 (4/5) — unchanged |
| bentuk_reward accuracy | 1.00 (10/10) | 1.00 (9/9) | 1.00 (9/9) — unchanged |
| decision_accuracy | 0.16 | 0.16 | 0.16 — unchanged (lihat §9 poin 6) |
| review_rate | 0.0 | 0.0 | **0.12** — dokumen ambigu sekarang jujur masuk REVIEW, bukan dipaksa OK/TOLAK |

File bukti: `eval_runs/evaluation_summary_v15_1_dy_fix.json` (FINAL, dipakai). File V15
awal (`v15_title_anchor`/`v15_final`) disimpan sbg bukti historis "kenapa
fix pertama tidak cukup" — JANGAN dipakai sbg baseline, lihat tabel di atas
utk perbandingan yg benar.

**Catatan jujur**: kenaikan kali ini NYATA & lebih besar (nomor_rekening naik
13 poin, nominal_penempatan naik 6.6 poin, measurable count ikut naik utk
ketiganya — artinya lebih banyak dokumen yg field-nya sekarang genuinely
ter-ekstrak, bukan cuma akurasi di antara yg sudah measurable). `review_rate`
naik dari 0 ke 0.12 adalah sinyal SEHAT (bukan regresi) — sebelumnya semua
dokumen dipaksa OK/TOLAK tegas walau datanya ambigu; sekarang beberapa
record correctly masuk status "perlu direview" krn ROI sekarang benar-benar
merefleksikan ketidakpastian yg genuine. `decision_accuracy` belum bergerak
krn 7-check-nya sendiri masih ketat — lihat §9 poin 6, bukan soal ROI lagi.

## 5. V15.2 — Output Structure + Field Quality Gate (Leakage) + Stop Always-Visible

Lanjutan permintaan user (lihat `update.md` utk analisis lengkap section A-F
asli) — dikerjakan Section A, C (sudah termasuk di §4), D, dan SEBAGIAN B.
Section E (notes-only, tidak ada gambar baru) otomatis terpenuhi krn desain
di bawah. **Section B (quality gate) HANYA SEBAGIAN — baca catatan jujur di
akhir sebelum menganggap ini "selesai".**

**Section A — struktur output (`pipeline.py`, `app.py`).** `debug_images`
sekarang jg mengembalikan `original` (dokumen mentah, SEBELUM preprocessing
apa pun — `filled_raw`, sudah dihitung `run_pipeline` tapi sebelumnya tidak
diekspos) dan `preprocessed` (`aligned_img`, SEBELUM kotak ROI digambar).
`app.py` menyimpan `{request_id}_original.jpg` / `_preprocessed.jpg` /
`_roi_document.jpg` per record + SATU `template_roi.jpg` bersama (fungsi baru
`_ensure_template_roi_image()`, cache-checked spt `template_preview.jpg` yg
sudah ada SEBELUMNYA & TETAP dipertahankan — beda fungsi, jangan hapus).
`_preview.jpg` lama (di-load+simpan terpisah SEBELUM `_run_ocr_and_format`
dipanggil, duplikat kerja) dihapus, di-alias ke `original_url` yg baru.
Diverifikasi lewat HTTP nyata (server test di port 8002, endpoint
`/api/sheet/*`) — bukan cuma baca kode: `outputs/` sesudah 1 record berisi
PERSIS `template_roi.jpg` (1x, dipakai bersama), `{id}_original.jpg`,
`{id}_preprocessed.jpg`, `{id}_roi_document.jpg`, `{id}_preview.jpg` (Tab 1
lama) — TIDAK ADA lagi `_roi_template.jpg` atau file debug lain.

**Section D — Stop button** (`static/index.html`): `display:none` diganti
`disabled` (tombol SELALU tampil, cuma status enabled/disabled berubah).
Enable SEGERA saat submit (bukan tunggu poll pertama), disable lagi lewat
poll `batch-status` (`!stillRunning || cancel_requested`). CSS `.btn-stop`/
`.btn-stop:disabled` baru (inline style lama akan KETIMPA `button:disabled`
global krn inline style menang atas class selector manapun kecuali
`!important` — ini kenapa harus pindah ke class, bukan cuma soal rapi).
"Individual/Index/Bulk Run" SUDAH lewat SATU jalur (`runRecords()` →
`/api/sheet/submit` → `_process_batch()`, dikonfirmasi lewat grep) — tidak
ada 3 mekanisme terpisah yg perlu disatukan, itu ASUMSI SALAH di permintaan
awal, arsitekturnya sudah begitu sejak V15.

**Diverifikasi EMPIRIK lewat HTTP nyata (bukan baca kode)**: upload dataset →
submit record 1-6 → cancel saat record 1 masih `processing` → tunggu record 1
selesai → hasil PERSIS sesuai spec: record 1 `done` (TIDAK diinterupsi),
record 2-6 `cancelled` (TIDAK pernah mulai), record 7+ tetap `pending`,
`active` jadi `False`, `cancel_requested` ke-reset `False` (siap run baru
tanpa refresh), hasil record 1 tetap bisa di-fetch (`/api/sheet/result/...`
mengembalikan fields + `template_roi_url`/`original_url`/`preprocessed_url`
lengkap).

**Section B — field quality gate (SEBAGIAN, HANYA leakage detection).**
`preprocessing.extract_template_static_text()` (baru): ekstrak fragmen teks
cetak dari `template.pdf` via **pymupdf** (SUDAH dependency — bukan
pdfplumber, komentar lama yg menyebut pdfplumber cuma soal proses ekstraksi
KOORDINAT manual satu-kali dulu, BUKAN dependency runtime; pdfplumber TIDAK
terpasang di venv ini, jangan install kalau tidak perlu). Di-cache in-memory
(1x per proses). `pipeline._detect_template_leakage(value)` (baru): cek tiap
field yg statusnya SUDAH "read" (OCR Primary percaya diri) apakah value-nya
ternyata cocok fuzzy dgn salah satu fragmen itu (label sendiri/tetangga/
judul/paragraf) — reuse `dynamic._fuzzy_contains` yg SUDAH ada (bukan bikin
fuzzy-match baru). Diwire ke `_text_fallback_reason()` (titik keputusan
eskalasi yg SUDAH ada, dipakai oleh SEMUA 3 tahap fallback existing) sbg
alasan baru `"template_leakage"` — bukan retry-engine terpisah, PERSIS sesuai
rekomendasi `update.md` (pakai infrastruktur fallback yg sudah ada).

Diverifikasi 2 lapis:
1. Unit-level: string kontaminasi yg SUDAH terkonfirmasi sesi ini
   ("Program BRI CUAN HADIAH 2026", "FORMULIR KEIKUTSERTAAN") terdeteksi
   BENAR; nama/nomor rekening/cabang asli TIDAK false-positive.
2. Full dataset (25 dokumen NYATA, bukan string sintetis): gate AKTIF
   BENERAN, nemu 4 field kontaminasi asli — record 5
   (`unit_kerja_pengelola_rekening` = `"3701-01-03219055 engelola Rekening"`,
   fragmen "Unit Kerja Pengelola Rekening"), record 7 (`nama_nasabah` =
   `"Program BRI CUAN HADIAH 2026 ah Ratem"`), record 10
   (`unit_kerja_pengelola_rekening` = `"Pengelola Rekening"`), record 24
   (`nama_nasabah` = `"Program BRI CUAN HADIAH 2026 Syanril Alamyah"`).

**Catatan JUJUR (baca sebelum lanjut ke Section B lainnya)**: full evaluation
(`v15_2_leakage_gate` vs `v15_1_dy_fix`) menunjukkan **angka akurasi PERSIS
IDENTIK, nol perubahan** — walau gate di atas TERBUKTI aktif & benar
mendeteksi 4 kontaminasi nyata. Kenapa: gate SUDAH benar mengeskalasi ke
fallback chain yg ADA (`_run_vlm_fullpage_fallback` → gagal senyap krn
torch/transformers tidak terpasang di environment ini, lihat exception
handling yg SUDAH ada, aman/tidak crash — lalu lanjut ke
`_run_roi_template_fallback`), TAPI fallback ROI/template yg ADA cuma
meng-OCR ULANG crop yg SAMA (`row["roi_bbox"]` yg sudah dikoreksi V15.1),
BUKAN mencoba posisi ROI yang BERBEDA. Utk kasus "partial merge" (konten asli
GENUINE bercampur dgn fragmen judul/label yg bocor, spt record 7/24 di atas)
OCR ulang di crop yg sama akan membaca campuran yg SAMA lagi. **Recovery
sesungguhnya (positional re-localization/local row snapping, "geometric
retry" di spec asli) BELUM dibangun** — itu kenapa akurasi tidak bergerak
meski deteksi sudah benar. Ini BUKAN bug, ini scope yg memang belum
dikerjakan — jangan diklaim "Section B selesai" sampai ini ada.

**Section B yang BELUM dikerjakan sama sekali (jangan asumsikan sudah ada)**:
- ROI clipping detection (B.C) — belum ada kode sama sekali.
- Positional re-localization / local row snapping sbg AKSI recovery baru
  (bukan cuma re-OCR crop yg sama) — belum ada.
- Group-aware sibling re-check (kalau 1 field mencurigakan, cek field
  seangkatan di region yg sama) — belum ada.
- Bounded retry count (`max 1 retry per field`) + notes vocabulary lengkap
  (`geometric_retry`/`shared_area_correction`/`retry_success`/`retry_failed`)
  — belum ada, baru `template_leakage` sbg satu-satunya notes key baru.

## 6. V16 — OCR Candidate Selection + Field Cleaning + Signature Thresholds (SELESAI, terverifikasi)

**Section 1 (OCR token selection).** `dynamic_extraction.py`'s asosiasi
token→field DULU mengumpulkan SEMUA token dlm window & menggabung apa
adanya (row-quantization kasar) — akar penyebab kontaminasi "Program BRI
CUAN HADIAH 2026 ah Ratem" (record 7, lihat §5). Diganti best-candidate
spatial assignment: (1) tolak token label field SENDIRI (SAMA spt
sebelumnya); (2) tolak token label field TETANGGA di region yg sama
(`neighbor_labels`, dari `FIELD_REGION_MAP`, bukan hardcode); (3) tolak
teks cetak statis template (`_is_static_template_text`, reuse
`prep.extract_template_static_text()`, guard panjang alnum≥4 spy angka
pendek asli tidak ikut tertolak); (4) tolak token yg PERSIS teks baris
pilihan choice cetak (mis. "1/3/6" tenor) via `_CHOICE_OPTION_ALNUM`
(diturunkan dari `prep.CHOICE_GROUPS`, exact-match alnum bukan substring —
**ditambahkan SETELAH full-eval pertama nemu regresi record 17**:
`nominal_penempatan` "20000000"→"13620000000" krn token "1/3/6" ke-gabung,
alnum="136" cuma 3 karakter jadi lolos guard #3 yg minimal 4 karakter);
(5) sisa token dikelompokkan jadi BARIS (`_cluster_token_rows`, overlap
vertikal antar token) & baris TERBAIK dipilih via `_score_row_group`
(Y-overlap thd target_px, arah kanan dari anchor, jarak, overlap ROI
window, kecocokan tipe field digit/alpha, confidence sbg SATU sinyal
tambahan berbobot kecil — BUKAN filter penolak, lihat Section 3).
`reward_tunai` (satu2nya `GROWABLE_FIELDS`) DIKECUALIKAN dari seleksi
1-baris ini (tetap gabung semua baris dlm window spt V15) krn
terbilang-nya terkonfirmasi overflow 2 baris.

**Section 2 (cleaning).** `postprocessing.build_text_field_result`: "raw"
TIDAK LAGI ditimpa hasil cleaning (dulu overwritten oleh
`_strip_terms`+`_strip_label_bleed`) — "raw" sekarang = kandidat terpilih
apa adanya (bukti mentah), "cleaned" (key BARU) = setelah strip_terms/
label-bleed/`_clean_field_residue` (baru — buang token residu MURNI tanda
baca sisa OCR, mis. artefak baris jadi "-"/":" sendirian, TIDAK PERNAH
mengubah karakter alnum asli), "value" = `normalize_value(cleaned)`.
`raw_tokens` (SEMUA token yg sempat masuk window sebelum filtering) ikut
tersimpan tembus ke `raw_results` utk audit (raw_json/debug saja, TIDAK
ada perubahan frontend — task eksplisit minta frontend tidak disentuh).

**Section 3 (low confidence).** TIDAK ADA perubahan logika terpisah
diperlukan — seleksi kandidat baru di atas SUDAH TIDAK PERNAH memakai
confidence sbg filter penolak (murni geometris+tipe; confidence cuma bobot
kecil tie-breaker, `0.3 * conf_score` di `_score_row_group`). Kandidat
geometris terbaik SELALU dipertahankan sbg value walau confidence rendah —
status tetap jatuh ke "review" via `OCR_REVIEW_THRESHOLD`/
`FALLBACK_LOW_CONFIDENCE`, **KEDUANYA TIDAK diubah, tetap 0.60/0.45**
(sesuai batasan eksplisit — jangan turunkan ke 0.001).

**Section 4-6 (signature, `signature_diagnostics.py` baru).** Script
standalone (spt `evaluation.py`) — jalankan pipeline penuh 25 dokumen, dump
`ink_area_ratio`/`spread_x`/`spread_y`/`component_count`/`change_ratio`/
status per field signature ke `eval_runs/signature_diagnostics.csv`. Temuan (dari
data, bukan tebakan):
- `signature_nasabah`: 25/25 "present", area_ratio TERENDAH 0.104 (jauh di
  atas `present_area_ratio` 0.012). Diverifikasi visual record 19 (tanda
  tangan tinta asli) & record 7 (ternyata materai Rp10.000 + cap jempol —
  kotak ini di `template.pdf` sendiri berlabel "Opsional materai
  Rp10.000", BUKAN murni kotak tanda tangan). TIDAK ADA kasus borderline
  di 25 dokumen → **threshold TIDAK diubah** (tidak ada bukti utk itu).
- `signature_atasan`: 14 dokumen PERSIS 0.0 (diverifikasi visual record 6,
  benar2 kosong) + 10 dokumen 0.063-0.132 (diverifikasi visual record 1
  crop/3, tanda tangan genuine) + SATU borderline: **record 23**,
  area_ratio=0.0035 (di bawah `absent_area_ratio` lama 0.004 → salah
  ke-klasifikasi "absent"), diverifikasi visual TERNYATA ada goresan tanda
  tangan asli (kemungkinan ke-crop sebagian, `roi_source="fallback"` —
  anchor lokal gagal match utk dokumen ini).
- `postprocessing.SIGNATURE_THRESHOLDS` (baru): profil per-field.
  `signature_atasan.absent_area_ratio` diturunkan 0.004→0.001 (dipakai sbg
  ambang ABSENT candidate sesuai rekomendasi, **BUKAN** dipakai sbg
  `present_area_ratio`) — memindahkan record 23 dari "absent" (salah) ke
  "uncertain" (jujur: ada tinta, bukti tanda tangan blm cukup yakin).
  `signature_nasabah` & sisa parameter `signature_atasan`
  (`present_area_ratio`/`min_spread_ratio`) TIDAK diubah — tidak ada bukti
  25-dokumen utk mengubahnya.
- **Limitation jujur (JANGAN coba "fix" lewat threshold)**: record 1
  `signature_atasan` (`roi_source="fallback"`) scr visual kelihatan ada
  sisa goresan tipis di tepi crop, tapi area_ratio TERUKUR = PERSIS 0.0
  (ke-filter `_remove_line_noise` sbg noise tepi/di luar crop) —
  kemungkinan false-absent akibat anchor gagal match, bukan soal
  threshold (0.0 tidak akan tertangkap threshold serendah apapun). Ini
  masalah GEOMETRI/anchor, di luar scope sesi ini.

**Validasi (urutan sesuai requirement — isolated dulu, baru full).** Git
stash utk banding thd baseline pd 6 record (1,2,5,7,10,24) SEBELUM full
eval → nemu & fix regresi record 17 (poin #4 di atas) SEBELUM jadi masalah
tersembunyi di full-run 25 dokumen. Full evaluation (`v16_final` vs
`v15_2_leakage_gate`):

| metrik | before | after |
|---|---|---|
| nama_nasabah accuracy | 0.3333 | **0.3889** |
| nomor_rekening accuracy | 0.5789 | 0.5789 (unchanged) |
| nominal_penempatan accuracy | 0.3043 | **0.4762** |
| tenor_penempatan accuracy | 0.80 | 0.80 (unchanged) |
| bentuk_reward accuracy | 1.00 | 1.00 (unchanged) |
| all_5_fields_exact | 0.00 | 0.04 |
| all_7_checks_exact | 0.04 | 0.04 (unchanged) |
| decision_accuracy | 0.16 | 0.16 (unchanged — lihat §9 poin 6, di luar scope sesi ini) |
| false_reject_rate | 0.8182 | 0.7727 |
| review_rate | 0.12 | 0.16 |

**Catatan jujur**: beberapa record individual bergeser dari "MISMATCH
lucky-wrong" ke "N/A/review honestly-uncertain" tanpa mengubah rasio
akurasi field itu scr spesifik (mis. record 2/9 `bentuk_reward`/
`nominal_penempatan` evidence-fallback yg dulu kebetulan menebak benar dari
teks noise, sekarang correctly blank/review) — net POSITIF utk kualitas
(tidak ada lagi value yg terlihat pasti benar padahal kebetulan salah),
walau tidak selalu tercermin sbg kenaikan angka. Tidak ada field yg
REGRESI di angka akhir manapun.

## 7. V17 — Identity ROI Collapse Rescue + Signature Static-Content Mask + Blowout Guard (SELESAI, terverifikasi)

Dipicu `nextupdate.md` (plan tertulis user dari sesi audit sebelumnya, 4
finding: identity ROI collapse, konfirmasi placement_area title-anchor,
signature_nasabah kelewat permisif, signature_atasan salah "absent"). Semua
4 finding diverifikasi ULANG thd kode existing sebelum implementasi
(instruksi eksplisit), lalu di-implementasi + diuji isolated + full
evaluation 25 dokumen.

**Finding 1 — identity ROI collapse (record 9/14/6/18/20).** Root cause versi
`nextupdate.md` (satu anchor section transform tidak bisa koreksi skew
per-baris → drift → `_clamp_field_box_y` motong jadi sliver) BENAR sbg
GEJALA, tapi fix yg direncanakan (rescue via `prep.resolve_roi()` per-field,
dianggap "evidence independen") **DIUJI isolated & TERBUKTI TIDAK CUKUP** —
lihat §10 utk detail debug trace. Root cause SEBENARNYA, ditemukan lewat debug
trace nilai dx/dy aktual (BUKAN dari nextupdate.md): `PREV_FIELD_BOTTOM_NORM`/
`NEXT_FIELD_TOP_NORM` (batas tetangga, dipakai `_clamp_field_box_y`) dihitung
di koordinat TEMPLATE (tanpa shift) — kalau section transform-nya sendiri
menggeser satu baris beberapa piksel scr SAH (bukan drift acak — dikonfirmasi
via baris tetangga `unit_kerja` yg SAMA-SAMA bergeser dgn dy identik & tetap
menghasilkan konten yg masuk akal), batas tetangga yg statis itu memotong box
yg SUDAH BENAR posisinya.

**Fix**: `preprocessing._clamp_field_box_y()` dapat parameter baru `dy=0.0`
(default menjaga PERSIS perilaku lama utk SEMUA pemanggil lain — HANYA
`dynamic_extraction._resolve_field_boxes()` utk `identity_area` yg memberi
nilai eksplisit). Sebelum clamp, hitung `raw_dy = target_px[1] -
template_px[1]` (pergeseran section transform utk field ITU SENDIRI), lalu
`_clamp_field_box_y(target_px, field_name, shape, dy=raw_dy)` — batas
tetangga ikut bergeser sejauh dy yg SAMA, jadi box yg genuinely bergeser
tidak lagi ke-clip oleh batas yg "ketinggalan". Rescue via `resolve_roi()`
(rencana awal) DIPERTAHANKAN sbg jaring pengaman TAMBAHAN (defensive,
`FIELD_BOX_COLLAPSE_RATIO=0.75`, diukur dari `eval_runs/identity_roi_baseline.csv` —
rasio TERBURUK yg masih broken 0.731 vs rasio TERKECIL yg tidak pernah
dilaporkan bermasalah 0.782, margin ~0.05 di kedua sisi) — TIDAK aktif lagi
utk 5 record yg diuji (dy-aware clamp sudah cukup), tapi tetap ada utk kasus
di luar sampel itu.

**Hasil isolated (debug trace + pipeline run, records 9/14/6/18 broken +
1/20 kontrol, SEBELUM full eval):**

| record | field | height SEBELUM | height SESUDAH |
|---|---|---|---|
| 9 | nama_nasabah | 9px (collapse) | 26px (penuh) |
| 6 | nama_nasabah | 19px | 26px |
| 14 | nomor_rekening | 3px | 22px |
| 18 | nomor_rekening | 9px | 22px |
| 20 | nomor_rekening | 6px | 22px |
| 1 (kontrol) | semua | tidak berubah | tidak berubah (dy≈0, clamp lama & baru identik) |

**Hasil full evaluation (`eval_runs/evaluation_summary_v17_final.json` vs
`_v16_final.json`, 25 dokumen):**

| metrik | V16 | V17 |
|---|---|---|
| nama_nasabah accuracy | 0.3889 (7/18) | **0.4000 (8/20)** |
| nomor_rekening accuracy | 0.5789 (11/19) | 0.5238 (11/21) |
| nominal_penempatan / tenor_penempatan / bentuk_reward | tidak berubah kecuali 1 record (lihat catatan jujur) | — |
| all_5/all_7 exact, decision_accuracy | 0.04/0.04/0.16 | tidak berubah |

**Catatan JUJUR (per-record, dari `eval_runs/evaluation_results_v17_final.csv` diff
thd `_v16_final.csv` — TIDAK ADA record yg berubah dari MATCH ke MISMATCH
utk 5 field yg diukur, nol regresi murni):**
- Record 14 `nama_nasabah`: N/A → **MATCH** ("Sopuah", benar).
- Record 9 `nama_nasabah` & `nomor_rekening`, record 14 `nomor_rekening`:
  N/A → MISMATCH — box SEKARANG geometris benar (dikonfirmasi visual thd
  baris `unit_kerja` yg sehat), TAPI tulisan tangan nasabah-nya sendiri
  TIDAK persis duduk di baris cetak yg diasumsikan template (record 9
  `nama_nasabah` membaca digit, bukan nama) — ini limitasi TERPISAH yg
  SUDAH didokumentasikan (§9 poin 5, per-baris ink-band localization),
  BUKAN regresi dari fix ini. `nomor_rekening` accuracy TURUN secara %
  (0.5789→0.5238) krn DENOMINATOR naik (19→21, 2 record yg dulu blank
  sekarang measurable) sementara match count TETAP 11 — lebih banyak
  dokumen genuinely mencoba, bukan lebih banyak yg salah dari yg sudah
  benar.
- Record 20 `bentuk_reward` (placement_area, TIDAK disentuh fix ini):
  MATCH → N/A. Diinvestigasi (reproduksi 2x dgn kode V17 yg SAMA → identik,
  jadi bukan noise run-to-run V17 sendiri; code-path SEPENUHNYA independen
  dari perubahan V17 yg identity_area/signature-only). Record 20 SUDAH
  punya `reward_internal_consistent=False` bahkan di V16 (choice detection
  vs teks OCR sendiri tidak sepakat) — indikasi record ini SUDAH
  borderline/fragile sebelum V17, kemungkinan besar sensitif thd
  variasi environment kecil (record 20 JUGA satu2nya yg title-anchornya
  gagal match, §9 poin 3). Dicatat sbg prioritas terpisah (§9 poin 9),
  BUKAN diklaim sbg "fixed" atau disembunyikan.

**Finding 2 — placement_area title-anchor, KONFIRMASI (tidak ada perubahan
kode).** Diverifikasi: `dynamic_extraction._resolve_section_transforms()`
memanggil `prep.estimate_section_transform(..., title_anchor=prep.
TITLE_ANCHOR_ENTRY, ...)` utk SEMUA region di `prep.SECTION_ANCHOR_BBOXES`
(loop yg sama), termasuk `placement_area` — mekanisme title-fallback SUDAH
simetris sejak V15.1, TIDAK butuh perubahan.

**Finding 3 — signature_nasabah static-content mask (Phase A).** Template
crop `signature_nasabah` value_bbox dikonfirmasi visual (crop langsung dari
`template.pdf`) berisi grafik cetak "Opsional (Tidak Wajib) materai
Rp10.000" (kotak bulat + 4 baris teks) — BUKAN area kosong spt
`signature_atasan`. `preprocessing.signature_static_content_mask()` (baru):
threshold intensitas crop TEMPLATE itu sendiri (`SIGNATURE_STATIC_MASK_
THRESHOLD=230`, diukur langsung dari histogram crop — background ~255,
border/teks cetak <220, gap jelas), dilasi 3×3/1 iterasi utk toleransi
registrasi, di-cache per field. Dipanggil UNCONDITIONAL utk KEDUA field di
`postprocessing.process_signatures()` (bukan cuma `signature_nasabah`) —
utk `signature_atasan` template-nya blank murni jadi mask-nya otomatis
kosong (no-op), TIDAK perlu percabangan per-field terpisah.

**Finding 4 — blowout guard (raw coverage → uncertain, bukan absent).**
Diukur (SEBELUM fix, `eval_runs/raw_coverage_baseline.csv` +
`eval_runs/nasabah_masked_coverage.csv`, 25×2 field) raw `difference_mask` (SETELAH mask Finding 3
dikurangi utk `signature_nasabah`, SEBELUM `_remove_line_noise`):
`signature_atasan` menunjukkan pola BIMODAL TAJAM — normal 0.063–0.132
ATAU PERSIS 1.0000 (blowout, **13/25 record**: 1/5/6/7/9/11/12/13/18/19/
20/22/25), SATU kasus di antaranya (record 23, 0.6508, `roi_source=
fallback` — anchor lokal gagal match, crop parsial). `signature_nasabah`
SETELAH mask Finding 3 dikurangi menunjukkan pola SAMA PERSIS (normal
0.0035–0.224, blowout 0.691–0.736, **14/25 record**) — mengonfirmasi
blowout ini properti PER-DOKUMEN (pencahayaan/eksposur di area bawah
halaman tempat KEDUA kotak tanda tangan berada), BUKAN spesifik satu field.
**Ini JAUH lebih luas drpd estimasi awal `nextupdate.md`** ("3 dari 4
record roi_source=fallback") — investigasi awal itu cuma cek record
`roi_source=fallback`, sedangkan blowout SEBENARNYA menimpa `roi_source=
anchor` juga (anchor lokal cocok dgn benar, tapi crop-nya sendiri tetap
gelap krn eksposur foto).

**Fix**: `SIGNATURE_RAW_COVERAGE_BLOWOUT_RATIO=0.50` (duduk di celah lebar
antara cluster normal tertinggi 0.224 dan cluster blowout terendah 0.6508,
margin ≥0.15 di kedua sisi utk SEMUA record terukur) — kalau raw coverage
(setelah mask Finding 3) ≥ ambang ini, LANGSUNG kembalikan `uncertain`
(reason `raw_mask_coverage_hampir_total_kemungkinan_pencahayaan_bukan_
kosong`) TANPA melewati `_remove_line_noise`/`_classify_signature` sama
sekali — mencegah filter garis menghapus SEMUA piksel (termasuk tinta
asli kalau ada) lalu hasil 0.0 pasca-filter disalahartikan sbg "absent".

**Hasil isolated (records 1/7/8/11/12/19/23, pipeline run penuh):** record
11 & 12 `signature_atasan`: absent → **uncertain** (sesuai target
verifikasi `nextupdate.md`). Record 8: TETAP `absent` utk kedua field
(genuinely kosong, area_ratio 0.0000/0.0035, di bawah `absent_area_ratio`
— TIDAK terpengaruh guard blowout). Record 1/7/19/23: `signature_atasan`
JUGA jadi uncertain (raw coverage 1.0/1.0/1.0/0.6508) — melebihi cakupan yg
diantisipasi rencana awal, tapi konsisten dgn data (lihat Finding 4 di
atas).

**Hasil full evaluation — `signature_status_distribution`:**

| field | status | V16 | V17 |
|---|---|---|---|
| signature_nasabah | present | 25 | **10** |
| signature_nasabah | absent | 0 | 1 |
| signature_nasabah | uncertain | 0 | **14** |
| signature_atasan | present | 10 | 10 (tidak berubah) |
| signature_atasan | absent | **14** | **1** |
| signature_atasan | uncertain | 1 | **14** |

**Catatan JUJUR**: pergeseran `signature_nasabah` present→uncertain (25→10)
JAUH lebih besar drpd yg diantisipasi `nextupdate.md` (yg memprediksi "tidak
ada kasus borderline, threshold tidak perlu diubah" utk field ini) — guard
blowout (Finding 4) TERNYATA berlaku SAMA kuat utk `signature_nasabah`
(bukan cuma `signature_atasan`), krn fenomenanya PER-DOKUMEN bukan
per-field (lihat Finding 4 di atas). Ini bukan penyimpangan dari rencana —
persyaratan eksplisit user ("kalau raw signature mask mengalami near-total
coverage akibat lighting, hasilkan uncertain, bukan absent") tidak
membatasi ke satu field, dan mekanisme kegagalan (`_remove_line_noise`
menghapus SEMUA piksel mask nyaris-total) SECARA STRUKTURAL identik utk
kedua field krn keduanya lewat fungsi `process_signatures()` yg sama.
`signature_atasan` `absent` 14→1 (13 record dipindah ke `uncertain`) TEPAT
sesuai tujuan Finding 4 — sebelumnya SEMUA 13 record itu salah diklaim
"tidak ada tanda tangan" padahal buktinya cuma tidak bisa dipercaya (lihat
Finding 4), bukan genuinely kosong.

**Dampak ke decision-level**: **NOL** — diverifikasi (`decision` per record,
`eval_runs/evaluation_results_v17_final.csv` vs `_v16_final.csv`) TIDAK ADA record yg
`decision` akhirnya berubah, `decision_accuracy`/`false_reject_rate`/
`review_rate` semua identik V16=V17. `validate_signature()`
(`comparison.py`) memetakan `absent→TOLAK` dan `uncertain→REVIEW` (BEDA
konsekuensi per-check), tapi 7-check decision pakai worst-case dari SEMUA
check — 13 record yg signature_atasan-nya berubah SUDAH punya check lain
yg gagal duluan (field accuracy dataset ini rendah, ~40-58%), jadi
perubahan status signature TIDAK mengubah keputusan akhir dokumen itu
(tapi TETAP perbaikan kualitas: alasan yg ditampilkan sekarang jujur,
bukan salah menyalahkan "tidak ada tanda tangan").

**Metodologi evaluasi (baca sebelum re-run)**: mesin ini mengalami OOM
BERULANG KALI sesi ini bahkan utk SATU proses evaluator (`--limit 5`
sekalipun) krn RAM sistem yg dipakai aplikasi lain user turun ke <1GB
sesaat — lihat §11 utk workaround batch-subprocess (evaluasi 25 dokumen
INI dijalankan lewat 6 batch kecil, hasil digabung via
`evaluation.build_summary()`/`write_csv()` yg SAMA, bukan reimplementasi
logic). **`average_processing_time_sec` di `evaluation_summary_v17_final.
json` (24.81s) TIDAK BISA dibandingkan langsung dgn V16 (8.8s)** — beda
metodologi (6 proses Python terpisah, tiap batch reload model OCR dari
awal) menaikkan rata-rata scr ARTIFISIAL, BUKAN indikasi pipeline jadi
lebih lambat. `eval_runs/signature_diagnostics.csv` sesi ini HANYA record 1-17
(OOM lagi saat mencoba 18-25 berulang kali) — distribusi status LENGKAP
25 record tetap ada di `eval_runs/evaluation_summary_v17_final.json` (`signature_
status_distribution` di atas), yg dipakai sbg sumber utama.

File bukti (semua di `eval_runs/`): `evaluation_summary_v17_final.json`/
`evaluation_results_v17_final.csv` (FINAL, dibanding thd `_v16_final`),
`identity_roi_baseline.csv` (pengukuran height SEBELUM fix, dasar
`FIELD_BOX_COLLAPSE_RATIO`), `raw_coverage_baseline.csv` +
`nasabah_masked_coverage.csv` (pengukuran raw coverage dasar
`SIGNATURE_RAW_COVERAGE_BLOWOUT_RATIO`), `signature_diagnostics.csv`
(partial, record 1-17).

## 8. V18 — Nomor Rekening Leading-Zero, Template-Leakage via Fallback, Identity Content-Type Gate (SELESAI, terverifikasi)

Dipicu user memeriksa VISUAL output/ dari batch run baru (`4dcb60c90763_*`,
25 dokumen, dijalankan via app.py stlh V17) & menemukan 2 masalah konkret:
teks cetak template ("Program BRI CUAN HADIAH 2026") tertulis sbg
`nama_nasabah` utk record 24 (Syahril Alamsyah), & nomor rekening tertulis sbg
`nama_nasabah` utk record 13 (Yadi Iskandar). User minta audit MENYELURUH +
cek eksplisit `nomor_rekening` salah tempat ke `nama_nasabah`/`unit_kerja`
di SEMUA 25 record.

**Metodologi sesi ini (penting utk direplikasi kalau serupa lagi terjadi)**:
3 Explore agent paralel (audit visual SEMUA 25 `*_roi_document.jpg`, dibagi
2, + 1 code root-cause trace) + 1 Plan agent (desain fix). **User secara
eksplisit menolak exit-plan pertama & minta "audit ulang, pakai yg BENAR2
perlu, jangan sampai turunkan performa yg sudah ada."** Audit ulang ini
men-cross-check klaim visual agent thd HARD DATA (`evaluation_results_
v17_final.csv`) — nemu klaim visual "record 18/23/24: nomor_rekening berisi
NAMA" TIDAK didukung data keras (nomor_rekening_extracted record 18='5',
record 23='N/A', record 24='511' — bukan nama sama sekali). Fix 3 di-rescope
HANYA berdasar bukti yg benar2 bertahan cross-check (record 9 & 13), teori
lebih luas utk 1/18/23/24 didemosikan jadi hipotesis-belum-terkonfirmasi.
**Pelajaran (baca §10 utk entry lengkap)**: laporan visual-only dari
sub-agent BISA salah baca isi kotak kecil/skew — SELALU cross-check thd data
keras (ground truth/extracted value) sebelum mendesain fix, jangan percaya
1 sumber laporan begitu saja walau detail & meyakinkan.

**Fix 1a — leading-zero false MISMATCH, `evaluation.py`.** `compare_field()`
utk `nomor_rekening` (`evaluation.py`) sebelumnya bandingkan digit-string
PERSIS (`_digits_only`) — record 21/25: extracted `'066601005283536'`/
`'051601002049566'` (BENAR, dgn leading zero) vs ground truth
`'66601005283536'`/`'51601002049566'` (angka SAMA, leading zero hilang di
spreadsheet sumber) → salah ke-MISMATCH murni krn leading zero. Fix: reuse
`_norm_numeric()` (int-cast, SUDAH dipakai `nominal_penempatan`) drpd
`_digits_only()` mentah — monotonic-safe (pasangan yg sudah sama persis
sbg string TETAP sama setelah int-cast).

**Fix 1b — bug SAMA di production, `comparison.py`.** Root cause SAMA
PERSIS ditemukan di `_normalize_for_compare()`'s branch `"numeric"` (dipakai
`nomor_rekening`, SATU-SATUNYA titik keputusan match/mismatch, dipakai BAIK
tabel UI MAUPUN decision backend OK/TOLAK/REVIEW) — branch `"currency"` 3
baris di bawahnya SUDAH punya fix ini (`lstrip("0")`), branch `"numeric"`
TIDAK. Ini artinya nasabah NYATA bisa salah ke-TOLAK kalau nomor rekening di
spreadsheet referensi kehilangan leading zero — bug PRODUKSI, bukan cuma
evaluator. Fix: `"numeric"` dipisah dari `"tenor"` (yg TIDAK boleh
tersentuh, jumlah bulan tidak butuh lstrip) jadi branch sendiri dgn
`lstrip("0")` sama spt `"currency"`.

**Fix 2 — leakage template bocor lewat fallback stage, `pipeline.py`.**
`_detect_template_leakage()` (V15.2) SEBELUMNYA cuma dicek di
`_text_fallback_reason()` thd nilai OCR PRIMARY (trigger eskalasi ke
fallback) — begitu eskalasi terjadi, KETIGA fallback stage (VLM full-page,
ROI/template, ROI second-pass VLM) TIDAK PERNAH menyaring ulang HASIL MEREKA
SENDIRI sebelum `row.update(candidate)`/`raw_results[name].update(candidate)`.
Record 24: `dynamic_extraction._is_static_template_text()` diuji LANGSUNG &
TERBUKTI benar menolak "Program BRI CUAN HADIAH 2026" di jalur OCR-primary
(bukan filter itu yg bocor) — leak terkonfirmasi datang dari salah satu
fallback stage (source=`vlm_fullpage`/`roi_template`/`roi_second_pass_vlm`,
independen dari OCR primary) yg tidak pernah disaring. Fix: helper baru
`_screen_fallback_leakage()` dipanggil di KETIGA titik "terima kandidat
fallback" — kalau leakage terdeteksi, status diturunkan jadi `"review"`
(BUKAN `"not_detected"` — ada bukti bacaan, cuma tak bisa dipercaya) SEBELUM
`row.update(...)`, sehingga tidak pernah tertulis & fallback chain lanjut ke
stage berikutnya.

**Fix 3 — content-type sanity gate, `postprocessing.py` (REVISI DESAIN
di-tengah-implementasi berdasar pengukuran runtime, BUKAN rencana awal).**
Rencana awal taruh gate di `dynamic_extraction.py`'s row-selection
(`_score_row_group`'s `type_score` yg sudah ada tapi tak pernah jadi gate
penolak). Pengukuran `raw_results[name]["raw"]` SEBENARNYA (via diagnostic
run per-record) mengungkap record 13's root cause BEDA dari asumsi: `raw` =
`'Yech iskamiar : 375801003255507'` (nama ASLI "Yadi Iskandar" — OCR noise
"Yech iskamiar" — TERTANGKAP, tergabung dgn nomor rekening via titik dua),
`source=roi_template` — datang dari FALLBACK ROI-TEMPLATE stage yg re-OCR
crop LANGSUNG (tidak lewat row-scoring `dynamic_extraction.py` sama sekali).
Kehilangan data sebenarnya terjadi di `postprocessing._strip_label_bleed()`:
heuristik titik-duanya (`if ":" in text: ambil bagian SETELAH titik dua
terakhir` — didesain utk kasus umum "asabah : Ramayana..." dari label yg
bocor) SALAH di sini krn titik dua BUKAN sisa label, tapi batas antara nama
asli & konten bocor dari baris `nomor_rekening` tetangganya — heuristik
membuang jawaban benar, menyisakan yg salah. Record 9 TETAP sesuai teori
awal (`source=ocr_primary`, row-selection murni digit, alpha_ratio=0.000).

**Konsekuensi**: fix di `dynamic_extraction.py` HANYA akan menolong record 9,
NOL efek utk record 13 (beda code path sama sekali). `postprocessing.
build_text_field_result()` adalah SATU-SATUNYA fungsi yg KEEMPAT jalur
(OCR Primary + ketiga fallback stage) lewati sebelum value diterima — choke
point yg lebih tepat & tidak perlu diduplikasi di tiap titik panggil (beda
dgn Fix 2 yg sengaja tetap di `pipeline.py` krn alasan arsitektur V15.2).
Fix: `_identity_type_mismatch(name, text, field_type)` (baru, scoped HANYA
`nama_nasabah`/`nomor_rekening`/`unit_kerja_pengelola_rekening` — TIDAK
generik ke semua field `type=="text"` krn `rentang_tenor`/
`tempat_tanggal_surat` JUGA "text" tapi LEGITIMATELY padat digit, mis.
"1/3/6 Bulan"/tanggal), dipanggil di `build_text_field_result` SETELAH
status dihitung — kalau `cleaned` badly type-mismatched (rasio digit utk
numeric / alpha utk text < `_TYPE_MISMATCH_REJECT_RATIO=0.5`), status
dipaksa `"not_detected"` & value `None` (memicu fallback chain lanjut,
BUKAN diam2 menampilkan value confidently salah bentuk). Ambang 0.5 diukur
dari data NYATA sesi ini: kasus broken 0.000 (record 9) & 0.444 (record 13)
vs kontrol bersih 1.000 (record 1/12/14/22) — margin besar di kedua sisi.

**Hasil isolated (records 9/13 target + 10 record kontrol bersih, pipeline
run penuh SETELAH Fix 3):**
- Record 9 `nama_nasabah`: `'1073 00 0892 568'` (digit) → **`'AGUSTNA NATAU'`**
  (nama asli, via `source=roi_template` — fallback chain berhasil pulih
  SETELAH kandidat digit ditolak; masih OCR-noisy drpd "Agustina Manalip"
  tapi SUDAH benar jenis kontennya, bukan lagi digit).
- Record 13 `nama_nasabah`: `'375801003255507'` (nomor rekening) →
  **`'Yec iskaniar'`** (nama asli, OCR-noisy dari "Yadi Iskandar" — KASUS
  PERSIS yg dilaporkan user, SEKARANG benar jenis kontennya).
- 10 record kontrol (1, 2, 4, 11, 12, 14, 15, 16, 22, 25): **NOL PERUBAHAN**
  — regression check paling penting krn Fix 3 tersentuh SEMUA dokumen,
  bukan cuma yg broken, TERVERIFIKASI aman.

**Hasil full evaluation (`eval_runs/evaluation_summary_v18_final.json` vs
`_v17_final.json`, 25 dokumen, dijalankan via 6 batch-subprocess terpisah —
lihat §11 catatan RAM):**

| metrik | V17 | V18 |
|---|---|---|
| nama_nasabah accuracy | 0.4000 (8/20) | 0.4211 (8/19) |
| nomor_rekening accuracy | 0.5238 (11/21) | **0.7222 (13/18)** |
| nominal_penempatan / tenor_penempatan / bentuk_reward | tidak berubah | tidak berubah |
| all_5/all_7 exact, decision_accuracy | 0.04/0.04/0.16 | tidak berubah |
| false_reject_rate | 0.7727 | 0.7273 (turun, lebih baik) |
| review_rate | 0.16 | 0.20 (naik — sinyal SEHAT, lihat catatan) |
| signature_status_distribution | tidak berubah (kode signature tidak disentuh sesi ini) | tidak berubah |

**Catatan JUJUR (per-record diff `eval_runs/evaluation_results_v18_final.csv` vs
`_v17_final.csv` — NOL REGRESI, tidak ada record MATCH→bukan-MATCH utk 5
field yg diukur, diverifikasi lewat script diff, bukan dibaca sekilas):**
- Record 21/25 `nomor_rekening`: MISMATCH→**MATCH** (Fix 1a bekerja tepat
  sesuai desain).
- Record 24 `nama_nasabah`: MISMATCH ("Program BRI CUAN HADIAH 2026")
  →N/A (Fix 2 bekerja — leak dari fallback stage berhasil disaring).
- Record 9/18/24 `nomor_rekening`: MISMATCH→N/A juga (dari `'07301000052550'`/
  `'5'`/`'511'` yg SEMUANYA sebenarnya salah, jadi honestly N/A). **INI TIDAK
  SEPENUHNYA bisa dijelaskan oleh Fix 1/2/3 scr langsung** (nilai lama bukan
  digit-heavy-salah-tipe yg Fix 3 targetkan utk field `nomor_rekening`
  SENDIRI, & bukan leakage template yg Fix 2 targetkan) — kemungkinan efek
  TIDAK LANGSUNG (perubahan hasil `nama_nasabah` di record yg sama mengubah
  urutan/isi token yg tersisa utk `nomor_rekening`), ATAU non-determinisme
  OCR/GPU run-to-run (terminal setiap run pipeline menampilkan warning
  "Paddle dikompilasi dgn CUDNN 8.9, tapi versi mesin ini JUGA 8.9, may
  cause serious incompatible bug" — TIDAK terverifikasi apakah ini benar2
  sumber variasinya, dicatat sbg kandidat penjelasan saja). **Tidak diklaim
  sbg "fixed oleh X" tanpa bukti pasti** — dicatat apa adanya, tetap net
  POSITIF (MISMATCH→N/A, bukan MISMATCH→lebih-salah).

**Temuan follow-up investigation (dijalankan sesi ini stlh Fix 1-3 selesai,
memakai raw_results langsung, bukan cuma evaluation.py):**
- **Hipotesis "unit_kerja kemasukan digit nomor rekening" (record 1/18/23/24)
  — SEBAGIAN TERKONFIRMASI, & Fix 3 SUDAH memperbaikinya sbg efek samping
  yg tidak direncanakan.** Record 1/18/23: `raw` unit_kerja MEMANG
  digit-heavy (`'1000-01.121212..56.3'`/`': 4059-01-002854-50-7'`/
  `': 0109.0.078059.50.4 BRI BO T'`) — SEKARANG `status=not_detected` (Fix 3
  gate menolaknya, krn `unit_kerja_pengelola_rekening` JUGA masuk
  `_IDENTITY_TYPE_GATE_FIELDS`). Record 24: `raw` unit_kerja = `'ngelola
  Rekening (ogur) Unit tes'` — BEDA jenis masalah (fragmen LABEL "Pengelola
  Rekening" yg bocor, alpha-dominant jadi LOLOS gate Fix 3 walau isinya tetap
  salah) — **BELUM diperbaiki, masalah BARU & berbeda dari digit-leak**,
  prioritas terpisah utk sesi depan (lihat §9).
- **Hipotesis "box unit_kerja/nama_nasabah hilang sama sekali" (record
  5/6/10/2) — TERBANTAH oleh data keras.** `roi_bbox` diukur langsung utk
  keempatnya: SEMUA berukuran NORMAL (24-26px, TIDAK collapse/degenerate) —
  record 5/6/2 genuinely `not_detected` (box benar posisinya, cuma tidak ada
  konten yg berhasil ter-OCR di situ — kemungkinan tulisan tangan terlalu
  samar/buram, bukan masalah ROI), record 10 malah `status=read` dgn value
  `'Pengelola Rekening'` — SAMA PERSIS pola label-bleed record 24 di atas
  (fragmen label bocor, bukan box hilang). Kesimpulan: laporan visual "kotak
  tidak ada" dari sub-agent audit TIDAK AKURAT utk ke-4 record ini —
  pelajaran metodologi, lihat catatan di atas & §10.
- **Record 19/20 (dugaan "seluruh blok identity_area salah tempat total")
  — record 20 TETAP kasus lama yg belum terselesaikan (`method=
  consensus_translation`, TIDAK berubah); record 19 TERNYATA BUKAN regresi
  dari klaim V15.1 (`method=title_fallback_translation` MEMANG aktif, sesuai
  klaim lama) — tapi `field_anchors_max_similarity=0.0` (kegagalan TOTAL
  anchor field lokal, lebih parah dari kasus lain yg pernah pakai title
  fallback) & KEDUA field identity (`nama_nasabah`/`nomor_rekening`) tetap
  `not_detected` scr total. Kesimpulan sementara: bukan masalah GEOMETRI
  (title fallback sudah benar secara desain), kemungkinan masalah KUALITAS
  GAMBAR (blur/eksposur) yg membuat OCR tidak menemukan APAPUN di area yg
  scr geometris sudah benar — BELUM diinvestigasi lebih lanjut, next step
  jelas: cek brightness/blur crop record 19 identity_area scr langsung.

File bukti (semua di `eval_runs/`): `evaluation_summary_v18_final.json`/
`evaluation_results_v18_final.csv` (FINAL, dibanding thd `_v17_final`).

## 9. Prioritas Berikutnya (berdasar data V15.2/V17/V18 di atas, BUKAN tebakan)

1. **Section B recovery, SEBAGIAN sudah selesai lewat V16 (§6)** — record
   7/24 (`nama_nasabah`, kontaminasi JUDUL) SUDAH fix lewat token-level
   candidate selection (bukan positional re-localization spt rencana
   semula — filtering token cukup krn judul & value adalah TOKEN TERPISAH
   yg bisa ditolak sebelum digabung). **Record 5/10
   (`unit_kerja_pengelola_rekening`) MASIH gagal** — beda root cause: OCR
   MENGGABUNG nomor rekening + sisa label jadi SATU token tunggal (mis.
   "3701-01-03219055 engelola Rekening"), tidak ada token terpisah utk
   ditolak filter apa pun. Ini genuinely butuh positional re-localization
   (geser window ROI, bukan filter token) — prioritas TERTINGGI berikutnya
   kalau mau lanjutkan Section B. Kandidat konkret: kalau
   `_detect_template_leakage`/token filtering tetap tidak bisa memisahkan
   (kasus SATU token campur), coba title-anchor-relative correction serupa
   V15.1 utk field itu sendiri, atau row-snapping ke arah field tetangga.
2. **ROI clipping detection (Section B.C)** — belum ada kode sama sekali.
   Perlu akses ke ink-coverage/crop boundary (bukan cuma `row["value"]` teks
   spt leakage detection) — kemungkinan perlu parameter tambahan ke
   `_text_fallback_reason` atau titik integrasi berbeda drpd leakage gate.
3. **Record 20 masih tidak terselesaikan** — anchor judul sendiri gagal
   match dgn cukup yakin (sim=0.310) di dokumen ini, beda root cause dari 5
   record block-shift lain yg sudah fix. Investigasi KENAPA judul gagal match
   di sini spesifik (blur? layout letterhead beda lagi? window pencarian
   `TITLE_ANCHOR_SEARCH_TOL_Y` masih kurang lebar utk kasus ini?) sebelum
   coba fix apa pun — jangan asumsikan sama dgn 5 record lain.
4. **ROI sudah benar (5/6 block-shift record) tapi OCR masih salah baca**
   tulisan tangan yg buram/miring pd dokumen2 itu — jalankan
   `stage_evaluation.py --label v15_2 --dataset assets/ocr_evaluation.xlsx`
   (belum sempat di-run sesi ini) utk pisahkan ROI-fail vs OCR-fail per field
   scr formal, atau evaluasi apakah VLM fallback (`pipeline.VLM_FALLBACK_ENABLED`,
   perlu `torch`+`transformers`, lihat `requirements.txt`) membantu — domain
   yg PAS utk VLM (deskriptif dari konteks visual), BEDA dari masalah ROI yg
   sudah selesai (deterministik, geometris).
5. **Per-baris localisasi independen utk identity_area** (bukan 1 shared
   section transform) — utk kasus drift-satu-blok spt record 1 (lihat §3,
   BEDA dari block-shift V15.1 — record 1 bukan soal letterhead, tapi rotasi
   residual kecil). Pendekatan: cari ink band lokal di sekitar window tiap
   baris (mirip `_find_ink_bands` yg sudah ada). **V17 dy-aware clamp (§7
   Finding 1) MENGURANGI gejala ini utk kasus translasi-seragam** (box tidak
   lagi collapse jadi sliver), TAPI TIDAK menyelesaikan kasus konten
   handwriting yg posisinya sendiri tidak persis di baris cetak (lihat poin
   9 di bawah) — ink-band-per-baris MASIH prioritas valid utk itu.
6. **decision_accuracy stuck di 0.16** walau field accuracy naik nyata —
   cek `decision_reasons` di `eval_runs/evaluation_results_v15_1_dy_fix.csv` utk lihat
   check mana yg paling sering gagal (7-check jauh lebih ketat drpd akurasi
   per-field individual — field yg sekarang benar blm tentu cukup utk lolos
   SEMUA 7 check sekaligus). **V17: masih 0.16, TIDAK bergerak sama sekali**
   walau signature_atasan 13 record absent→uncertain — diverifikasi (§7
   Finding 4) SEMUA 25 record `decision` PERSIS SAMA V16 vs V17, krn record2
   yg terkena SUDAH punya check lain yg gagal duluan (7-check pakai
   worst-case, signature bukan satu2nya alasan TOLAK/REVIEW di record2 itu).
7. Mesin evaluasi ini hanya ~8GB RAM — **JANGAN** jalankan 2 evaluator penuh
   (25 dokumen) bersamaan (sudah beberapa kali trigger OOM/MemoryError sesi
   lalu). Jalankan satu-satu, atau pakai `--limit` utk uji cepat dulu. **V17:
   kondisi bisa LEBIH PARAH drpd ini** (free RAM sistem sempat <1GB krn app
   lain milik user, di luar kendali evaluator) — lihat §11 utk workaround
   batch-subprocess kalau kejadian lagi.
8. **(V17, RESOLVED lewat V18 §8 Fix 3) Record 9 `nama_nasabah` dulu
   ter-ekstrak PENUH tapi ISINYA SALAH** (baca digit dari row lain) — SEKARANG
   `'AGUSTNA NATAU'` (nama asli, OCR-noisy tapi jenis kontennya benar) stlh
   `_identity_type_mismatch` gate (V18) menolak kandidat digit & fallback
   chain berhasil pulih. **Catatan**: ini menangani GEJALA (digit salah tipe
   ditolak, bukan diterima), BUKAN akar masalah "ink-band per-baris" di
   prioritas #5 — kalau muncul lagi kasus serupa yg fallback-nya JUGA gagal
   pulih (tetap `not_detected`), prioritas #5 (ink-band lokal) masih relevan
   sbg next step, gate V18 cuma jaring pengaman thd OUTPUT yg confidently
   salah bentuk.
9. **(V17, MASIH BELUM JELAS setelah V18) Record 20 `bentuk_reward`** —
   TIDAK disentuh V17 ATAU V18 (keduanya identity_area/signature/nomor_
   rekening-only), TETAP borderline sejak V16 (`reward_internal_
   consistent=False`). Belum diselidiki lebih lanjut sesi V18 — tetap
   prioritas rendah, terkait dgn masalah title-anchor record 20 yg sama
   (poin 3 di atas).
10. **(V18, BARU) Label cetak "Pengelola Rekening" bocor ke `unit_kerja_
    pengelola_rekening`, ALPHA-dominant jadi LOLOS gate V18 Fix 3** — record
    10 (`raw='Pengelola Rekening'`, `status=read`) & record 24
    (`raw='ngelola Rekening (ogur) Unit tes'`) BEDA dari kasus digit-leak yg
    sudah tertangani (poin 8/§8 Fix 3): ini fragmen LABEL CETAK yg bocor
    (bukan konten field tetangga), lolos gate krn alpha-ratio-nya TINGGI
    (bukan digit). Kandidat fix: `_is_static_template_text`/`neighbor_labels`
    rejection (`dynamic_extraction.py`) SEHARUSNYA sudah menolak fragmen
    label spt ini di jalur OCR-primary, tapi record 10/24 keduanya via
    fallback stage (cek `source` dulu) yg TIDAK melewati filter itu sama
    sekali — mirip pola akar masalah Fix 2 (V18), pertimbangkan apakah
    `neighbor_labels`/`_is_static_template_text` PERLU direplikasi di
    fallback choke point yg sama spt `_screen_fallback_leakage`, ATAU cukup
    tambahkan sbg bagian dari `_identity_type_mismatch`-style check (bukan
    shape digit/alpha, tapi fuzzy-match ke label cetak). BELUM diukur/
    didesain — jangan asumsikan salah satu pendekatan tanpa data dulu.
11. **(V18, BARU) Record 19 identity_area: `title_fallback_translation`
    AKTIF (BUKAN regresi dari klaim V15.1 §4a), tapi `nama_nasabah` DAN
    `nomor_rekening` tetap `not_detected` total** — `field_anchors_max_
    similarity=0.0` (kegagalan TOTAL anchor field lokal, lebih parah dari
    kasus title-fallback lain yg sudah diverifikasi visual benar). Diduga
    masalah KUALITAS GAMBAR (blur/eksposur) di area itu, BUKAN penempatan
    ROI (title fallback sudah bekerja sesuai desain scr geometris) — next
    step: ukur brightness/blur crop identity_area record 19 langsung
    (`cv2.Laplacian` variance utk blur, `.mean()` utk exposure, pola yg
    SAMA dgn `_needs_enhancement`/V17 Finding 4's raw-coverage diagnostic)
    sebelum menebak fix apa pun.
12. **(V18, BARU, prioritas rendah) 3 record (9/18/24) `nomor_rekening`
    berubah MISMATCH→N/A antara V17 & V18 tanpa penjelasan pasti** — nilai
    lama (`'07301000052550'`/`'5'`/`'511'`) semuanya sudah salah, jadi N/A
    net POSITIF (jujur, bukan lebih buruk), tapi TIDAK ada jalur kode V18
    (Fix 1/2/3 identity_area-only utk field lain) yg secara langsung
    menjelaskan PERUBAHAN ini utk field `nomor_rekening` itu SENDIRI —
    kemungkinan efek tidak langsung (hasil `nama_nasabah` yg berubah di
    record sama menggeser token yg tersisa) atau non-determinisme OCR/GPU
    (lihat §8 catatan CUDNN di atas). TIDAK perlu dikejar kecuali muncul
    pola serupa yg lebih luas.

## 10. Sudah Dicoba / Jangan Diulang (hemat waktu+token sesi berikutnya)

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
- **(V15) Menggabung anchor judul ke pool anchor field yg sama** (median atau
  rata-rata berbobot similarity) → DIBUANG, lihat §4a. Anchor judul & anchor
  field bisa sama-sama valid tapi mengukur offset lokal BERBEDA di posisi
  halaman berbeda — jangan dirata-rata, pakai sbg fallback translasi terpisah
  (`title_anchor` param di `estimate_section_transform`, tidak pernah campur).
- **(V15) `_grow_window_downward` diterapkan ke SEMUA field** (bukan hanya yg
  terkonfirmasi overflow) → TERBUKTI regresi `nominal_penempatan`, lihat §4b.
  Field angka 1-baris tidak butuh growth; JANGAN tambah field baru ke
  `GROWABLE_FIELDS` (`dynamic_extraction.py`) tanpa bukti overflow spesifik
  spt reward_tunai (zoom debug image, bukan tebakan).
- **(V15) Trigger title-fallback berdasar "similarity anchor field individual
  rendah"** (`FIELD_ANCHORS_WEAK_MAX_SIMILARITY`) → TERBUKTI tidak cukup:
  placement_area bisa lolos jadi `section_affine` yg confidently WRONG walau
  semua anchor individualnya "biasa saja" (bukan rendah). Diganti V15.1 (lihat
  §4a) — jangan kembalikan trigger berbasis similarity-individual.
- **(V15.1, percobaan 1) Cross-check pakai jarak translasi 2D penuh**
  (`np.linalg.norm` antara `field_result.M[:,2]` dan `title_result.M[:,2]`) →
  DIBUANG. Record 2 (harus dipertahankan) jarak ~36px vs record 13 (harus
  di-override) jarak ~30px — kontradiksi matematis, satu ambang tidak bisa
  memisahkan keduanya. Jangan coba lagi dgn ambang berbeda, masalahnya bukan
  di angka ambang, tapi di bentuk perbandingannya (2D penuh, bukan Y saja).
- **(V15.1, percobaan 2) Cross-check pakai overlap kotak kandidat vs kotak
  judul** (area tumpang tindih geometris) → DIBUANG. Benar utk identity_area
  TAPI placement_area yg salah nangkring di PARAGRAF LAIN (bukan judul), jadi
  tidak pernah terdeteksi lewat overlap dgn judul. Diganti perbandingan
  translasi-Y polos (lihat §4a) yg tidak bergantung pd APA yg ada di lokasi
  yg salah.
- **(V15.2) `pdfplumber` DIKIRA dependency runtime** krn komentar lama di
  `preprocessing.py` menyebutnya — SALAH, `pdfplumber` TIDAK terpasang di
  venv ini & TIDAK ada di `requirements.txt`. Komentar itu soal proses
  ekstraksi KOORDINAT manual satu-kali dulu (di luar aplikasi), bukan
  dependency yg jalan tiap request. Pakai **pymupdf** (`fitz`, SUDAH
  dependency) utk ekstrak teks — `preprocessing.extract_template_static_text()`.
  Jangan install pdfplumber kalau tidak benar-benar perlu fitur yg pymupdf
  tidak punya.
- **(V15.2) Menganggap "deteksi leakage terpasang" = "Section B selesai"**
  → SALAH, lihat §5 catatan jujur. Deteksi tanpa aksi recovery yg bisa ubah
  POSISI ROI (bukan cuma re-OCR crop yg sama) TIDAK mengubah akurasi apa pun
  — 4 kontaminasi nyata terdeteksi BENAR sesi ini, nol yg ke-fix, krn
  fallback ROI/template yg ADA cuma re-OCR di crop yg SAMA. Jangan laporkan
  gate ini sbg "fix" tanpa recovery yg benar2 mengubah posisi.
- **(V16) Guard panjang minimum utk template-text rejection (alnum≥4) TIDAK
  CUKUP utk baris pilihan choice cetak** (mis. "1/3/6" alnum="136", cuma 3
  karakter) → lolos filter, ke-gabung ke field tetangga
  (`nominal_penempatan` "20000000"→"13620000000", record 17). Ditemukan
  LEWAT full-evaluation (bukan isolated check — isolated check pakai record
  yg beda, tidak nyentuh kasus ini), fixed dgn exact-match terpisah
  `_CHOICE_OPTION_ALNUM` (dari `prep.CHOICE_GROUPS`, bukan hardcode "1/3/6").
  **Pelajaran**: filter berbasis panjang-minimum SAJA tidak cukup utk teks
  cetak PENDEK yg posisinya berdekatan scr vertikal dgn field lain — kalau
  nambah filter template-text serupa lagi, pikirkan token PENDEK (label
  angka/kode) yg mungkin lolos guard panjang, bukan cuma teks panjang.
- **(V16) Metodologi git-stash utk A/B isolated check TANPA commit** →
  TERBUKTI efisien: `git stash` (kembali ke baseline persis) → jalankan
  script isolated check yg SAMA → catat hasil → `git stash pop` (kembali ke
  versi baru). Lebih cepat & lebih akurat drpd membandingkan thd angka lama
  dari sesi sebelumnya (kondisi environment/model cache bisa beda). Pakai
  pola ini lagi kalau perlu banding cepat sebelum full evaluation 25 dokumen.
- **(V17) Rencana awal Finding 1 (rescue box collapse LEWAT `prep.resolve_roi()`
  per-field SAJA, tanpa dy-aware clamp)** → DIUJI isolated (debug trace
  langsung, bukan tebakan) & TERBUKTI TIDAK CUKUP: pada 4 dari 5 record
  broken yg diuji (9/14/18/6), anchor lokal `resolve_roi()` field itu
  SENDIRI menghitung dx/dy yg PERSIS SAMA dgn section transform yg katanya
  "collapse" — BUKAN evidence independen spt diasumsikan rencana awal,
  keduanya menemukan pergeseran GENUINE yg sama pada baris itu, sehingga
  hasil rescue ke-clamp dgn cara & collapse yg SAMA (tidak menolong sama
  sekali). Root cause SEBENARNYA (lihat §7 Finding 1): batas tetangga
  (`PREV_FIELD_BOTTOM_NORM`/`NEXT_FIELD_TOP_NORM`) dihitung di koordinat
  TEMPLATE (tanpa shift) & TIDAK ikut bergeser walau box-nya sendiri SAH
  bergeser turun/naik beberapa piksel — clamp lama menabrak box yg SUDAH
  BENAR. Fix yg benar2 bekerja: geser batas tetangga dgn dy YANG SAMA
  SEBELUM clamp (`prep._clamp_field_box_y(..., dy=...)`), bukan mengganti
  SUMBER box-nya. **Pelajaran**: "dua metode independen setuju" TIDAK selalu
  berarti keduanya independen scr evidence — kalau keduanya mengukur
  fenomena fisik yg SAMA (di sini: pergeseran vertikal genuine section),
  keduanya akan setuju walau salah satu bukan "second opinion" yg valid;
  verifikasi via debug trace nilai aktual (dx/dy/similarity), jangan cuma
  percaya "sumber lain dipakai = independen".
- **(V17) Menjalankan `evaluation.py` (bahkan `--limit 5`) di mesin ini saat
  free RAM sistem <1GB (dipakai app lain user)** → OOM-killed BERULANG kali,
  bahkan sendirian (tanpa proses evaluator lain jalan bersamaan) → lihat
  §11 utk workaround batch-subprocess yg terbukti bekerja di kondisi sama.
- **(V18) Mempercayai laporan visual-only dari sub-agent TANPA cross-check
  data keras** → sesi ini mendesain rencana Fix 3 awal berdasar laporan 2
  Explore agent yg mengaudit visual 25 `*_roi_document.jpg` (deskripsi
  "record 18/23/24: nomor_rekening berisi NAMA, unit_kerja berisi digit") —
  TERNYATA SALAH utk separuh klaimnya begitu di-cross-check thd
  `eval_runs/evaluation_results_v17_final.csv` (nomor_rekening_extracted record 18=`'5'`,
  23=`'N/A'`, 24=`'511'` — BUKAN nama sama sekali). Begitu juga laporan
  "box unit_kerja/nama_nasabah HILANG SAMA SEKALI" (record 5/6/10/2) —
  TERBANTAH oleh `roi_bbox` yg diukur langsung (semua NORMAL, 24-26px, tidak
  collapse). **Pelajaran**: sub-agent visual (bahkan yg detail & percaya
  diri) BISA salah baca kotak kecil/skew/tulisan buram di gambar statis
  TANPA ground truth utk cross-check — SELALU verifikasi klaim visual thd
  data keras (extracted value/ground truth/raw_results langsung) SEBELUM
  mendesain fix berdasarkan itu, bahkan kalau laporannya panjang & spesifik.
  User secara eksplisit meminta audit-ulang ini ("jangan sampai turunkan
  performa yg sudah ada") — hasilnya Fix 3 di-rescope jadi HANYA berdasar
  bukti yg bertahan cross-check (record 9/13), bukan seluruh klaim awal.
- **(V18) Rencana taruh gate content-type di `dynamic_extraction.py`'s
  row-selection (row/score-based)** → SETELAH implementasi dimulai, diuji
  langsung dgn diagnostic run per-record (`raw_results[name]["raw"]`) &
  TERNYATA record 13 (kasus utama yg mau diperbaiki) SAMA SEKALI tidak lewat
  jalur itu (`source=roi_template`, fallback stage re-OCR crop langsung,
  BUKAN row-scoring `dynamic_extraction.py`) — root cause SEBENARNYA ada di
  `postprocessing._strip_label_bleed()`'s heuristik titik-dua yg salah
  interpretasi. Fix dipindah ke `postprocessing.build_text_field_result()`
  (choke point yg SEMUA 4 jalur — OCR primary + 3 fallback stage — lewati),
  BUKAN `dynamic_extraction.py`. **Pelajaran**: SEBELUM mengunci lokasi fix
  berdasar teori "row-selection yg salah", ukur DULU `raw_results[name]
  ["raw"]`/`["source"]` sesungguhnya utk record yg mau diperbaiki — jalur
  kode yg BENAR2 memproduksi value yg salah bisa beda dari yg diasumsikan,
  bahkan kalau gejala akhirnya (value tampak "collapsed"/salah) sama persis.

## 11. Cara Re-run Evaluasi (perintah persis)

**(V18) Semua output (`evaluation_results_*.csv`/`evaluation_summary_*.json`/
`stage_evaluation_*`/`compare_runs_*`/`signature_diagnostics.csv`) SEKARANG
otomatis ditulis ke folder `eval_runs/` (dibuat otomatis kalau belum ada),
BUKAN root project lagi** — sebelumnya berserakan campur dgn file kode,
menyulitkan cari file evaluasi/diagnostic terbaru. `eval_runs/` di-gitignore
(sama spt `evaluation_summary*`/`evaluation_result*` sebelumnya) — file di
situ HANYA utk referensi lokal, TIDAK di-commit. Semua file historis (V13-V18)
SUDAH dipindah ke sana sesi ini, lihat §12.

```bash
# end-to-end (field accuracy + decision accuracy) -- paling relevan utk "akurasi asli"
python evaluation.py --dataset assets/ocr_evaluation.xlsx --label <nama_run> --debug-dir none

# 3-gate funnel (PREPROCESSING -> ROI -> OCR), CER, dst -- utk isolasi root cause per stage
python stage_evaluation.py --dataset assets/ocr_evaluation.xlsx --label <nama_run> --no-xlsx --debug-dir none

# uji cepat sebelum full run (hemat waktu ~12s/dokumen)
python evaluation.py --dataset assets/ocr_evaluation.xlsx --label smoke --limit 5 --debug-dir none

# bandingkan beberapa run (lihat §2, poin 4) -- path SEKARANG di eval_runs/
python compare_runs.py eval_runs/evaluation_summary_<a>.json eval_runs/evaluation_summary_<b>.json

# audit diagnostik signature_nasabah/signature_atasan (V16, §6) -- CSV per
# record: ink_area_ratio/spread_x/spread_y/component_count/change_ratio/status
# -- ditulis ke eval_runs/signature_diagnostics.csv
python signature_diagnostics.py --dataset assets/ocr_evaluation.xlsx
```

**Ingat**: `evaluation.py` TANPA `--debug-dir none` MENULIS folder
`evaluation_debug/` (crop per-record, bisa ~100MB utk 25 dokumen) — sesi
V16 sempat lupa pakai flag ini & harus dibersihkan manual (`rm -rf
evaluation_debug`) sesudahnya. Selalu sertakan `--debug-dir none` kecuali
memang butuh crop debug visual utk investigasi.

**V17 — mesin bisa lebih parah dari ~8GB yg didokumentasikan §9 poin 7.**
Sesi V17 mengalami `evaluation.py` full-25-dokumen (bahkan `--limit 5`)
di-OOM-kill BERULANG KALI oleh sistem (free RAM sempat turun ke <1GB krn
aplikasi lain user, di luar kendali proses evaluator) — bukan disebabkan
kode V17, tapi tetap memblokir evaluator SATU proses sekalipun (bukan cuma
soal "jangan jalankan 2 evaluator bersamaan"). Workaround yg TERBUKTI
bekerja sesi ini: jalankan per-batch record kecil (mis. 5 record) sbg
proses Python TERPISAH tiap batch (supaya model OCR di-release penuh antar
batch, bukan diakumulasi dlm satu proses panjang), reuse
`evaluation.evaluate_record()`/`build_summary()`/`write_csv()` apa adanya
lewat script wrapper kecil (TIDAK mengubah `evaluation.py` itu sendiri),
gabung hasil tiap batch di akhir. Kalau `evaluation.py` biasa (bahkan
`--limit` kecil) di-OOM-kill lagi di sesi depan, pakai pola ini drpd
mengulang retry penuh berkali-kali.

## 12. Peta File

**Aktif (produksi, di-import `app.py`/`pipeline.py`):** `preprocessing.py`, `ocr.py`,
`postprocessing.py`, `dynamic_extraction.py`, `comparison.py`, `vlm.py`, `pipeline.py`,
`data_input.py`, `app.py`, `static/index.html`.

**Evaluasi (standalone CLI, tidak di-import app.py):** `evaluation.py`,
`stage_evaluation.py`, `compare_runs.py` (V13), `signature_diagnostics.py` (V16).
Output KEEMPAT tool ini (csv/json hasil run) ditulis ke folder **`eval_runs/`**
(V18 — dibuat otomatis, di-gitignore, SATU tempat utk semua evaluasi/diagnostic
drpd berserakan di root spt sebelumnya), isinya: `evaluation_results_*.csv`/
`evaluation_summary_*.json` (V15-V18, riwayat lengkap), `stage_evaluation_
results_*`/`stage_evaluation_summary_*` (kalau `stage_evaluation.py` dijalankan),
`compare_runs_*` (`compare_runs.py`), `signature_diagnostics.csv` (V16-V18),
plus 3 file diagnostic pengukuran threshold V17 (`identity_roi_baseline.csv`,
`raw_coverage_baseline.csv`, `nasabah_masked_coverage.csv` — bukti dasar
`FIELD_BOX_COLLAPSE_RATIO`/`SIGNATURE_RAW_COVERAGE_BLOWOUT_RATIO`/
`SIGNATURE_STATIC_MASK_THRESHOLD`, lihat §7).

**Eksperimen/versi lama, TIDAK di-import mana pun secara aktif (verifikasi ke user
sebelum dihapus — mungkin masih dipakai manual utk banding V10/11 vs V12 lewat
`--pipeline pipeline1`):** `pipeline1.py`/`pipeline2.py`, `comparison1.py`/`comparison2.py`,
`postprocessing1.py`, `vlm1.py`, `index1.html`, `static/index1.html`,
`static/serep2.html`/`serep3.html`/`serep4.html`/`serep6.html`.

## 13. Field & Rules (tidak berubah dari versi sebelumnya)

Field: Nama (fuzzy match 80%), Nomor Rekening, Unit Kerja, Nominal Penempatan, Tenor
(dari rentang tanggal, <1 bulan = tidak sesuai), Bentuk Reward (tunai/non-tunai +
fallback), TTD Nasabah & TTD Unit Kerja/BRI (presence only, bukan baca isi).

Rule rural (`Urban` != "urban" di data): cukup validasi Nama, Nomor Rekening, Nominal
Penempatan, TTD Nasabah, TTD BRI — kalau semua lolos, `OK Lanjut Proses`.

Environment: Windows, FastAPI `127.0.0.1:8001`, GPU NVIDIA MX230 2GB (PaddleOCR
mayoritas jalan CPU di environment evaluasi), `MAX_CANVAS_HEIGHT_GPU=2200` /
`_CPU=6000`, `text_recognition_batch_size=1` (low VRAM GPU kecil).

## 14. Instruksi utk Claude (sesi berikutnya)

1. Baca §1-9 dulu — itu cukup utk lanjut kerja tanpa re-explore dari nol.
2. Jangan ulangi §10 (sudah dicoba & terbukti gagal/terbukti benar).
3. Sebelum ubah kode: jalankan evaluator (§11), baca angkanya, baru putuskan.
4. **Default: satu perubahan kecil per sesi, langsung diverifikasi full
   25-dokumen** (bukan sapuan banyak fix sekaligus). V15 & V17 adalah
   PENGECUALIAN krn user eksplisit minta beberapa hal sekaligus dlm satu
   pesan — kalau itu terjadi lagi, tetap verifikasi tiap perubahan YANG
   BERISIKO REGRESI (mis. logika ROI/transform) satu-satu sebelum lanjut ke
   perubahan berikutnya, spt yg dilakukan utk 4a/4b (V15) dan Finding 1 (V17)
   di atas (masing2 ketahuan overreach/rencana-tidak-cukup-nya justru krn
   dites terpisah, bukan digabung langsung/dipercaya begitu saja).
4b. **(V17) Rencana tertulis (mis. `nextupdate.md`) adalah HIPOTESIS, bukan
   jaminan** — Finding 1 V17 adalah contoh konkret: root cause & fix yg
   direncanakan TERBUKTI SALAH saat diuji isolated (debug trace nilai
   aktual), root cause SEBENARNYA berbeda. Selalu verifikasi via ukuran
   nyata (debug trace / isolated test) SEBELUM percaya sebuah rencana
   "seharusnya bekerja", walau rencananya eksplisit & detail.
5. Setelah ubah kode: update §2/§3/§4/§5/§6/§7/§8 (state tiap versi) di file
   ini dgn angka baru — bukan cuma cerita naratif, supaya sesi berikutnya
   bisa langsung baca tabel.
6. Jangan jalankan 2 evaluator penuh (25 dokumen) bersamaan (§9 poin 7,
   resource mesin cuma ~8GB RAM, sudah beberapa kali OOM/MemoryError). **V17:
   bisa OOM bahkan SATU evaluator/`--limit` kecil kalau RAM sistem lagi
   dipakai app lain user** — lihat §11 utk workaround batch-subprocess.
