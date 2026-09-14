# HANDOVER — OCR Pipeline Preprocessing & Evaluation

**Last update:** 14 September 2026 (V19e -- local Qwen3-VL-2B pada GPU 2GB TERBUKTI tidak usable, engine "qwen3_vl_local" ditambahkan sbg opsi terpisah, lihat §9e; V19c -- ganti engine "qwen2_vl" lokal jadi "qwen3_vl" Hugging Face HOSTED, lihat §9c; V19 -- Qwen2-VL direct-semantic pipeline + Gemini token/cost tracking, lihat §9b)
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
lihat §10 poin 7 soal RAM):

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
lihat instruksi eksplisit di §15):

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
| decision_accuracy | 0.16 | 0.16 | 0.16 — unchanged (lihat §10 poin 6) |
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
krn 7-check-nya sendiri masih ketat — lihat §10 poin 6, bukan soal ROI lagi.

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
| decision_accuracy | 0.16 | 0.16 (unchanged — lihat §10 poin 6, di luar scope sesi ini) |
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
lihat §11 utk detail debug trace. Root cause SEBENARNYA, ditemukan lewat debug
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
  SUDAH didokumentasikan (§10 poin 5, per-baris ink-band localization),
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
  gagal match, §10 poin 3). Dicatat sbg prioritas terpisah (§10 poin 9),
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
sesaat — lihat §12 utk workaround batch-subprocess (evaluasi 25 dokumen
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
**Pelajaran (baca §11 utk entry lengkap)**: laporan visual-only dari
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
lihat §12 catatan RAM):**

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
  prioritas terpisah utk sesi depan (lihat §10).
- **Hipotesis "box unit_kerja/nama_nasabah hilang sama sekali" (record
  5/6/10/2) — TERBANTAH oleh data keras.** `roi_bbox` diukur langsung utk
  keempatnya: SEMUA berukuran NORMAL (24-26px, TIDAK collapse/degenerate) —
  record 5/6/2 genuinely `not_detected` (box benar posisinya, cuma tidak ada
  konten yg berhasil ter-OCR di situ — kemungkinan tulisan tangan terlalu
  samar/buram, bukan masalah ROI), record 10 malah `status=read` dgn value
  `'Pengelola Rekening'` — SAMA PERSIS pola label-bleed record 24 di atas
  (fragmen label bocor, bukan box hilang). Kesimpulan: laporan visual "kotak
  tidak ada" dari sub-agent audit TIDAK AKURAT utk ke-4 record ini —
  pelajaran metodologi, lihat catatan di atas & §11.
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

## 9. V19 — Multi-Extractor Benchmark (V18 + Gemini 3.8/2.5 Flash + Mistral OCR 4.1)

**Tujuan**: benchmark V18 (pipeline deterministik: template alignment + ROI +
PaddleOCR + fallback VLM) thd 3 API AI eksternal (2 model Gemini, 1 model
Mistral OCR), pakai LOGIC PERBANDINGAN/KEPUTUSAN yg SAMA PERSIS
(`comparison.py`, TIDAK diubah sesi ini), dgn jaminan TEPAT SATU extractor
jalan per dokumen/request — TIDAK PERNAH fan-out, TIDAK PERNAH fallback diam2
antar engine. V18 TETAP baseline, internal-nya (preprocessing/alignment/ROI/
PaddleOCR/VLM fallback) TIDAK disentuh sama sekali.

**Arsitektur — modul baru `extractors.py`**: `run(engine, document_path,
progress_callback=None, reference_record=None)` adalah SATU-SATUNYA titik
masuk yg dipanggil `app.py`/`evaluation.py` — dispatcher if/elif/else, TEPAT
1 cabang jalan per panggilan:
- `run_v18(...)` — passthrough TIPIS ke `pipeline.run_pipeline()` (SAMA
  persis argumen, TIDAK ada logika baru). **Diverifikasi (bukan diasumsikan)**:
  `extractors.run("v18", doc_path)` vs `pipeline.run_pipeline(doc_path)`
  langsung, dibandingkan field-per-field (`fields`/`raw_results`/
  `choice_groups`/`alignment_status`) — **MATCH persis**, jalur V18 SUDAH
  terbukti tidak berubah perilakunya sama sekali.
- `run_gemini(...)`/`run_mistral(...)` — baca BYTES dokumen asli LANGSUNG
  (tanpa `preprocessing.load_document`/alignment/ROI/PaddleOCR sama sekali —
  itu justru poinnya: mengukur ekstraksi AI NATIVE), panggil API dgn
  instruksi ekstraksi TUNGGAL yg SAMA (`EXTRACTION_INSTRUCTION`, verbatim dari
  spesifikasi) + structured JSON output native (BUKAN free-text+regex),
  parse ke skema kanonik, lalu `adapt_common_to_pipeline_shape()` membangun
  return-shape yg SAMA PERSIS dgn `pipeline.run_pipeline()`
  (`fields`/`raw_results`/`choice_groups`/`debug_images`/`alignment`/dst) —
  dgn REUSE `postprocessing.build_fields_table()` (TIDAK diimplementasi
  ulang). **`reference_record` TIDAK PERNAH masuk ke signature
  `run_gemini`/`run_mistral` sama sekali** — kebocoran data referensi scr
  STRUKTURAL tidak mungkin terjadi utk kedua engine ini (beda dgn `run_v18`
  yg TETAP menerima param itu krn `pipeline.run_pipeline()` sendiri sudah
  lama memakainya utk `_run_vlm_reference_check`, fitur V10 yg TIDAK terkait
  sesi ini).
- `ExtractorError(message, stage)` — `stage` salah satu `API_AUTH`/
  `API_RATE_LIMIT`/`API_TIMEOUT`/`API_RESPONSE`. TIDAK PERNAH ditelan diam2,
  TIDAK PERNAH memicu fallback ke engine lain — cukup propagate, KEDUA
  pemanggil (`app.py`'s `process_document`/`_process_one_record`, SUDAH ADA
  sblm sesi ini) SUDAH membungkus panggilan dgn `try/except Exception ->
  sanitize_error(exc)` generik, jadi kegagalan API otomatis tertangkap TANPA
  kode error-handling baru di `app.py`.

**Adapter design decision (dokumentasikan jujur, JANGAN disembunyikan)**:
`comparison.validate_tenor()`/`validate_reward()` didesain utk FORM KERTAS
("2 bukti independen harus sepakat" — coretan pilihan + rentang tanggal
tulisan tangan utk tenor; coretan pilihan + field detail terisi utk reward) —
TIDAK map 1:1 ke satu respons JSON AI.
- **Tenor — reuse JUJUR, TANPA fabrikasi**: skema kanonik SUDAH punya 2 sinyal
  independen dari respons AI yg SAMA — `tenor_penempatan` (angka) &
  `tanggal_mulai`/`tanggal_selesai` (tanggal). Adapter format 2 tanggal itu
  jadi teks (mis. "1 Januari 2026 s/d 1 April 2026") yg **diverifikasi
  parseable** oleh `postprocessing.derive_tenor_from_range()`/
  `_extract_dates()` yg SUDAH ADA (REUSE, bukan ditulis ulang) — kalau angka
  tenor & rentang tanggal versi AI sendiri TIDAK sepakat, `validate_tenor`
  BENAR menandai TOLAK (menangkap inkonsistensi internal AI sungguhan, bukan
  workaround).
- **Reward — simplifikasi jujur, BUKAN fabrikasi data baru**: skema kanonik
  cuma punya SATU nilai `bentuk_reward` (tidak ada field detail terpisah spt
  form kertas), tapi `validate_reward` butuh sinyal KEDUA "field detail
  terisi". Adapter mencerminkan nilai `bentuk_reward` yg SAMA ke slot
  `reward_tunai`/`reward_non_tunai` yg sesuai (BUKAN data baru, cuma
  menaruh nilai yg SAMA di slot ke-2 yg diminta check lama) supaya check
  tidak salah melaporkan REVIEW utk SETIAP record engine API. **Ini
  simplifikasi yg diketahui & disengaja** — tidak ada sinyal independen
  kedua yg genuine utk ekstraksi AI seluruh-dokumen, dicatat di komentar
  `extractors.py` (`adapt_common_to_pipeline_shape`'s docstring) & di sini.

**Verifikasi (SEMUA dijalankan, bukan diasumsikan — lihat catatan jujur di
bawah soal yg BELUM bisa diverifikasi):**
1. Syntax + import check `extractors.py`/`app.py`/`evaluation.py` — semua OK.
2. `extractors.run("v18", doc_path)` vs `pipeline.run_pipeline(doc_path)`
   langsung, record 12 — **MATCH persis** (lihat di atas).
3. Adapter end-to-end: JSON kanonik sintetis (`nama_nasabah`/`nomor_rekening`/
   dst lengkap) → `adapt_common_to_pipeline_shape()` → `comparison.
   _compute_all_checks()` LANGSUNG — SEMUA 5 field OK/checks bekerja benar,
   tenor cross-check (angka vs rentang tanggal) BERHASIL sepakat (3 bulan =
   3 bulan), signature "uncertain" BENAR jadi REVIEW. Membuktikan integrasi
   adapter->comparison.py bekerja, bukan cuma "tidak crash".
4. `extractors.run("gemini-3.8-flash", ...)`/`("mistral-ocr-4-1", ...)` TANPA
   `GEMINI_API_KEY`/`MISTRAL_API_KEY` (env kosong) — KEDUANYA raise
   `ExtractorError(stage="API_AUTH")` dgn pesan jelas, BUKAN crash/stack
   trace mentah.
5. `python evaluation.py --engine gemini-3.8-flash --limit 1` &
   `--engine mistral-ocr-4-1 --limit 1` (tanpa API key) — KEDUANYA selesai
   TANPA crash, `failure_stage_breakdown.API_AUTH == 1`, summary JSON valid
   (`"engine"` tercatat benar).
6. `python evaluation.py --engine v18 --limit 1` (SETELAH restrukturisasi
   `evaluate_record`) — `pipeline_success_rate=1.0`/`alignment_success_rate=
   1.0` SAMA spt sebelum sesi ini (record 1, hasil field akurasi SAMA dgn
   histori) — membuktikan cabang `if engine=="v18"` di `evaluate_record`
   (ALIGNMENT/LOCALIZATION/`pipeline_module.run_pipeline` verbatim, TIDAK
   diubah) genuinely tidak ter-regresi oleh restrukturisasi jadi if/else.
7. JS: syntax-check via Node (`new Function(code)`) + HTML tag-balance check
   utk `static/index.html` — keduanya OK.

**BELUM diverifikasi (jujur, TIDAK diklaim selesai)**: TIDAK ADA
`GEMINI_API_KEY`/`MISTRAL_API_KEY` nyata tersedia sesi ini — panggilan API
SUNGGUHAN (`client.models.generate_content(...)` utk Gemini,
`client.ocr.process(...)` utk Mistral) **belum pernah benar2 dieksekusi**.
Bentuk panggilan SDK persis (nama method, kwarg) didasarkan pengetahuan
SDK `google-genai`/`mistralai` versi terkini SAAT PENULISAN — DIBERI KOMENTAR
JELAS di `extractors.py` (`run_gemini`/`run_mistral`) utk diverifikasi ulang
thd versi package yg BENAR2 terpasang sebelum dipercaya di produksi, krn
`google-genai`/`mistralai` tidak sempat diinstall (tidak ada akses jaringan
sesi ini utk `pip index`/`pip install` — lihat `requirements.txt`, versi
sengaja TIDAK di-pin krn tidak bisa diverifikasi thd PyPI). UI selector
(Tab 1/Tab 2, gate Run+disable saat processing) diverifikasi lewat
pembacaan kode + syntax/tag-balance check, **BUKAN** lewat klik manual di
browser sungguhan — kalau ada waktu sesi depan, jalankan `app.py` &
klik-test langsung.

**File yg diubah/baru**: `extractors.py` (BARU — router + adapter + skema
kanonik + 2 engine API), `requirements.txt` (+`google-genai`/`mistralai`/
`python-dotenv`, TIDAK di-pin, lihat catatan di atas), `.env.example`
(BARU), `.gitignore` (+`.env`), `app.py` (`_run_ocr_and_format` route lewat
`extractors.run()` bukan `pipeline.run_pipeline()` langsung; `ProcessPayload`/
`SheetSubmitPayload.engine` baru; `_new_session`/`submit_sheet_batch`/
`_process_one_record` thread `engine` lewat; beberapa `_ERROR_HINTS` baru),
`static/index.html` (`engineSelect`/`sheetEngineSelect` baru, `setTab1RunState`/
`setProcessingState` gate Run+selector pd pilihan extractor, `runRecords`/
Tab1 fetch kirim `engine`), `evaluation.py` (`--engine` CLI, `evaluate_record`
di-restrukturisasi if/else per engine SAMBIL mempertahankan cabang v18
verbatim, `FAILURE_STAGES` +4 stage API, `_blank_result`/summary +`"engine"`).

**TIDAK disentuh (sesuai scope eksplisit)**: `comparison.py` (nol baris
diubah), `signature_diagnostics.py`/`stage_evaluation.py` (hardcode
`pipeline.run_pipeline`, di luar scope benchmark (spesifikasi tugas V19 §15) — HANYA
`evaluation.py --engine` yg diminta), semua masalah V18 yg sudah ada (§10,
tidak ada satupun di-fix sesi ini).

**Update (sesi lanjutan, `GEMINI_API_KEY`/`mistralai` SUNGGUHAN sudah
tersedia) — 2 bug nyata ketemu PERSIS di titik yg sudah diperingatkan jujur
di atas ("verifikasi ulang thd versi package yg BENAR2 terpasang"):**

1. **`run_gemini` crash `pydantic ValidationError` (9 errors) di
   `client.models.generate_content(...)`.** Root cause: `response_schema`
   divalidasi google-genai sbg `types.Schema` (Pydantic, OpenAPI-style) —
   field `type`-nya enum TUNGGAL (`STRING`/`NUMBER`/`INTEGER`/.../`NULL`),
   TIDAK BISA menerima union JSON-Schema standar (`"type": ["string",
   "null"]`, yg dipakai `_response_json_schema()` utk SETIAP field Optional).
   **Fix**: pakai `response_json_schema` (bukan `response_schema`) di
   `config={}` — field ini dikonfirmasi ADA di `GenerateContentConfig`
   (`google-genai` 2.22.0), diteruskan APA ADANYA lewat `t_json_schema()`
   (no-op passthrough, dibaca langsung dari source `_transformers.py`) —
   BUKAN divalidasi sbg `types.Schema` sama sekali, jadi dialek JSON-Schema
   standar (termasuk union nullable) valid tanpa konversi apa pun. Skema
   asli `_response_json_schema()` dipakai APA ADANYA utk Gemini juga
   sekarang (sebelumnya sempat ada fungsi konversi `_to_gemini_schema()` yg
   me-rewrite ke dialek `nullable: true` — SUDAH DIHAPUS, jadi dead code
   sejak `response_json_schema` dipakai; Mistral TIDAK terpengaruh, sudah
   dari awal pakai dialek JSON-Schema standar yg sama).
2. **`run_mistral` crash `ImportError: cannot import name 'Mistral' from
   'mistralai'`.** Root cause: `mistralai` 2.x (yg ter-install dari
   `mistralai>=1.0` di `requirements.txt`, resolve ke 2.10.0) merestrukturisasi
   paket — `mistralai` level atas jadi PURE NAMESPACE PACKAGE (tanpa
   `__init__.py`, `mistralai.azure`/`mistralai.gcp` jadi varian cloud-specific
   sejajar), class `Mistral` pindah ke `mistralai.client` & TIDAK di-re-export
   lagi di level atas — import lama (`from mistralai import Mistral`, valid
   di 1.x) diam2 resolve ke namespace kosong. **Fix**: `from mistralai.client
   import Mistral`. `client.ocr.process(...)`/`document_annotation_format`
   dikonfirmasi masih kompatibel (signature method dicek via `inspect.
   signature`, key `"schema"` dlm payload dikonfirmasi match lewat Pydantic
   alias `schema_definition` punya `alias="schema"` — TIDAK perlu diubah).

**Verifikasi fix ini**: `types.GenerateContentConfig(response_mime_type=...,
response_json_schema=schema)` dibangun tanpa error (dicek langsung, bukan
diasumsikan) — SEBELUM & SESUDAH `pip install --upgrade google-genai
pydantic` (hasil: google-genai TETAP 2.22.0 [sudah terbaru], pydantic
2.13.4→2.13.5). End-to-end SUNGGUHAN dicoba (`extractors.run_gemini(doc_path,
model="gemini-2.5-flash")` thd record nyata dari `assets/ocr_evaluation.xlsx`,
`GEMINI_API_KEY` asli terbaca dari `.env`) — **schema error TIDAK MUNCUL
LAGI**, request berhasil dibangun & benar2 dikirim (lolos sampai lapisan
HTTP), TAPI gagal di `httpx.ConnectError: [Errno 11001] getaddrinfo failed`
— **sandbox dev environment sesi ini TIDAK PUNYA akses internet keluar**
(sama kategori limitasi dgn VLM/torch yg sudah dicatat berkali-kali di
handover versi sebelumnya) — BUKAN bug kode, respons SUNGGUHAN dari Gemini
belum bisa diverifikasi round-trip penuh di sini. **Jangan klaim "Gemini
sudah jalan end-to-end"** sampai dicoba di environment dgn akses internet.

**Item yg SENGAJA TIDAK diubah (diminta, tapi premisnya tidak cocok kode
nyata — dicek dulu, bukan dituruti buta)**:
- Tidak ada `ExtractionResult` (Pydantic model) di mana pun di codebase ini
  — skema dibangun manual sbg dict via `_response_json_schema()`. Parsing
  `json.loads(response.text)` (sudah ada) tetap dipakai, BUKAN diganti
  `ExtractionResult.model_validate_json(...)` — tidak ada model itu utk
  divalidasi. Kalau validasi tipe-kuat via Pydantic memang diinginkan, itu
  perubahan desain terpisah (bikin model baru + migrasi adapter), bukan
  bagian dari fix bug ini.
- `_classify_api_exception` SUDAH BENAR (dicek ulang): hanya keyword auth
  eksplisit (`unauthorized`/`invalid api key`/`401`/`403`/dst) yg
  menghasilkan `API_AUTH` — error skema/response SUDAH jatuh ke
  `API_RESPONSE` (bukan `API_AUTH`) SEBELUM sesi ini. Traceback yg
  menunjukkan `"GEMINI_API_KEY environment variable is not set"`
  berdampingan dgn error skema kemungkinan besar hasil paste 2 run TERPISAH
  yg tercampur (kode-nya cuma raise pesan itu di 1 titik, SEBELUM panggilan
  API apa pun, independen dari error skema) — bukan bug klasifikasi nyata.
- `ocr_meta["engine"] = "v18"` yg dicurigai ada di `run_gemini` — DICEK,
  baris itu ADA tapi di dalam `run_v18()` (§9 poin `run_v18`), BUKAN
  `run_gemini`. `run_gemini` SUDAH benar via `adapt_common_to_pipeline_shape`
  (`ocr_meta = {"engine": source, ...}` dgn `source=model`, mis.
  "gemini-2.5-flash") — granularitas per-MODEL ini SENGAJA (tujuan V19 adl
  benchmark ANTAR MODEL, bukan cuma antar provider), bukan bug.

## 9b. V19 (lanjutan) — Qwen2-VL Direct-Semantic Pipeline + Gemini Token/Cost Tracking (SELESAI, terverifikasi sebagian)

Lanjutan §9 di atas (engine router/adapter/canonical schema SUDAH ada dari
sesi sebelumnya, TIDAK diubah strukturnya) -- menambahkan 2 hal terpisah
diminta user: (1) `qwen2_vl` sbg engine ke-5 (PRIMARY EXPERIMENTAL, direct
semantic, TANPA ROI/PaddleOCR/label coordinate -- beda total dari V10/V12
VLM FALLBACK yg sudah ada di `pipeline.py`/`vlm.py`, itu tetap tidak
disentuh), (2) Gemini token usage + cost tracking (config.py baru + UI +
export).

**File baru:** `config.py` (Gemini pricing/USD-IDR rate, MUDAH diupdate,
lihat komentar di file itu -- angka BELUM diverifikasi thd halaman pricing
resmi Gemini terkini, sandbox sesi ini sempat tanpa akses jaringan lalu
TERNYATA ada akses -- lihat poin torchvision di bawah -- tapi harga Gemini
sendiri tidak dicek ulang thd dokumentasi resmi, cuma best-effort).

**1. Qwen2-VL primary engine (`extractors.run_qwen_vlm`, `vlm.
extract_direct_semantic`).** Alur: `extractors._qwen_prepare_image()`
(PDF->image via `prep.load_document` yg SUDAH ada -- reuse, TIDAK
alignment; foto: `PIL.ImageOps.exif_transpose` utk EXIF orientation, TANPA
deskew/threshold/binarize/crop) -> `vlm.extract_direct_semantic()` (SATU
panggilan model, prompt PERSIS spesifikasi task + skema JSON `{value,
confidence}` per field dilampirkan supaya model consistent -- model TIDAK
support constrained/structured JSON decoding native spt Gemini, jadi
skema dilampirkan sbg teks, pola yg SAMA dgn `INDEPENDENT_READ_PROMPT_
TEMPLATE`/`FULLPAGE_PROMPT_TEMPLATE` yg sudah ada) -> `extractors.
_adapt_qwen_to_common()` (map ke skema kanonik `COMMON_FIELDS` yg SAMA dgn
Gemini/Mistral, field yg TIDAK ada di skema Qwen -- unit_kerja/tanggal_
mulai/tanggal_selesai -- diisi not_detected/null, BUKAN ditebak) ->
`adapt_common_to_pipeline_shape()` (REUSE PERSIS, tanpa cabang baru).

**Fallback (task spec §6, SATU-SATUNYA pengecualian thd prinsip "1 engine
per panggilan, tidak pernah fallback diam2 antar engine" §9 di atas):**
`run_qwen_vlm` menangkap SEMUA exception dari persiapan gambar ATAU
inferensi Qwen (model/dependensi tidak ada, JSON tidak bisa di-parse, dll)
-> panggil `run_v18()` (pipeline OCR/ROI yg SUDAH ADA, TIDAK diubah) sbg
fallback, TIDAK PERNAH ke Gemini/Mistral, TIDAK PERNAH dijalankan
mendahului Qwen. Label fallback (`ocr_meta["fallback_source"]`) dipetakan
dari `source` per-field yg SUDAH ADA di `pipeline.py` (`ocr_primary`/
`vlm_fullpage` -> "OCR Fallback", `roi_template`/`roi_second_pass_vlm` ->
"ROI Fallback") -- perkiraan KASAR (SATU label utk seluruh dokumen, bukan
per-field), didokumentasikan sbg itu, bukan klaim presisi granular. Kalau
Qwen SUKSES, `fallback_source = "Qwen VLM"`.

**Batasan jujur, disengaja (BUKAN bug, DIDOKUMENTASIKAN di kode
`extractors._adapt_qwen_to_common` docstring juga):** skema wajib Qwen
(task spec §2) TIDAK punya tanggal_mulai/tanggal_selesai -- akibatnya
`comparison.validate_tenor()` (TIDAK diubah, tetap butuh 2 bukti independen
choice+rentang tanggal) SELALU jatuh ke cabang "rentang tanggal tidak
terbaca" -> `tenor_penempatan` **SELALU REVIEW** utk engine `qwen2_vl`,
terlepas seberapa yakin/benar choice number-nya. Ini konsekuensi nyata dari
skema Qwen yg fixed sesuai spesifikasi (sama kategori dgn simplifikasi
Reward Gemini/Mistral di §9), BUKAN sesuatu yg "diperbaiki" dgn
mengarang rentang tanggal yg Qwen tidak pernah diminta ekstrak.

**Verifikasi NYATA (bukan cuma baca kode -- dijalankan sungguhan sesi
ini):**
1. Model lokal (`models/Qwen2-VL-2B-Instruct/`, weights LENGKAP ada di
   mesin ini) berhasil di-load LEWAT `vlm._load()` yg SUDAH ADA (reuse,
   TIDAK diubah loading logic-nya) -- **SETELAH fix dependency nyata**:
   `transformers` 5.17.0 di venv ini butuh `torchvision` (import error
   `Qwen2VLVideoProcessor requires the Torchvision library`, muncul saat
   `AutoProcessor.from_pretrained` resolve video-processor Qwen2-VL walau
   TIDAK ada video yg dipakai) -- **TERNYATA sandbox sesi ini PUNYA akses
   jaringan** (beda dari kondisi "tidak ada akses internet" yg dicatat sesi
   V19 sebelumnya utk Gemini/Mistral, lihat §9 di atas -- kemungkinan
   kondisi jaringan environment berubah antar sesi, JANGAN asumsikan salah
   satu kondisi tanpa cek ulang), `pip install torchvision` berhasil ->
   ditambahkan ke `requirements.txt` (`torchvision>=0.19`, dgn komentar
   penjelasan root cause).
2. Inferensi SUNGGUHAN dijalankan (`extractors.run_qwen_vlm` end-to-end,
   `downloaded_documents/13EQ77C6tOZoWZ-vN4HEmn6BO7qsB0hNB`) -- **CPU-only
   (mesin sandbox ini TIDAK punya CUDA, beda dari environment produksi yg
   punya GPU MX230 2GB, lihat §14), elapsed 1640 detik (~27 menit) utk SATU
   dokumen** -- jauh lebih lambat drpd PaddleOCR/ROI (~11 detik/dokumen,
   §2). **CATATAN PENTING utk sesi depan**: JANGAN coba jalankan
   `evaluation.py --engine qwen2_vl` 25-dokumen penuh di mesin CPU-only spt
   ini tanpa estimasi waktu dulu (~27 menit x 25 = ~11+ jam) -- kalau perlu
   benchmark akurasi Qwen scr formal, jalankan di mesin dgn GPU dulu.
3. Model MENGHASILKAN JSON valid & bisa di-parse (`vlm._extract_json`
   berhasil, termasuk membuang markdown code-fence \`\`\`json yg
   dihasilkan model) -- TAPI menjawab **flat** (`{"nama": null, ...}`),
   BUKAN wrapped `{"value":.., "confidence":..}` spt yg diminta skema di
   prompt. **Bug NYATA ditemukan & diperbaiki dari temuan ini**:
   `extract_direct_semantic` sebelumnya HANYA menerima bentuk wrapped
   (`isinstance(item, dict)`), kalau model menjawab flat scalar maka value
   asli DIBUANG diam2 jadi `None`/confidence 0 walau model sebenarnya
   menjawab sesuatu yg valid (mis. angka/boolean). Fix: fallback ke value
   flat (pola yg SAMA dgn `extract_fields_independent`/`extract_fields_
   fullpage` yg sudah ada di file yg sama) kalau bentuknya bukan dict --
   confidence tetap 0.0 utk kasus ini (model tidak memberi angka self-report
   sendiri). Diverifikasi via unit test (mock `_infer`, bukan re-run
   inferensi 27 menit lagi) utk kedua bentuk (flat DAN wrapped) -- keduanya
   sekarang benar.
4. Utk dokumen NYATA yg diuji, model menjawab **SEMUA field null/false**
   (genuinely uncertain menurut model sendiri, sesuai instruksi prompt "use
   null when uncertain, do not guess" -- BUKAN crash/parsing gagal).
   **Jujur, BELUM disimpulkan** apakah ini krn (a) resolusi turun ke
   `MAX_SIDE_FULL=1600` (sama dgn V12 fullpage fallback yg sudah ada)
   terlalu kecil utk tulisan tangan dokumen ini scr spesifik, (b) model 2B
   genuinely terlalu kecil utk baca dokumen form Indonesia penuh dlm SATU
   pass tanpa ROI/crop (ITU JUSTRU pertanyaan eksperimen utama V19, lihat
   §1 spesifikasi task), atau (c) idiosinkrasi 1 dokumen ini saja. **Sample
   size = 1 dokumen** -- BUKAN kesimpulan benchmark akurasi, cuma bukti
   pipeline-nya BEKERJA end-to-end tanpa crash. Benchmark akurasi formal
   (`evaluation.py --engine qwen2_vl` full 25 dokumen) BELUM dijalankan
   (lihat poin waktu di atas) -- prioritas sesi depan kalau ada akses GPU.
5. Hasil model (SEMUA null, dari poin 4) diverifikasi end-to-end lewat
   `extractors.run("qwen2_vl", ...)` PENUH (pakai `_qwen_prepare_image`
   ASLI thd dokumen nyata, `vlm._infer` di-mock supaya tidak re-run 27
   menit) -> `adapt_common_to_pipeline_shape` -> `comparison.
   compute_decision()` -- **TIDAK crash**, SEMUA field jatuh ke status yg
   BENAR (`not_detected`/`review`), decision akhir **REVIEW** dgn alasan
   per-field yg masuk akal ("Nama Nasabah tidak terbaca dari dokumen",
   dst) -- BUKAN salah dipaksa OK/TOLAK. Membuktikan integrasi validasi
   bekerja BENAR walau model-nya sendiri tidak yakin.
6. Jalur fallback (Qwen gagal total -> v18) diverifikasi terpisah (mock
   `vlm.extract_direct_semantic` raise exception) -- `run_v18()` terpanggil
   benar, `ocr_meta["fallback_source"]` terisi label yg masuk akal,
   `primary_engine_attempted`/`primary_engine_error` tercatat.
7. Regresi Gemini/Mistral: `extractors.run("gemini-2.5-flash"/
   "mistral-ocr-4-1", ...)` TANPA API key -- KEDUANYA tetap
   `ExtractorError(stage="API_AUTH")` yg bersih, SAMA spt sebelum sesi ini
   (`_field_entry`/`put()` yg diubah utk menyalurkan confidence Qwen TIDAK
   mengubah perilaku Gemini/Mistral -- `.get("confidence")` mengembalikan
   `None` sesuai desain, dicek scr eksplisit).
8. `run_v18()` (fungsi itu sendiri) **TIDAK disentuh sama sekali** sesi ini
   (dicek lewat riwayat edit sesi ini, BUKAN git diff -- `extractors.py`
   sendiri masih untracked dari sesi sebelumnya jadi `git diff` thd file
   itu tidak menunjukkan apa2 scr default). **Dicoba** membandingkan
   `extractors.run("v18", doc)` vs `pipeline.run_pipeline(doc)` langsung
   (2x eksekusi PENUH pipeline V18+VLM fallback internal pd dokumen yg
   sama) -- `raw_results` MATCH persis, TAPI `fields` (list) TIDAK persis
   sama. **Proses kedua sempat di-kill sistem krn low memory** (mesin ini
   ~8GB RAM, SUDAH didokumentasikan §10 poin 7/§12 -- JANGAN ulangi
   pola ini, 2 instance PaddleOCR+VLM sekaligus). Diagnosis TIDAK
   dituntaskan (butuh proses ke-3 yg dibatalkan krn risiko OOM), tapi
   `raw_results` (dict, sumber data mentah) MATCH persis mengindikasikan
   ini kemungkinan besar **non-determinisme run-to-run yg SUDAH
   didokumentasikan sebelumnya** (§8/§10 poin 12: "kemungkinan efek tidak
   langsung...atau non-determinisme OCR/GPU") pada pipeline V18 ITU
   SENDIRI (dipicu jalur VLM fallback internal V10/V12 yg SUDAH ada,
   `vlm_fullpage_fallback.used=True` pd dokumen ini) -- **BUKAN regresi
   dari sesi ini** (`run_v18()` verbatim tidak diubah, dicek di atas).
   **Belum dituntaskan jujur** -- kalau ada waktu sesi depan & RAM cukup,
   ulangi dgn 2 proses Python TERPISAH (bukan 1 proses 2x panggilan, pola
   §12) supaya lebih aman scr memori, lalu diff field-per-field spt yg
   sempat dimulai sesi ini.

**File yg diubah/baru (sesi ini):**
- `config.py` (BARU) -- `GEMINI_PRICING`/`GEMINI_DEFAULT_PRICING`/
  `USD_IDR_RATE`.
- `vlm.py` -- `DIRECT_SEMANTIC_FIELDS`/`DIRECT_SEMANTIC_PROMPT`/
  `_clean_confidence`/`extract_direct_semantic` (BARU, lihat poin 3 di atas
  utk bugfix flat-vs-wrapped).
- `extractors.py` -- `_field_entry`/`put()` diperluas thread confidence
  (backward-compatible, Gemini/Mistral tetap `None`); `_compute_gemini_cost`
  (BARU, baca `config.py`); `run_gemini` capture `response.usage_metadata`
  -> `result["gemini_usage"]` (top-level, HANYA ada kalau Gemini beneran
  dipanggil); `QWEN_FALLBACK_SOURCE_LABELS`/`QWEN_UNCERTAIN_CONFIDENCE`/
  `_qwen_prepare_image`/`_adapt_qwen_to_common`/`run_qwen_vlm` (BARU);
  `ENGINE_LABELS`/`ENGINES` +`"qwen2_vl"`; router `run()` skrg assign lalu
  return SATU kali (bukan return langsung tiap cabang) supaya bisa stamp
  `ocr_meta["version"]="v19"` utk SEMUA engine termasuk v18 (task spec §5).
- `app.py` -- `_run_ocr_and_format` return `gemini_usage`/`fallback_source`
  baru; fallback source Qwen di-append ke `decision_v9_2["notes"]` (HANYA
  informational, TIDAK mengubah decision -- `comparison.py` nol perubahan);
  `_compute_session_gemini_summary` (BARU) diselipkan ke response `/api/
  sheet/batch-status/{id}` sbg `gemini_usage_summary`; `/api/sheet/export/
  {id}` +9 kolom (`engine`/`model`/`input_tokens`/`output_tokens`/
  `thinking_tokens`/`total_tokens`/`cost_usd`/`cost_idr`/`fallback_source`,
  blank/0 utk record non-Gemini).
- `static/index.html` -- opsi `qwen2_vl` di KEDUA dropdown extractor (Tab 1
  & Tab 2); panel `.gemini-usage-box` (BARU, CSS+JS) tampil HANYA kalau
  `gemini_usage`/`gemini_usage_summary` ada (Tab 1 per-dokumen, Tab 2
  per-record + total sesi via polling `batch-status`).
- `requirements.txt` -- `+torchvision>=0.19` (fix nyata, lihat poin 1 di
  atas), komentar VLM section diperbarui sebut V19 Qwen direct-semantic jg
  pakai dependency yg sama.

**TIDAK disentuh (scope eksplisit, sama spt §9):** `comparison.py`
(nol baris), `pipeline.py`/`postprocessing.py`/`preprocessing.py` (nol
baris -- V18 internals tetap utuh), `stage_evaluation.py`/`signature_
diagnostics.py` (hardcode `pipeline.run_pipeline`, di luar scope).
`evaluation.py` **TIDAK perlu diubah sama sekali** utk dukung `qwen2_vl`
-- cabang `else` (non-v18) di `evaluate_record` SUDAH generic (panggil
`extractors.run(engine, doc_path)` langsung), `--engine` CLI choices sudah
otomatis ambil dari `extractors.ENGINES` (tuple, skrg 5 entri).

**Prioritas sesi depan (V19 lanjutan, urutan berdasar temuan sesi ini):**
1. Benchmark akurasi Qwen formal (`evaluation.py --engine qwen2_vl`) di
   mesin ber-GPU -- 1 sample (poin 4 verifikasi) TIDAK CUKUP utk simpulkan
   apa pun soal akurasi, cuma bukti pipeline jalan.
2. Kalau akurasi Qwen rendah scr sistematis stlh benchmark formal, coba
   naikkan resolusi (`vlm.MAX_SIDE_FULL`, env var `VLM_MAX_SIDE_FULL`
   SUDAH ada, tinggal ubah env, TIDAK perlu ubah kode) SEBELUM ubah prompt
   -- ukur dulu efeknya scr terpisah drpd gabung banyak perubahan sekaligus
   (prinsip §15).
3. Tuntaskan diagnosis "fields match: False, raw_results match: True" utk
   `run_v18` (poin 8 verifikasi di atas) dgn 2 proses Python TERPISAH
   (bukan 1 proses 2x panggilan) supaya aman dari OOM -- kemungkinan besar
   cuma non-determinisme lama yg sudah didokumentasikan, tapi belum
   dikonfirmasi tuntas root cause-nya field mana yg beda.

## 9c. V19c — Ganti Qwen2-VL LOKAL (§9b) dgn Qwen3-VL HOSTED via Hugging Face
(SELESAI, terverifikasi 2 dokumen)

User eksplisit minta engine "qwen3_vl" (dulu "qwen2_vl", §9b) TIDAK LAGI
memuat model lokal (torch/transformers/`vlm._load()`) -- diganti SATU
panggilan HTTP ke `Qwen/Qwen3-VL-2B-Instruct` yg di-host Hugging Face,
via `huggingface_hub.InferenceClient`. **Beda desain sengaja dari §9b**: engine
ini TIDAK LAGI fallback diam-diam ke `run_v18` kalau gagal -- error asli
(auth/timeout/bad response) langsung dilempar sbg `ExtractorError`, sesuai
instruksi eksplisit "no silent fallback for this test".

**File diubah:**
- `.env.example`/`requirements.txt` -- `HF_TOKEN`/`QWEN_MODEL`/`HF_PROVIDER`
  (opsional) + `huggingface_hub>=0.26` (TIDAK butuh torch/transformers utk
  jalur ini -- itu tetap dipakai HANYA oleh V10/V12 VLM fallback lokal yg
  ADA di dalam `pipeline.py`, tidak disentuh sesi ini).
- `config.py` -- `QWEN_MODEL_DEFAULT`/`HF_REQUEST_TIMEOUT_S` (60 detik,
  jangan biarkan proses menggantung lama -- sesuai instruksi).
- `vlm.py` -- `extract_direct_semantic_hosted()` (baru): encode gambar
  halaman penuh jadi JPEG data-URI, kirim via `InferenceClient.
  chat_completion()` pakai prompt/skema `DIRECT_SEMANTIC_PROMPT`/
  `DIRECT_SEMANTIC_FIELDS` YANG SAMA dgn versi lokal (parsing JSON respons
  di-share lewat `_parse_direct_semantic_json()`, factored out dari
  `extract_direct_semantic()` lama supaya kedua jalur identik cara baca
  JSON-nya). `extract_direct_semantic`/`_load`/`_infer` versi lokal
  DIBIARKAN ada (tidak dipanggil lagi dari `extractors.py`, didokumentasikan
  sbg "superseded" di docstring-nya) -- bukan dihapus, supaya perubahan tetap
  minimal/reversible.
- `extractors.py` -- `run_qwen_vlm` (lokal, §9b) diganti
  `run_qwen3_vl_hosted()`; engine id `"qwen2_vl"` -> `"qwen3_vl"` di
  `ENGINE_LABELS`/dispatch `run()` supaya hasil jelas ter-stamp
  `source`/`engine` = `"qwen3_vl"` (sesuai permintaan "clearly identify the
  result as qwen3_vl"). `QWEN_FALLBACK_SOURCE_LABELS` (dead code stlh
  fallback dihapus) ikut dihapus.
- `app.py`/`static/index.html` -- update komentar/label dropdown yg
  menyebut `qwen2_vl`/"Qwen VLM" jadi `qwen3_vl`/"Qwen3-VL Hosted".

**Bug nyata ditemukan & diperbaiki dari testing SUNGGUHAN (bukan baca
dokumentasi HF)**: `InferenceClient(provider=None)` (default "auto" routing)
GAGAL utk `Qwen/Qwen3-VL-2B-Instruct` dgn error `"not supported by any
provider you have enabled"` -- dicek via `HfApi().model_info(model,
expand="inferenceProviderMapping")`, model ini TERNYATA cuma di-serve oleh
SATU provider (`featherless-ai`) saat sesi ini berjalan, dan "auto" tidak
otomatis menjangkaunya. Fix: `_resolve_live_provider()` (baru, `vlm.py`) --
kalau panggilan pertama (provider="auto") gagal dgn pesan itu, query Hub API
utk provider `status="live"` yg sebenarnya, retry SATU KALI dgn provider
eksplisit itu. Ini BUKAN pelanggaran "no silent OCR/ROI fallback" (task
eksplisit cuma melarang fallback ke pipeline OCR/ROI) -- masih Qwen, masih
Hugging Face hosted, dicatat programmatically (`ocr_meta.qwen_hosted_request.
provider`) & lewat `print()`. Provider bisa dipaksa manual via env
`HF_PROVIDER` kalau HF berubah lagi ke depannya (jangan hardcode
`"featherless-ai"` di kode -- itulah kenapa fix-nya lookup dinamis, bukan
konstanta).

**Verifikasi NYATA (2 dokumen dari `downloaded_documents/`, foto + PDF, TIDAK
full 25 -- sesuai instruksi eksplisit user "test with 1-3 data, dont use all
the data"):**
1. Tanpa `HF_TOKEN`: `extractors.run("qwen3_vl", ...)` melempar
   `ExtractorError(stage="API_AUTH", "HF_TOKEN environment variable is not
   set")` SEBELUM panggilan jaringan apa pun -- TIDAK fallback ke `run_v18`.
2. Dgn `HF_TOKEN` asli, 1 dokumen foto (`13EQ77C6...`): request sukses
   (elapsed ~9.2s stlh retry provider), respons JSON valid ter-parse,
   ter-map ke `COMMON_FIELDS` yg SAMA dgn Gemini/Mistral/Qwen2-VL lokal
   (`source`/`engine` = `"qwen3_vl"` di SEMUA row `raw_results` +
   `ocr_meta`), nama/nomor rekening/nominal/reward/kedua tanda tangan
   terbaca `"detected"`/`"present"`. `tenor_penempatan` jatuh ke `"review"`
   krn model menjawab teks non-numerik ("3 /6 Bulan", bukan salah satu
   choice 1/3/6 murni) -- `int()` cast di `_adapt_qwen_to_common` gagal,
   fallback ke `None` -> `review` (PERILAKU YG SUDAH ADA/DIDOKUMENTASIKAN,
   sama kategorinya dgn keterbatasan tenor di §9b, BUKAN bug baru).
3. 1 dokumen PDF (`1H_AVgmXHQVoIuGOpfwzm7JeIcI1R2cz_`, lewat cabang
   `prep.load_document` di `_qwen_prepare_image`, BUKAN cabang foto):
   sukses juga (elapsed ~7.1s), model menjawab literal teks placeholder
   cetak formulir yg belum dicoret nasabah utk tenor & reward ("1-3/6 Bulan
   (*corel salah satu)", "tunai / non-tunai (*corel salah satu)") --
   kedua-duanya correctly jatuh ke `null`/`not_detected`/`review` lewat
   validasi yg SUDAH ADA (bukan salah, model genuinely tidak bisa
   membedakan mana yg dicoret di kualitas render PDF ini -- BUKAN
   disimpulkan sbg kesimpulan akurasi umum, sample size cuma 1).

**Lanjutan sesi ini -- prompt cleanup + template image (SELESAI, verifikasi
1 dokumen tambahan, record 6 yg sama):** user minta rapikan
`DIRECT_SEMANTIC_PROMPT` (bahasa Inggris, hapus rule yg redundan) & kasih
tahu model layout template asli. Perubahan:
- `DIRECT_SEMANTIC_PROMPT` ditulis ulang full Inggris + hint layout eksplisit
  (`signature_nasabah` = kotak KIRI, `signature_bri` = kotak KANAN, diambil
  presisi dari `preprocessing.SIGNATURE_CONFIG`'s value_bbox x-range, BUKAN
  tebakan). `extract_direct_semantic()` (versi lokal, sudah 0 caller sejak
  fix hosted-path sesi sebelumnya) DIHAPUS -- bukan cuma didiamkan lagi.
  `INDEPENDENT_READ_PROMPT_TEMPLATE`/`FULLPAGE_PROMPT_TEMPLATE`/`_TENOR_RULE`/
  `_SIGNATURE_RULE` (Indonesia) SENGAJA TIDAK disentuh -- itu punya sistem
  LAIN yg masih aktif (V10/V12 local VLM fallback dipakai `pipeline.py`,
  eksplisit di luar scope task ini sejak awal).
- `vlm._get_template_data_uri()` (baru): render `assets/template.pdf` KOSONG
  sekali (cache in-memory), dikirim sbg GAMBAR KEDUA (sebelum gambar dokumen
  terisi) di message hosted -- model diberi tahu eksplisit gambar 1 = blank
  reference, gambar 2 = yg harus dibaca.

**Hasil re-test record 6 (`1NlvkF3UYsoe7MePHisu85kfUs94eyTsR`, GT tenor "3
Bulan", `signature_atasan` genuinely kosong per V16):**
- `tenor_penempatan` TETAP benar (`3`) -- fix prompt sebelumnya robust thd
  perubahan ini.
- `signature_bri`/atasan: BENAR sekarang (`false`) -- match GT.
- `signature_nasabah`: SEKARANG `false` -- padahal GT (V16) bilang SEMUA
  25 dokumen punya tanda tangan nasabah asli, jadi ini SALAH (sebelumnya,
  dgn prompt-tanpa-template, nasabah benar `true` tapi atasan salah `true`
  juga). **Kesimpulan jujur: signature detection BELUM stabil di 3 varian
  yg dicoba** (selalu true -> swap kiri/kanan -> sekarang selalu false).

**Eksperimen resolusi (DICOBA & DIBUANG, jangan diulang tanpa alasan baru):**
`vlm.HOSTED_MAX_SIDE_FULL` (konstanta baru, terpisah dari `MAX_SIDE_FULL`
lokal supaya TIDAK menyentuh V10/V12 fallback) dinaikkan 1600 -> 2600 utk
jalur hosted saja (aman dicoba krn tidak lagi dibatasi CPU/RAM lokal, cuma
biaya/latency HTTP). Hasil re-test record 6 yg SAMA (temperature=0):
signature JUSTRU flip lagi (`nasabah=false` tetap salah, `bri=true` balik
salah lagi) DAN muncul regresi baru -- `nominal_penempatan` misread jadi
"Rp. 500.000" (10x lebih kecil dari 2 run sebelumnya yg konsisten ~5 juta).
**Kesimpulan: resolusi BUKAN akar masalahnya** -- pola flip-flop di
temperature=0 pada 4 varian berbeda (prompt lama, prompt+template, +res
tinggi) lebih konsisten dgn hosted serving stack yg genuinely tidak
deterministik/reliable utk penilaian visual sekecil ini, BUKAN sesuatu yg
bisa diperbaiki dgn menaikkan resolusi gambar semata. `HOSTED_MAX_SIDE_FULL`
DIKEMBALIKAN ke 1600 (default sama dgn `MAX_SIDE_FULL`) sesuai instruksi
eksplisit user setelah melihat hasil ini -- konstanta & env var
(`HF_MAX_SIDE_FULL`) TETAP ada terpisah kalau eksperimen resolusi mau
dicoba lagi nanti dgn alasan/data baru, tapi jangan diulang tanpa itu.

**Root cause SEBENARNYA (dikonfirmasi user, BUKAN tebakan lagi):** user
mencoba manual di HF Space demo Qwen3-VL yg SAMA (bukan model/provider yg
lebih besar/beda) tapi dgn gambar YANG SUDAH DI-CROP ke area relevan (bukan
halaman penuh) -- dan hasilnya BENAR. Ini mengonfirmasi: masalahnya BUKAN
resolusi halaman penuh (makanya eksperimen `HOSTED_MAX_SIDE_FULL=2600` di
atas tidak membantu), BUKAN model/provider quality -- tapi TASK FRAMING:
menyuruh model MENCARI DULU baru MENILAI tanda tangan kecil di tengah
halaman penuh jauh lebih sulit drpd menilai area yg SUDAH di-zoom ke tempat
itu. Field teks (nama/rekening/nominal/reward) tidak kena masalah ini krn
tidak butuh menemukan lokasi presisi dulu -- itu kenapa field2 itu konsisten
benar sementara tenor (sebelum fix prompt) & signature (sampai sekarang)
konsisten bermasalah -- keduanya butuh menemukan tanda/lokasi presisi kecil.

**Keputusan eksplisit user (mulanya):** 2 opsi fix ditawarkan -- (1) tambah
crop kecil khusus area tanda tangan, atau (2) model self-locate dulu baru
re-query pd crop itu. User AWALNYA memilih TETAP full-page-only, terima
signature detection tidak reliable. **User kemudian minta lanjut ("push
further")** -- lihat sub-bagian di bawah, SUDAH ADA fix yg bekerja tanpa
melanggar batasan full-page/no-ROI (tidak jadi perlu opsi 1/2 di atas).

**Grounded prompt rewrite (SELESAI, sebelum majority-vote di bawah):**
`DIRECT_SEMANTIC_PROMPT` ditulis ulang lagi, kali ini digroundkan ke bukti
visual NYATA (bukan tebakan) dgn me-render `assets/template.pdf` DAN
dokumen asli record 6 lalu di-zoom manual (`prep.load_document` + crop
manual, dilihat langsung sbg gambar). Temuan konkret:
1. Konvensi tenor/reward BUKAN "coret/circle" generik -- printed text PERSIS
   `"1 / 3 / 6 Bulan (*coret salah satu)"`, dan customer men-CORET (garis
   strike-through) angka yg TIDAK dipilih (dikonfirmasi visual: record 6
   angka "6" ada garis coret, "3" bersih -> jawaban benar "3"). Prompt lama
   cuma bilang "marked/circled/crossed out" (3 kemungkinan berbeda) -- SEKARANG
   spesifik: "look for a strike-through line ON TOP of the digit(s) NOT
   chosen, report the ONE digit with NO line through it".
2. Kotak `signature_nasabah` (kiri) TERNYATA berisi GRAFIK CETAK ("Opsional
   (Tidak Wajib) materai Rp10.000" dlm kotak bulat abu-abu) yg BUKAN tanda
   tangan -- tanda tangan asli biasanya goresan kursif KECIL yg overlap dgn
   stempel meterai fisik di kotak yg SAMA (dikonfirmasi visual record 6:
   goresan biru tipis di atas stempel Rp10.000). Prompt SEKARANG eksplisit
   suruh abaikan grafik cetak itu & cari goresan tulisan tangan asli.
   Kotak `signature_bri` (kanan) dikonfirmasi POLOS (cuma garis titik-titik
   cetak, tanpa grafik lain) -- prompt bilang ini eksplisit jg (box lbh
   sederhana drpd nasabah, BUKAN alasan knp keduanya sama sulitnya).

**Majority-vote utk signature (SELESAI, TERBUKTI BEKERJA -- fix nyata,
bukan sekadar dokumentasi limitasi lagi):** root cause instabilitas
(5 percobaan sebelumnya di record 6 yg SAMA, temperature=0, hasil signature
berubah-ubah tiap kali walau prompt makin presisi) didiagnosis BUKAN soal
kata-kata prompt lagi tapi genuinely non-determinisme di serving stack
hosted utk 2 field spesifik ini. Fix: `vlm.extract_direct_semantic_hosted_
majority()` (baru) -- panggil `extract_direct_semantic_hosted()` SEBANYAK
`HOSTED_SIGNATURE_VOTE_COUNT` kali (default 3, env `HF_SIGNATURE_VOTE_COUNT`,
ganjil spy tdk pernah seri), ambil MAJORITY VOTE HANYA utk
`signature_nasabah`/`signature_bri` (field teks/angka lain diambil dari run
PERTAMA saja -- voting exact-match utk teks tidak masuk akal krn variasi
format harmless spt "5000000" vs "Rp 5.000.000" akan merusak vote).
`extractors.run_qwen3_vl_hosted` dipindah ke fungsi ini (bukan versi single-
call lagi). Detail tiap vote (list bool per field) disimpan di
`ocr_meta.qwen_hosted_request.signature_votes` utk transparansi, +di-print
ke stdout per vote.

**Hasil verifikasi (2 dokumen, biaya 3x panggilan/API per dokumen skrg,
~21-26 detik total drpd ~7-9 detik sblmnya -- trade-off yg jelas dinaikkan
sengaja):**
- Record 6 (`1NlvkF3UYsoe7MePHisu85kfUs94eyTsR`, GT signature_atasan KOSONG,
  signature_nasabah ADA): votes nasabah=`[False,True,True]`->majority
  **True** (BENAR), votes bri=`[False,False,False]`->**False** (BENAR).
  **PERTAMA KALI kedua field benar SEKALIGUS** dari 6 percobaan di dokumen
  yg sama.
- Dokumen ke-2 (`13EQ77C6tOZoWZ-vN4HEmn6BO7qsB0hNB`, foto, GT signature tidak
  diverifikasi eksplisit tapi hasil konsisten/masuk akal): votes
  nasabah=`[True,True,True]` (unanimous), votes bri=`[True,False,False]`
  ->majority **False** -- menunjukkan mekanisme voting genuinely resolve
  disagreement per-kotak, bukan cuma kebetulan.

**Catatan jujur**: baru diverifikasi di 2 dokumen (bukan benchmark formal),
tapi ini PERTAMA KALI ada pendekatan yg benar2 mengatasi akar masalah
(non-determinisme) drpd cuma memoles kata-kata prompt. Biaya 3x lipat
latency/token per dokumen utk engine `qwen3_vl` -- worth it utk tahap test
ini, tapi kalau nanti masuk produksi sungguhan perlu dipikir ulang trade-off
biaya vs akurasi (mis. turunkan `HF_SIGNATURE_VOTE_COUNT` kalau biaya jadi
masalah, atau majority-vote SEMUA field bukan cuma signature kalau field
lain jg mulai kelihatan tidak stabil di sampel lebih besar).

**TIDAK disentuh (scope eksplisit sesuai instruksi task):**
`pipeline.py`/`preprocessing.py`/`postprocessing.py`/`comparison.py` (nol
baris), `evaluation.py` (generic, otomatis dukung `"qwen3_vl"` dari
`extractors.ENGINES` tanpa perlu diubah, SAMA spt §9b), V10/V12 VLM fallback
lokal di `vlm.py` (`_load`/`_infer`/`extract_fields_independent`/
`extract_fields_fullpage`, dipakai `pipeline._run_vlm_fallback`, TIDAK
terkait sama sekali dgn engine `qwen3_vl`).

**Belum dikerjakan (di luar scope sesi ini, eksplisit dilarang task)**:
benchmark akurasi formal `evaluation.py --engine qwen3_vl` 25-dokumen,
local Qwen deployment/quantization, multi-model selection UI.

## 9d. V19d — Tenor date-range + reward non-tunai detail (SELESAI SEBAGIAN,
kemenangan nyata + 1 keterbatasan baru terkonfirmasi & didokumentasikan)

User (lihat `vlm.py` langsung) minta 2 field yg dulu SELALU null utk
`qwen3_vl` diisi sungguhan: rentang tanggal tenor (`tanggal_mulai`/
`tanggal_selesai`) & detail barang `reward_non_tunai` (dulu cuma mirror
string "non_tunai", bukan isi barang sungguhan).

**Fix 1 -- tanggal_mulai/tanggal_selesai (BERHASIL, terverifikasi dgn
contoh nyata).** Ditambahkan ke `DIRECT_SEMANTIC_FIELDS`/`DIRECT_SEMANTIC_
PROMPT` (`vlm.py`), `extractors._adapt_qwen_to_common` diisi sungguhan
(dulu hardcode null) -- machinery derive `rentang_tenor`/`comparison.
validate_tenor` SUDAH ADA & TERBUKTI (dipakai Gemini/Mistral sejak §9),
TIDAK perlu diubah. **Ground truth dataset (`ocr_evaluation.xlsx`) record 6
blank-nya KOSONG** (dicek visual) jadi tidak bisa jadi bukti positif --
tapi record LAIN (`1a7X78Si8DTJCpf6Czq33IdnokoXoFRuA`, ditemukan via spot-
check 3 dokumen) TERNYATA terisi ("14 Agustus 2026 s/d 14 Februari 2027")
& model membaca KEDUANYA benar, PLUS otomatis konsisten scr kalender dgn
`tenor_penempatan=6` yg dibaca independen (6 bulan kalender persis) --
**bukti nyata pertama fitur ini bekerja end-to-end**, bukan cuma tidak
crash.

**Fix 2 -- reward_non_tunai_detail (DIPINDAH jadi follow-up call
KONDISIONAL, bukan bagian prompt utama -- root cause bug asli
ditemukan & diperbaiki via testing nyata, bukan tebakan).**
1. Percobaan PERTAMA: field ditambahkan LANGSUNG ke `DIRECT_SEMANTIC_
   PROMPT` utama (sama call dgn `bentuk_reward`), dgn instruksi eksplisit
   "abaikan teks contoh cetak 'iPhone 17 Pro Max...'". **GAGAL TOTAL** --
   diuji di record `1a7X78Si8DTJCpf6Czq33IdnokoXoFRuA` (GT: `tunai`,
   "tunai" DILINGKARI tanpa coretan sama sekali di baris itu): model
   menjawab `bentuk_reward="non_tunai"` (SALAH) DAN meng-copy PERSIS teks
   contoh cetak "iPhone 17 Pro Max 128 GB Warna Silver" sbg jawaban --
   diulang 2x (termasuk stlh instruksi anti-copy diperkuat lebih eksplisit
   lagi), TETAP gagal kedua-duanya.
2. **Diagnosis via eliminasi (dibuktikan, bukan tebakan)**: field
   `reward_non_tunai_detail` DIHAPUS TOTAL dari prompt utama (extractors
   tidak lagi menyebutnya sama sekali) & di-test ulang di dokumen yg SAMA
   -- `bentuk_reward` **TETAP** `"non_tunai"` (salah) di SEMUA 3 vote,
   1x elapsed. Ini MEMBUKTIKAN hipotesis "priming dari pertanyaan detail"
   SALAH -- model genuinely tidak bisa membaca lingkaran "tunai" itu scr
   independen, terlepas dari field lain apa yg ditanyakan. Kesimpulan:
   ini masalah PERSEPSI VISUAL (kualitas/resolusi gambar penuh 1600px
   menghilangkan detail lingkaran tipis), SATU KATEGORI dgn keterbatasan
   signature yg SUDAH diterima sesi sebelumnya (§9c) -- BUKAN sesuatu yg
   bisa diperbaiki dgn kata-kata prompt lagi.
3. **Percobaan generalisasi konvensi tanda (DICOBA & DIBUANG -- regresi
   nyata, jangan diulang tanpa data baru)**: rule tenor/reward diperluas
   utk terima KEDUA konvensi (coret salah satu ATAU lingkari yg benar).
   Hasil: TIDAK memperbaiki dokumen target (`bentuk_reward` tetap salah)
   DAN merusak record 6 yg SEBELUMNYA selalu benar (`bentuk_reward` jadi
   `null`/`"non tunai"` dgn spasi bukan underscore, gagal exact-match).
   **DIKEMBALIKAN ke wording strike-through-only yg terbukti reliable**
   di SEMUA dokumen lain yg pernah diuji sesi ini -- trade-off yg diterima
   SADAR: model tidak bisa baca konvensi lingkaran-saja utk reward, tapi
   itu lebih baik drpd merusak kasus yg sudah benar demi 1 dokumen edge
   case yg TETAP tidak sepenuhnya ke-fix bahkan dgn rule yg lebih rumit.
4. **Solusi akhir utk reward_non_tunai_detail (BEKERJA, arsitektur
   final)**: `vlm.REWARD_DETAIL_PROMPT` + `_extract_reward_detail_hosted()`
   (baru) -- follow-up call TERPISAH, hanya dipanggil (dari
   `extract_direct_semantic_hosted_majority`) KALAU `bentuk_reward` hasil
   call utama SUDAH `"non_tunai"` -- TIDAK PERNAH membebani/mem-bias call
   utama lagi. Karena ground truth dataset ini 100% `tunai`/Cashback,
   follow-up ini PRAKTIS TIDAK PERNAH terpicu di 25 dokumen yg ada --
   biaya tambahan nyaris nol utk kasus umum. Ada guard tambahan
   (`_REWARD_DETAIL_PRINTED_EXAMPLE`) yg buang jawaban kalau PERSIS sama
   dgn teks contoh cetak, sbg lapis pertahanan terakhir.

**Ringkasan jujur**: Fix 1 (tanggal tenor) = kemenangan bersih, terverifikasi
nyata. Fix 2 (reward detail) = arsitektur sudah benar & aman (tidak pernah
dipanggil kalau tidak perlu, tidak lagi bias call utama), TAPI akar masalah
`bentuk_reward` salah baca lingkaran pada 1 dokumen spesifik TETAP belum
terpecahkan -- didokumentasikan sbg keterbatasan, BUKAN diklaim selesai.
**Jangan coba lagi generalisasi rule coret/lingkar tanpa dataset uji lebih
besar** (1 dokumen positif tidak cukup dasar utk ubah rule yg dipakai
SEMUA dokumen lain) -- kalau mau coba lagi, uji DULU thd minimal 5-10
dokumen sebelum mengganti wording yg sudah terbukti, PERSIS prinsip §15
("satu perubahan kecil, satu pengukuran, baru lanjut") yg session ini
sempat langgar (2 perubahan besar diuji sekaligus tanpa baseline
per-dokumen dulu) sebelum akhirnya kembali ke disiplin itu.

## 9e. V19e — Local Qwen3-VL-2B on GPU (MX230 2GB) + "qwen3_vl_local" engine
(SELESAI, hasil JUJUR: TIDAK USABLE di hardware ini, tapi engine tetap
ditambahkan sbg opsi terpisah atas permintaan eksplisit user)

User minta download Qwen3-VL agar bisa dipakai LOKAL di GPU sendiri
(NVIDIA MX230). Dikerjakan penuh (bukan cuma rencana): environment
disiapkan, model didownload, dimuat di GPU, dites end-to-end thd 9 dokumen
nyata dgn ground truth -- SEMUA langkah diverifikasi via eksekusi nyata,
bukan tebakan.

**Setup environment (real, terverifikasi):**
- `nvidia-smi` konfirmasi GPU fisik ADA: NVIDIA GeForce MX230, driver
  536.67, **CUDA Version: 12.2 (maksimum yg didukung driver ini)**, VRAM
  **2048 MiB (2GB)**, mode WDDM (shared dgn desktop).
- Torch terpasang sblmnya `2.14.0+cpu` (CPU-only, `torch.cuda.is_available()
  == False`) -- diganti CUDA build. **Penting utk sesi depan**: cu124/cu126/
  cu128/cu130 SEMUA tersedia di index pytorch tapi TIDAK kompatibel dgn
  driver 536.67 (CUDA runtime > 12.2 yg didukung driver) -- yg BENAR2
  kompatibel & tersedia HANYA `cu118` (`torch==2.7.1+cu118`, versi lebih
  baru tdk ada di index cu118/cu121 utk cu121). `torchvision` HARUS
  di-reinstall match persis (`0.22.1+cu118` utk torch `2.7.1`) -- pip TIDAK
  otomatis downgrade dependency yg sudah kepasang, dibiarkan mismatch bakal
  crash saat import.
- `transformers` 5.17.0 yg SUDAH terpasang **ternyata SUDAH punya native
  class** `Qwen3VLForConditionalGeneration`/`Qwen3VLMoeForConditionalGeneration`
  -- TIDAK perlu upgrade transformers sama sekali utk Qwen3-VL.
  `AutoConfig.from_pretrained(...).model_type` utk `Qwen/Qwen3-VL-2B-Instruct`
  = `"qwen3_vl"` (dikonfirmasi via `hf_hub_download` config.json saja, tanpa
  download model penuh dulu).
- `bitsandbytes==0.50.2` terpasang bersih via pip di Windows (isu lama
  "bitsandbytes Windows kurang reliable" TIDAK terjadi di versi ini).
- Model `Qwen/Qwen3-VL-2B-Instruct` (fp16 asli, BUKAN GGUF -- lihat catatan
  desain di bawah) didownload via `huggingface_hub.snapshot_download` ke
  `models/Qwen3-VL-2B-Instruct/` (~4GB, ~26 menit), sudah otomatis
  ter-gitignore (`models/` + `*.safetensors` sudah ada di `.gitignore`
  sblm sesi ini).

**Keputusan desain (dgn user, PENTING utk sesi depan):** user awalnya
jawab 2 pertanyaan terpisah dgn jawaban yg SECARA TEKNIS tidak bisa
digabung -- "2B GGUF quantized" (format llama.cpp) TAPI "integrasikan ke
vlm.py via transformers+bitsandbytes" (format berbeda total, transformers
TIDAK bisa load GGUF multimodal dgn cara standar). Diselesaikan dgn
memprioritaskan pilihan RUNTIME (lebih spesifik/actionable): download
repo safetensors NORMAL (bukan GGUF), kuantisasi 4-bit dilakukan SAAT
LOAD via `bitsandbytes` (`BitsAndBytesConfig`), BUKAN pre-kuantisasi GGUF
di disk. Ini flagged eksplisit ke user saat itu, tidak dikoreksi -- jadi
dianggap konfirmasi implisit.

**Perubahan kode (`vlm.py`):**
- `DEFAULT_LOCAL_CANDIDATES` tambah `models/Qwen3-VL-2B-Instruct` (dicek
  LEBIH DULU drpd Qwen2-VL lama).
- `LOAD_IN_4BIT_ENV` (baru, `VLM_LOAD_IN_4BIT`, default `"auto"`) -- 4-bit
  otomatis AKTIF kalau device CUDA, TIDAK aktif di CPU (V10/V12 fallback
  CPU-only TIDAK terpengaruh sama sekali).
- `_load()`: tambah branch eksplisit `model_type == "qwen3_vl"` ->
  `Qwen3VLForConditionalGeneration`, `"qwen3_vl_moe"` ->
  `Qwen3VLMoeForConditionalGeneration` (pola SAMA persis dgn branch
  qwen2_vl/qwen2_5_vl yg sudah ada, bukan rewrite). Load 4-bit pakai
  `device_map={"": 0}` (BUKAN `.to(device)` manual sesudahnya -- restriksi
  transformers utk model quantized).
- `extract_direct_semantic_local()` (baru) -- reuse PERSIS
  `DIRECT_SEMANTIC_PROMPT`/`_parse_direct_semantic_json`/`_infer()` yg SAMA
  dgn jalur hosted, kirim 2 gambar (template + dokumen) yg SAMA lewat
  `_infer()` yg SUDAH mendukung multi-image. `_get_template_data_uri()`
  di-refactor jadi reuse `_get_template_image()` (baru, cache gambar
  mentah) supaya tidak duplikasi `prep.load_document(prep.TEMPLATE_PATH)`.

**Perubahan kode (`extractors.py`):** engine baru `"qwen3_vl_local"` (label
"Qwen3-VL-2B (Local GPU, EXPERIMENTAL)") ditambahkan **DI SAMPING**
`"qwen3_vl"` (hosted) -- TIDAK menggantikan, TIDAK mengubah default,
sesuai keputusan eksplisit user setelah ditanya (lihat hasil eval di bawah
knp menggantikan hosted SALAH). `run_qwen3_vl_local()`: prep gambar SAMA
(`_qwen_prepare_image`), TIDAK ADA majority-vote (beda dgn hosted -- 1x
inferensi lokal sudah >5 menit, 3x tidak masuk akal), TIDAK fallback ke
`run_v18` kalau gagal (SAMA persis kebijakan hosted). `static/index.html`
kedua dropdown (Tab 1 & 2) tambah opsi ini.

**HASIL EVALUASI JUJUR (9 dokumen nyata dgn ground truth, `eval_runs/
local_qwen_eval_results.csv`, script `local_qwen_eval.py` di root --
STANDALONE, TIDAK dipanggil `evaluation.py`/`extractors.py`):**

Resolusi `VLM_MAX_SIDE_FULL=1600` (default) -> **CUDA OOM** (butuh puncak
3.88GB drpd 2GB tersedia) -- weight model SENDIRI cuma 1.58GB (4-bit
BERHASIL muat), tapi vision encoder + activation utk halaman penuh
resolusi tinggi yg menghabiskan sisa VRAM. Diturunkan ke `640` supaya
tidak OOM (puncak 1.86GB, muat) -- TAPI hasil ekstraksi PRAKTIS 0% akurat:

| Field | Akurasi (9 dokumen) | Catatan |
|---|---|---|
| nama_nasabah | 0/9 | Mulai dari label generik ("Nasabah") sampai nama Indonesia yg TERDENGAR masuk akal tapi SEPENUHNYA karangan |
| nomor_rekening | 0/9 | **7 dari 9 jawaban PERSIS string "1234567890"/"123456789"** -- BUKAN salah baca, ini fallback halusinasi TETAP |
| nominal_penempatan | 1/9 | Sisanya meleset 10x-1000x |
| tenor | "5/9 benar" DI ATAS KERTAS, TAPI model jawab "3 bulan" di 7 dari 9 dokumen TANPA PEDULI isi gambar -- kebetulan 5 dari 9 GT memang "3 Bulan" (base rate), BUKAN genuinely membaca. 2 kasus GT="6 Bulan" SELALU salah. |
| bentuk_reward | ~0/9 | Jawaban tak masuk akal ("100%", "1x", "10000000") |
| signature (2 field) | N/A (GT tdk di-load script ini) | Jawaban IDENTIK ("ada"/"kosong") di SEMUA 9 dokumen tanpa variasi -- pola constant-output yg sama dgn tenor |

Proses full-25-dokumen SEMPAT dicoba tapi **di-kill sistem krn low
memory** setelah 12 dokumen (9 sukses + 3 gagal `imread` -- bug SENDIRI di
script test, `cv2.imread` tdk bisa baca PDF, PERBAIKAN sudah dimasukkan ke
`local_qwen_eval.py` tapi TIDAK di-rerun). 9 dokumen dinilai CUKUP -- pola
sudah 100% konsisten & konklusif, melanjutkan 13 dokumen sisanya (~1.5 jam
lagi) tidak akan mengubah kesimpulan.

**Kesimpulan JUJUR, JANGAN diulang tanpa data baru**: pada GPU 2GB spt
MX230, TIDAK ADA resolusi yg SEKALIGUS (a) cukup kecil utk muat di VRAM
DAN (b) cukup besar utk model benar2 membaca tulisan tangan dokumen ini.
Weight 4-bit muat (1.58GB), tapi begitu resolusi diturunkan cukup jauh
utk sisa VRAM cukup utk vision encoder, modelnya sendiri sudah TIDAK BISA
membaca apa pun lagi -- ini BUKAN masalah prompt/kuantisasi yg bisa
diperbaiki lebih lanjut, ini keterbatasan HARDWARE murni. Engine
`qwen3_vl_local` TETAP ada di kode (opsi eksplisit, EXPERIMENTAL) utk GPU
dgn VRAM lebih besar atau evaluasi ulang nanti -- **JANGAN dijadikan
default/pengganti `qwen3_vl` hosted**, TIDAK USABLE di hardware sesi ini.

**File baru:** `local_qwen_eval.py` (root, standalone), `eval_runs/
local_qwen_eval_results.csv` (9 baris hasil, bukti mentah).
**TIDAK disentuh:** `qwen3_vl` (hosted, tetap default/utama), `pipeline.py`/
`preprocessing.py`/`postprocessing.py`/`comparison.py`, V10/V12 local VLM
fallback (`extract_fields_independent`/`extract_fields_fullpage`, masih
CPU-only spt sblmnya krn `LOAD_IN_4BIT_ENV` auto-off di CPU).

## 10. Prioritas Berikutnya (berdasar data V15.2/V17/V18 di atas, BUKAN tebakan)

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
   lain milik user, di luar kendali evaluator) — lihat §12 utk workaround
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

## 11. Sudah Dicoba / Jangan Diulang (hemat waktu+token sesi berikutnya)

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
  §12 utk workaround batch-subprocess yg terbukti bekerja di kondisi sama.
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

## 12. Cara Re-run Evaluasi (perintah persis)

**(V18) Semua output (`evaluation_results_*.csv`/`evaluation_summary_*.json`/
`stage_evaluation_*`/`compare_runs_*`/`signature_diagnostics.csv`) SEKARANG
otomatis ditulis ke folder `eval_runs/` (dibuat otomatis kalau belum ada),
BUKAN root project lagi** — sebelumnya berserakan campur dgn file kode,
menyulitkan cari file evaluasi/diagnostic terbaru. `eval_runs/` di-gitignore
(sama spt `evaluation_summary*`/`evaluation_result*` sebelumnya) — file di
situ HANYA utk referensi lokal, TIDAK di-commit. Semua file historis (V13-V18)
SUDAH dipindah ke sana sesi ini, lihat §13.

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

**V17 — mesin bisa lebih parah dari ~8GB yg didokumentasikan §10 poin 7.**
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

## 13. Peta File

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

## 14. Field & Rules (tidak berubah dari versi sebelumnya)

Field: Nama (fuzzy match 80%), Nomor Rekening, Unit Kerja, Nominal Penempatan, Tenor
(dari rentang tanggal, <1 bulan = tidak sesuai), Bentuk Reward (tunai/non-tunai +
fallback), TTD Nasabah & TTD Unit Kerja/BRI (presence only, bukan baca isi).

Rule rural (`Urban` != "urban" di data): cukup validasi Nama, Nomor Rekening, Nominal
Penempatan, TTD Nasabah, TTD BRI — kalau semua lolos, `OK Lanjut Proses`.

Environment: Windows, FastAPI `127.0.0.1:8001`, GPU NVIDIA MX230 2GB (PaddleOCR
mayoritas jalan CPU di environment evaluasi), `MAX_CANVAS_HEIGHT_GPU=2200` /
`_CPU=6000`, `text_recognition_batch_size=1` (low VRAM GPU kecil).

## 15. Instruksi utk Claude (sesi berikutnya)

1. Baca §1-9 dulu — itu cukup utk lanjut kerja tanpa re-explore dari nol.
2. Jangan ulangi §11 (sudah dicoba & terbukti gagal/terbukti benar).
3. Sebelum ubah kode: jalankan evaluator (§12), baca angkanya, baru putuskan.
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
6. Jangan jalankan 2 evaluator penuh (25 dokumen) bersamaan (§10 poin 7,
   resource mesin cuma ~8GB RAM, sudah beberapa kali OOM/MemoryError). **V17:
   bisa OOM bahkan SATU evaluator/`--limit` kecil kalau RAM sistem lagi
   dipakai app lain user** — lihat §12 utk workaround batch-subprocess.
