# HANDOVER — OCR Pipeline Preprocessing & Evaluation

**Last update:** 17 September 2026 (V22 addendum -- fix bug mirrored-placeholder `reward_tunai`/`reward_non_tunai` [literal string "tunai"/"non_tunai" muncul sbg nilai palsu kalau detail extraction gagal, SEKARANG null jujur utk Qwen; Gemini/Mistral placeholder TETAP dipertahankan by design] + perkuat instruksi null-default utk SEMUA choice field [tenor_penempatan/bentuk_reward] di prompt, lihat §9l. V22 -- decision-accuracy metric baru di popup "Evaluasi" [OK vs TOLAK, prefix-based, Review dilipat ke Tolak], 3 perbaikan nomor_rekening [prompt/crop-padding/length-gate: 0.56->0.60], + validasi digit-vs-terbilang BARU utk nominal_penempatan/reward_tunai [modul parser Indonesia BARU `terbilang.py`, diagnostic-only, consistency rate nyata 0.9412] -- lihat §9k utk detail lengkap & catatan jujur soal noise run-to-run. V21 -- runtime+accuracy quick-wins pd "qwen3_vl_local_yolos": YOLOS dipindah ke GPU [sebelumnya diam-diam CPU-only], attn_implementation="sdpa" dipin eksplisit, mitigasi "Extraction Collapsed" via retry SATU KALI di subprocess BARU [collapse deterministic per-proses, jadi retry in-process percuma], + close-up crop ROI statis utk 3 field terlemah [nama_nasabah/nomor_rekening/nominal_penempatan]. Full 25-dokumen PERTAMA KALI utk engine ini: nama_nasabah 0.60, nomor_rekening 0.56, nominal_penempatan 0.84, avg 30.83s/dok. Ditemukan JUGA bug lingkungan terpisah: PaddleOCR gagal import (circular import) independen dari perubahan sesi ini -- lihat §9j utk detail lengkap. V20 -- HYBRID engine baru "qwen3_vl_local_yolos" [Qwen3-VL Local + YOLOS signature detector, hard local-only gate, additive thd qwen3_vl_local yg TIDAK diubah], vendor asli Tech4Humans YOLOv8 dibatalkan krn repo HF gated [tidak ada akses scriptable], diganti `mdefrance/yolos-tiny-signature-detection` [non-gated, Apache-2.0, NOL dependency pip baru] atas pilihan eksplisit user, kode selesai DAN diuji end-to-end NYATA tanpa mocking [Qwen asli + YOLOS asli, termasuk sesi live server sungguhan], lihat §9i; V20 -- Live Evaluation feature di Record Table (tombol "Evaluasi", modal metrik agregat dari hasil TERSIMPAN saja, tidak pernah OCR ulang), terverifikasi end-to-end nyata (server sungguhan, upload+proses+rerun asli), lihat §9h; V20 -- root-cause investigasi "Extraction Collapsed" pd local Qwen3-VL-4B: CONFIRMED non-determinism run-to-run di level inference runtime (BUKAN dokumen/prompt/schema) -- dokumen yg sama collapse 100% di satu peluncuran proses & berhasil 100% di peluncuran lain, direplikasi 3x independen, lihat §9g; V19f -- V18 DIBEKUKAN sampai user minta lagi, Qwen3-VL 4B disamakan antara local & hosted, signature majority-vote ON by default utk local juga, kanonikalisasi tenor/reward, lihat §9f; V19e -- local Qwen3-VL-2B pada GPU 2GB TERBUKTI tidak usable, engine "qwen3_vl_local" ditambahkan sbg opsi terpisah, lihat §9e; V19c -- ganti engine "qwen2_vl" lokal jadi "qwen3_vl" Hugging Face HOSTED, lihat §9c; V19 -- Qwen2-VL direct-semantic pipeline + Gemini token/cost tracking, lihat §9b)
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

## 9f. V19f — Qwen3-VL 4B Alignment (Local <-> Hosted), Signature Majority-Vote utk Local, Tenor/Reward Canonicalization, V18 Freeze (SELESAI, spot-check direkomendasikan -- BUKAN full benchmark)

**V18 DIBEKUKAN mulai sesi ini** (permintaan eksplisit user): pipeline
deterministik V18 (`pipeline.py`/`preprocessing.py`/`postprocessing.py` --
template alignment + ROI + PaddleOCR + fallback VLM lokal) TIDAK BOLEH
disentuh internalnya lagi sampai user secara eksplisit minta lagi ("saya
akan mention lagi kalau perlu"). Ini men-supersede framing "Prioritas
Berikutnya" di §10 di bawah -- semua item di situ masih soal internal
V18/preprocessing/ROI, JANGAN dikerjakan tanpa izin baru dari user,
terlepas dari seberapa jelas root cause-nya kelihatan. Fokus kerja mulai
sesi ini DIALIHKAN ke Qwen3-VL (`qwen3_vl`/`qwen3_vl_local`), dgn kebijakan
BARU yg men-supersede §9c/§9e ("local changes must not affect hosted
defaults"): apa pun yg diubah utk local HARUS ikut diterapkan ke hosted
juga, supaya kedua engine tetap sejalan.

**Retroactive documentation -- commit `1909f86` ("fix - current fix, needed
to evaluate the reward and tenor"), TIDAK PERNAH ditulis di handover ini
sblm sesi ini:**
- `vlm.py`: field `unit_kerja` ditambahkan ke `DIRECT_SEMANTIC_PROMPT`
  (rule anti-leakage sendiri thd nomor_rekening di atasnya); rule baru utk
  `nama` (prefiks badan usaha CV/PT/UD/Koperasi dibaca sbg singkatan baku,
  bukan dieja huruf per huruf); rule signature DIKETATKAN lagi (revert
  dari wording "LENIENT/generous" balik ke strict -- jawab true HANYA utk
  goresan tanda tangan asli, false kalau kosong) + kalibrasi confidence
  eksplisit (low <0.5 utk ambigu, high >=0.8 HANYA utk yg jelas); status
  per-field non-signature diganti dari confidence 0.0-1.0 mentah jadi
  kategori `detected/uncertain/not_detected` (`_parse_direct_semantic_
  json` ditulis ulang, `_ALLOWED_DIRECT_SEMANTIC_STATUSES` baru);
  follow-up call BARU `REWARD_TUNAI_DETAIL_PROMPT`/`_extract_reward_
  tunai_detail_hosted`/`_extract_reward_tunai_detail_local` (simetris dgn
  follow-up non-tunai yg sudah ada, §9d) utk baca nominal reward tunai;
  follow-up reward (tunai maupun non-tunai) PERTAMA KALI disambungkan ke
  `extract_direct_semantic_local()` juga -- sebelumnya HANYA hosted yg
  dapat detail asli, local selalu dapat placeholder kata pilihan
  ("tunai"/"non_tunai" mentah). Comment (BELUM diverifikasi lewat eval
  formal di sesi manapun sblm sesi ini) juga mengklaim local Qwen3-VL
  sudah di-re-verify di hardware RTX 4060 Laptop (8.6GB VRAM) pakai model
  4B 4-bit -- lihat detail di bawah, klaim ini yg jadi pemicu utama fix
  sesi ini.
- `comparison.py`: threshold match nama diganti dari 1 angka
  (`NAME_FUZZY_MATCH_THRESHOLD=80`) jadi 3-state (`NAME_PASS_
  THRESHOLD=90`/`NAME_MISMATCH_THRESHOLD=55`, status tengah "uncertain"
  -> "perlu_review", TIDAK PERNAH diam2 ditampilkan sbg match).
  `_calendar_month_diff` tidak lagi pakai `abs()` + guard baru
  `reversed_date_range` di `derive_tenor_from_range` (postprocessing.py)
  supaya rentang tanggal terbalik TIDAK diam2 ditukar jadi tenor karangan.
  Badge UI baru via `_extra_validation_notes()` ("Name Mismatch", "Name
  Uncertain", "Invalid/Reversed Date Range", "Tenor Conflict", "Reward
  Uncertain", "Signature Absent").
- `postprocessing.py`: nama bulan singkatan (Indonesia+Inggris, mis.
  "Sep"/"Mar") ditambah ke `_ID_MONTHS`; regex baru utk buang label bocor
  "No Rek:" sblm substitusi huruf->digit di `normalize_numeric`.

**Kenapa sesi ini terjadi (bug aktif, dikonfirmasi via `ls models/`):**
`vlm.DEFAULT_LOCAL_CANDIDATES` mendaftar `models/Qwen3-VL-2B-Instruct`
LEBIH DULU drpd `Qwen2-VL-2B-Instruct` -- tapi folder itu **TIDAK PERNAH
ADA** di disk (hanya `Qwen2-VL-2B-Instruct` dan `Qwen3-VL-4B-Instruct` yg
benar2 ada). Akibatnya `qwen3_vl_local` diam2 SELALU jalan pakai
`Qwen2-VL-2B-Instruct` lama, BUKAN model 4B yg diklaim comment `1909f86`
sudah di-re-verify -- inkonsistensi nyata antara comment & perilaku kode
yg TIDAK PERNAH diperiksa ulang sblm sesi ini.

**Perubahan kode sesi ini:**
- `vlm.DEFAULT_LOCAL_CANDIDATES`: `Qwen3-VL-2B-Instruct` (tdk pernah ada)
  dihapus, `Qwen3-VL-4B-Instruct` jadi kandidat pertama, `Qwen2-VL-2B-
  Instruct` tetap fallback. `_resolve_model_path()`'s online-fallback repo
  id & pesan error disesuaikan.
- `config.QWEN_MODEL_DEFAULT` & `.env.example`'s `QWEN_MODEL`: diubah dari
  `Qwen/Qwen3-VL-2B-Instruct` ke `Qwen/Qwen3-VL-4B-Instruct` (permintaan
  eksplisit user: hosted disamakan ke model 4B yg sama dgn local, drpd
  mengejar referensi "30B-A3B" di comment yg tidak pernah menyebut repo id
  konkret). **Sudah di-live-test** sesi ini thd HF Inference API nyata
  (`evaluation.py --engine qwen3_vl --limit 1`, `eval_runs/
  v19f_hosted_spotcheck.csv/json`) -- request BERHASIL (33.6s incl. 3x vote
  + 1 follow-up, no auth/model-not-found error), jadi id ini dikonfirmasi
  resolvable via provider auto-routing saat ini.
- `ENGINE_LABELS` (`extractors.py`): label "Qwen3-VL-2B" -> "Qwen3-VL-4B"
  utk kedua engine (hosted & local).
- Majority-vote signature: helper baru `vlm._aggregate_signature_votes()`
  (kebijakan lenient any-present-wins, PERSIS sama dgn yg sudah ada utk
  hosted) di-share oleh hosted (`extract_direct_semantic_hosted_majority`,
  di-refactor pakai helper ini, perilaku TIDAK berubah) DAN fungsi BARU
  `extract_direct_semantic_local_majority()` utk local. Env var baru
  `VLM_SIGNATURE_VOTE_COUNT` (`vlm.LOCAL_SIGNATURE_VOTE_COUNT`), default
  **"3" (ON by default, keputusan eksplisit user)** -- supersede kebijakan
  §9e ("TIDAK ADA majority-vote lokal, 1x inferensi sudah >5 menit di
  hardware lama"); dgn ~22-26s/dok di hardware RTX4060 yg diklaim comment
  `1909f86`, 3x vote jadi ~1 menit/dok, dianggap terjangkau.
  `run_qwen3_vl_local()` di `extractors.py` disambungkan ke fungsi
  majority ini (bukan single-call lagi); `ocr_meta["qwen_local_request"]`
  baru utk transparansi votes/signature_votes, sejajar dgn
  `qwen_hosted_request` yg sudah ada.
- Kanonikalisasi minimal utk `tenor_penempatan`/`bentuk_reward` di
  `vlm._parse_direct_semantic_json()` (`_canonicalize_tenor_penempatan`/
  `_canonicalize_bentuk_reward`) -- SEBELUMNYA jawaban model yg near-miss
  (mis. "3 bulan" bukan "3", "non tunai" [pakai spasi] bukan "non_tunai")
  lolos mentah lalu diam2 hilang jadi `None` DUA langkah di hilir
  (`extractors._adapt_qwen_to_common`'s `int()` cast murni / exact-tuple
  check) SEBELUM normalisasi `comparison.py` yg sudah teruji sempat jalan
  sama sekali. Scope kanonikalisasi ini SENGAJA minimal (cuma strip
  suffix satuan + normalisasi spasi/underscore/case) -- fold leksikal yg
  lebih luas (mis. "cash"->"tunai") TETAP TUGAS `comparison._normalize_
  for_compare` saja, supaya tidak ada 2 salinan aturan "apa itu tunai" yg
  bisa divergen. Kalau tidak match set tertutup `{"1","3","6"}`/
  `{"tunai","non_tunai"}`, value jadi `None` & status dipaksa
  `"uncertain"` (bukan silently lolos atau `"not_detected"`).
- Ambiguitas konvensi coret/lingkar utk `bentuk_reward`/`tenor_penempatan`
  (§9d: pernah dicoba digeneralisasi & di-revert krn regresi "record 6")
  -- DIKONFIRMASI (baca ulang `DIRECT_SEMANTIC_PROMPT` langsung) prompt yg
  LIVE sekarang SUDAH minta model cek strike-through, circle, DAN
  bold-retrace utk kedua field itu. TIDAK ADA perubahan kode di sesi ini
  utk ini -- HANYA re-konfirmasi wording yg sudah ada, belum di-spot-check
  ulang thd dokumen "record 6" yg dulu regresi (lihat item terbuka di
  bawah).

**Spot-check 1-dokumen sesi ini (`eval_runs/v19f_spotcheck.*` [local],
`eval_runs/v19f_hosted_spotcheck.*` [hosted], record "BO MEDAN GATOT
SUBROTO"), BUKAN benchmark formal (1 dokumen saja, sesuai permintaan
eksplisit user utk menunda benchmark):**
- Hosted (`Qwen/Qwen3-VL-4B-Instruct`): request BERHASIL, 33.6s total (3x
  vote + 1 reward-detail follow-up), TIDAK ada error auth/model-not-found
  -- id BARU ini dikonfirmasi resolvable via provider auto-routing saat
  ini.
- Local (mesin sesi ini KEBETULAN persis RTX 4060 Laptop yg direferensikan
  comment `1909f86`, tapi VRAM bebas cuma ~3.1GB drpd 8.6GB penuh krn
  proses lain sedang jalan): BERHASIL load model 4B + selesai tanpa OOM,
  197.29s utk 3x vote + 1 follow-up (~66s/vote) -- LEBIH LAMBAT dari klaim
  comment (~22-26s/dok utk 1 call), kemungkinan besar krn VRAM/GPU
  ternyata dipakai bersama proses lain sesi ini, BUKAN direplikasi di
  kondisi VRAM penuh/idle. Anggap timing asli comment sbg BELUM
  direplikasi persis; behavior fungsional (tidak OOM, hasil benar) SUDAH
  dikonfirmasi.
- Kedua engine: `tenor_penempatan` & `bentuk_reward` cocok ground truth
  (kanonikalisasi baru bekerja pd data nyata, bukan cuma unit test).
  Signature: local 3 vote SEPAKAT (True/True utk kedua field pd dokumen
  ini -- tidak ada perpecahan vote pd sample n=1 ini). `nama_nasabah`/
  `nomor_rekening`/`nominal_penempatan` MASIH salah baca di kedua engine
  pd dokumen ini -- konsisten dgn keterbatasan tulisan-tangan yg sudah
  didokumentasikan (§9d/9e), BUKAN regresi baru dari perubahan sesi ini.
- **BELUM dikerjakan**: re-verify dokumen "record 6" (§9d, regresi
  circle-marking) scr spesifik, dan full re-run `evaluation.py` thd
  seluruh dataset (25 dokumen) utk kedua engine -- tetap ditunda sesuai
  permintaan user, bukan lupa dikerjakan.

**TIDAK disentuh sesi ini:** V18 internal (`pipeline.py` core,
`postprocessing.py` ink-diff, `preprocessing.py` ROI/alignment) dan
fallback lokal V12 lama (`FULLPAGE_PROMPT_TEMPLATE`/`_TENOR_RULE`/
`_SIGNATURE_RULE` di `vlm.py`) -- keduanya di luar scope permintaan user
sesi ini.

## 9g. V20 — Root-Cause Investigation: "Extraction Collapsed" pada Local Qwen3-VL-4B (SELESAI utk pertanyaan diagnostik utama; mekanisme akar TIDAK sepenuhnya diketahui)

**Scope eksplisit dari user**: fokus HANYA local (`qwen3_vl_local`), TIDAK menyentuh/menguji hosted, TIDAK menghabiskan kredit HF, V18 tetap dibekukan (§9f). Tujuan: cari akar masalah "Extraction Collapsed" (§9f/§20 -- 6/25 dokumen pada run 25-dokumen `qwen3_vl_local` mengembalikan JSON valid tapi SEMUA field inti null) SEBELUM lanjut ke perbaikan akurasi nama/nomor rekening (42%/47%).

**Alat baru**: `diagnose_collapse.py` (root, standalone spt `local_qwen_eval.py` -- TIDAK dipanggil app.py/pipeline manapun). Reimplementasi `_infer_instrumented()` (paralel `vlm._infer`, hanya menambah capture tensor-shape/model-identity/raw-response, TIDAK mengubah `vlm.py`) + harness eksperimen. Failure set (6 dokumen, dari run 25-dokumen V19f): indramayu, pamanukan, situmekar_sukabumi (PDF), reska_cibadak, ngawi, ciamis. Control set (6 dokumen sukses): medan_gatot_subroto, mangga_dua (PDF), muntilan, subang, jakarta_kalideres (PDF), tanjung_tabalong.

**Pertanyaan diagnostik utama dari user**: "Apakah model 4B lokal GAGAL memahami dokumen scr visual, ATAU dia memahami dokumen tapi collapse krn prompt/schema/generation/inference handling?"

### JAWABAN: bukan keduanya -- ini masalah run-to-run non-determinism di level inference runtime, BUKAN dokumen atau prompt.

**CONFIRMED (bukti langsung, direplikasi berkali-kali independen):**
1. **Model BISA membaca dokumen-dokumen yang collapse ini.** Dikonfirmasi 2 cara: (a) inspeksi visual LANGSUNG thd gambar persis yg dikirim ke model utk `reska_cibadak` -- terbaca jelas 100% oleh mata (nama, nomor rekening, nominal, tenor, tanda tangan semua terlihat) meski hasil model NULL semua; (b) dokumen YANG SAMA, kode YANG SAMA, prompt YANG SAMA berhasil dibaca benar scr KONSISTEN (0 collapse dari puluhan panggilan) di sesi proses yg berbeda.
2. **Collapse SANGAT reproducible per-DOKUMEN via chain produksi asli** (`extract_direct_semantic_local_majority` -> `_extract_direct_semantic_local_core` -> `vlm._infer`, TIDAK dimodifikasi): 3 replikasi independen (proses baru tiap kali) SEMUA cocok persis pola asli run 25-dokumen -- **6/6 dokumen failure-set collapse lagi, 6/6 dokumen control-set berhasil lagi** (`eval_runs/collapse_diagnosis/real_majority_replication.jsonl`), termasuk 1 replikasi TERISOLASI (`indramayu` SENDIRIAN, proses baru, tanpa dokumen lain sama sekali) yg tetap collapse.
3. **BUKAN beda kode**: `_infer_instrumented()` (reimplementasi diagnostik) dibandingkan LANGSUNG head-to-head dgn `vlm._infer()` asli, di PROSES & panggilan yg SAMA persis (`indramayu`, 3x berpasangan) -- **raw text output IDENTIK BYTE-FOR-BYTE** tiap pasang (`identical raw text: True` x3), dan KEDUANYA collapse bersamaan di proses itu. Menyingkirkan kemungkinan reimplementasi diagnostik "sengaja"/tidak sengaja beda dari produksi.
4. **BUKAN akumulasi panggilan dlm 1 proses**: collapse muncul pd panggilan PERTAMA (`indramayu`, cumulative_calls_after=3, yaitu 3 vote dokumen pertama) di proses yg baru dimulai -- tidak ada state "terkumpul" dari dokumen sebelumnya yg mungkin jadi penyebab.
5. **BUKAN tekanan VRAM semata**: 36 panggilan berurutan (3 pass x 12 dokumen, TERMASUK ke-6 dokumen failure-set) dijalankan sengaja di bawah tekanan memori GPU yg direkonstruksi (~2.96GB bebas, meniru kondisi asli run 25-dokumen yg ~3.1GB bebas krn proses lain) via `eval_runs/_gpu_hog.py` -- **0/36 collapse**, termasuk sebagian pass berjalan setelah hog mati (GPU bebas total 7.9GB). Tekanan memori BUKAN pemicu tunggal/cukup.
6. **Hasil KONSISTEN dalam 1 proses, TAPI BERBEDA antar proses**: dokumen yg sama, kode yg sama, input yg sama bisa 100% berhasil di satu peluncuran proses dan 100% collapse di peluncuran proses lain -- deterministik SELAMA proses itu hidup, tapi outcome-nya sendiri tidak bisa diprediksi sebelum proses dimulai.

**HYPOTHESIS (belum dikonfirmasi mekanismenya, TAPI konsisten dgn SEMUA temuan di atas)**: non-determinism di level CUDA/kernel bitsandbytes 4-bit -- kemungkinan pemilihan algoritma/kernel (heuristik autotuning cuBLAS/attention-backend) atau urutan reduksi floating-point yg TIDAK stabil antar proses (dipengaruhi state GPU/driver/alokator memori saat load model atau inferensi pertama), yg kemudian "terkunci" utk sisa umur proses itu. Di bawah jalur numerik yg "buruk", sebagian dokumen (dgn konten/dimensi visual yg entah kenapa dekat suatu titik ambang internal) jatuh ke respons default kosong; di jalur numerik yg "baik", dokumen yg SAMA terbaca benar. **Mekanisme PERSIS (kernel/operasi mana yg bertanggung jawab) TIDAK diselidiki lebih lanjut sesi ini** -- perlu profiling level lebih rendah (mis. `CUBLAS_WORKSPACE_CONFIG`/determinism flags, uji tanpa kuantisasi 4-bit/fp16 penuh, `torch.use_deterministic_algorithms`) yg di luar scope sesi ini.

**REJECTED (diuji langsung, terbukti salah/tidak cukup sbg penyebab):**
- Orientasi/kualitas gambar dokumen -- gambar yg dites terbukti tegak & terbaca jelas scr visual langsung.
- Kompleksitas prompt/schema sbg penyebab TUNGGAL -- **BELUM diuji formal** via eksperimen A/B prompt-minimal yg direncanakan (lihat "TIDAK dikerjakan" di bawah) krn ditemukan lebih dulu bahwa hasil single-shot TIDAK valid tanpa mengontrol variabel proses ini -- status: **tertunda, bukan rejected**, ditulis terpisah supaya tidak disalahpahami sbg sudah diuji.
- Akumulasi panggilan berturut-turut dlm 1 proses (memory drift dsb) -- REJECTED, collapse muncul di panggilan pertama proses baru.
- Tekanan VRAM/kontensi GPU sbg penyebab TUNGGAL/CUKUP -- REJECTED, 36 panggilan di bawah tekanan setara kondisi asli tidak menghasilkan collapse sama sekali.
- Perbedaan kode antara reimplementasi diagnostik & `vlm._infer` asli -- REJECTED, output identik byte-for-byte saat dibandingkan langsung.

**UNRESOLVED (belum terjawab, perlu kerja lanjutan):**
- Mekanisme non-determinism level-rendah yg PERSIS (kernel/operasi CUDA mana).
- KENAPA dokumen SPESIFIK ini (bukan acak) yg jatuh ke collapse dlm proses yg "buruk", sementara yg lain selalu selamat -- apakah terkait `image_grid_thw`/jumlah token gambar tertentu, atau sesuatu lain di konten visualnya? Tidak diuji sesi ini.
- Apakah hosted engine (`qwen3_vl`) punya masalah serupa -- DI LUAR SCOPE sesi ini (instruksi eksplisit: jangan sentuh/uji hosted, jangan pakai kredit HF).
- Eksperimen A/B yg direncanakan user (prompt minimal -> progresif, gambar asli vs preprocessed, resolusi lebih tinggi, generation config minimal, plain-text vs JSON) **TIDAK dikerjakan** sesi ini -- desain aslinya (1 percobaan per variabel per dokumen) TIDAK valid lagi setelah temuan non-determinism antar-proses ini ditemukan: satu percobaan single-shot tidak bisa dibedakan dari sekadar "proses ini kebetulan baik/buruk". Eksperimen itu, KALAU mau dilanjutkan, perlu diulang BANYAK kali (proses baru tiap ulangan) per sel matriks utk kesimpulan yg valid scr statistik -- usaha jauh lebih besar dari rencana awal, BELUM disetujui user.

**Rekomendasi (BELUM diimplementasikan, perlu keputusan user)**: krn akar masalah ada di lapisan inference-runtime (bukan prompt/schema), mitigasi paling masuk akal BUKAN prompt yg lebih baik, melainkan salah satu dari: (a) retry di PROSES BARU (bukan sekadar panggilan ulang di proses yg sama -- itu akan mengulang state "buruk" yg sama) saat `extraction_collapsed` terdeteksi -- costly (perlu subprocess/restart model per retry); (b) uji apakah menonaktifkan kuantisasi 4-bit (fp16/bf16 penuh) menghilangkan variabilitas antar-proses ini (belum diuji, VRAM 4B fp16 ~8GB mepet dgn kapasitas GPU 8.19GB sesi ini, risiko OOM); (c) terima sbg keterbatasan reliabilitas jalur lokal 4-bit yg diketahui, cukup andalkan deteksi+badge `extraction_collapsed` (§9f) yg sudah ada utk review manual, tanpa remediasi otomatis. TIDAK ada yg dipilih/dikerjakan sesi ini -- keputusan diserahkan ke user.

**File baru**: `diagnose_collapse.py` (root, standalone, infrastruktur eksperimen msh valid utk dipakai kalau mau lanjut dgn desain "banyak ulangan per sel"), `eval_runs/_gpu_hog.py` (scratch, alat uji tekanan VRAM), `eval_runs/collapse_diagnosis/*.jsonl` (bukti mentah semua run).
**TIDAK disentuh sesi ini**: hosted engine (`qwen3_vl`, sesuai instruksi eksplisit), V18 (`pipeline.py`/`preprocessing.py`/`postprocessing.py`, tetap dibekukan §9f), identity-block crop / perbaikan akurasi nama-rekening (ditunda sampai pertanyaan collapse ini selesai, sesuai instruksi eksplisit user).

## 9h. V20 — Live Evaluation Feature di Record Table (SELESAI, terverifikasi end-to-end nyata)

Tombol "Evaluasi" baru di toolbar Record Table (Tab 2) yang membuka modal berisi metrik agregat (accuracy/precision/recall/F1 per field, confusion matrix utk field kategorikal, statistik runtime, statistik Extraction Collapsed) dari record yang SUDAH diproses pada sesi berjalan -- TIDAK PERNAH menjalankan OCR/VLM ulang.

**Arsitektur:**
- `_SESSIONS[session_id]["results"]` (in-memory, `app.py`) SUDAH secara alami menyimpan HANYA hasil TERBARU per record_no (rerun menimpa di tempat, `app.py:481-482` -- record hanya masuk `results` setelah `status` jadi `"done"`, gagal/cancelled TIDAK pernah masuk `results`). Ini berarti `len(session["results"])` SUDAH PERSIS "unique records dgn hasil valid terbaru" -- tidak perlu bookkeeping baru sama sekali utk hitungan ini.
- `_process_one_record` (`app.py:448-497`) adalah SATU funnel utk single run/bulk run/index-range run/rerun -- semua otomatis konsisten lewat titik ini.

**Perubahan kode:**
- `app.py` `_run_ocr_and_format`: tambah `t0 = time.monotonic()` di awal fungsi + `processing_time_seconds` (round 3 desimal) di dict hasil, sebelum `return` -- SATU-SATUNYA perubahan pada fungsi ini, murni instrumentasi waktu, TIDAK mengubah logika ekstraksi/comparison/decision apa pun.
- `live_evaluation.py` (BARU, root, mengimpor HANYA `comparison`+`extractors`, TIDAK mengimpor `app.py`): `build_session_evaluation(session)` -- satu-satunya entry point, dipanggil endpoint baru. Menggunakan ULANG `match`/`ocr_result`/`data_entry`/`status` yang SUDAH dihitung `comparison.attach_data_entry` saat proses (bukan re-derive dari nol).
- `app.py`: endpoint baru `GET /api/sheet/evaluation/{session_id}` (antara `sheet_result` dan `sheet_export`), pure read, 404 kalau sesi tidak ada (format error SAMA dgn endpoint sheet lain).
- `static/index.html`: tombol `#sheetEvalBtn` baru di toolbar Record Table (setelah grup Export Excel); modal pertama di codebase ini (`#evalModalBackdrop`/`.eval-modal-card`, overlay `position:fixed` + dim backdrop) -- HANYA memakai token visual yang SUDAH ada (font Segoe UI, biru `#2563eb`, radius/shadow `.workspace-block`, warna badge hijau/merah/abu yang sudah ada) -- TIDAK ada palet baru, TIDAK ada library chart baru.

**Field yang dievaluasi (Akurasi per Field) -- HANYA 5 field yang punya ground truth nyata** (`comparison.COLUMN_FIELD_MAP`): `nama_nasabah`, `nomor_rekening`, `nominal_penempatan`, `tenor_penempatan`, `bentuk_reward`. `unit_kerja_pengelola_rekening` TIDAK PERNAH punya kolom GT di skema manapun -- dikecualikan total, bukan alpa.

**Definisi metrik:**
- Accuracy per field = matched/measurable, dari `match` yang SUDAH dihitung `attach_data_entry` saat proses (`True`=matched, `False`=mismatched, `None`=tidak measurable/dilewati). `match=="uncertain"` (HANYA `nama_nasabah`, fuzzy-match state ke-3) DIKECUALIKAN dari numerator MAUPUN denominator accuracy -- dilaporkan terpisah sbg `uncertain_rate` ("jangan anggap GT yang belum pasti sbg salah").
- Precision/Recall/F1 HANYA valid utk `tenor_penempatan` (kelas `{1,3,6}`) dan `bentuk_reward` (kelas `{tunai,non_tunai}`) -- satu-satunya field kategorikal dgn kelas tetap. 3 field teks/angka lain SELALU `"N/A"` utk kolom ini (tidak pernah diisi angka karangan).
- `overall.accuracy` = rata-rata (unweighted) accuracy dari SEMUA field yang measurable >=1. `overall.precision_macro/recall_macro/f1_macro` = rata-rata HANYA dari macro-score `tenor_penempatan` dan `bentuk_reward` sendiri -- accuracy 3 field teks TIDAK PERNAH ikut dirata-rata ke sini atau menggantikan P/R/F1 yang tidak ada (keputusan eksplisit user).
- Runtime = `processing_time_seconds` (per DOKUMEN, bukan per field) dari `time.monotonic()` yang membungkus SELURUH `_run_ocr_and_format` (ekstraksi + simpan gambar debug + comparison/decision). Hasil lama (sebelum fitur ini ada) yang belum punya field ini otomatis dilewati (`isinstance` guard), TIDAK di-nol-kan/crash.
- `extraction_collapsed` = pakai ULANG `extractors._is_extraction_collapsed()`/`ocr_meta["extraction_collapsed"]` APA ADANYA -- TIDAK ada definisi kedua.

**Confusion matrix -- Tenor & Bentuk Reward (GT nyata):** dibangun dari `ocr_result`/`data_entry` mentah tiap row (sudah tersimpan), dinormalisasi ULANG lewat `comparison._normalize_for_compare()` (fungsi yang SAMA dipakai `attach_data_entry`, bukan reimplementasi) ke kelas tertutup; record yang salah satu sisinya di luar kelas valid (kosong/tak terbaca) DIKECUALIKAN dari matrix, bukan dihitung sbg kelas salah.

**Confusion matrix -- TTD Nasabah & TTD BRI (GT PENDEKATAN/SEMENTARA, keputusan eksplisit user):** tidak ada kolom GT terstruktur utk signature di skema manapun. Atas permintaan user ("pakai kolom verif dulu, dataset proper menyusul"), `live_evaluation._derive_signature_gt_from_verification()` membaca kolom teks bebas `"Status Verifikasi Form Pendaftaran Nasabah"` scr KONSERVATIF: teks berawalan "ok" -> kedua TTD GT="present"; TTD spesifik hanya diklaim "absent" kalau teks EKSPLISIT menyebutnya hilang (regex "ttd/tanda tangan ... nasabah" / "... bri/unit kerja" + "tidak ada"); selain itu GT=None (dikecualikan, TIDAK PERNAH ditebak). Diverifikasi thd data nyata (`assets/ocr_evaluation.xlsx`, 4 nilai unik kolom ini): "OK, Lanjut Proses" -> present/present; "Tolak, Tidak Ada Tanda Tangan Pihak BRI" -> atasan=absent, nasabah=None; "Tolak, Tunai/Non Tunai?" (tanpa sebut TTD) -> keduanya None. **Ini PENDEKATAN, BUKAN dataset per-signature yang sebenarnya** -- caption di modal & di sini menyatakan ini eksplisit. Kolom tidak ada di sheet upload lain -> matrix otomatis `"available": false`, tidak fabrikasi.

**CTA enable/disable:** `updateEvalButtonState()` (`sheetEvalBtn.disabled = !Object.values(batchStatus).some(s => s === "done")`), dipanggil dari 2 titik yang SUDAH ADA (`updateStatusCells()` -- tiap poll tick + sesudah rerun; `renderSheetTable()` -- saat sesi baru DAN saat restore dari localStorage) -- menutupi semua skenario wajib (single/bulk/index-range run, rerun, sesi dipulihkan) tanpa wiring baru per aksi. Record yang cuma "loaded" (belum diproses, status `pending`/`queued`) TIDAK PERNAH mengaktifkan tombol.

**API contract:** `GET /api/sheet/evaluation/{session_id}` -> 404 `{"error":...}` kalau sesi tak ada, else objek dgn keys `engine`/`engine_label`/`documents_evaluated`/`documents_total`/`overall`/`field_accuracy`/`confusion_matrices`/`runtime`/`extraction_collapsed`. Sesi 0-record diproses -> bentuk sama, semua angka `"N/A"`/0 (bukan cabang khusus di kode -- setiap helper degradasi alami dgn list kosong).

**Test yang BENAR-BENAR dijalankan (bukan rencana):**
1. Modul `live_evaluation.py` diuji unit dgn 4 record sintetis (match/mismatch/uncertain/collapsed campuran) -- semua angka (accuracy per field, confusion matrix, macro P/R/F1, overall) dicocokkan manual dgn hitungan tangan, SEMUA cocok persis.
2. Server dijalankan nyata (`uvicorn app:app`), upload `assets/ocr_evaluation.xlsx` via `/api/sheet/upload` -> `GET /api/sheet/evaluation/{id}` sblm proses apa pun -> `documents_evaluated: 0`, bentuk lengkap `"N/A"`, TIDAK error.
3. `POST /api/sheet/submit` dgn `engine="v18"` -> GAGAL (`AttributeError: partially initialized module 'paddle'...`) -- **bug lingkungan PaddleX/paddle circular-import PRE-EXISTING, TIDAK terkait perubahan sesi ini** (traceback murni di dalam `paddlex`/`paddle` internal, V18 sendiri dibekukan §9f & tidak disentuh). Dikonfirmasi: record error TIDAK masuk `session["results"]`, `documents_evaluated` TETAP 0 -- jalur pengecualian record gagal bekerja benar.
4. Restart server bersih, proses 1 record nyata dgn `engine="qwen3_vl_local"` (~97 detik) -> `GET /api/sheet/evaluation/{id}` mengembalikan `documents_evaluated:1`, `engine_label` dari `extractors.ENGINE_LABELS` (bukan hardcode), `runtime.average_seconds:96.765` (timing instrumentasi baru TERBUKTI jalan), confusion matrix TTD terisi dari heuristik verifikasi (record ini "OK, Lanjut Proses" -> present/present, cocok dgn prediksi -> akurasi 1.0).
5. Rerun record yang sama -> `documents_evaluated` TETAP 1 (bukan 2), `runtime.average_seconds` berubah ke `74.64` (run baru) -- perilaku "hasil terbaru menggantikan, jumlah tidak bertambah" terkonfirmasi NYATA, bukan cuma dianalisis dari kode.
6. Fungsi render modal (`renderEvaluationModal`/`renderEvalConfusionMatrix`) diekstrak & dijalankan via Node.js langsung thd response API NYATA dari test #4 -- HTML yang dihasilkan diperiksa: tidak ada `"undefined"`/`"NaN"` bocor, semua 4 confusion matrix + tabel per-field + runtime + extraction status ter-render lengkap.
7. `node --check` thd seluruh `<script>` block -- sintaks JS valid.
8. `python -m py_compile app.py live_evaluation.py extractors.py comparison.py` -- semua valid.

**BELUM diuji (keterbatasan sesi ini, bukan diklaim selesai):** klik tombol sungguhan di browser asli (Chrome tool ditolak user sesi ini) -- verifikasi UI dilakukan via ekstraksi+eksekusi Node thd fungsi render yang SAMA persis, bukan screenshot visual nyata. Skenario "proses 3 lagi jadi 4", "sesi baru mereset ke 0", dan "restore session" TIDAK diuji live (mekanismenya identik dgn yang SUDAH diuji nyata di atas -- `len(dict)` Python & `batchStatus` yang sama -- tapi belum di-observasi langsung).

**File berubah:** `app.py`, `live_evaluation.py` (baru), `static/index.html`. **TIDAK disentuh:** `comparison.py`/`extractors.py` (dipakai ulang, bukan dimodifikasi), V18/ROI/preprocessing, logika Qwen/hosted, threshold comparison, fallback logic.

## 9i. V20 — Hybrid Engine "qwen3_vl_local_yolos" (Qwen3-VL Local + YOLOS Signature Detector) (SELESAI, DIVERIFIKASI NYATA thd bobot yg benar-benar berjalan lokal)

Engine BARU, ADDITIVE thd `qwen3_vl_local` (TIDAK diubah/diganti sama sekali) -- id `qwen3_vl_local_yolos`, label `"Qwen3-VL Local + YOLOS Signature"`. Tujuan: Qwen3-VL Local tetap membaca SEMUA field semantik (nama/rekening/unit kerja/nominal/tenor/reward), tapi `signature_nasabah`/`signature_atasan` diambil dari detektor objek lokal (YOLOS) sbg sumber OTORITATIF, bukan penilaian Qwen sendiri -- utk kerahasiaan dokumen (tanda tangan tidak boleh lewat jalur model apa pun yg BISA menyentuh infrastruktur hosted) dan perbandingan akurasi vs `qwen3_vl_local` polos.

**Riwayat vendor (penting utk konteks, bukan lagi arsitektur aktif)**: target awal sesi ini adalah `tech4humans/yolov8s-signature-detector` (YOLOv8s via `ultralytics`), TAPI repo HF-nya **GATED** -- `hf_hub_download` gagal dgn `GatedRepoError` (401) MESKI dgn `HF_TOKEN` valid, mewajibkan user login manual di huggingface.co dan menyetujui akses via browser, TIDAK BISA lewat API/skrip apa pun. User diberi pilihan (AskUserQuestion): tunggu approval manual, ATAU cari alternatif non-gated. **User memilih alternatif** ("Look for an alternative non-gated source instead"), lalu dari kandidat yg ditemukan (web research sesi ini), user memilih scr eksplisit **`mdefrance/yolos-tiny-signature-detection`** ("Swap to mdefrance/yolos-tiny-signature-detection -- Recommended"). Ini vendor & arsitektur BERBEDA (YOLOS/ViT via `transformers`, BUKAN YOLOv8/ultralytics) -- semua penamaan "Tech4Humans" di kode/env/UI diganti total ke penamaan generik (`signature_detector.py`, `SIGNATURE_DETECTOR_*`) supaya tidak salah merepresentasikan sumbernya. Modul lama `tech4humans.py` **DIHAPUS**, bukan disimpan berdampingan.

**Arsitektur:**
```
Dokumen
  |
  +-- Qwen3-VL Local (vlm.extract_direct_semantic_local, SATU panggilan,
  |   BUKAN _majority -- skip 3x vote sinyal tanda tangan sepenuhnya)
  |     -> nama/rekening/unit_kerja/nominal/tenor/reward
  |
  +-- YOLOS signature detector (signature_detector.detect_signatures, HANYA
      dokumen, TANPA template) -> deteksi bbox+confidence
        -> extractors._assign_signature_roles (geometri SIGNATURE_CONFIG
           yg SUDAH ada, kiri=nasabah/kanan=atasan) -> signature_nasabah/atasan
  |
  v
common schema (extractors._adapt_qwen_to_common, override 2 baris signature
SAJA) -> adapt_common_to_pipeline_shape (TIDAK diubah) -> comparison.py
(TIDAK diubah) -> keputusan akhir
```

**Hard local-only gate (kerahasiaan dokumen)**: `extractors._hybrid_readiness()` mengecek DUA dependency LOKAL SEBELUM apa pun lain dijalankan (baris PERTAMA `run_qwen3_vl_local_yolos`, sebelum `_qwen_prepare_image` sekalipun): `vlm._resolve_model_path()` (Qwen lokal) dan `signature_detector.is_ready()` (YOLOS). Kalau salah satu `"blocked"`, `ExtractorError` (stage `"API_AUTH"`) langsung dilempar dgn pesan PERSIS yg diminta user -- TIDAK PERNAH Qwen jalan dulu baru gagal belakangan. Diverifikasi NYATA DUA KALI: sekali thd modul lama (`tech4humans.py`, sblm dihapus) dan sekali lagi thd `signature_detector.py` setelah swap (path model sengaja dikosongkan -> `ExtractorError` langsung, dokumen TIDAK sempat diproses).

Kode engine ini TIDAK PERNAH mengimpor/memanggil apa pun yg menyentuh HF hosted inference, Gemini, atau API manapun -- `signature_detector.py`'s satu-satunya titik yg menyentuh file model adalah `AutoImageProcessor.from_pretrained(local_dir)`/`AutoModelForObjectDetection.from_pretrained(local_dir)` dgn `local_dir` berupa PATH LOKAL (bukan repo id), yg dikonfirmasi tidak pernah menghubungi jaringan.

**Model YOLOS**: `mdefrance/yolos-tiny-signature-detection` di Hugging Face Hub -- **TIDAK gated**, lisensi **Apache-2.0** (lebih permisif dari AGPL-3.0 Tech4Humans), arsitektur YOLOS (ViT-based object detector) single-class (`"signature"`), diakses via `transformers.AutoImageProcessor`/`AutoModelForObjectDetection` -- API yg SAMA persis yg SUDAH jadi dependency wajib project ini utk Qwen, jadi **NOL dependency pip baru** (beda dgn upaya Tech4Humans/`ultralytics` sebelumnya). Bobot diunduh SATU KALI sesi ini via `from_pretrained(repo_id)` lalu `.save_pretrained("models/yolos-tiny-signature-detection/")` (konvensi penyimpanan sama spt `models/Qwen3-VL-4B-Instruct/`, gitignored) -- langkah manual satu kali, BUKAN sesuatu yg dijalankan runtime code. Path default `models/yolos-tiny-signature-detection/`, override via `SIGNATURE_DETECTOR_MODEL_PATH` (nama generik, bukan nama vendor, krn detektor mungkin berganti lagi nanti -- sama spt `vlm.py`'s `VLM_MODEL_PATH`).

**Dependency**: TIDAK ADA yg baru. Blok `ultralytics` yg sempat ditambahkan ke `requirements.txt` utk upaya Tech4Humans DIHAPUS (diganti komentar penjelasan riwayat + peringatan eksplisit "Do not reintroduce `ultralytics` without re-reading handover.md's V20 section first" -- mencegah pengulangan insiden bentrok `opencv-python`/`opencv-python-headless` yg sempat nyata merusak instalasi `cv2` sesi ini sebelum diperbaiki). Paket `ultralytics` & dependency eksklusifnya di-uninstall dari environment supaya cocok dgn `requirements.txt` yg sudah diperbarui.

**Signature role assignment (geometri yg SUDAH ADA, BUKAN sistem ROI baru, TIDAK berubah dari upaya Tech4Humans)**: `extractors._assign_signature_roles()` memakai `preprocessing.SIGNATURE_CONFIG`'s `value_bbox` (nasabah x∈[0.1678,0.4530], atasan x∈[0.5872,0.8305], y∈[0.7981,0.8622] sama utk keduanya) sbg GERBANG REGION LONGGAR (margin x=0.06, y=0.08) thd gambar hasil `_qwen_prepare_image` (HANYA EXIF+rotasi kasar, TANPA perspective warp) -- BUKAN crop presisi piksel. Deteksi dgn confidence tertinggi per region dipilih; kalau tidak ada deteksi di region manapun -> `absent` (confidence=None, TIDAK ditebak). Logika ini detector-agnostic dan TIDAK disentuh sama sekali saat swap vendor.

**Status/confidence logic (DUA threshold terpisah)**: (1) `SIGNATURE_DETECTOR_CONF_THRESHOLD` (default 0.25) -- threshold PROPOSAL deteksi YOLOS sendiri, dipakai `signature_detector.detect_signatures()`'s `processor.post_process_object_detection(outputs, threshold=conf, ...)`; (2) begitu ada deteksi, confidence-nya diteruskan APA ADANYA ke `extractors._signature_state()` yg SUDAH ADA (threshold 0.6) -- TIDAK ADA threshold "final" baru diciptakan sesi ini. **0.25 HANYA spot-checked thd SATU dokumen nyata sesi ini** (lihat Validasi #4) -- belum ditera thd sampel lebih besar.

**Qwen 3x signature vote**: DI-SKIP SEPENUHNYA utk engine ini -- `run_qwen3_vl_local_yolos` memanggil `vlm.extract_direct_semantic_local()` (single-call) bukan `extract_direct_semantic_local_majority()`, krn jawaban tanda tangan Qwen toh akan ditimpa YOLOS. Jawaban tanda tangan Qwen sendiri TETAP disimpan sbg diagnostik (`ocr_meta["qwen_raw_signature_diagnostic"]`), TIDAK PERNAH jadi hasil akhir.

**Extraction Collapsed**: TIDAK diubah -- `extractors._is_extraction_collapsed(qwen_result)` dipanggil PERSIS spt engine lain, thd `qwen_result` SEBELUM override sinyal tanda tangan (fungsi ini HANYA baca 5 field semantik inti). Skenario "semantik collapsed TAPI tanda tangan present" valid & didukung.

**Evaluation compatibility**: `live_evaluation.py` **TIDAK disentuh sama sekali** -- `_engine_info()` sudah generik (baca string id engine APA PUN dari hasil tersimpan), dikonfirmasi NYATA DUA KALI sesi ini (sekali sblm swap dgn id `qwen3_vl_local_tech4humans`, sekali lagi setelah swap dgn id final `qwen3_vl_local_yolos` lewat sesi live server sungguhan): `GET /api/sheet/evaluation/{id}` melaporkan `engine: "qwen3_vl_local_yolos"` scr independen dari `qwen3_vl_local`, tanpa satu baris pun kode `live_evaluation.py` berubah.

**File berubah**: `extractors.py` (`ENGINE_LABELS`, `run()` dispatch, `_hybrid_readiness`, `_assign_signature_roles`, `run_qwen3_vl_local_yolos`), `signature_detector.py` (BARU, menggantikan `tech4humans.py` yg DIHAPUS), `requirements.txt` (blok `ultralytics` dihapus, komentar riwayat ditambahkan), `.env.example` (`TECH4HUMANS_*` diganti `SIGNATURE_DETECTOR_MODEL_PATH`/`SIGNATURE_DETECTOR_CONF_THRESHOLD`), `static/index.html` (opsi dropdown di KEDUA Tab 1 & Tab 2 diganti nama final), `models/yolos-tiny-signature-detection/` (bobot BARU, diunduh nyata). **TIDAK disentuh**: `qwen3_vl_local` (fungsi lamanya, byte-for-byte sama), V18, Gemini, Qwen hosted, `comparison.py`, `live_evaluation.py`, `preprocessing.py`/`postprocessing.py`, `app.py`.

**Validasi yg BENAR-BENAR dijalankan (server/proses nyata, bukan rencana, TANPA mocking utk bagian ini):**
1. `python -m py_compile app.py extractors.py signature_detector.py vlm.py comparison.py live_evaluation.py` -- semua valid.
2. Bobot `mdefrance/yolos-tiny-signature-detection` diunduh NYATA (repo memang tidak gated, berhasil tanpa hambatan) dan disimpan ke `models/yolos-tiny-signature-detection/`.
3. Hard local-only gate diuji ulang dgn `signature_detector.py`: path model dikosongkan/tidak ada -> `is_ready()` -> `False` dgn pesan jelas; `extractors.run("qwen3_vl_local_yolos", ...)` LANGSUNG melempar `ExtractorError` (stage `API_AUTH`), TIDAK ada dokumen yg sempat diproses.
4. **Uji end-to-end NYATA, ZERO mocking** thd dokumen nyata: `signature_detector.detect_signatures()` dijalankan APA ADANYA (kode produksi persis) thd gambar dokumen asli -- menghasilkan **5 deteksi nyata** dgn confidence bervariasi (termasuk satu kasus tepi yg jatuh sedikit DI LUAR margin y region tanda tangan -- dgn sengaja DITOLAK oleh gerbang region, bukti gerbang region bekerja thd data nyata bukan cuma sintetis). `run_qwen3_vl_local_yolos` penuh dijalankan thd dokumen yg sama dgn Qwen3-VL Local ASLI (bukan mock): **field semantik IDENTIK 100%** (`SEMANTIC MATCH: True`) dgn `qwen3_vl_local` polos pd dokumen yg SAMA; `signature_nasabah`/`signature_atasan` keduanya `"present"` dgn confidence NYATA dari YOLOS (nasabah 0.9586, atasan 0.9962) -- BEDA dari confidence self-rated Qwen di `qwen_raw_signature_diagnostic`, membuktikan override BENAR-BENAR terjadi dari deteksi objek nyata, bukan kebetulan cocok.
5. **Sesi live via server FastAPI sungguhan** (upload `assets/ocr_evaluation.xlsx`, submit record 1 dgn `engine=qwen3_vl_local_yolos` lewat `POST /api/sheet/submit`, poll `GET /api/sheet/batch-status` sampai selesai): record selesai `status=done` dlm 47.016 detik, `signature_nasabah`/`signature_atasan` keduanya `"present"`; `GET /api/sheet/evaluation/{id}` mengonfirmasi `documents_evaluated: 1` dan `engine: "qwen3_vl_local_yolos"` / `engine_label: "Qwen3-VL Local + YOLOS Signature"` scr independen, tanpa perubahan kode `live_evaluation.py`.
6. `ENGINE_LABELS`/`ENGINES` dikonfirmasi memuat id final; opsi dropdown dikonfirmasi muncul persis 2x (Tab 1 & Tab 2) dgn label final di `static/index.html`.
7. Environment dibersihkan: `ultralytics` & dependency eksklusifnya di-uninstall, `models/tech4humans-yolov8s-signature-detector/` (placeholder kosong) dihapus.

**BELUM diuji / keterbatasan diketahui (JANGAN dianggap selesai):**
- Margin region (0.06/0.08) hanya diamati thd SATU dokumen nyata (5 deteksi, termasuk 1 kasus tepi di luar margin) -- belum ditera thd sampel lebih besar utk memastikan margin tidak memotong tanda tangan asli atau meloloskan noise di dekat tepi region.
- `SIGNATURE_DETECTOR_CONF_THRESHOLD=0.25` default HANYA spot-checked thd satu dokumen -- belum divalidasi thd sampel lebih besar/variasi kualitas scan.
- Skenario "area kosong -> BUKAN false present" dan "kasus ambigu -> uncertain" thd dokumen tanpa tanda tangan sungguhan belum diuji nyata sesi ini -- baru dokumen DENGAN tanda tangan jelas yg diuji end-to-end.
- Bulk/Index-Range run & rerun utk engine baru ini pakai jalur `_process_batch`/`_process_one_record` yg SAMA persis dgn engine lain (tidak ada percabangan khusus di `app.py`) -- secara desain otomatis berfungsi, TAPI belum di-klik-coba scr manual lewat UI (Chrome tool ditolak user sesi lalu, sama spt fitur Live Evaluation sebelumnya) atau lewat batch >1 dokumen via API.

## 9j. V21 — Runtime + Accuracy Quick-Wins pd "qwen3_vl_local_yolos" (SELESAI, terverifikasi full 25 dokumen)

Dipicu permintaan user: "extractor yg combine qwen dgn signature detection sudah bagus di runtime, ada ide utk tingkatkan runtime atau bikin qwen baca lebih presisi?". 4 perubahan quick-win (low-risk, additive, HANYA menyentuh `qwen3_vl_local_yolos` kecuali disebutkan lain), disepakati via plan mode sebelum implementasi:

**1 — YOLOS ke GPU (`signature_detector.py`).** Root cause: modul ini TIDAK PERNAH memanggil `.to(device)` sama sekali sejak V20 -- YOLOS selalu jalan di CPU walau Qwen jalan di CUDA dlm proses yg SAMA (beda dgn `vlm.py` yg sudah punya pola `_choose_device`). Fix: `SIGNATURE_DETECTOR_DEVICE` env var (pola SAMA persis dgn `VLM_DEVICE`), model+input dipindah ke CUDA kalau tersedia. Diverifikasi nyata: `_device == "cuda"`, inference warm ~26ms/panggilan (vs beberapa detik cold-start CPU), override `SIGNATURE_DETECTOR_DEVICE=cpu` tetap berfungsi sbg fallback.

**2 — Attention backend dipin eksplisit (`vlm.py` `_load()`).** Diverifikasi LANGSUNG thd model yg sudah termuat: `model.config._attn_implementation == "sdpa"` SUDAH jadi default transformers di stack ini (torch 2.14.0+cu126, transformers 5.16.1) -- TIDAK ada regresi/perubahan perilaku, cuma dipin eksplisit (`attn_implementation="sdpa"`, dgn fallback `TypeError` kalau versi lebih lama tidak kenal argumen ini) supaya upgrade library nanti tidak diam-diam mengganti default. `flash_attention_2` SENGAJA TIDAK dipakai: tidak terpasang (`No module named 'flash_attn'`), rapuh dibuild di Windows, dan sesuai kesepakatan risk-conservative dgn user.

**3 — Mitigasi "Extraction Collapsed" via subprocess retry (`extractors.py` + `collapse_retry_worker.py` BARU).** §9g menemukan collapse itu deterministic DI DALAM satu proses tapi bervariasi ANTAR peluncuran proses -- artinya retry in-process PASTI mengulang hasil collapse yg sama (greedy decoding, `do_sample=False`). Fix: `_retry_collapsed_in_subprocess()` menulis image+crops+manifest JSON ke temp dir, spawn `sys.executable collapse_retry_worker.py <manifest> <output>` (proses Python BENAR-BENAR baru -> model dimuat ulang dari nol, kemungkinan real dpt kernel-selection CUDA/bitsandbytes yg beda), timeout `COLLAPSE_RETRY_TIMEOUT_S` (default 240s, generous krn nanggung load model ulang ~15-19s + 1x inference). Hasil retry: kalau BUKAN collapse lagi -> dipakai sbg qwen_result final (`ocr_meta.collapse_retry = "recovered"`); kalau gagal/timeout/collapse lagi -> fallback ke hasil asli TANPA crash (`ocr_meta.collapse_retry = "failed"`). Diverifikasi dgn SIMULASI (monkeypatch `vlm.extract_direct_semantic_local` utk memaksa collapse palsu di panggilan pertama): jalur recovered terbukti bekerja (nilai field kembali benar), jalur failed jg terbukti graceful (status tetap `not_detected`, tidak crash, tidak ada temp file bocor).

**4 — Close-up crop utk 3 field terlemah (`vlm.py` + `extractors.py`).** Target: `nama_nasabah`/`nomor_rekening`/`nominal_penempatan` (V16 full-eval: 38.9%/57.9%/47.6%, jauh di bawah `bentuk_reward` 100%). Desain: `extractors._get_weak_field_crops()` menjalankan `prep.prepare_and_align()` (warp perspektif ke template) LALU crop `prep.FIELD_CONFIG[field]["value_bbox"]` statis dgn padding `WEAK_FIELD_CROP_PAD_RATIO=0.5` -- crop ini dikirim sbg image TAMBAHAN (bukan panggilan kedua) dlm SATU panggilan Qwen yg sama (`vlm.extract_direct_semantic_local(image_bgr, extra_crops=[...])`), prompt `DIRECT_SEMANTIC_PROMPT` diperluas dinamis via `_build_direct_semantic_prompt()` HANYA kalau ada crop (byte-identical utk semua engine lain, diverifikasi via assert). Gambar utama (full-page, coarse-rotation-only) TIDAK disentuh sama sekali.

**Ditemukan & dibuang sebelum versi final (jangan diulang):**
- Rencana awal memakai `dynamic.extract_dynamic_fields()` (localisasi ber-OCR V14-V17 yg lebih presisi) utk bbox crop -- DIBUANG krn `PaddleOCR` TERBUKTI gagal diimpor di environment ini (`import ocr; ocr.get_ocr_engine()` SENDIRIAN, tanpa vlm/torch apa pun, melempar `AttributeError: partially initialized module 'paddle' has no attribute 'tensor' (circular import)`). Ini bug LINGKUNGAN yg sudah ADA SEBELUM sesi ini (mempengaruhi jg engine V18 & evaluation.py/stage_evaluation.py kapan pun mereka lewat jalur OCR ini), BUKAN sesuatu yg dirusak sesi ini -- di luar scope diperbaiki (risk-conservative). Diganti box statis `FIELD_CONFIG` saja (tanpa OCR).
- Padding 0.5 + hard-clamp ke batas tetangga (`prep._clamp_field_box_y`, pola SAMA dgn V14/V17) DICOBA lalu DIBUANG: diverifikasi VISUAL bahwa 3 baris identity_area (`nama_nasabah`/`nomor_rekening`/`unit_kerja`) punya GAP NOL persis di `value_bbox` template (dikonfirmasi numerik: batas bawah nama_nasabah == batas atas nomor_rekening, 311px == 311px pd satu dokumen nyata), dan tulisan tangan asli SERING jatuh DI BAWAH batas administratif itu (temuan row-drift V14/V15 yg sama) -- clamp ke batas tetangga memotong crop jadi LEBIH KECIL drpd box mentah tanpa padding sama sekali, menyembunyikan justru tulisan yg ingin diperjelas. Keputusan akhir: padding TANPA clamp (risiko sebaliknya -- sedikit bocoran baris tetangga -- diterima krn ini cuma citra SUPLEMEN utk VLM yg jg melihat halaman penuh & diberi tahu apa yg harus difokuskan, bukan jendela OCR pixel-diff yg gampang bingung oleh tinta ekstra).

**Hasil full 25-dokumen PERTAMA KALI utk `qwen3_vl_local_yolos`** (`eval_runs/evaluation_summary_v21_optim.json`; dijalankan via wrapper batch 4x proses terpisah krn `evaluation.py` biasa/`--limit` kena OOM-kill BERULANG oleh sistem walau RAM bebas ~8.6GB saat dicek manual -- pola SAMA persis dgn §10 poin 7/§12 V17, BUKAN regresi baru; wrapper reuse `evaluate_record()`/`build_summary()`/`write_csv()` apa adanya sesuai pola yg sudah didokumentasikan, TIDAK mengubah `evaluation.py`):

| Field | Akurasi (25 dok, V21) | Referensi historis terdekat |
|---|---|---|
| nama_nasabah | **0.60 (15/25)** | V16 (`v18`) 0.3889; `qwen3_vl_local` polos §9g (kena collapse) 0.42 |
| nomor_rekening | **0.56 (14/25)** | V16 (`v18`) 0.5789; `qwen3_vl_local` polos §9g 0.47 |
| nominal_penempatan | **0.84 (21/25)** | V16 (`v18`) 0.4762 |
| tenor_penempatan | 0.76 (19/25) | tidak dapat crop (di luar scope V21) |
| bentuk_reward | 1.00 (24/24) | tidak dapat crop (di luar scope V21) |
| pipeline_success_rate | 1.00 (0 gagal/crash) | -- |
| average_processing_time_sec | **30.83s** | live-server §9i (SEBELUM V21) 47.016s; §9h `qwen3_vl_local` biasa 74-97s |

**Catatan jujur**: ini BUKAN perbandingan before/after terkontrol thd kode identik (4 perubahan diimplementasi sekaligus sebelum eval penuh pertama drpd 1-per-1 spt disiplin biasa sesi ini -- trade-off eksplisit krn tiap eval 25-dokumen makan ~15-20 menit runtime nyata & mesin ini gampang OOM, lihat di atas), jadi kontribusi individual tiap dari 4 perubahan thd angka akurasi TIDAK bisa dipisah scr pasti dari angka ini saja -- yang bisa diklaim dgn percaya diri: (a) tidak ada regresi/crash pd 25 dokumen manapun, (b) waktu proses turun nyata dibanding satu-satunya angka live-server sebelumnya utk engine yg SAMA, (c) 2 dari 3 field target crop (nama_nasabah, nominal_penempatan) naik SUBSTANSIAL dibanding referensi historis manapun yg ada, (d) `nomor_rekening` kira-kira setara/stabil, BUKAN regresi. Referensi historis "V16 (v18)" di atas jg BUKAN apple-to-apple (engine BEDA, `v18`=pipeline OCR+ROI klasik) -- dicantumkan sbg titik pembanding kasar SATU-SATUNYA yg tersedia, bukan baseline formal.

**Temuan terpisah, PENTING, di luar scope V21 -- laporkan ke user, jangan diabaikan:** PaddleOCR rusak di environment ini (lihat poin "Ditemukan & dibuang" di atas) -- traceback lengkap berakhir di `paddlex/utils/import_guard.py``import_paddle_module` -> `paddle/base/layers/math_op_patch.py:962` `AttributeError: partially initialized module 'paddle' has no attribute 'tensor'`. Reproducible via `import ocr; ocr.get_ocr_engine()` SENDIRIAN di proses Python baru, TANPA torch/vlm/extractors apa pun diimpor duluan -- jadi ini BUKAN konflik order-import dgn torch, kemungkinan besar masalah versi paddlex/paddleocr/numpy di venv ini. Ini memblokir jalur OCR apa pun (`ocr.py`/`dynamic_extraction.py`) yg dipakai engine `v18` DAN `stage_evaluation.py`/skrip lain yg lewat situ -- BELUM diinvestigasi/diperbaiki sesi ini (di luar scope "quick win" yg disepakati), next step yg valid utk sesi terpisah.

**File berubah**: `signature_detector.py` (device selection), `vlm.py` (`attn_implementation`, `_build_direct_semantic_prompt`, `extra_crops` param di `_extract_direct_semantic_local_core`/`extract_direct_semantic_local`), `extractors.py` (`_retry_collapsed_in_subprocess`, `_get_weak_field_crops`, `_pad_bbox`, wiring di `run_qwen3_vl_local_yolos`), `collapse_retry_worker.py` (BARU). **TIDAK disentuh**: `qwen3_vl_local`/`qwen3_vl` (majority-vote path lama, prompt tanpa crop byte-identical), V18, Gemini, `app.py`, `static/index.html`, `comparison.py`, `postprocessing.py`.

## 9k. V22 — Decision-Accuracy Metric, Perbaikan nomor_rekening, Validasi Digit-vs-Terbilang (SELESAI, terverifikasi full 25 dokumen)

Dipicu 3 permintaan user sekaligus: (1) popup "Evaluasi" perlu skor akurasi OK/TOLAK keseluruhan (prefix-based: GT mulai dgn "OK"/"Oke" vs prediksi "OK" -> benar; GT mulai dgn "Tolak" vs prediksi "Tolak" -> benar); (2) `nomor_rekening` field terlemah, minta beberapa pendekatan perbaikan + diagnosis apakah ini soal postprocessing; (3) validasi cross-check digit-vs-terbilang (mis. "10.000" harus konsisten dgn "sepuluh ribu") utk field yg punya 2 representasi nilai di form.

**Temuan awal penting (sebelum implementasi, mengoreksi asumsi user):**
- Popup "Evaluasi" (`live_evaluation.py`) TERNYATA **belum punya metrik decision-accuracy sama sekali** — hanya per-field accuracy + confusion matrix. `evaluation.py` (CLI) SUDAH punya prefix-based GT mapping (`_map_ground_truth_decision`, "ok"->OK/"tolak"->TOLAK) sejak lama — bukan bug yg perlu diperbaiki, tapi metrik yg perlu DITAMBAHKAN ke popup live, reuse logic yg sama.
- `nomor_rekening` **BUKAN terutama bug postprocessing** — `postprocessing.normalize_numeric` cuma strip non-digit + perbaiki huruf OCR-confusable (O->0 dst), justru membantu. Root cause asli: (a) `DIRECT_SEMANTIC_PROMPT` cuma kasih SATU baris instruksi utk field ini (`"nomor_rekening - account number"`), TIDAK ADA aturan anti-bleed spt `unit_kerja` yg sudah py; (b) field ini MIDDLE dari 3 baris identity_area ber-gap NOL (bocor dari ATAS & BAWAH, beda dgn `nama_nasabah` yg cuma bocor ke bawah); (c) tidak ada anchor visual khas (beda dgn `nominal_penempatan` yg ada "Rp"+kurung); (d) `validate_value()` cuma terima 4-30 digit -- nyaris tanpa filter. Diverifikasi LANGSUNG thd `assets/ocr_evaluation.xlsx`: nomor rekening BRI asli KONSISTEN 14-16 digit.
- Terbilang **TIDAK berlaku utk nomor_rekening** (nomor rekening tidak punya representasi kata di form) — berlaku utk `nominal_penempatan` & detail nilai `reward_tunai` (keduanya punya format cetak `"Rp………… (…………)"`, diverifikasi langsung thd `assets/template.pdf`). Teks terbilang SEBELUMNYA dibuang (`postprocessing.normalize_currency`: `text.split("(")[0]`), tidak pernah diparse balik. Pola desain "2 bukti independen harus konsisten" SUDAH ADA di `comparison.validate_tenor()`/`validate_reward()` (utk pasangan field berbeda) — dipakai ulang, bukan pola baru.

**Bagian 1 — Decision-accuracy metric (`live_evaluation.py`, `static/index.html`).** `_decision_accuracy_stats()` (baru) reuse `evaluation._map_ground_truth_decision` verbatim (import langsung, tanpa duplikasi logic) utk GT, baca `result["decision_v9_2"]["decision"]` utk prediksi, lipat ke biner (`"OK"` exact vs SISANYA jadi `"TOLAK"` — **Review dihitung SAMA spt Tolak per keputusan eksplisit user**, krn keduanya bukan auto-approval; GT di dataset asli TIDAK PERNAH mengandung "Review" jadi pelipatan ini cuma menyentuh sisi prediksi). Record tanpa GT ATAU tanpa keputusan (pipeline gagal) dikecualikan dari numerator MAUPUN denominator (prinsip "jangan hitung data hilang sbg salah" yg sudah dipakai di seluruh file ini). Output: `accuracy`, `false_accept_rate`/`false_reject_rate` (+count), `review_folded_into_tolak_count` (transparansi). Diverifikasi UNIT TEST manual (5 record buatan, semua kombinasi OK/TOLAK/REVIEW/missing) — angka cocok persis hitungan tangan. Frontend: 1 tile baru di summary row + 1 tabel detail (Accuracy/FAR/FRR + catatan Review-folding), reuse `evalFmtPct` yg sudah ada, TIDAK mengubah render branch lain.

**Bagian 2 — Perbaikan nomor_rekening**, scoped ke engine aktif `qwen3_vl_local_yolos` saja (V18 tetap beku):
- **(a) Prompt** (`vlm.py` `DIRECT_SEMANTIC_PROMPT`): tambah 1 aturan baru persis di antara aturan `nama`/`unit_kerja` yg sudah ada (bukan pola baru) — jelaskan posisi baris (di antara nama & unit_kerja), estimasi panjang nyata (15-16 digit), toleransi format titik, dan larangan bocor ke baris tetangga.
- **(b) Crop padding per-field** (`extractors.py`): `WEAK_FIELD_CROP_PAD_RATIO` diubah dari 1 rasio flat (0.5) jadi dict per-field `(pad_x, pad_y)` — `nomor_rekening` dapat `(0.5, 0.2)` (vertikal lebih sempit drpd `nama_nasabah`/`nominal_penempatan` yg tetap `(0.5, 0.5)`), krn field ini SATU-SATUNYA yg diapit baris tetangga di KEDUA sisi (dikonfirmasi ulang pixel-exact sesi V21 sebelumnya). Catatan jujur: sempat dicoba TANPA padding sama sekali + hard-clamp ke batas tetangga (`prep._clamp_field_box_y`, pola V14/V17) — DIBUANG, krn diverifikasi visual clamp memotong crop jadi LEBIH KECIL drpd box mentah tanpa padding (tulisan tangan asli sering jatuh DI BAWAH batas administratif baris, temuan row-drift V14/V15 yg sama) — padding tanpa clamp dipertahankan sbg keputusan akhir.
- **(c) Length sanity gate** (`extractors.py`, BUKAN `postprocessing.py` -- lihat catatan jujur di bawah): `_is_plausible_account_number()` baru, range 13-17 digit (margin dari 14-16 digit nyata), demote ke `not_detected` (BUKAN reject diam-diam) kalau di luar range — filosofi sama dgn V18 Fix 3 (`_identity_type_mismatch`). **Diverifikasi NYATA menangkap 1 kasus sungguhan**: model membaca "430531539" (9 digit, sebagian dari nomor asli) pd satu dokumen — gate benar menolaknya jadi review drpd diam-diam salah.
- **Catatan jujur PENTING (koreksi rencana awal)**: rencana semula menaruh gate ini di `postprocessing.build_text_field_result()` (titik singgah 4 jalur V18) — SALAH utk engine aktif. Diverifikasi LANGSUNG dgn membaca kode: `extractors.adapt_common_to_pipeline_shape()` (dipakai SEMUA engine non-V18) menaruh nilai Qwen LANGSUNG ke `final_results` via closure `put()`, TIDAK PERNAH lewat `build_text_field_result`/`normalize_numeric`/`validate_value` sama sekali — seluruh pipeline cleaning V18 itu HANYA berlaku utk engine `v18`. Gate dipindah ke `extractors.py` persis di titik `nomor_rekening` diproses di `adapt_common_to_pipeline_shape`, agar benar2 aktif utk engine yg dipakai sekarang.

**Bagian 3 — Validasi digit-vs-terbilang (`terbilang.py` BARU, `vlm.py`, `extractors.py`, `comparison.py`, `evaluation.py`, `live_evaluation.py`).**
- `terbilang.py` (baru, 0 dependency eksternal): `parse_terbilang(text) -> int|None`, grammar Indonesia standar (satuan/belasan/puluhan/ratusan/ribu/juta/miliar), dukung kontraksi "se-" ("seribu"/"sejuta") DAN 2 ejaan belasan ("limabelas" fused ATAU "lima belas" 2 kata). Diuji STANDALONE dulu (27 kasus unit test, termasuk nilai nyata dari sesi ini spt "empat puluh delapan juta tujuh ratus lima puluh ribu rupiah"->48750000) SEBELUM disambungkan ke apa pun, sesuai rencana verifikasi.
- `vlm.py`: `DIRECT_SEMANTIC_PROMPT` dapat field baru `nominal_penempatan_terbilang` (transkripsi kata dlm kurung, TERPISAH dari tugas baca digit); `REWARD_TUNAI_DETAIL_PROMPT` dapat field baru `reward_tunai_terbilang` (SEBELUMNYA instruksi promptnya eksplisit MELARANG model melaporkan teks terbilang -- instruksi itu diganti jadi "laporkan sbg field terpisah" utk kasus BARU ini saja; instruksi digit-only utk field aslinya tetap sama). `max_tokens` follow-up reward-tunai dinaikkan 40->80 (2 field skrg, bukan 1).
- `extractors.py`: 2 field terbilang baru diteruskan lewat `_adapt_qwen_to_common`/`adapt_common_to_pipeline_shape` sbg entry `raw_results` TERPISAH (BUKAN bagian `FIELD_ORDER`/tabel field yg ditampilkan -- diagnostic only, tidak mengubah UI).
- `comparison.validate_nominal_terbilang()` (baru): pola SAMA persis dgn `validate_tenor`/`validate_reward` ("2 bukti independen harus setuju"), tapi **SENGAJA belum disambungkan ke `_decide_from_checks`/keputusan OK-TOLAK-REVIEW** — verdict pakai vocabulary BEDA (`consistent`/`inconsistent`/`evidence_missing`, bukan OK/TOLAK/REVIEW) supaya tidak pernah tertukar dgn keputusan bisnis sungguhan. Alasan konservatif: cek BARU tanpa validasi skala-nyata dulu — promosi jadi decisive adalah follow-up terpisah stlh akurasinya diukur.
- `evaluation.py`/`live_evaluation.py`: `nominal_terbilang_both_evidence_read_count`/`nominal_terbilang_internal_consistency_rate`, pola SAMA dgn `tenor_both_evidence_read_count` yg sudah ada.
- **Diverifikasi NYATA menangkap 1 bug sungguhan** (bukan cuma lolos unit test): pd satu dokumen nyata, model membaca `nominal_penempatan`="5.000.00" (->500rb salah) TAPI `nominal_penempatan_terbilang`="Lima Juta rupiah" (->5.000.000 benar, dikonfirmasi visual crop sesi V21 sblmnya) — `validate_nominal_terbilang` BENAR melaporkan `"inconsistent"` dgn detail digit vs terbilang yg jelas, persis skenario yg diminta user.

**Hasil full 25-dokumen** (`eval_runs/evaluation_summary_v22_optim.json`, dibanding `v21_optim` sesi sebelumnya, wrapper batch 4x proses sama spt V21 krn `evaluation.py` masih OOM-kill di sistem ini):

| Metrik | V21 | V22 |
|---|---|---|
| nama_nasabah | 0.60 | 0.56 |
| **nomor_rekening** | 0.56 | **0.60** |
| nominal_penempatan | 0.84 | 0.84 |
| tenor_penempatan | 0.76 | 0.68 |
| bentuk_reward | 1.00 | 1.00 |
| decision_accuracy (CLI, 7-check strict) | (tdk diukur sesi V21) | 0.12 |
| decision-accuracy BARU (OK vs TOLAK biner, Review->Tolak) | -- (belum ada) | **0.16** (4/25 -- FRR 0.9091, sangat tinggi) |
| nominal_terbilang_internal_consistency_rate | -- (belum ada) | **0.9412** (16/17 evidence lengkap setuju) |
| avg_processing_time_sec | 30.83 | 34.46 |

**Catatan jujur (WAJIB dibaca sebelum menganggap sesi ini "berhasil/gagal")**: 4 perubahan (a/b/c Bagian 2 + skema Bagian 3) diuji SEKALIGUS dlm satu eval 25-dokumen, BUKAN 1-per-1 sesuai disiplin biasa sesi ini — trade-off eksplisit sama spt V21 (tiap eval 25-dokumen makan ~15-20 menit nyata + mesin ini gampang OOM, lihat §9j). Kontribusi individual TIDAK bisa dipisah pasti dari angka gabungan ini. Yang bisa diklaim percaya diri: (1) tidak ada crash/gagal di 25 dokumen manapun; (2) `nomor_rekening` naik +4 poin (56%->60%), TAPI sampel 25 dokumen + non-determinism model (dikonfirmasi ulang: menambah field ke prompt/schema mengubah SELURUH urutan token generate, jadi bahkan field yg TIDAK disentuh spt `tenor_penempatan` & `nama_nasabah` ikut bergeser -- turun 76%->68% & 60%->56%) berarti ini BUKAN bukti kuat, perlu run ulang/sampel lebih besar utk konfirmasi; (3) metrik decision-accuracy biner BARU **bekerja benar secara teknis** (diverifikasi unit test + data nyata) dan mengungkap temuan JUJUR yg sudah lama terdokumentasi (§10 poin 6): False Reject Rate 90.91% berarti mayoritas dokumen GT-OK tetap di-Tolak/Review oleh sistem — field-level accuracy yg lumayan TIDAK otomatis jadi decision-accuracy yg baik, krn 7-check butuh SEMUA benar sekaligus; ini BUKAN regresi dari sesi ini, cuma metrik baru yg pertama kali mengukurnya scr eksplisit; (4) validasi terbilang TERBUKTI menangkap kesalahan nyata (bukan cuma teori), consistency rate 94% pd kasus yg datanya lengkap adalah sinyal SEHAT bahwa fitur ini tidak berisik/salah-alarm berlebihan, tapi 17/25 "both evidence read" berarti masih ~1/3 dokumen blm dapat 2 bukti sekaligus (evidence_missing) -- belum diinvestigasi lebih lanjut, next step yg valid.

**File berubah**: `terbilang.py` (BARU), `live_evaluation.py` (`_decision_accuracy_stats`, `_nominal_terbilang_stats`, import `evaluation`), `static/index.html` (tile + tabel Decision Accuracy baru), `vlm.py` (prompt `nomor_rekening`+`nominal_penempatan_terbilang`+`reward_tunai_terbilang`, `DIRECT_SEMANTIC_FIELDS`), `extractors.py` (`_is_plausible_account_number`, `WEAK_FIELD_CROP_PAD_RATIO` per-field, `_pad_bbox` 2-ratio, wiring 2 field terbilang baru), `comparison.py` (`validate_nominal_terbilang`, import `terbilang`), `evaluation.py` (`nominal_terbilang_*` metrics). **TIDAK disentuh**: `postprocessing.py`/`preprocessing.py`/`dynamic_extraction.py` (V18 tetap beku), `app.py`, `pipeline.py`, `_decide_from_checks`/keputusan OK-TOLAK-REVIEW produksi (terbilang tetap diagnostic-only).

## 9l. V22 addendum — Fix Mirrored-Placeholder reward_tunai/reward_non_tunai + Perkuat Null-Default Choice Field (SELESAI)

Dipicu laporan user: "sebagian record sudah terekstrak benar, sebagian lain field reward_tunai/reward_non_tunai masih berisi literal string 'tunai'/'non_tunai'". Root cause dikonfirmasi di `extractors.adapt_common_to_pipeline_shape`: kapan pun `bentuk_reward` terbaca (mis. "tunai") TAPI follow-up detail extraction (`reward_tunai_detail`, panggilan Qwen terpisah) gagal/None, kode LAMA fallback ke STRING PILIHAN itu sendiri ("tunai") sbg placeholder — bukan nilai nominal asli, tapi terlihat spt "sudah terisi". Ini jg diam-diam merusak `comparison.validate_reward()`'s evidence check (`tunai_filled = bool(value)` — placeholder non-kosong bikin `tunai_filled=True` walau amount-nya TIDAK PERNAH benar2 terbaca, melewati validasi "2 bukti independen" yg jadi tujuan asal fungsi itu).

**Fix** (`extractors.py`): deteksi `"reward_tunai_detail" in common` utk bedakan engine yg BENAR2 mencoba ekstraksi detail terpisah (Qwen, key SELALU ada krn `_adapt_qwen_to_common` selalu set-nya lewat `wrap()`, walau value-nya None) vs yg TIDAK PUNYA konsep detail terpisah sama sekali (Gemini/Mistral, key TIDAK ADA di schema mereka). Utk Qwen: None tetap None, TIDAK PERNAH overwrite dgn placeholder. Utk Gemini/Mistral: placeholder LAMA dipertahankan (satu2nya cara `validate_reward` py bukti utk engine itu, "known deliberate simplification" yg sudah terdokumentasi sblm sesi ini — SENGAJA tidak diubah). Status field jg diperbaiki ikut: `"detected"` HANYA kalau value-nya benar2 terisi (sblmnya `"detected"` dipasang tiap kali choice cocok, walau value-nya kosong).

Diverifikasi NYATA (bukan cuma unit test sintetis): 1 dokumen dgn detail terbaca ("1000000") -> `reward_tunai` = nilai asli (unchanged, behavior lama sdh benar utk kasus ini); 1 dokumen dgn detail GAGAL terbaca (`bentuk_reward="non_tunai"`, follow-up return None) -> `reward_non_tunai` skrg `None` (SEBELUM fix: literal `"non_tunai"`).

**Perkuat null-default utk SEMUA choice field** (`vlm.py` `DIRECT_SEMANTIC_PROMPT`, `tenor_penempatan` & `bentuk_reward`): ditambah kalimat eksplisit "kalau TIDAK ADA mark sama sekali (bukan cuma ambigu) -> null, JANGAN default ke pilihan manapun" per permintaan user ("this happen for all the choice field we will extract") — sebelumnya prompt sudah bilang "null if unclear" tapi tidak eksplisit membedakan "ambigu/ragu" vs "benar2 tidak ada tanda apa pun", jadi diperjelas keduanya sama2 -> null.

**File berubah**: `extractors.py` (`detail_extraction_attempted` gate), `vlm.py` (prompt, tidak ada perubahan field/schema baru).

## 10. Prioritas Berikutnya (berdasar data V15.2/V17/V18 di atas, BUKAN tebakan)

**CATATAN V19f (§9f): V18 DIBEKUKAN** -- semua item di bawah ini adalah
perubahan internal V18/preprocessing/ROI, dan TIDAK BOLEH dikerjakan
sampai user secara eksplisit meminta lagi. Dibiarkan di bawah sbg catatan
riwayat/rencana, bukan instruksi aktif.

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

**(Restructure) Semua perintah di bawah sekarang pakai prefix `eval_tools/`**
-- file-nya pindah folder, isi/argumen persis sama, lihat catatan restructure
di akhir file ini.

```bash
# end-to-end (field accuracy + decision accuracy) -- paling relevan utk "akurasi asli"
python eval_tools/evaluation.py --dataset assets/ocr_evaluation.xlsx --label <nama_run> --debug-dir none

# 3-gate funnel (PREPROCESSING -> ROI -> OCR), CER, dst -- utk isolasi root cause per stage
python eval_tools/stage_evaluation.py --dataset assets/ocr_evaluation.xlsx --label <nama_run> --no-xlsx --debug-dir none

# uji cepat sebelum full run (hemat waktu ~12s/dokumen)
python eval_tools/evaluation.py --dataset assets/ocr_evaluation.xlsx --label smoke --limit 5 --debug-dir none

# bandingkan beberapa run (lihat §2, poin 4) -- path SEKARANG di eval_runs/
python eval_tools/compare_runs.py eval_runs/evaluation_summary_<a>.json eval_runs/evaluation_summary_<b>.json

# audit diagnostik signature_nasabah/signature_atasan (V16, §6) -- CSV per
# record: ink_area_ratio/spread_x/spread_y/component_count/change_ratio/status
# -- ditulis ke eval_runs/signature_diagnostics.csv
python eval_tools/signature_diagnostics.py --dataset assets/ocr_evaluation.xlsx
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

**(Restructure) File pindah folder, lihat catatan restructure di akhir file
ini utk detail lengkap. Peta di bawah ini SUDAH folder baru — kalau baca
referensi file:line di bagian atas handover ini (§1-§12, semua ditulis
SEBELUM restructure), nomor barisnya sudah tidak match lagi, cari fungsinya
pakai grep nama fungsi, bukan percaya nomor baris.**

- `core/` — `config.py`, `data_input.py`, `terbilang.py`, `paths.py` (baru,
  satu sumber utk semua path folder project).
- `pipeline/` — `preprocessing.py`, `ocr.py`, `postprocessing.py`,
  `dynamic_extraction.py`, `comparison.py`, `vlm.py`, `pipeline.py`.
- `extractors/` — `extractors.py`, `signature_detector.py`,
  `collapse_retry_worker.py`.
- `eval_tools/` — `evaluation.py`, `stage_evaluation.py`, `compare_runs.py`,
  `signature_diagnostics.py`, `live_evaluation.py`, `diagnose_collapse.py`,
  `local_qwen_eval.py`. Semua standalone CLI KECUALI `live_evaluation.py`
  (di-import `app.py` langsung, dipakai fitur popup "Evaluasi" di UI).
- Tetap di root: `app.py` (entrypoint, `python app.py` tidak berubah),
  `static/` (frontend, tidak disentuh), `prd.md`, `handover.md`.
- Data/model (tidak disentuh): `assets/`, `downloaded_documents/`, `models/`,
  `outputs/`, `uploads/`, `eval_runs/`.

Output evaluasi (csv/json hasil run) tetap di folder **`eval_runs/`** seperti
sebelumnya (V18, dibuat otomatis, di-gitignore) — restructure ini tidak
mengubah ke mana output ditulis, cuma dari mana script-nya dijalankan
(`python eval_tools/evaluation.py ...`, lihat §12).

File eksperimen versi lama (`pipeline1.py`/`comparison1.py`/`postprocessing1.py`/
`vlm1.py`/`index1.html`/dst, disebut di versi lama peta file ini) SUDAH TIDAK
ADA di repo — sudah dihapus/dibersihkan di sesi sebelum restructure ini,
paragraf lama yang menyebut file-file itu masih ada dihapus dari sini.

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
7. (Restructure) Tulis update singkat & lugas — kayak reminder ke diri
   sendiri, bukan laporan. Jangan tulis ulang apa yang sudah ada di bagian
   lain file ini, cukup pointer ke section-nya. Baca `prd.md` dulu kalau
   butuh gambaran scope/goals, baru masuk ke sini kalau butuh detail
   eksperimen/root-cause.

---

## 16. Restructure — Kode dipindah ke folder (2026-09-18)

Ini BUKAN perubahan fitur/logic. Semua ~20 file `.py` yang sebelumnya rata di
root project sekarang dikelompokkan per tanggung jawab, biar gampang dicari.
Detail lengkap ada di `prd.md` (bagian struktur project) — di sini cuma
ringkasan apa yang berubah & cara verifikasinya.

**Yang pindah:**
- `core/` — `config.py`, `data_input.py`, `terbilang.py`, + `paths.py` baru
  (satu tempat definisi `PROJECT_ROOT`/`ASSETS_DIR`/`OUTPUT_DIR`/dst, ganti 9
  file yang sebelumnya masing-masing hitung `BASE_DIR = Path(__file__)...`
  sendiri-sendiri).
- `pipeline/` — `preprocessing.py`, `ocr.py`, `postprocessing.py`,
  `dynamic_extraction.py`, `comparison.py`, `vlm.py`, `pipeline.py`.
- `extractors/` — `extractors.py`, `signature_detector.py`,
  `collapse_retry_worker.py`.
- `eval_tools/` — `evaluation.py`, `stage_evaluation.py`, `compare_runs.py`,
  `signature_diagnostics.py`, `live_evaluation.py`, `diagnose_collapse.py`,
  `local_qwen_eval.py`.
- Tetap di root: `app.py`, `static/`, `prd.md`, `handover.md`, data/model
  dirs (`assets/`, `outputs/`, `uploads/`, `eval_runs/`, `models/`,
  `downloaded_documents/`).

**Yang TIDAK berubah:** isi logic tiap fungsi, nama file (tidak ada rename),
struktur `eval_runs/`/`outputs/`/dll, cara `app.py` dijalankan
(`python app.py`, uvicorn mount tetap sama).

**Yang berubah (dampak ke cara pakai):** command CLI eval tools sekarang
pakai prefix `eval_tools/` (lihat §12 yang sudah diupdate), mis.
`python eval_tools/evaluation.py --dataset ...` bukan `python evaluation.py
--dataset ...` lagi.

**Kenapa `BASE_DIR = Path(__file__).resolve().parent` jadi masalah:** 9 file
pakai pola ini utk cari folder root project (assets/models/eval_runs/dst) —
benar SELAMA file-nya di root. Setelah pindah ke folder, `Path(__file__).parent`
nunjuk ke folder BARU (mis. `pipeline/`), bukan root lagi. Fix: satu file
`core/paths.py` yang hitung root dgn `.parent.parent`, semua modul lain
import dari situ. Untuk script yang dijalankan langsung dari command line
(semua isi `eval_tools/` + `extractors/collapse_retry_worker.py`), ditambah
satu baris `sys.path.insert(0, ...)` di paling atas SEBELUM import
`core`/`pipeline`/`extractors` — soalnya begitu file-nya sendiri dijalankan
(`python eval_tools/evaluation.py`), Python taruh folder `eval_tools/` (bukan
root) di `sys.path[0]`.

**Cara import berubah, cara pakai fungsi TIDAK:** semua `import preprocessing
as prep` dst berubah jadi `from pipeline import preprocessing as prep` (dan
serupa utk `core`/`extractors`/`eval_tools`), tapi setelah baris import itu,
semua pemanggilan fungsi (`prep.resolve_roi(...)` dst) PERSIS sama, tidak ada
yang diubah. `pipeline.py` (di dalam folder `pipeline/`) dan `extractors.py`
(di dalam folder `extractors/`) pakai bentuk `import pipeline.pipeline as
pipeline` / `import extractors.extractors as extractors` supaya pemanggil
lama yang sudah terbiasa `pipeline.run_pipeline(...)`/`extractors.run_v18(...)`
tidak perlu ubah apa-apa selain baris import-nya.

**Verifikasi yang sudah dilakukan sesi ini:**
1. Smoke-import tiap paket satu-satu (`core`, lalu `pipeline`, lalu
   `extractors`, lalu `eval_tools`, terakhir `app.py`) — semua OK, tidak ada
   `ImportError`/`ModuleNotFoundError`.
2. Tiap script CLI di `eval_tools/` dites `--help` (evaluation.py,
   stage_evaluation.py, compare_runs.py, signature_diagnostics.py) — semua
   jalan.
3. `python extractors/collapse_retry_worker.py` (tanpa argumen) dites —
   sampai ke pesan usage, artinya `from pipeline import vlm` di dalamnya
   berhasil (subprocess worker ini sensitif krn dipanggil `subprocess.run`
   dari `extractors.py`, bukan diimport biasa).
4. Grep seluruh repo (bukan cuma folder baru) utk pola import lama
   (`import preprocessing`, `import data_input`, dst tanpa prefix) — nol
   hasil, tidak ada yang ketinggalan.
5. Smoke test end-to-end nyata: `python eval_tools/evaluation.py --dataset
   assets/ocr_evaluation.xlsx --label restructure_smoke --limit 2
   --debug-dir none`. Hasilnya JUJUR: 2/2 dokumen gagal di stage OCR, TAPI
   root cause-nya BUKAN restructure — error persis
   `AttributeError: partially initialized module 'paddle' has no attribute
   'tensor'`, yang SUDAH terdokumentasi sebelumnya (lihat teks lama di §9,
   soal bug lingkungan PaddleX/paddle circular-import independen dari kode
   proyek). Dikonfirmasi ulang sesi ini dgn cara paling telanjang: import
   `paddleocr` sendirian tanpa kode proyek sama sekali (`from paddleocr
   import PaddleOCR; PaddleOCR(...)`) — GAGAL dgn traceback yang berujung ke
   `OSError: [WinError 127] ... Error loading
   ".../nvidia/cudnn/bin/cudnn_cnn64_9.dll"`. Ini masalah DLL CUDA/cuDNN di
   environment ini, sama sekali tidak menyentuh kode yang dipindah sesi ini.
   Belum diperbaiki — di luar scope restructure, catat sbg known issue
   lingkungan (lihat juga §10 kalau mau lanjut investigasi ini nanti).

**Kesimpulan:** restructure selesai, wiring import terverifikasi bekerja,
tidak ada logic yang berubah. Kegagalan OCR di smoke test adalah bug
lingkungan yang sudah ada sebelum sesi ini, bukan regresi dari restructure.

### 16.1 Fix — satu import lolos dari verifikasi awal (sesi lanjutan)

User lapor app tidak bisa jalan sama sekali (`No module named 'preprocessing'`)
langsung setelah restructure di atas — verifikasi §16 TERNYATA tidak cukup.

**Root cause:** `pipeline/vlm.py` fungsi `_get_template_image()` (dipanggil
oleh jalur Gemini/Qwen hosted & local utk ambil gambar template pembanding)
punya `import preprocessing as prep` di DALAM badan fungsi, bukan di atas
file. Verifikasi restructure kemarin cuma grep pola `^import`/`^from` (anchor
ke awal baris) utk cari sisa import lama — lolos krn baris ini diindentasi
(nested di dalam `if`). Smoke-import test (`python -c "import ..."`) juga
tidak menyentuhnya krn `_get_template_image()` baru jalan saat dipanggil,
bukan saat modul di-import.

**Fix:** ganti jadi `from pipeline import preprocessing as prep` (baris
986). Lalu ulang pencarian TANPA anchor `^` (`\s+(import|from) <nama modul
lama>`) ke SELURUH repo — nol hasil lain, ini satu-satunya yang lolos.

**Pelajaran:** kalau verifikasi restructure/import-path lagi (sesi
berikutnya, folder mana pun), grep pola import HARUS termasuk yang
diindentasi (lazy/deferred import di dalam fungsi), bukan cuma baris paling
atas file — dan smoke-import saja tidak cukup, harus benar-benar PANGGIL
fungsi yang isinya lazy-import (atau jalankan end-to-end lewat
`eval_tools/evaluation.py`) sebelum yakin "sudah aman".

**Diverifikasi ulang setelah fix:** smoke-import semua paket lagi (OK),
panggil `vlm._get_template_image()` langsung (OK, kembalikan gambar template
asli), re-run `eval_tools/evaluation.py --limit 2` (perilaku identik dgn
sebelum fix ini — masih kena bug lingkungan paddle/cuDNN yang sama, TIDAK
ada kegagalan baru), start `uvicorn app:app` + `GET /` (200 OK). Bersihkan jg
`__pycache__/` sisa dari SEBELUM restructure di root project (isinya .pyc
modul lama spt `preprocessing.cpython-311.pyc` yang file sumbernya sudah
pindah — cuma bytecode cache basi, aman dihapus, di-gitignore, TIDAK
memengaruhi Python manapun cari modul).

## 17. Fix — reward_tunai/reward_non_tunai detail masih null (crop-based, engine qwen3_vl_local_yolos/qwen3_vl_local) (2026-09-18)

User lapor lagi: `reward_tunai`/`reward_non_tunai` (nilai Rp di baris "Nilai
Reward (termasuk pajak)") masih sering null di hasil. Root cause: follow-up
call yang bertugas baca baris itu (`vlm.REWARD_TUNAI_DETAIL_PROMPT`/
`REWARD_DETAIL_PROMPT`, dipanggil dari `_run_local_reward_detail_followup`)
cuma pernah kirim gambar HALAMAN PENUH (`MAX_SIDE_FULL=1600px`) — baris
tulisan tangan yang ditanya cuma ~19px tinggi di gambar itu, sama persis
kelas masalah yang sudah pernah diperbaiki utk nama_nasabah/nomor_rekening/
nominal_penempatan lewat close-up crop (V21).

**Fix**: `extractors._get_weak_field_crops()` sekarang JUGA crop
`reward_tunai`/`reward_non_tunai` (pakai `preprocessing.FIELD_CONFIG`'s
`value_bbox` yang SUDAH ada dari era V18, cuma belum pernah disambungkan ke
jalur VLM). Crop ini dikirim sbg gambar ke-3 (TAMBAHAN, bukan pengganti) ke
follow-up call reward — persis pola yg sudah terbukti di V21, lewat
parameter baru `reward_crops` (`extractors.py` -> `vlm.
extract_direct_semantic_local`/`_majority` -> `_run_local_reward_detail_
followup` -> `_extract_reward_tunai_detail_local`/`_extract_reward_detail_
local`, semua `None`-default supaya `qwen3_vl_local` tanpa crop TETAP
persis sama seperti sebelumnya). Dithread juga ke
`_retry_collapsed_in_subprocess`/`collapse_retry_worker.py` biar collapse-
retry tidak kehilangan fix ini. **Scope: HANYA `qwen3_vl_local_yolos` +
`qwen3_vl_local`** (keduanya pakai helper follow-up yang sama) — hosted
`qwen3_vl` & `gemini-3.8-flash` TIDAK disentuh (keputusan eksplisit user,
Gemini bahkan tidak punya mekanisme follow-up sama sekali, beda pekerjaan
lebih besar). V18 (`pipeline/`) juga TIDAK disentuh, cuma DIBACA
(`FIELD_CONFIG`).

**Diverifikasi NYATA, bukan cuma baca kode** (semua run pakai model lokal
sungguhan, `downloaded_documents/` asli):
1. Baseline SEBELUM fix: 3 dokumen (`bentuk_reward` tunai/non_tunai
   campuran) — SEMUA `reward_tunai_detail`/`reward_non_tunai_detail`
   kembali `None`. Cocok persis dgn laporan user.
2. Investigasi 1 dokumen (`13EQ77C6tOZoWZ...`) yg TETAP `None` walau
   crop-nya sendiri terbukti LEGIBLE (dicek visual langsung, isinya "Rp
   600.000 (Enam Ratus Ribu Rupiah)") — dgn prompt LENGKAP (`REWARD_TUNAI_
   DETAIL_PROMPT`), model tetap jawab null baik dikasih 1/2/3 gambar
   (crop-saja, template+crop, atau template+halaman+crop). Ganti ke prompt
   PENDEK/minimal (TANPA instruksi "jangan menebak") baru model mau
   menjawab — TAPI jawabannya SALAH (1000000/"satu juta", bukan 600000).
   **Kesimpulan jujur**: utk dokumen INI, bukan resolusi yg jadi penghalang
   — model genuinely kesulitan baca tulisan tangan kursifnya, dan prompt
   yg ketat ("jangan menebak") membuatnya milih null drpd menebak salah.
   Prompt ketat itu TIDAK diubah/dilonggarkan (menjawab salah dgn percaya
   diri lebih buruk drpd null jujur — prinsip yg sama dgn fix V22 mirrored-
   placeholder, lihat §9l) — ini didokumentasikan sbg keterbatasan yg
   tersisa, BUKAN dicoba "diperbaiki" dgn melonggarkan instruksi.
3. Test lebih luas: 8 dokumen (GT "Cashback"/tunai), SATU proses (model
   dimuat sekali), `reward_tunai_detail` dibandingkan TANPA crop vs DENGAN
   crop:
   | hasil | jumlah dokumen |
   |---|---|
   | sudah benar SEBELUM fix, TETAP sama sesudah (tidak berubah) | 3/8 |
   | **null SEBELUM fix -> nilai asli terbaca SESUDAH fix** | **3/8** |
   | tetap null baik sebelum maupun sesudah (kasus sulit, spt poin 2) | 2/8 |
   | jadi salah/regresi krn fix ini | **0/8** |

   3 nilai baru yg berhasil diekstrak (32500000/"tiga puluh dua juta lima
   ratus ribu rupiah", 500000/"lima ratus ribu rupiah", 40000/"empat puluh
   ribu rupiah") SEMUA konsisten digit-vs-terbilang saat dicek manual
   (dikonfirmasi jg oleh `comparison.validate_nominal_terbilang` yg
   sekarang disambungkan ke pasangan reward juga, lihat di bawah) — bukan
   angka asal tebak. Catatan jujur terpisah (BUKAN akibat fix ini, sudah
   ada sebelumnya di kedua kolom tanpa-crop/dengan-crop): 1 dokumen
   (`1a7X78Si8DTJ...`) hasil digit (48750000) tidak cocok dgn terbilang-nya
   sendiri ("empat puluh delapan juta" = 48000000, hilang bagian "tujuh
   ratus lima puluh ribu") — transkripsi terbilang yg terpotong, di luar
   scope sesi ini.
4. Regresi ke field lain: `eval_tools/evaluation.py --engine
   qwen3_vl_local_yolos --limit 5 --debug-dir none` (label
   `reward_fix_smoke`) — 0 kegagalan pipeline di 5 dokumen manapun
   (`failure_stage_breakdown` semua nol), `bentuk_reward` 1.0,
   `nama_nasabah`/`nomor_rekening`/`nominal_penempatan`/`tenor_penempatan`
   dalam rentang wajar (bukan bukti definitif n=5 kecil, tapi cukup utk
   pastikan tidak ada crash/collapse baru — perubahan ini scr kode memang
   tidak pernah menyentuh gambar/prompt main call field-field itu).

**Tambahan diagnostic (measurement-only, TIDAK masuk keputusan OK/TOLAK/
REVIEW)**: `comparison.validate_nominal_terbilang` (sudah ada sejak V21/V22
utk `nominal_penempatan`) sekarang JUGA dipanggil dgn
`digit_field="reward_tunai_detail", terbilang_field="reward_tunai_terbilang"`
di `evaluation.py` (`reward_terbilang_verdict`/`reward_terbilang_both_
evidence_read_count`/`reward_terbilang_internal_consistency_rate` di
summary) dan `live_evaluation.py` (`reward_terbilang` stat di popup
"Evaluasi"). Alasan: `reward_tunai` tidak punya kolom ground truth di
spreadsheet, jadi consistency-rate ini SATU-SATUNYA sinyal terukur yg ada
utk memantau apakah fix ini masih bekerja di sesi-sesi berikutnya.

**Yang TIDAK diklaim selesai**: 2/8 dokumen di test poin 3 TETAP tidak
terbaca (keterbatasan model/handwriting genuine, bukan soal ROI/resolusi
lagi — lihat poin 2). Hosted `qwen3_vl` & Gemini TIDAK mendapat perbaikan
ini sama sekali (lihat scope di atas). `reward_non_tunai_detail` (deskripsi
barang non-tunai) dapat perbaikan KODE yg sama (crop `reward_non_tunai`)
tapi belum diverifikasi seluas `reward_tunai_detail` di atas (dataset ini
mayoritas tunai, jalur non-tunai jarang ke-exercise) — kalau ada laporan
serupa utk non-tunai spesifik, ukur dulu sebelum asumsi sama-sama membaik.

## 18. Sesi lanjutan — nama_nasabah threshold, tenor date-range abstraction, choice-field investigation (2026-09-18)

User lapor "no different on performance" utk fix reward di §17 — dikonfirmasi
lewat pertanyaan: yang dicek adalah dashboard field/decision-accuracy, yang
memang TIDAK BISA bergerak dari fix itu (`reward_tunai` tidak punya kolom
ground truth, sejak awal tidak masuk `FIELD_GT_MAP`). Bukan tanda fix-nya
gagal — bukti riil session lalu (3/8 dokumen null→nilai benar, 0 regresi)
tetap berlaku. Tidak ada aksi kode di poin ini, cuma catatan supaya tidak
disalahpahami lagi sesi depan.

**18a. `NAME_PASS_THRESHOLD` 90 → 80 (`pipeline/comparison.py`), SELESAI.**
User minta threshold "uncertain" utk nama_nasabah diturunkan supaya lebih
banyak match borderline-tapi-benar jatuh ke "sesuai", TAPI bucket uncertain
sendiri harus tetap ada (bukan dihapus, beda dari mismatch). Root cause:
kode (90) TERNYATA sudah beda dari spec yang sudah lama didokumentasikan
(`handover.md` §14 / `prd.md` §7: "Nama (fuzzy match 80%)") — 90 tidak
pernah disamakan ke situ. Diverifikasi thd similarity SUNGGUHAN yang sudah
tersimpan di `eval_runs/*.csv` (bukan tebakan): 6 pasangan nama jatuh di
rentang [80,90) — 5/6 GENUINE match cuma beda noise OCR ("Qurib
Ramadhani"/"Qurbi ramadhan"=89.7, 3 varian "CV PERMATA AGROTANI
KENCANA"=87-89.3), yang SEBELUMNYA salah kena "perlu_review". 1/6 lebih
borderline ("CV. Agrotani Kencana" vs "CV PERMATA AGROTANI KENCANA"=80.9,
kehilangan 1 kata "PERMATA" sepenuhnya) — dicatat jujur, bukan alasan utk
batal, krn rasio 5:1 & sudah sesuai spec yang sudah disepakati.
`NAME_MISMATCH_THRESHOLD` (55) TIDAK diubah — bucket uncertain (55-79
sekarang, dulu 55-89) tetap ada, cuma diperkecil. Diverifikasi ulang lewat
`comparison.is_field_match` langsung: `("Qurib Ramadhani","Qurbi
ramadhan")` sekarang `(True, 89.7)` (dulu `("uncertain", 89.7)`), sementara
`("Nuryani","Mulyani")=71.4` tetap `"uncertain"` seperti seharusnya.

**18b. Tenor date-range: hapus round-trip digit→teks→re-parse
(`extractors.py` + `pipeline/postprocessing.py`), SELESAI.** User bilang
transformasi dd/mm/yy ke "alphabetical" terasa "abstract" — dikonfirmasi
NYATA: alur lama VLM keluarkan ISO `YYYY-MM-DD` → `extractors._format_id_date`
ubah ke teks nama-bulan Indonesia ("14 Agustus 2026") → digabung jadi satu
string → `postprocessing._extract_dates` regex-parse teks itu BALIK jadi
objek `date` → hitung selisih bulan. Konversi teks itu HANYA ada supaya
parser lama (didesain utk teks OCR gabungan V18) bisa dipakai lagi —
murni indirection utk data yang sebenarnya sudah bersih (ISO). Bug
tersembunyi: `_format_id_date`'s `except Exception: return None` diam-diam
menelan SEMUA kegagalan format, membuat `rentang_tenor.status=
"not_detected"` yang TIDAK BISA dibedakan dari "field memang kosong" di
output eval (dikonfirmasi: 12/25 baris `tenor_date_range_derived` di satu
eval run nyata semuanya `N/A`, sumbernya campur aduk).

Fix: `postprocessing.tenor_from_date_pair(d1, d2)` (baru) — inti logic
`derive_tenor_from_range` (selisih bulan kalender → opsi 1/3/6 terdekat)
dipisah jadi fungsi sendiri yang terima 2 objek `date` LANGSUNG.
`derive_tenor_from_range(raw_text)` (V18/OCR, TIDAK berubah perilakunya)
tetap `_extract_dates` lalu delegasi ke fungsi baru ini. `extractors.py`
sekarang parse ISO string VLM LANGSUNG (`_parse_iso_date`, pakai
`datetime.date.fromisoformat`) & simpan sbg `rentang_tenor["dates_iso"]`
(STRING, bukan objek `date` — sengaja, supaya tetap aman di-JSON-kan kalau
`raw_json`/export menyentuhnya suatu saat). `comparison.validate_tenor`
skrg cek `dates_iso` dulu (parse ulang + `tenor_from_date_pair` langsung)
SEBELUM fallback ke `derive_tenor_from_range(raw_text)` yang lama — V18
sama sekali tidak ke-touch krn tidak pernah mengisi `dates_iso`.
Efek samping positif: reason kegagalan sekarang beda antara "field kosong"
vs `"iso_tanggal_tidak_valid"` (VLM keluar format aneh) — tidak collapse
jadi satu "N/A" lagi.

**Investigasi false-alarm (dicatat supaya tidak diulang)**: sempat curiga
ada bug type-mismatch (`choice_value` string "3" vs `derived_value` int 3
di `validate_tenor`, bikin `!=` SELALU True) berdasar trace parsial
`vlm._canonicalize_tenor_penempatan` (return string). TERNYATA SALAH —
`extractors._adapt_qwen_to_common` (baris ~1086) sudah cast `int()` sebelum
disimpan, jadi `choice_value` di production memang int, cocok dgn
`derived_value`. Dikonfirmasi lewat test langsung end-to-end
(`_adapt_qwen_to_common({"tenor_penempatan": {"value": "3", ...}})` →
`{"value": 3, ...}`, tipe int). **Pelajaran**: jangan simpulkan bug dari
SATU fungsi return type saja — trace SAMPAI titik pemakaian sungguhan,
persis prinsip §11 yang sudah berkali-kali dicatat di file ini.

Diverifikasi: unit test `tenor_from_date_pair`/`validate_tenor` (dates_iso
vs text-fallback, keduanya hasil IDENTIK utk pasangan tanggal yang sama;
reversed-range & invalid-ISO masing2 dapat reason yang benar), plus
`eval_tools/evaluation.py --engine qwen3_vl_local_yolos --limit 5` (label
`session2_smoke`) — 0 kegagalan, angka field accuracy identik dgn baseline
sebelum sesi ini (`nama_nasabah` 0.2, `nomor_rekening` 0.4, `nominal_
penempatan`/`tenor_penempatan` 0.8, `bentuk_reward` 1.0) — tidak ada
regresi dari 18a maupun 18b.

**18c. Choice-field (`tenor_penempatan`/`bentuk_reward`) — 2 pendekatan
DICOBA, KEDUANYA tidak dipromosikan, dgn bukti nyata knp.** User pilih 2
dari 3 opsi yang diajukan: crop close-up + ink-diff V18 sbg authoritative
override. Sengaja dicoba SATU-SATU (bukan digabung) supaya efeknya bisa
diukur terpisah.

*Crop close-up (`extractors.py`, DICOBA & DIHAPUS — bukan cuma dimatikan).*
Sama pola dgn fix reward (§17): crop garis pilihan (union `anchor_bbox` +
semua `options` dari `preprocessing.CHOICE_GROUPS`, fungsi
`_choice_group_bbox`), ditambahkan ke `extra_crops` MAIN call (bukan
follow-up terpisah, krn tenor/reward dibaca di call utama). Butuh padding
JAUH lebih besar dari perkiraan awal (0.3→1.5) krn ditemukan drift
block-shift asli ~30-35px pada satu dokumen (`13EQ77C6tOZoWZ...`) — mirip
persis temuan V15 lama, cuma sekarang kena box yang lebih tipis sehingga
tidak lolos begitu saja spt `reward_tunai`/`nominal_penempatan` (yang
kebetulan selamat krn padding mereka sudah 0.5-1.0 utk alasan lain).
Divisualisasikan LANGSUNG (crop di-render, dibaca manual) sebelum & sesudah
naikkan padding — dikonfirmasi bekerja pada 2/3 dokumen spot-check.

**TAPI tes NYATA end-to-end (8 dokumen, model lokal sungguhan, extra_crops
dgn vs tanpa crop pilihan) nunjukkin HASIL BERSIH: 7/8 IDENTIK (termasuk
`bentuk_reward` 8/8 TIDAK berubah sama sekali — prioritas utama TIDAK
regresi TERPENUHI), TAPI 1/8 REGRESI nyata** (record 8: GT tenor=6, crop
divisualisasikan CRYSTAL CLEAR — "1/3" dicoret, "6" dilingkari jelas —
TETAP dibaca salah jadi "3" DENGAN crop, padahal benar "6" TANPA crop).
Beda kelas masalah dgn fix reward: di situ resolusi memang penghalang
utama; di sini model bisa salah baca WALAU gambarnya sudah jelas & besar —
menambah gambar redundan kadang malah mengganggu bacaan halaman-penuh yang
sudah benar, bukan cuma menambah info. **Kode DIHAPUS TOTAL** (bukan
dibiarkan idle) — `CHOICE_CROP_TARGETS`/`_choice_group_bbox`/
`CHOICE_CROP_PAD_RATIO` semua dibuang dari `extractors.py`, `vlm.py`'s
`_WEAK_FIELD_CROP_LABELS` dikembalikan ke 3 field asli. **Jangan diulang
tanpa sampel lebih besar & hipotesis konkret knp regresi ini terjadi.**

*Ink-diff V18 sbg cross-check (`extractors.py`, DIBANGUN sbg
DIAGNOSTIC-ONLY, TIDAK dipromosikan jadi override).* Sesuai rencana
staged-rollout eksplisit (ukur dulu sebelum percaya, persis pelajaran dari
percobaan crop di atas): `_get_weak_field_crops` sekarang JUGA memanggil
`postprocessing.process_choices()` (mekanisme ink-diff piksel V18 yang
frozen, BUKAN OCR) pakai alignment yang SAMA (satu pass, tidak dobel),
return berubah jadi `(crops, choice_ink_diff)`. `_choice_ink_diff_diagnostic`
(baru) bandingkan verdict ink-diff vs bacaan Qwen sendiri, disimpan di
`ocr_meta["choice_ink_diff_diagnostic"]` — **TIDAK PERNAH mengubah
`final_results`/keputusan**, murni observasi.

**Hasil ukur NYATA (12 dokumen, murni geometri, tanpa panggil VLM sama
sekali — cepat)**: ink-diff mentah (`process_choices`) balik `status=
"review"` (ambigu) utk **KEDUA field, di SEMUA 12/12 dokumen, tanpa
kecuali** — beberapa malah dgn skor SANGAT TINGGI di SEMUA opsi sekaligus
(mis. record 5: skor tenor 0.94/0.93/0.91 utk opsi 1/3/6 — ketiganya
"berubah banyak", bukan cuma satu). Diselidiki apakah ini bug pemanggilan
(cross-check urutan argumen thd `pipeline.py`'s pemanggilan V18 sendiri) —
TERNYATA IDENTIK, bukan bug. Kesimpulan yang lebih masuk akal (dibaca dari
`postprocessing.resolve_bentuk_reward`'s docstring sendiri): akurasi tinggi
V18 utk `bentuk_reward` SELAMA INI bukan dari ink-diff mentah yang selalu
yakin, tapi dari LAPISAN KEDUA (`resolve_bentuk_reward`/`resolve_tenor_source`
fallback ke bukti teks OCR `reward_tunai`/`reward_non_tunai`/`rentang_tenor`
saat ink-diff sendiri ambigu) — lapisan kedua itu TIDAK ADA utk engine VLM
ini (tidak ada field OCR terpisah yang setara). Jadi ink-diff mentah SAJA,
tanpa lapisan kedua itu, memang wajar sering ambigu — bukan mekanismenya
rusak, tapi memang didesain utk dipakai BERSAMA bukti lain yang di sini
tidak tersedia.

**Keputusan**: TIDAK dipromosikan jadi authoritative override — dgn data
100% ambigu, override itu tidak akan pernah aktif sama sekali (aman, tapi
percuma). Kode diagnostic-only DIPERTAHANKAN (murah, reuse alignment yang
sudah ada, tidak pernah mengubah hasil) sbg sinyal terukur kalau nanti ada
yang mau membangun lapisan kedua yang setara (mis. VLM's own tenor/reward
reading SEBAGAI bukti teks pengganti OCR, dipasang ke `resolve_bentuk_
reward`-style resolver yang baru). **Next step yang valid** kalau mau
lanjutkan choice-field ini: bangun lapisan kedua itu, BUKAN ulangi crop
(§18c poin 1) atau menaikkan ink-diff jadi override tanpa lapisan kedua
(sudah terbukti percuma di sini).

**Ringkasan jujur sesi ini**: 2/4 item selesai bersih dgn perbaikan nyata &
nol regresi (18a, 18b). 1/4 item DICOBA & dibuang dgn bukti konkret knp
(18c crop). 1/4 item dibangun sbg fondasi diagnostic yang benar tapi belum
bisa dipromosikan krn prasyaratnya (lapisan kedua bukti) belum ada (18c
ink-diff). Tidak ada yang dipaksakan "selesai" tanpa bukti — sesuai prinsip
§15 file ini.
serupa utk non-tunai spesifik, ukur dulu sebelum asumsi sama-sama membaik.
