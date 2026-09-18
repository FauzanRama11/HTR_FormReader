# PRD — HTR Form Reader (Validasi Surat Pernyataan Nasabah)

Dokumen ini scoping, bukan spek desain ulang. Isinya rangkuman dari
`handover.md` (source of truth eksperimen/history) + kode aktual saat ini.
Kalau ada yang beda antara dokumen ini dan `handover.md`/kode, percaya
`handover.md`/kode — update dokumen ini, jangan sebaliknya.

## 1. Ringkasan

Sistem membaca **Surat Pernyataan nasabah** (foto HP atau PDF hasil scan),
mengekstrak beberapa field (nama, nomor rekening, nominal, tenor, bentuk
reward, tanda tangan), membandingkan dengan data referensi spreadsheet, lalu
menghasilkan keputusan: `OK Lanjut Proses` / `Tolak` (+ alasan) / `Review`.

Alur inti:
```
Document -> Preprocessing (align ke template) -> ROI per field -> OCR/VLM
         -> Field Comparison (vs data referensi) -> Final Status
```

## 2. Masalah

Verifikasi manual Surat Pernyataan (cocokkan isi form tulisan tangan dengan
data pendaftaran di spreadsheet) lambat dan rawan salah baca kalau dilakukan
satu-satu oleh manusia untuk banyak dokumen. Sistem ini otomatisasi proses
itu supaya bisa diproses massal dari satu spreadsheet/Google Drive.

## 3. Tujuan

- Ekstrak 5 field inti (nama, nomor rekening, nominal penempatan, tenor,
  bentuk reward) dari dokumen tulisan tangan.
- Deteksi keberadaan (bukan baca isi) tanda tangan nasabah & tanda tangan
  unit kerja/BRI.
- Bandingkan hasil ekstraksi dengan data referensi spreadsheet, hasilkan
  status `OK Lanjut Proses` / `Tolak` (+ alasan) / `Review`.
- Bisa diproses satu dokumen lewat UI, atau bulk lewat spreadsheet
  Excel/Google Sheet + Google Drive.
- Evaluasi akurasi terukur & reproducible (dataset ground truth 25 dokumen,
  lihat §7) supaya tiap perubahan bisa dibandingkan before/after.

## 4. Bukan tujuan (non-goals)

- Tidak membaca ISI tanda tangan (cuma presence: ada/tidak/tidak yakin).
- Tidak menangani dokumen selain template Surat Pernyataan ini (`assets/
  template.pdf`) — alignment/ROI semuanya diukur relatif ke template ini.
- Tidak ada training model baru — semua engine (lihat §6) pakai model
  OCR/VLM yang sudah ada (PaddleOCR, Gemini, Qwen3-VL, YOLOS), bukan
  dilatih ulang di project ini.
- V18 (engine PaddleOCR + VLM fallback asli) sedang DIBEKUKAN atas
  permintaan eksplisit user (lihat `handover.md` §9f) — perubahan
  internal ke jalur itu (preprocessing/ROI/dynamic_extraction) tidak
  dikerjakan lagi sampai diminta ulang.

## 5. Pengguna

Staff yang perlu memvalidasi form pendaftaran nasabah program (mis. "BRI
CUAN HADIAH 2026") terhadap data pendaftaran yang sudah masuk. Dua cara
pakai:
1. **Single document** — upload satu file lewat UI, proses, lihat hasil.
2. **Bulk dari spreadsheet** — upload Excel atau load Google Sheet publik
   (kolom link Google Drive per dokumen), sistem download semua dokumen,
   proses satu-satu (bisa distop di tengah jalan), lihat hasil per record +
   metrik agregat + export ke Excel.

## 6. Alur & fitur saat ini

Rute API (`app.py`):
- `GET /` — serve UI (`static/index.html`).
- `POST /api/upload`, `POST /api/process/{request_id}` — jalur single
  document.
- `POST /api/sheet/upload`, `POST /api/sheet/load-url` — masukkan dataset
  (Excel upload / Google Sheet URL).
- `POST /api/sheet/submit/{session_id}`, `POST /api/sheet/cancel/{session_id}`,
  `GET /api/sheet/batch-status/{session_id}` — jalankan/stop/pantau batch.
- `GET /api/sheet/result/{session_id}/{record_no}` — hasil per record.
- `GET /api/sheet/evaluation/{session_id}` — popup "Evaluasi": metrik
  agregat dari hasil yang SUDAH tersimpan (tidak pernah OCR ulang).
- `GET /api/sheet/export/{session_id}` — export hasil ke Excel.

**5 engine ekstraksi** (dipilih per proses, `extractors.py`):
| engine id | Cara kerja |
|---|---|
| `v18` | Jalur asli: PaddleOCR per-ROI + VLM (Qwen2-VL lokal) sbg fallback kalau OCR tidak yakin. DIBEKUKAN (§4). |
| `gemini-3.8-flash` | Panggil Gemini API langsung (full-page, tanpa ROI/template/OCR V18), butuh `GEMINI_API_KEY`. |
| `qwen3_vl` | Qwen3-VL hosted (Hugging Face Inference), butuh `HF_TOKEN`. |
| `qwen3_vl_local` | Qwen3-VL jalan lokal on-device (default `models/Qwen3-VL-4B-Instruct`), majority-vote. |
| `qwen3_vl_local_yolos` | Sama seperti `qwen3_vl_local` utk field semantik, TAPI signature_nasabah/signature_atasan pakai YOLOS detector lokal (`signature_detector.py`) sbg sumber kebenaran, bukan penilaian Qwen. |

Tiap proses menghasilkan SATU engine per run (tidak pernah campur beberapa
engine dalam satu proses yang sama).

## 7. Field & rules

(Sumber: `handover.md` §14, jangan diduplikasi beda di sini kalau berubah —
update di sana dulu.)

- Field: Nama (fuzzy match 80%), Nomor Rekening, Unit Kerja, Nominal
  Penempatan, Tenor (dari rentang tanggal, <1 bulan = tidak sesuai), Bentuk
  Reward (tunai/non-tunai + fallback), TTD Nasabah & TTD Unit Kerja/BRI
  (presence only, bukan baca isi).
- Rule rural (kolom `Urban` di data referensi != "urban"): cukup validasi
  Nama, Nomor Rekening, Nominal Penempatan, TTD Nasabah, TTD BRI — kalau
  semua lolos, `OK Lanjut Proses`.

## 8. Metrik sukses (baseline saat ini)

Dataset ground truth: `assets/ocr_evaluation.xlsx` (25 record) +
`downloaded_documents/` (25 dokumen ter-cache). Angka di bawah dari
`handover.md` §16 (V16, engine `v18`) — cek `handover.md` §2-§9 untuk angka
per-versi & versi terbaru tiap engine sebelum dipakai sbg acuan baru:

| Metrik | Nilai |
|---|---|
| pipeline_success_rate | 1.00 |
| alignment_success_rate | 1.00 |
| nama_nasabah accuracy | 0.3889 |
| nomor_rekening accuracy | 0.5789 |
| nominal_penempatan accuracy | 0.4762 |
| tenor_penempatan accuracy | 0.80 |
| bentuk_reward accuracy | 1.00 |
| decision_accuracy | 0.16 |
| review_rate | 0.16 |

Engine lain (`gemini-3.8-flash`, `qwen3_vl*`) punya angka sendiri di
`handover.md` §9 (V19-V22) — belum tentu bisa dibandingkan apple-to-apple
dengan tabel di atas (metodologi/tanggal beda), baca detailnya di sana.

## 9. Known gaps / risiko

Ringkas dari `handover.md` §10 (Prioritas Berikutnya) — baca §10 untuk detail
lengkap tiap poin sebelum mengerjakan salah satunya:
- Beberapa kasus kontaminasi label cetak masih bocor ke field tetangga
  (butuh positional re-localization, belum ada).
- ROI clipping detection belum ada sama sekali.
- 1 dokumen (record 20) title-anchor gagal match, belum diselidiki
  root cause-nya.
- `decision_accuracy` stuck di 0.16 walau field accuracy naik — 7-check
  keputusan lebih ketat dari akurasi per-field individual.
- Mesin evaluasi ~8GB RAM, gampang OOM kalau jalankan 2 evaluator penuh
  bersamaan atau RAM sistem lagi dipakai app lain.
- Bug lingkungan (bukan bug kode): PaddleOCR gagal jalan di environment ini
  krn masalah DLL CUDA/cuDNN (`paddle` circular-import symptom) — lihat
  `handover.md` §16 poin verifikasi. Berdampak ke engine `v18` saja.

## 10. Out of scope

- File eksperimen/versi lama yang sudah dihapus dari repo (lihat
  `handover.md` §13, riwayat lama).
- Engine `v18` internal changes (dibekukan, §4).
- Redesign UI (`static/index.html`) — restructure sesi ini cuma
  merapikan kode Python, HTML tidak disentuh.

## 11. Struktur project (setelah restructure)

Kode dikelompokkan per tanggung jawab (sebelumnya semua rata di root):

```
core/         config.py, data_input.py, terbilang.py, paths.py
pipeline/     preprocessing.py, ocr.py, postprocessing.py, dynamic_extraction.py,
              comparison.py, vlm.py, pipeline.py
extractors/   extractors.py, signature_detector.py, collapse_retry_worker.py
eval_tools/   evaluation.py, stage_evaluation.py, compare_runs.py,
              signature_diagnostics.py, live_evaluation.py, diagnose_collapse.py,
              local_qwen_eval.py
app.py        entrypoint (tetap di root, `python app.py`)
static/       frontend (tetap di root)
```

Aturan gampang buat nyari sesuatu: kalau soal ROI/OCR/ekstraksi field dasar
-> `pipeline/`. Kalau soal pilih engine (Gemini/Qwen/YOLOS) -> `extractors/`.
Kalau soal ukur akurasi/diagnostic -> `eval_tools/`. Kalau soal
path/config/baca spreadsheet -> `core/`. Detail lengkap kenapa & cara
migrasinya: `handover.md` §16.
