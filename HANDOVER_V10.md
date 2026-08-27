# Handover — HTR Form Verification V10 (geometry + Paddle/VLM fallback)

For the next session: read this file first. It is current state, not history.

## ROOT CAUSE (V9.3 -> V10)
- Per-field `dx/dy` anchor correction (V9.x) let a single bad/local anchor drag one field
  off target even when nearby fields in the same section were fine — no shared, robust
  correction per section.
- Global alignment accepted transforms based only on `good_matches` + `inlier_ratio`, so
  matches clustered in one small area (e.g. header) could pass as a "good" full-page fit.
- Cropped/partial photos silently produced blank fields — nothing tracked whether a
  field's ROI actually fell inside photographed pixels vs. warp letterbox/border.
- PaddleOCR-only recognition had no fallback: correct ROI + wrong text (e.g. `Indo
  Transport` -> `Ramayana`, `CV` -> `cy`) had no second opinion.
- `comparison.py`'s decision buckets (blank/review/conflict) had no place for "field is
  simply not visible" vs. "two readers disagree" — both would otherwise risk being
  treated as an ordinary mismatch/OK.

## CHANGED FILES
`preprocessing.py`, `dynamic_extraction.py`, `postprocessing.py`, `comparison.py`,
`vlm.py`, `pipeline.py`, `app.py` — all included in this folder, ready to replace in
place. No other files touched (`ocr.py`, `data_input.py`, `static/index.html` are
untouched by this backend change; `static/index.html` has a **separate** set of
frontend fixes, see `HANDOVER_FRONTEND.md` if present).

### preprocessing.py
- `HEADER_EXCLUDE_RATIO`: ORB keypoints are now masked away from the top of the page
  (letterhead/branch/address) before matching, so global alignment is driven by the
  stable form body, not variable header text.
- `_match_geometry_diagnostics`: adds **inlier coverage ratio** (are matches spread
  across the page, not clustered in a corner?) and **mean reprojection error** to the
  existing good_matches/inlier_ratio gate. A transform now needs matches, ratio,
  spread, *and* low error to be marked `"ok"`; otherwise `"warning"`.
- `estimate_section_transform(...)`: NEW. Collects multiple printed anchors per
  section (identity: nama/nomor rekening/unit kerja; placement: nominal/tenor/reward),
  rejects low-confidence anchors and offset outliers (median-distance based), then
  fits ONE robust local transform per section:
  `estimateAffinePartial2D` (>=3 clean anchors) -> consensus translation (1-2 anchors)
  -> `"none"` (0 anchors, caller falls back further).
- `resolve_roi_section(...)`: applies the section transform to a field's template box;
  falls back to the old single-anchor `resolve_roi` if no section transform is usable.
- `align_to_template(...)`: now also returns a `coverage_mask` in the alignment meta —
  a warp of an all-white source-sized mask using the *same* transform used for the
  document, so any canvas pixel that is actually letterbox/border (not real source
  content) is `0`. `region_coverage()` / `is_out_of_frame()` read this mask.

### dynamic_extraction.py
- `_resolve_section_transforms` / `_resolve_field_boxes` now build field ROIs from the
  new section transform first (see fallback hierarchy above), not independent
  per-field `resolve_roi`.
- `extract_dynamic_fields(...)` signature changed (**6-tuple now**, was 4):
  `(result, roi_boxes, regions_debug, ocr_calls, field_statuses, section_meta)`.
  `field_statuses[field] = "out_of_frame"` for any field whose region coverage is
  below `preprocessing.COVERAGE_MIN_RATIO` — that field is *not* sent to OCR at all.

### postprocessing.py
- `build_out_of_frame_result()`: NEW helper, produces a `status="out_of_frame"` field
  result (value/raw/confidence all `None`) — explicitly distinct from "blank".
- `process_choices` / `process_signatures` accept an optional `coverage_mask=None`
  kwarg (backward compatible) and return `status="out_of_frame"` when the choice/
  signature area isn't actually covered by source pixels.
- Debug overlay: out-of-frame regions/fields get a distinct color + `"(out_of_frame)"`
  label.

### comparison.py
- `INCOMPLETE_FIELD_STATUSES` gained `"out_of_frame"`; `UNCERTAIN_FIELD_STATUSES`
  gained `"recognition_conflict"` — both map to `REVIEW`, never silently to
  `TOLAK`/`OK`, in both `compute_final_status` and `compute_decision_v9_2`.
- `_validate_simple`, `validate_tenor`, `validate_reward`, `validate_signature`: all
  now short-circuit to `REVIEW` with an explicit message when the field's status is
  `out_of_frame` or `recognition_conflict`, instead of comparing a possibly-stale
  value against the spreadsheet reference.

### vlm.py
- Existing loader (`_load`, lazy singleton), model path resolution, and
  `extract_full_page`/`extract_detail` are **unchanged**.
- `extract_fields_independent(images_bgr, fields, max_new_tokens=None, max_sides=None)`:
  NEW. Fixed prompt that (a) never includes any reference/spreadsheet value, (b) asks
  the model to read *only* what's visually written, (c) requires one of exactly three
  statuses per field: `readable` / `uncertain` / `not_visible`. Supports multiple
  images (full page + crops) in one call.

### pipeline.py
Orchestration is now, in order:
1. Load + align (+ `coverage_mask`).
2. Section-transform localization -> ONE batched Paddle OCR call for
   identity_area + placement_area (unchanged from V9.3, just fed better ROIs).
3. `tempat_tanggal_surat` (V9.1 path, coverage-checked).
4. Normalize; fields with `field_statuses[...] == "out_of_frame"` skip normalization
   entirely and get `build_out_of_frame_result()`.
5. **NEW — Paddle fallback** (`_run_paddle_fallback`): for any *non-optional* field
   still `review`/`not_detected` (and not `out_of_frame`/`error`), crop a tighter,
   padded ROI and OCR all of them together in ONE extra Paddle batch. Only overwrites
   the field if the fallback actually reaches `status="read"`.
6. Choices/signatures (coverage-aware).
7. **NEW — VLM fallback** (`_run_vlm_fallback`): triggers only for
   `nama_nasabah` / `nomor_rekening` / `nominal_penempatan`, only when: still
   review/not_detected (`"recognition"`), OR confidently mismatches
   `reference_record` (`"mismatch_verification"` — **reference value is never put in
   the VLM prompt**, only used to decide *whether* to call), OR the field's section
   transform was `"none"` while the field nonetheless reads `"read"`
   (`"geometry"`). At most **one** VLM call per document, with the full aligned page
   + one crop per targeted field. Arbitration (`_arbitrate_vlm_result`) is
   conservative: Paddle-reliable+VLM-disagrees -> `recognition_conflict`
   (`REVIEW`, never auto `TOLAK`/`OK`); Paddle-blank+VLM-readable -> adopt VLM value;
   VLM `not_visible` -> leave Paddle result untouched.
8. `run_pipeline(...)` gained an **optional** `reference_record=None` kwarg (old
   callers unaffected) — used only for step 7's mismatch-verification trigger.
9. `ocr_meta` now also reports `ocr_calls_paddle_fallback`, `paddle_fallback`,
   `vlm_fallback` (used/reason/calls/fields), `out_of_frame_fields`, `section_meta`.

### app.py
- Two small, additive changes only:
  - `_run_ocr_and_format` now passes `reference_record=data_entry_record` into
    `pipeline.run_pipeline(...)` (still optional/backward compatible).
  - The batch/single response dict now also includes `"ocr_meta": result.get("ocr_meta")`.
- No endpoint, schema field removal, or signature break for existing consumers.

## RUN / TEST
1. `pip install -r requirements.txt` (unchanged deps; Qwen2-VL deps only needed if VLM
   fallback is actually exercised — `pipeline.VLM_FALLBACK_ENABLED = False` disables
   VLM calls entirely without touching any other code path if the model isn't
   available in an environment).
2. `uvicorn app:app --reload --port 8000` from the project root (same as before).
3. Upload a normal, well-scanned document via Tab 1 — expect `ocr_meta.vlm_fallback.used
   == false`, `ocr_meta.ocr_calls_paddle_fallback == 0`.
4. Upload a deliberately cropped/partial photo (missing e.g. the placement section) —
   expect the missing fields' `status == "out_of_frame"` in `raw_json`, and final
   decision `REVIEW`, not a false `TOLAK`/blank.
5. Run a batch (Tab 2) with a document where a decisive text field is hard to read —
   check `ocr_meta.paddle_fallback.used_fields` and, if still unresolved,
   `ocr_meta.vlm_fallback.used == true` with `reason` per field (`"recognition"` /
   `"mismatch_verification"` / `"geometry"`).

## EXPECTED (per acceptance criteria)
- Normal document: 1 Paddle batch call (dynamic) + 0-1 fallback + 0 VLM calls.
- Difficult document: 1 Paddle batch + up to 1 Paddle fallback batch + **at most 1**
  VLM call, covering only the fields that actually need it.
- Qwen2-VL loaded once per process (existing `vlm._load()` singleton, untouched).
- Reference/spreadsheet values are never included in any VLM prompt (see
  `vlm.INDEPENDENT_READ_PROMPT_TEMPLATE` — takes only `fields`, never values).
- Paddle vs VLM disagreement -> `recognition_conflict` -> `REVIEW`, never silent
  `TOLAK`/`OK`.
- Cropped/partial photos -> `out_of_frame`, distinct from `blank`, never OCR'd/guessed.
- All changes are additive/optional-kwarg where they touch shared call sites; no
  existing endpoint, response schema, or public function signature was removed.

## Known scope limits (explicit, not oversights)
- VLM fallback triggers are scoped to the 3 text decisive fields (`nama_nasabah`,
  `nomor_rekening`, `nominal_penempatan`). Tenor/reward keep their existing
  deterministic dual-evidence (image-diff) logic as primary and are **not** wired to
  VLM in this pass — they already have out_of_frame handling via `coverage_mask`, but
  no VLM verification step. If eval data shows tenor/reward choice errors are still a
  material source of TOLAK/REVIEW, that is the next likely extension point in
  `pipeline._run_vlm_fallback` / `postprocessing.process_choices`.
- Runtime validation in this environment was import/syntax-level only (`paddleocr`,
  `pymupdf`, `torch`/`transformers` are not installed in this sandbox). Recommend
  running the 5 steps above against real sample documents (incl. one deliberately
  cropped photo) before considering this fully verified end-to-end.
