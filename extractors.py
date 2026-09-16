"""
extractors.py
==============
V19 -- Multi-extractor benchmark router. Adds 3 alternative extractors
(Gemini 3.8 Flash, Qwen3-VL hosted, Qwen3-VL local) alongside the existing,
UNCHANGED V18 pipeline (`pipeline.run_pipeline`), so `app.py`/`evaluation.py`
can benchmark all four under the SAME comparison/decision logic
(`comparison.py`, not touched by this module). Gemini 2.5 Flash and Mistral
OCR 4.1 were removed from the selectable engines (explicit user request) --
`run_mistral()` and its `mistralai` dependency were deleted entirely;
`run_gemini()` stays (still used by gemini-3.8-flash) but "gemini-2.5-flash"
is no longer a reachable engine id. "qwen3_vl_local" was added alongside
(not instead of) "qwen3_vl" -- initially measured to perform very poorly on
a VRAM-constrained (2GB) GPU, but re-verified since as usable on adequate
hardware with the current 4B model (see `run_qwen3_vl_local`'s docstring and
handover.md); still kept as an explicit, separate opt-in rather than the
default engine.

Design (see handover.md for the full writeup):
  - `run(engine, document_path, ...)` is the ONLY entry point other modules
    should call -- a single if/elif/else dispatch (see bottom of this file).
    Exactly ONE engine executes per call, NEVER a fan-out, NEVER an automatic
    fallback from one engine to another.
  - `run_v18` is a thin passthrough to `pipeline.run_pipeline()` -- V18's own
    internals (preprocessing/template alignment/ROI/PaddleOCR/VLM fallback)
    are NOT touched by this session's work.
  - `run_gemini` reads ONLY the original document bytes (no
    `preprocessing.load_document`/template alignment/ROI/PaddleOCR involved --
    the whole point is to measure NATIVE AI extraction) and NEVER receives
    `reference_record` (structurally impossible for it to leak ground
    truth into the prompt -- its signature doesn't even accept it).
  - `run_qwen3_vl_hosted` does MINIMAL image preparation only (PDF->image,
    EXIF orientation, safe resize -- NO deskew/threshold/binarize/ROI crop/
    label detection/PaddleOCR) and asks Qwen3-VL-2B-Instruct, running on
    Hugging Face's HOSTED inference infrastructure (no local model weights,
    `vlm.extract_direct_semantic_hosted`), to read the WHOLE document
    semantically in one call. Unlike every other engine here, it does NOT
    fall back to anything on failure -- a hosted-call error propagates as a
    real `ExtractorError` (task requirement: no silent OCR/ROI fallback for
    this test).
  - All four non-v18 engines parse their response into the CANONICAL schema
    (`COMMON_FIELDS`, task spec) via `adapt_common_to_pipeline_shape()`, which
    builds the EXACT SAME top-level return shape `pipeline.run_pipeline()`
    produces (`fields`/`raw_results`/`choice_groups`/`debug_images`/etc) by
    REUSING `postprocessing.build_fields_table()` -- so `app.py`/
    `evaluation.py` need almost no changes downstream of `run()`.
  - Gemini calls additionally capture real token usage (`gemini_usage`, top
    level of the result dict) from the API response's `usage_metadata` and
    compute cost from `config.py`'s pricing (never hardcoded here) -- see
    `_compute_gemini_cost`.
"""

import json
import os
import threading
import time
from pathlib import Path

import cv2

import config
import pipeline
import postprocessing as post
import preprocessing as prep
import signature_detector
import vlm

# ============================================================================
# CANONICAL SCHEMA (task spec) + SHARED EXTRACTION INSTRUCTION
# ============================================================================

COMMON_FIELDS = (
    "nama_nasabah", "nomor_rekening", "unit_kerja_pengelola_rekening",
    "nominal_penempatan", "tenor_penempatan", "tanggal_mulai", "tanggal_selesai",
    "bentuk_reward", "signature_nasabah", "signature_atasan",
)

# Verbatim from the task spec -- kept as ONE shared constant, used by BOTH
# Gemini and Mistral prompts, so the two engines are given the identical
# instruction (only the structured-output mechanism differs per SDK).
EXTRACTION_INSTRUCTION = """You are a document data extraction engine.

Read the attached Indonesian banking declaration/participation document visually.

Extract only information actually visible in the document.

Extract:
- customer name
- account number
- managing unit
- placement amount
- placement tenor
- start date
- end date
- reward type
- customer signature presence
- bank/unit representative signature presence

Use document labels, handwriting, typed text, checkboxes, marks, dates, currency values and visual context.

Do not infer missing values.
Do not use external/reference data.
Do not include printed labels or headers as field values.
Do not guess unreadable characters.
If ambiguous, return uncertain.
If absent, return not_detected.

Signature detection means presence only, not biometric signature verification.

Return only the required structured schema."""


def _response_json_schema():
    """JSON schema for native structured output (Gemini response_schema /
    Mistral document_annotation_format) -- each extracted field is wrapped as
    {"value": ..., "status": "detected"|"uncertain"|"not_detected"} so the AI
    can express uncertainty WITHOUT corrupting the value's type (task's flat
    canonical example is the CONCEPTUAL contract; this wrapper is the wire
    format that keeps typing strict for structured-output validation).
    Signatures are a bare enum since their "value" already IS the status."""
    status_enum = ["detected", "uncertain", "not_detected"]

    def wrapped(value_schema):
        return {
            "type": "object",
            "properties": {"value": value_schema, "status": {"type": "string", "enum": status_enum}},
            "required": ["value", "status"],
        }

    return {
        "type": "object",
        "properties": {
            "nama_nasabah": wrapped({"type": ["string", "null"]}),
            "nomor_rekening": wrapped({"type": ["string", "null"]}),
            "unit_kerja_pengelola_rekening": wrapped({"type": ["string", "null"]}),
            "nominal_penempatan": wrapped({"type": ["integer", "null"]}),
            "tenor_penempatan": wrapped({"type": ["integer", "null"]}),
            "tanggal_mulai": wrapped({"type": ["string", "null"], "description": "ISO date YYYY-MM-DD"}),
            "tanggal_selesai": wrapped({"type": ["string", "null"], "description": "ISO date YYYY-MM-DD"}),
            "bentuk_reward": wrapped({"type": ["string", "null"], "enum": ["tunai", "non_tunai", None]}),
            "signature_nasabah": {"type": "string", "enum": ["present", "absent", "uncertain"]},
            "signature_atasan": {"type": "string", "enum": ["present", "absent", "uncertain"]},
        },
        "required": list(COMMON_FIELDS),
    }


# ============================================================================
# ERRORS (task §14) -- one exception type, `stage` maps directly onto the 4
# new evaluation.py FAILURE_STAGES entries. Never silently swallowed, never
# triggers a fallback to another engine (callers just let it propagate).
# ============================================================================


class ExtractorError(Exception):
    """stage: one of API_AUTH / API_RATE_LIMIT / API_TIMEOUT / API_RESPONSE."""

    def __init__(self, message, stage="API_RESPONSE"):
        super().__init__(message)
        self.stage = stage


class InvalidExtractor(ExtractorError):
    def __init__(self, engine):
        super().__init__(f"Unknown extractor engine: {engine!r}", stage="API_RESPONSE")


def _classify_api_exception(exc):
    """Best-effort classification of an SDK exception into one of the 4
    stages, from its type name + message -- same string-matching style
    already used by app.py's sanitize_error()/_ERROR_HINTS, not a new
    pattern introduced here."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(k in text for k in ("unauthorized", "invalid api key", "authentication", "permission_denied", "401", "403", "api key not valid")):
        return "API_AUTH"
    if any(k in text for k in ("rate limit", "quota", "resource_exhausted", "429", "too many requests")):
        return "API_RATE_LIMIT"
    if any(k in text for k in ("timeout", "timed out", "connection", "network", "unavailable", "deadline", "dns")):
        return "API_TIMEOUT"
    return "API_RESPONSE"


# ============================================================================
# SHARED HELPERS
# ============================================================================


def _check_cancel(cancel_event, step):
    """V19f: checked at each API-engine function's existing prepare/extract/
    validate checkpoints (see run_gemini/run_qwen3_vl_hosted/
    run_qwen3_vl_local) -- same PipelineCancelled signal pipeline.run_pipeline
    raises (via its `emit` closure) at its own 14 checkpoints, so app.py has
    ONE exception type to catch regardless of engine. Can't abort an
    in-flight network/GPU call itself, but stops immediately after it
    returns instead of continuing to validate/adapt/save."""
    if cancel_event is not None and cancel_event.is_set():
        raise pipeline.PipelineCancelled(f"Dibatalkan pada tahap '{step}'")


def _progress(callback, step, percent, message):
    """Mirrors pipeline._emit_progress()'s exact behavior (clamp 0-100, never
    let a callback exception break the extractor) so app.py's existing
    progress-polling plumbing works unchanged for every engine."""
    if callback is None:
        return
    try:
        callback({"step": step, "percent": int(max(0, min(100, percent))), "message": message})
    except Exception:
        pass


class _ProgressHeartbeat:
    """V19f: background ticker that nudges progress_callback's MESSAGE (never
    percent -- real execution stages already set percent, this never fakes
    one) every `interval` seconds while a single long call is in flight (a
    hosted/local Qwen or Gemini inference request, which today sits under
    one frozen checkpoint for its whole duration -- ~22-26s for local Qwen,
    longer once a conditional reward-detail follow-up call also fires).
    Reuses the SAME progress_callback plumbing already threaded through
    every layer -- no new state, no new endpoint. A callback exception (or
    callback=None) is swallowed the same way `_progress()` already does."""

    def __init__(self, callback, step, percent, label, interval=2.5):
        self._callback = callback
        self._step = step
        self._percent = percent
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        start = time.monotonic()
        while not self._stop.wait(self._interval):
            elapsed = int(time.monotonic() - start)
            _progress(self._callback, self._step, self._percent, f"{self._label} ({elapsed} dtk)")

    def __enter__(self):
        if self._callback is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        return False


def _read_file_for_upload(document_path):
    """Raw bytes + best-effort MIME type -- NO cv2/pymupdf rasterization, NO
    template alignment. Reuses prep.is_pdf_document()'s existing magic-header
    sniff (needed because data_input.get_document_path() returns downloaded
    files WITHOUT an extension) for the PDF/image split; falls back to a
    simple PNG-signature check, defaulting to JPEG (the common case for
    phone-camera uploads in this dataset)."""
    with open(document_path, "rb") as f:
        raw_bytes = f.read()
    if prep.is_pdf_document(document_path):
        return raw_bytes, "application/pdf"
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return raw_bytes, "image/png"
    return raw_bytes, "image/jpeg"


_ID_MONTHS = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}


def _format_id_date(iso_str):
    """'2026-01-01' -> '1 Januari 2026' -- confirmed parseable by the
    EXISTING, REUSED postprocessing._extract_dates() (Indonesian month-name
    regex), so derive_tenor_from_range() works unmodified on AI-extracted
    dates exactly as it does on OCR'd handwritten text."""
    if not iso_str:
        return None
    try:
        y, m, d = (int(p) for p in str(iso_str).split("-")[:3])
        return f"{d} {_ID_MONTHS[m]} {y}"
    except Exception:
        return None


def _field_entry(common, name):
    """Read one wrapped {"value":..., "status":..., "confidence":...} field
    from the parsed AI response; defensive flat-value fallback in case a
    response ever comes back unwrapped. Returns (value, status, confidence).
    "confidence" is None for Gemini/Mistral (their wire schema,_response_
    json_schema(), has no confidence key -- .get() harmlessly returns None,
    preserving the V18-session behavior of put()'s confidence=None exactly);
    only Qwen's canonical dict (see _adapt_qwen_to_common) populates it."""
    raw = common.get(name) if isinstance(common, dict) else None
    if isinstance(raw, dict):
        value = raw.get("value")
        status = raw.get("status") or ("not_detected" if value in (None, "") else "detected")
        return value, status, raw.get("confidence")
    return raw, ("not_detected" if raw in (None, "") else "detected"), None


# ============================================================================
# ADAPTER -- canonical AI JSON -> EXACT pipeline.run_pipeline() return shape
# ============================================================================


def adapt_common_to_pipeline_shape(common, source, document_path, prepared_image_bgr=None):
    """`common`: parsed JSON matching `_response_json_schema()`. `source`:
    engine/model name, stamped onto every field's "source" + `ocr_meta`.
    NEVER receives/reads a reference_record -- structurally cannot leak
    ground truth (see module docstring).

    V19 adapter design decision (documented in full in handover.md): V18's
    comparison.validate_tenor()/validate_reward() are built around the PAPER
    FORM's "two independent pieces of evidence must agree" design (a
    checkbox mark + a hand-written date range for tenor; a checkbox mark + a
    filled detail field for reward) -- this doesn't map 1:1 onto a single AI
    JSON response.
      - Tenor: genuinely honest reuse. The canonical schema already gives TWO
        independently-extracted signals from the SAME AI response --
        tenor_penempatan (a number) and tanggal_mulai/tanggal_selesai (dates).
        Both are fed in as-is; if the AI's own number and its own date range
        disagree, validate_tenor correctly flags TOLAK -- catching genuine AI
        self-inconsistency, a real feature of reusing this check.
      - Reward: `common`'s canonical `bentuk_reward` has no separate "detail"
        field (Gemini/Mistral's schema doesn't produce one), so the SAME
        choice value is mirrored into whichever of reward_tunai/
        reward_non_tunai it matches, purely so the existing "detail field
        filled" check has a witness instead of reporting a false REVIEW.
        This is STILL true for Gemini/Mistral -- a known, deliberate
        simplification, flagged here and in handover.md. For Qwen
        specifically, `common` MAY additionally carry OPTIONAL
        `reward_non_tunai_detail`/`reward_tunai_detail` keys (see
        extractors._adapt_qwen_to_common, sourced from vlm.py's conditional
        reward-detail follow-up calls -- see the section comments above
        vlm.REWARD_DETAIL_PROMPT/REWARD_TUNAI_DETAIL_PROMPT) with the
        customer's actual handwritten item description / cash amount; when
        present and non-empty, THAT real value is used instead of the
        mirrored placeholder -- checked below via `_field_entry`, which
        harmlessly returns nothing for Gemini/Mistral's `common` (no such
        key), so their behavior is completely unchanged."""
    final_results = {}

    def put(name, value, status, reason=None, confidence=None):
        final_results[name] = {
            "value": value,
            "raw": (str(value) if value is not None else None),
            "status": status, "confidence": confidence, "source": source, "reason": reason,
        }

    nama, nama_status, nama_conf = _field_entry(common, "nama_nasabah")
    put("nama_nasabah", nama, nama_status, confidence=nama_conf)

    rekening, rekening_status, rekening_conf = _field_entry(common, "nomor_rekening")
    put("nomor_rekening", str(rekening) if rekening is not None else None, rekening_status, confidence=rekening_conf)

    unit, unit_status, unit_conf = _field_entry(common, "unit_kerja_pengelola_rekening")
    put("unit_kerja_pengelola_rekening", unit, unit_status, confidence=unit_conf)

    nominal, nominal_status, nominal_conf = _field_entry(common, "nominal_penempatan")
    put("nominal_penempatan", nominal, nominal_status, confidence=nominal_conf)

    tenor, tenor_status, tenor_conf = _field_entry(common, "tenor_penempatan")
    tanggal_mulai, _tm_status, _tm_conf = _field_entry(common, "tanggal_mulai")
    tanggal_selesai, _ts_status, _ts_conf = _field_entry(common, "tanggal_selesai")

    range_raw = None
    d1, d2 = _format_id_date(tanggal_mulai), _format_id_date(tanggal_selesai)
    if d1 and d2:
        range_raw = f"{d1} s/d {d2}"
    final_results["rentang_tenor"] = {
        "value": range_raw, "raw": range_raw,
        "status": "detected" if range_raw else "not_detected",
        "confidence": None, "source": source, "reason": "derived_from_tanggal_mulai_selesai",
    }
    tenor_choice_status = "detected" if tenor is not None else "review"
    final_results["tenor_penempatan"] = {
        "value": tenor, "status": tenor_choice_status, "source": source, "reason": "ai_extracted",
        "confidence": tenor_conf,
    }

    reward, _reward_status, reward_conf = _field_entry(common, "bentuk_reward")
    reward = reward if reward in ("tunai", "non_tunai") else None
    # reward_non_tunai_detail/reward_tunai_detail: OPTIONAL keys, only
    # present when the caller is Qwen (extractors._adapt_qwen_to_common) --
    # absent for Gemini/Mistral, whose schema has no separate detail field,
    # so _field_entry harmlessly returns (None, "not_detected", None) for
    # them and behavior below is UNCHANGED (falls straight to the
    # mirrored-placeholder branch, same as before these fields existed).
    reward_detail, _detail_status, _detail_conf = _field_entry(common, "reward_non_tunai_detail")
    reward_tunai_detail, _tunai_detail_status, _tunai_detail_conf = _field_entry(common, "reward_tunai_detail")
    non_tunai_value = reward_detail if (reward == "non_tunai" and reward_detail) else (
        reward if reward == "non_tunai" else None
    )
    tunai_value = reward_tunai_detail if (reward == "tunai" and reward_tunai_detail) else (
        reward if reward == "tunai" else None
    )
    put("reward_tunai", tunai_value,
        "detected" if reward == "tunai" else "not_detected")
    put("reward_non_tunai", non_tunai_value,
        "detected" if reward == "non_tunai" else "not_detected")
    reward_choice_status = "detected" if reward is not None else "review"
    final_results["bentuk_reward"] = {
        "value": reward, "status": reward_choice_status, "source": source, "reason": "ai_extracted",
        "confidence": reward_conf,
    }

    _SIG_LABEL = {"present": "Ada tanda tangan", "uncertain": "Perlu verifikasi manual", "absent": "Kosong"}
    for name in ("signature_nasabah", "signature_atasan"):
        raw_sig = common.get(name) if isinstance(common, dict) else None
        state = raw_sig if raw_sig in ("present", "absent", "uncertain") else "uncertain"
        final_results[name] = {
            "value": _SIG_LABEL[state], "status": state, "present": state == "present",
            "reason": "ai_extracted", "source": source,
        }

    fields_table = post.build_fields_table(final_results, post.FIELD_TYPES)

    choice_groups = {
        "tenor_penempatan": {"value": tenor, "status": tenor_choice_status, "reason": "ai_extracted"},
        "bentuk_reward": {"value": reward, "status": reward_choice_status, "reason": "ai_extracted"},
    }

    # No alignment/ROI concept for API engines -- reuse the SAME raw loaded
    # document for "template"/"document" (no ROI to show) and for "original"
    # (the untouched file, for a true before/after comparison in the UI).
    # "preprocessed" is the SAME raw image UNLESS the caller passes
    # `prepared_image_bgr` -- V19f: Qwen's `_qwen_prepare_image` already runs
    # preprocessing.align_with_orientation_correction's coarse 0/90/180/270
    # ROTATION ONLY (never the full template-warp/perspective alignment --
    # that stays exclusively V18's job) before sending the image to the
    # model; this surfaces that SAME corrected image in app.py's existing
    # "Preprocessed" preview box instead of silently re-loading the raw file
    # a second time, so a rotated-photo case (e.g. the Reska Nurdianti
    # landscape-captured document, see preprocessing.py's comment above
    # align_with_orientation_correction) is visibly upright in the UI, not
    # just correctly upright inside the model's own input.
    raw_img = prep.load_document(str(document_path))
    preprocessed_img = prepared_image_bgr if prepared_image_bgr is not None else raw_img
    debug_images = {"template": raw_img, "document": raw_img, "original": raw_img, "preprocessed": preprocessed_img}

    return {
        "alignment": {"status": "not_applicable", "method": "api_extraction", "engine": source},
        "input_preparation": {"rectified": False, "reason": "not_applicable_api_engine"},
        "fields": fields_table,
        "groups": {},
        "choice_groups": choice_groups,
        "raw_results": final_results,
        "field_errors": {},
        "debug_images": debug_images,
        "dynamic_regions_debug": {},
        "ocr_meta": {"strategy": f"api_extraction_{source}", "engine": source, "ocr_calls": 1},
    }


# ============================================================================
# ENGINE: V18 (thin passthrough, V18 internals untouched)
# ============================================================================


def run_v18(document_path, progress_callback=None, reference_record=None, cancel_event=None):
    """reference_record IS forwarded here (unlike run_gemini) because
    pipeline.run_pipeline() already uses it internally in an existing,
    previously-audited way (_run_vlm_reference_check, predates this
    session) -- unrelated to Gemini's no-reference-leakage guarantee, which
    is about NOT giving ground truth to an external API."""
    result = pipeline.run_pipeline(
        str(document_path), progress_callback=progress_callback, reference_record=reference_record,
        cancel_event=cancel_event,
    )
    result.setdefault("ocr_meta", {})["engine"] = "v18"
    return result


# ============================================================================
# ENGINE: Gemini (3.8 Flash / 2.5 Flash -- same function, different model id)
# ============================================================================


def _compute_gemini_cost(model, input_tokens, output_tokens, thinking_tokens):
    """V19 sec 8 -- reads config.py (GEMINI_PRICING/GEMINI_DEFAULT_PRICING/
    USD_IDR_RATE), NEVER hardcodes a price here -- update config.py when
    Gemini's pricing changes, this function stays untouched."""
    pricing = config.GEMINI_PRICING.get(model, config.GEMINI_DEFAULT_PRICING)
    input_cost = input_tokens / 1_000_000 * pricing["input_per_1m"]
    output_cost = (output_tokens + thinking_tokens) / 1_000_000 * pricing["output_per_1m"]
    cost_usd = input_cost + output_cost
    return cost_usd, cost_usd * config.USD_IDR_RATE


def run_gemini(document_path, model, progress_callback=None, cancel_event=None):
    _progress(progress_callback, "prepare", 10, "Preparing document")
    _check_cancel(cancel_event, "prepare")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ExtractorError("GEMINI_API_KEY environment variable is not set", stage="API_AUTH")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ExtractorError(
            f"google-genai package not installed ({exc}); see requirements.txt", stage="API_AUTH"
        ) from exc

    raw_bytes, mime_type = _read_file_for_upload(document_path)

    extract_label = f"Extracting with {ENGINE_LABELS.get(model, model)}"
    _progress(progress_callback, "extract", 35, extract_label)
    heartbeat = _ProgressHeartbeat(progress_callback, "extract", 35, extract_label)
    client = genai.Client(api_key=api_key)
    # `response_schema` is validated as `google.genai.types.Schema`, an
    # OpenAPI-style Pydantic model whose `type` field is a SINGLE strict enum
    # (STRING/NUMBER/INTEGER/.../NULL) -- it cannot represent a standard JSON
    # Schema nullable union (`"type": ["string", "null"]`, what
    # `_response_json_schema()` emits for every Optional field), which is
    # exactly what crashed here before. `response_json_schema` (confirmed
    # present on the installed google-genai's `GenerateContentConfig`, wired
    # straight through to the request via `t_json_schema()`, a no-op
    # passthrough -- NOT validated as `types.Schema`) accepts plain JSON
    # Schema natively, so the ORIGINAL dict is used as-is here -- no dialect
    # conversion, no stripped nullability.
    schema = _response_json_schema()

    # NOTE: exact SDK call shape (client.models.generate_content, Part.from_bytes)
    # matches the google-genai SDK's documented structured-output usage at the
    # time this was written -- verify against the currently-installed package
    # version if this raises an unexpected TypeError/AttributeError, since SDK
    # method signatures can change between releases.
    response = None
    last_exc = None
    with heartbeat:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=raw_bytes, mime_type=mime_type),
                        EXTRACTION_INSTRUCTION,
                    ],
                    config={"response_mime_type": "application/json", "response_json_schema": schema},
                )
                break
            except Exception as exc:
                last_exc = exc
                stage = _classify_api_exception(exc)
                if stage != "API_TIMEOUT" or attempt == 1:
                    raise ExtractorError(str(exc), stage=stage) from exc
                time.sleep(1.5)  # bounded retry, ONLY for transient transport errors
    if response is None:
        raise ExtractorError(str(last_exc), stage="API_TIMEOUT")
    _check_cancel(cancel_event, "extract")

    _progress(progress_callback, "validate", 70, "Validating extraction")
    try:
        common = json.loads(response.text)
    except Exception as exc:
        raise ExtractorError(f"invalid structured response from Gemini: {exc}", stage="API_RESPONSE") from exc

    result = adapt_common_to_pipeline_shape(common, source=model, document_path=document_path)

    # V19 sec 7-8 -- actual usage from the API response's usage_metadata
    # (NEVER estimated when this is available, per task spec), cost from
    # config.py (see _compute_gemini_cost). Stored at TOP LEVEL of the
    # result (not nested in ocr_meta) so app.py/export can read it directly
    # -- absent entirely for every other engine, so "don't show Gemini cost
    # for non-Gemini records" falls out naturally (nothing to show).
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    total_tokens = getattr(usage, "total_token_count", 0) or (input_tokens + output_tokens + thinking_tokens)
    cost_usd, cost_idr = _compute_gemini_cost(model, input_tokens, output_tokens, thinking_tokens)
    result["gemini_usage"] = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost_usd, 6),
        "estimated_cost_idr": round(cost_idr, 2),
    }

    _progress(progress_callback, "done", 100, "Complete")
    return result


# ============================================================================
# ENGINE: Qwen3-VL (Hugging Face HOSTED test -- direct semantic extraction.
# NO PaddleOCR, NO fixed ROI/label coordinates, NO field crop before
# inference, NO local model weights. Deliberately does NOT fall back to the
# existing v18 OCR/ROI pipeline on failure -- a hosted-call error propagates
# as a real ExtractorError instead (this test's explicit requirement); never
# fans out to Gemini/Mistral either.
# ============================================================================

def _qwen_self_assessment(qwen_result):
    """Per-field self-assessment straight from Qwen's own response, for
    ocr_meta logging/debugging -- status (detected/uncertain/not_detected)
    for the text/date/choice fields, confidence (0.0-1.0) for the two
    signature fields, matching vlm._parse_direct_semantic_json's contract
    exactly (see that function's docstring for why the two differ)."""
    out = {}
    for f in vlm.DIRECT_SEMANTIC_FIELDS:
        item = qwen_result.get(f) or {}
        out[f] = item.get("confidence") if f in ("signature_nasabah", "signature_bri") else item.get("status")
    return out


def _qwen_prepare_image(document_path):
    """Minimal preparation ONLY (task spec section 1): PDF -> rendered image
    (reuses prep.load_document's existing PyMuPDF render -- NO template
    alignment), EXIF orientation correction for photos (PIL.ImageOps.
    exif_transpose -- cv2.imread applies EXIF too on most builds, but this is
    explicit per the task spec's requirement), safe resize (handled
    downstream by vlm.extract_direct_semantic_hosted's existing
    HOSTED_MAX_SIDE_FULL cap at inference time, see vlm.py). Deliberately
    NOT doing: deskew,
    threshold, binarize, ROI crop, label detection first, PaddleOCR first --
    the model inspects the WHOLE, un-cropped document.

    V19f: exif_transpose is a no-op whenever the camera/device wrote no EXIF
    Orientation tag, or an invalid one (real case: a HUAWEI MatePad photo
    with Orientation=0 -- not one of the 8 standard EXIF codes -- see
    downloaded_documents/1NLnzLKXVlI-1lrn7WF0T1OgKWSYKHFkK, "Reska
    Nurdianti"), leaving a genuinely sideways-captured document sideways for
    the model to read. Since this engine deliberately sends the WHOLE
    uncropped document (no ROI/perspective warp, per the docstring above),
    fixing this needs coarse whole-image rotation, not a full template
    alignment -- reuses preprocessing.align_with_orientation_correction
    SOLELY to pick the best 0/90/180/270 rotation (via the same ORB-vs-
    template scoring used for the v18 pipeline), discarding the
    warped/aligned candidate it also computes -- that result is a no-op
    (same image back) whenever the document is already upright, so this
    costs nothing extra for the common case."""
    if prep.is_pdf_document(document_path):
        return prep.load_document(str(document_path))
    import numpy as np
    from PIL import Image, ImageOps
    pil_img = ImageOps.exif_transpose(Image.open(str(document_path))).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    try:
        template_img = vlm._get_template_image()
        _, _, _, image_bgr, _ = prep.align_with_orientation_correction(template_img, image_bgr)
    except Exception:
        pass  # best-effort -- never let orientation detection break extraction
    return image_bgr


# V19f: DIRECT_SEMANTIC_PROMPT's signature fields already return a genuine
# 0.0-1.0 self-rated confidence (vlm._parse_direct_semantic_json) -- this
# threshold recovers a real "uncertain" 3rd state from it instead of
# collapsing straight to present/absent (see _signature_state below). Real
# eval evidence (eval_runs/evaluation_summary_qwen_local_v2_fixed.json):
# BEFORE this fix, signature status was 100% present / 0% uncertain across
# all 10 real documents -- tunable, not a measured-optimal value.
SIGNATURE_CONFIDENCE_UNCERTAIN_THRESHOLD = 0.6


# V20 diagnostic finding: on a 25-document run of qwen3_vl_local, 6/25 (24%)
# came back with EVERY core field -- nama/nomor_rekening/nominal_penempatan/
# tenor_penempatan/bentuk_reward -- reported as "not_detected" in a single,
# perfectly well-formed JSON response, on documents independently confirmed
# (by direct visual inspection of the exact image sent to the model) to be
# correctly oriented, legible, and structurally matching the template. This
# is NOT the same thing as a genuinely hard-to-read document (which usually
# still yields a PARTIAL read) -- it's the model collapsing to its own
# "nothing here" default across every field simultaneously. Root cause is
# still open (ruled out so far: image orientation, resolution, and image
# order -- none of which changed the outcome on the one document tested);
# one of the 6 collapsed documents was independently re-tried through the
# HOSTED engine and did NOT collapse there, suggesting (not yet confirmed
# across more than 1 document -- HF credits ran out mid-check) this may be
# specific to the local 4-bit-quantized model rather than the shared prompt.
# Per explicit user decision: do NOT auto-escalate to hosted (cost/quota
# risk, hosted was separately found to be running a much larger/costlier
# model than configured) -- only DETECT and surface this distinctly so it
# isn't silently indistinguishable from "field genuinely blank/unreadable".
_COLLAPSE_CORE_FIELDS = ("nama", "nomor_rekening", "nominal_penempatan", "tenor_penempatan", "bentuk_reward")


def _is_extraction_collapsed(qwen_result):
    """True when EVERY field in _COLLAPSE_CORE_FIELDS came back status=
    'not_detected' in the SAME response -- see the module-level note above.
    Deliberately excludes unit_kerja/tanggal_mulai/tanggal_selesai (routinely
    legitimately blank even on a successful read, per DIRECT_SEMANTIC_PROMPT)
    and the two signature fields (separate {value, confidence} contract)."""
    return all(
        (qwen_result.get(f) or {}).get("status") == "not_detected"
        for f in _COLLAPSE_CORE_FIELDS
    )


def _signature_state(sig, threshold=SIGNATURE_CONFIDENCE_UNCERTAIN_THRESHOLD):
    """present/absent/uncertain from a DIRECT_SEMANTIC signature field's
    {value, confidence}. A low self-rated confidence -- regardless of which
    way the boolean `value` leans -- means the model itself wasn't sure the
    ink was a real signature (vs. a print artifact/faint mark), so it's
    reported as uncertain rather than trusted as a firm present/absent."""
    conf = sig.get("confidence")
    if conf is not None and conf < threshold:
        return "uncertain"
    return "present" if sig.get("value") else "absent"


def _adapt_qwen_to_common(qwen_result):
    """Map vlm.DIRECT_SEMANTIC_FIELDS (nama/nomor_rekening/unit_kerja/
    nominal_penempatan/tenor_penempatan/tanggal_mulai/tanggal_selesai/
    bentuk_reward/reward_non_tunai_detail/signature_nasabah/signature_bri)
    onto the COMMON_FIELDS canonical schema shared with Gemini/Mistral, so
    adapt_common_to_pipeline_shape() is reused UNCHANGED.

    tanggal_mulai/tanggal_selesai, unit_kerja and reward_non_tunai_detail
    were added specifically so Qwen's evidence stops being permanently
    incomplete relative to Gemini/Mistral (see adapt_common_to_pipeline_
    shape's docstring for the shared tenor/reward reuse design -- this
    function now feeds it real data instead of hardcoded nulls/placeholders).

    Every non-signature field in vlm.DIRECT_SEMANTIC_FIELDS now reports its
    own detected/uncertain/not_detected status directly (see vlm._parse_
    direct_semantic_json) instead of a raw 0.0-1.0 confidence that used to
    be thresholded here -- wrap() below just passes it through, falling
    back to _field_entry's own defensive derive-from-value rule (same as
    extractors._field_entry) for the one field that has no status of its
    own (reward_non_tunai_detail, sourced from a separate follow-up call,
    see vlm._extract_reward_detail_hosted)."""
    def wrap(value, status=None):
        if status not in ("detected", "uncertain", "not_detected"):
            status = "not_detected" if value in (None, "") else "detected"
        return {"value": value, "status": status, "confidence": None}

    nama = qwen_result.get("nama") or {}
    rekening = qwen_result.get("nomor_rekening") or {}
    unit = qwen_result.get("unit_kerja") or {}
    nominal = qwen_result.get("nominal_penempatan") or {}
    tenor = qwen_result.get("tenor_penempatan") or {}
    tanggal_mulai = qwen_result.get("tanggal_mulai") or {}
    tanggal_selesai = qwen_result.get("tanggal_selesai") or {}
    reward = qwen_result.get("bentuk_reward") or {}
    reward_detail = qwen_result.get("reward_non_tunai_detail") or {}
    reward_tunai_detail = qwen_result.get("reward_tunai_detail") or {}
    sig_nasabah = qwen_result.get("signature_nasabah") or {}
    sig_bri = qwen_result.get("signature_bri") or {}

    # V19f: vlm._parse_direct_semantic_json now canonicalizes both of these
    # to their exact closed-set values (or None) before they ever get here
    # -- the checks below are now a defense-in-depth safety net, not the
    # primary filter. They used to be the ONLY filter, which meant a
    # near-miss model answer (e.g. "3 bulan", "non tunai" with a space)
    # was silently discarded here instead of being normalized upstream.
    tenor_value = tenor.get("value")
    try:
        tenor_value = int(tenor_value) if tenor_value is not None else None
    except (TypeError, ValueError):
        tenor_value = None

    reward_value = reward.get("value")
    reward_value = reward_value if reward_value in ("tunai", "non_tunai") else None

    rekening_value = rekening.get("value")
    rekening_value = str(rekening_value) if rekening_value not in (None, "") else None

    return {
        "nama_nasabah": wrap(nama.get("value"), nama.get("status")),
        "nomor_rekening": wrap(rekening_value, rekening.get("status")),
        "unit_kerja_pengelola_rekening": wrap(unit.get("value"), unit.get("status")),
        "nominal_penempatan": wrap(nominal.get("value"), nominal.get("status")),
        "tenor_penempatan": wrap(tenor_value, tenor.get("status")),
        "tanggal_mulai": wrap(tanggal_mulai.get("value"), tanggal_mulai.get("status")),
        "tanggal_selesai": wrap(tanggal_selesai.get("value"), tanggal_selesai.get("status")),
        "bentuk_reward": wrap(reward_value, reward.get("status")),
        "reward_non_tunai_detail": wrap(reward_detail.get("value")),
        "reward_tunai_detail": wrap(reward_tunai_detail.get("value")),
        "signature_nasabah": _signature_state(sig_nasabah),
        "signature_atasan": _signature_state(sig_bri),
    }


def run_qwen3_vl_hosted(document_path, progress_callback=None, cancel_event=None):
    """Hugging Face HOSTED Qwen3-VL-2B-Instruct test (engine id "qwen3_vl").
    No `reference_record` parameter -- same no-leakage guarantee as
    run_gemini (vlm.extract_direct_semantic_hosted's signature doesn't
    accept one either). Deliberately does NOT catch failures to fall
    back to run_v18 -- any error (missing HF_TOKEN, auth, timeout, bad
    response) propagates as an ExtractorError with the REAL underlying
    message, per this test's explicit "no silent fallback" requirement."""
    _progress(progress_callback, "prepare", 10, "Preparing document (minimal, no ROI)")
    _check_cancel(cancel_event, "prepare")
    image_bgr = _qwen_prepare_image(document_path)

    model = os.getenv("QWEN_MODEL", config.QWEN_MODEL_DEFAULT).strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise ExtractorError("HF_TOKEN environment variable is not set", stage="API_AUTH")
    timeout = int(os.getenv("HF_REQUEST_TIMEOUT_S", str(config.HF_REQUEST_TIMEOUT_S)))

    extract_label = f"Extracting with {ENGINE_LABELS.get('qwen3_vl')}"
    _progress(progress_callback, "extract", 35, extract_label)
    try:
        # Multi-call LENIENT-vote wrapper, not the plain single-call
        # function -- the two signature fields (and only those two) were
        # observed to give a DIFFERENT true/false verdict on the IDENTICAL
        # document across repeated single-call tests at temperature=0
        # (handover.md); checking 3x and treating ANY "present" as present
        # (explicit user policy: absence should be the harder conclusion)
        # mitigates that. Costs 3x the API calls of a single extraction.
        with _ProgressHeartbeat(progress_callback, "extract", 35, extract_label):
            qwen_result, raw_text, request_meta = vlm.extract_direct_semantic_hosted_majority(
                image_bgr, model=model, hf_token=hf_token, timeout=timeout,
            )
    except vlm.HostedInferenceError as exc:
        raise ExtractorError(str(exc), stage=_classify_api_exception(exc)) from exc
    _check_cancel(cancel_event, "extract")

    _progress(progress_callback, "validate", 70, "Validating extraction")
    common = _adapt_qwen_to_common(qwen_result)
    result = adapt_common_to_pipeline_shape(
        common, source="qwen3_vl", document_path=document_path, prepared_image_bgr=image_bgr,
    )
    result["ocr_meta"]["strategy"] = "qwen3_vl_hosted_direct_semantic"
    result["ocr_meta"]["fallback_source"] = "Qwen3-VL Hosted"
    result["ocr_meta"]["qwen_confidence"] = _qwen_self_assessment(qwen_result)
    result["ocr_meta"]["extraction_collapsed"] = _is_extraction_collapsed(qwen_result)
    # Task requirement: log model/request status/processing time/raw
    # response/parsed JSON. vlm.extract_direct_semantic_hosted already prints
    # these to stdout; also surfaced here via ocr_meta (returned verbatim by
    # app.py) so they're visible in the API response, not just the console.
    result["ocr_meta"]["qwen_hosted_request"] = request_meta
    result["ocr_meta"]["qwen_raw_response"] = raw_text
    _progress(progress_callback, "done", 100, "Complete")
    return result


def run_qwen3_vl_local(document_path, progress_callback=None, cancel_event=None):
    """LOCAL Qwen model test (default models/Qwen3-VL-4B-Instruct, see
    vlm.VLM_MODEL_PATH env var; engine id "qwen3_vl_local"), separate from
    and additive to the hosted "qwen3_vl" engine above. Uses the SAME
    minimal image prep (_qwen_prepare_image) and adapter (_adapt_qwen_to_
    common/adapt_common_to_pipeline_shape) as the hosted path, but runs
    entirely on-device via vlm.extract_direct_semantic_local_majority (no
    network, no HF_TOKEN needed). Deliberately does NOT fall back to run_v18
    on failure, same "no silent fallback" policy as the hosted engine --
    model load errors (missing local weights/dependencies) or CUDA OOM
    propagate as a real ExtractorError.

    V19f: now runs the SAME lenient signature majority-vote as the hosted
    engine (LOCAL_SIGNATURE_VOTE_COUNT, default 3, env var VLM_SIGNATURE_
    VOTE_COUNT), per explicit user policy that local and hosted Qwen3-VL
    should be kept aligned going forward. This supersedes the earlier
    reasoning here ("local greedy decoding is fully deterministic, so
    voting would just repeat the same answer for no benefit") -- that
    determinism claim is still true given IDENTICAL input, but doesn't
    preclude voting being useful for hosted-parity and because the timing
    tradeoff that originally ruled it out (handover.md SS9e: ">5 minutes for
    1 call, 3x tidak masuk akal") no longer holds on the hardware the 4B
    model was validated on (~22-26s/doc, see vlm.py's comment above
    extract_direct_semantic_local). Set VLM_SIGNATURE_VOTE_COUNT=1 to
    restore a single deterministic call on slower local hardware.

    Re-verified on real hardware (RTX 4060 Laptop, 8.6GB VRAM), see
    vlm.py's module comment above extract_direct_semantic_local for the
    full measurement: ~22-26s/document, correct field values on real
    documents -- this supersedes an EARLIER finding here and in
    handover.md ("~0% accuracy... on a 2GB-VRAM GPU"), which was measured
    on different (much more constrained) hardware with a smaller 2B model
    and does not describe this engine's behavior in general. Actual
    accuracy/speed on whatever hardware this is deployed to should still be
    re-measured there rather than assumed from either finding."""
    _progress(progress_callback, "prepare", 10, "Preparing document (minimal, no ROI)")
    _check_cancel(cancel_event, "prepare")
    image_bgr = _qwen_prepare_image(document_path)

    extract_label = f"Extracting with {ENGINE_LABELS.get('qwen3_vl_local')}"
    _progress(progress_callback, "extract", 35, extract_label)
    try:
        with _ProgressHeartbeat(progress_callback, "extract", 35, extract_label):
            qwen_result, raw_text, request_meta = vlm.extract_direct_semantic_local_majority(image_bgr)
    except Exception as exc:
        raise ExtractorError(f"Local Qwen3-VL inference failed ({type(exc).__name__}): {exc}",
                              stage=_classify_api_exception(exc)) from exc
    _check_cancel(cancel_event, "extract")

    _progress(progress_callback, "validate", 70, "Validating extraction")
    common = _adapt_qwen_to_common(qwen_result)
    result = adapt_common_to_pipeline_shape(
        common, source="qwen3_vl_local", document_path=document_path, prepared_image_bgr=image_bgr,
    )
    result["ocr_meta"]["strategy"] = "qwen3_vl_local_direct_semantic"
    result["ocr_meta"]["fallback_source"] = "Qwen3-VL Local"
    result["ocr_meta"]["qwen_confidence"] = _qwen_self_assessment(qwen_result)
    result["ocr_meta"]["qwen_raw_response"] = raw_text
    result["ocr_meta"]["qwen_local_request"] = request_meta
    result["ocr_meta"]["extraction_collapsed"] = _is_extraction_collapsed(qwen_result)
    _progress(progress_callback, "done", 100, "Complete")
    return result


# ============================================================================
# V20 -- HYBRID: Qwen3-VL Local (semantic fields) + local YOLOS signature
# detector (signature fields ONLY). Engine id "qwen3_vl_local_yolos". Does
# NOT modify/replace "qwen3_vl_local" above in any way -- purely additive, a
# separate opt-in engine. See signature_detector.py for the detector itself
# and handover.md for the full architecture writeup.
#
# NOTE: originally targeted Tech4Humans' YOLOv8 signature detector
# (tech4humans/yolov8s-signature-detector), but that repo turned out to be
# GATED on Hugging Face Hub (requires manually accepting access on the
# website -- confirmed no scriptable path around it, even with a valid
# HF_TOKEN). Swapped to mdefrance/yolos-tiny-signature-detection (Apache-2.0,
# NOT gated, verified downloadable) per explicit user decision -- see
# signature_detector.py's module docstring for the full story.
#
# Confidentiality requirement (explicit): this engine must NEVER touch
# hosted/API inference of any kind. Both local dependencies (Qwen3-VL local
# weights, local signature-detector weights) are checked BEFORE any image
# preparation or inference starts -- a missing dependency blocks the whole
# request with a clear error, never a partial Qwen-then-fail.
# ============================================================================


def _hybrid_readiness():
    """Pre-flight check for BOTH local dependencies the hybrid engine needs.
    Cheap -- resolves paths only, never loads either model (unless one is
    already cached from a prior call in this process). Returns a dict:
    {"qwen_local": "ready"|"blocked: <reason>",
     "signature_detector": "ready"|"blocked: <reason>",
     "engine": "ready"|"blocked"}"""
    try:
        vlm._resolve_model_path()
        qwen_status = "ready"
    except Exception as exc:
        qwen_status = f"blocked: {exc}"

    sig_ready, sig_reason = signature_detector.is_ready()
    signature_detector_status = "ready" if sig_ready else f"blocked: {sig_reason}"

    engine_status = "ready" if (qwen_status == "ready" and signature_detector_status == "ready") else "blocked"
    return {"qwen_local": qwen_status, "signature_detector": signature_detector_status, "engine": engine_status}


def _assign_signature_roles(detections, image_shape):
    """Buckets local signature-detector detections into signature_nasabah
    (LEFT) vs signature_atasan (RIGHT) by REUSING preprocessing.
    SIGNATURE_CONFIG's normalized value_bbox geometry -- the SAME
    coordinates vlm.py's own prompt already cites verbally for Qwen's
    benefit -- as a generous region gate rather than a tight pixel-perfect
    crop, since this engine's image prep (_qwen_prepare_image) only does
    EXIF+coarse-rotation, no perspective alignment/warp. NOT a second ROI
    system -- no new geometry is invented here, only a looser tolerance
    applied to the existing numbers.

    Margins (0.06 x, 0.08 y) are chosen so the two regions never overlap
    (the real gap between the two boxes' x-ranges is ~0.134, so 0.06 margin
    on each side leaves a safety margin). Spot-checked this session against
    ONE real document's real detections (see handover.md): 4 of 5 real
    detections landed inside/outside the expected regions exactly as
    intended, but one plausible detection fell ~2 percentage points outside
    the y-margin -- flagged as a real, observed case suggesting the y-margin
    may need widening once more real documents are available, not yet acted
    on without more evidence.

    For each role, the highest-confidence detection whose center falls
    inside that role's expanded region is converted via the EXISTING,
    unchanged extractors._signature_state() (so the detector's confidence
    flows through the SAME present/uncertain boundary Qwen's own confidence
    already uses) -- no detection in a region -> confidence=None -> a clean
    "absent" (never a false "uncertain").

    Returns {"signature_nasabah": {"state": ..., "diagnostic": {...}},
             "signature_atasan": {"state": ..., "diagnostic": {...}}}."""
    h, w = image_shape[0], image_shape[1]
    margin_x, margin_y = 0.06, 0.08
    roles = {}
    for name, cfg in prep.SIGNATURE_CONFIG.items():
        x1, y1, x2, y2 = cfg["value_bbox"]
        region = (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y)
        best = None
        for det in detections:
            bx1, by1, bx2, by2 = det["bbox"]
            cx, cy = ((bx1 + bx2) / 2) / w, ((by1 + by2) / 2) / h
            if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                if best is None or det["confidence"] > best["confidence"]:
                    best = det

        if best is not None:
            state = _signature_state({"value": True, "confidence": best["confidence"]})
            diagnostic = {
                "detector": "yolos_local", "confidence": best["confidence"],
                "bbox": best["bbox"], "status": state,
            }
        else:
            state = _signature_state({"value": False, "confidence": None})
            diagnostic = {"detector": "yolos_local", "confidence": None, "bbox": None, "status": state}
        roles[name] = {"state": state, "diagnostic": diagnostic}
    return roles


def run_qwen3_vl_local_yolos(document_path, progress_callback=None, cancel_event=None):
    """HYBRID engine (id "qwen3_vl_local_yolos"): Qwen3-VL Local for ALL
    semantic fields (nama/nomor_rekening/unit_kerja/nominal_penempatan/
    tenor_penempatan/bentuk_reward), a local YOLOS signature detector
    (signature_detector.py) as the AUTHORITATIVE signature detector for
    signature_nasabah/signature_atasan -- Qwen's own signature judgment is
    discarded from the final result (kept only as a diagnostic, see
    ocr_meta["qwen_raw_signature_diagnostic"] below).

    Hard local-only gate: BOTH local dependencies are checked via
    _hybrid_readiness() as the FIRST thing this function does, before any
    image preparation or inference -- a missing dependency raises
    ExtractorError immediately (never a partial Qwen-then-fail), and this
    engine never falls back to run_qwen3_vl_local/run_qwen3_vl_hosted/any
    other engine, matching the "no silent fallback" policy already used by
    every other Qwen-based engine here.

    Skips Qwen's own 3x signature majority-vote entirely (calls
    vlm.extract_direct_semantic_local -- the existing single-call, no-vote
    building block, NOT extract_direct_semantic_local_majority) since
    Qwen's signature answer is going to be overridden anyway -- avoids 3x
    the local inference cost for a result that would be discarded. This
    does NOT touch run_qwen3_vl_local's own behavior in any way; both
    functions call independent vlm.py entry points.

    extraction_collapsed is computed from the SAME qwen_result (before any
    signature override) via the SAME extractors._is_extraction_collapsed
    definition used everywhere else -- unaffected by the signature source
    (that function only ever reads the 5 non-signature core fields)."""
    readiness = _hybrid_readiness()
    if readiness["engine"] != "ready":
        if readiness["signature_detector"] != "ready":
            raise ExtractorError(
                "Local signature detector is unavailable. Hybrid processing blocked to protect "
                f"document confidentiality. ({readiness['signature_detector']})",
                stage="API_AUTH",
            )
        raise ExtractorError(
            f"Qwen3-VL local model is unavailable. Hybrid processing blocked. ({readiness['qwen_local']})",
            stage="API_AUTH",
        )

    _progress(progress_callback, "prepare", 10, "Preparing document (minimal, no ROI)")
    _check_cancel(cancel_event, "prepare")
    image_bgr = _qwen_prepare_image(document_path)

    extract_label = f"Extracting with {ENGINE_LABELS.get('qwen3_vl_local_yolos')}"
    _progress(progress_callback, "extract", 35, extract_label)
    try:
        with _ProgressHeartbeat(progress_callback, "extract", 35, extract_label):
            qwen_result, raw_text = vlm.extract_direct_semantic_local(image_bgr)
    except Exception as exc:
        raise ExtractorError(f"Local Qwen3-VL inference failed ({type(exc).__name__}): {exc}",
                              stage=_classify_api_exception(exc)) from exc
    _check_cancel(cancel_event, "extract")

    _progress(progress_callback, "detect_signatures", 60, "Detecting signatures (local YOLOS)")
    try:
        detections = signature_detector.detect_signatures(image_bgr)
    except Exception as exc:
        raise ExtractorError(f"Local signature detection failed ({type(exc).__name__}): {exc}",
                              stage="API_AUTH") from exc
    _check_cancel(cancel_event, "detect_signatures")
    signature_roles = _assign_signature_roles(detections, image_bgr.shape)

    _progress(progress_callback, "validate", 70, "Validating extraction")
    common = _adapt_qwen_to_common(qwen_result)
    qwen_raw_signature_diagnostic = {
        "signature_nasabah": qwen_result.get("signature_nasabah"),
        "signature_bri": qwen_result.get("signature_bri"),
    }
    # The actual override -- everything else about `common` (all semantic
    # fields) is untouched, exactly what plain qwen3_vl_local would have
    # produced from this same Qwen call.
    common["signature_nasabah"] = signature_roles["signature_nasabah"]["state"]
    common["signature_atasan"] = signature_roles["signature_atasan"]["state"]

    result = adapt_common_to_pipeline_shape(
        common, source="qwen3_vl_local_yolos", document_path=document_path, prepared_image_bgr=image_bgr,
    )
    result["ocr_meta"]["strategy"] = "qwen3_vl_local_yolos_hybrid"
    result["ocr_meta"]["fallback_source"] = "Qwen3-VL Local + YOLOS Signature"
    result["ocr_meta"]["qwen_confidence"] = _qwen_self_assessment(qwen_result)
    result["ocr_meta"]["qwen_raw_response"] = raw_text
    result["ocr_meta"]["extraction_collapsed"] = _is_extraction_collapsed(qwen_result)
    result["ocr_meta"]["signature_detector_signatures"] = {
        "signature_nasabah": signature_roles["signature_nasabah"]["diagnostic"],
        "signature_atasan": signature_roles["signature_atasan"]["diagnostic"],
    }
    # Qwen's own (superseded) signature judgment kept purely for
    # diagnostics/comparison -- NEVER used as the final signature result.
    result["ocr_meta"]["qwen_raw_signature_diagnostic"] = qwen_raw_signature_diagnostic
    _progress(progress_callback, "done", 100, "Complete")
    return result


# ============================================================================
# ROUTER -- the ONLY function app.py/evaluation.py should call
# ============================================================================

ENGINE_LABELS = {
    "v18": "V18 Existing",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "qwen3_vl": "Qwen3-VL-4B (HF Hosted)",
    "qwen3_vl_local": "Qwen3-VL-4B (Local GPU, EXPERIMENTAL)",
    # V20 -- HYBRID, additive to qwen3_vl_local above (does not replace it):
    # same Qwen3-VL Local semantic extraction, but signature_nasabah/
    # signature_atasan come from a local YOLOS signature detector
    # (signature_detector.py) instead of Qwen's own judgment. Hard
    # local-only gate (see run_qwen3_vl_local_yolos) -- never touches
    # hosted/API inference.
    "qwen3_vl_local_yolos": "Qwen3-VL Local + YOLOS Signature",
}

ENGINES = tuple(ENGINE_LABELS)


def run(engine, document_path, progress_callback=None, reference_record=None, cancel_event=None):
    """SINGLE dispatch point -- exactly one branch executes per call, never a
    fan-out to multiple engines, never an automatic fallback between them.

    cancel_event (OPSIONAL, V19f): threading.Event forwarded to whichever
    engine actually runs -- checked at that engine's own existing
    prepare/extract/validate checkpoints (or, for run_v18,
    pipeline.run_pipeline's 14 checkpoints). Raises pipeline.PipelineCancelled
    when set; app.py catches it to mark the record "cancelled" instead of
    "error"."""
    if engine == "v18":
        result = run_v18(document_path, progress_callback=progress_callback, reference_record=reference_record,
                          cancel_event=cancel_event)
    elif engine == "gemini-3.8-flash":
        result = run_gemini(document_path, model="gemini-3.8-flash", progress_callback=progress_callback,
                             cancel_event=cancel_event)
    elif engine == "qwen3_vl":
        result = run_qwen3_vl_hosted(document_path, progress_callback=progress_callback, cancel_event=cancel_event)
    elif engine == "qwen3_vl_local":
        result = run_qwen3_vl_local(document_path, progress_callback=progress_callback, cancel_event=cancel_event)
    elif engine == "qwen3_vl_local_yolos":
        result = run_qwen3_vl_local_yolos(document_path, progress_callback=progress_callback,
                                           cancel_event=cancel_event)
    else:
        raise InvalidExtractor(engine)
    # V19 sec 5 -- metadata.version stamp, same for every engine (comparison
    # mode wants this consistently present, task spec section 11).
    result.setdefault("ocr_meta", {})["version"] = "v19"
    return result
