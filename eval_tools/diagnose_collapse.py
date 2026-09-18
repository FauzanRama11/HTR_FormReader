"""
diagnose_collapse.py
=====================
STANDALONE diagnostic script (like local_qwen_eval.py / signature_diagnostics.py --
NOT imported by app.py/pipeline.py/extractors.py, never touches production code
paths). Root-causes the "Extraction Collapsed" failure mode found in the V20
25-document qwen3_vl_local run: 6/25 documents, all independently confirmed
legible/correctly-oriented, returned a single well-formed JSON response with
EVERY core field (nama/nomor_rekening/nominal_penempatan/tenor_penempatan/
bentuk_reward) reported as "not_detected" simultaneously.

Deliberately does NOT modify vlm.py/extractors.py -- reuses their internals
(vlm._load/_to_pil/_get_template_image/_extract_json, extractors._qwen_
prepare_image/_is_extraction_collapsed) read-only, and re-implements ONLY the
inference call itself (_infer_instrumented, a parallel of vlm._infer) so every
tensor shape / generation kwarg / raw response can be captured without adding
instrumentation plumbing to the production function.

Simplifications vs production (explicit user instruction): ONE inference call
per experiment cell (no 3x signature vote), no reward-detail follow-up, no
fallback/escalation of any kind. This script only ever calls the LOCAL model
-- no HF_TOKEN, no network, no hosted credits spent.

Usage: python diagnose_collapse.py [--quick]
  --quick runs only the baseline + minimal-prompt experiments (skips the
  slower progressive/resolution/generation-config/plain-text sweeps), for a
  fast sanity pass before committing to the full ~40-60 min run.

Output: eval_runs/collapse_diagnosis/*.jsonl (one line per experiment cell,
written incrementally so a crash mid-run loses nothing already completed) and
a final eval_runs/collapse_diagnosis_summary.csv matrix.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

# Dijalankan langsung (`python eval_tools/diagnose_collapse.py ...`), jadi
# sys.path[0] = folder eval_tools/ ini sendiri -- root harus ditambah manual.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

import extractors.extractors as ex
from pipeline import preprocessing as prep
from pipeline import vlm

OUT_DIR = Path("eval_runs/collapse_diagnosis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = OUT_DIR / "runs.jsonl"
SUMMARY_CSV = Path("eval_runs/collapse_diagnosis_summary.csv")

# ============================================================================
# Document sets (from the V20 25-record qwen3_vl_local run, eval_runs/
# v20_diag_local.csv) -- failure set = the 6 records that came back with
# EVERY core field not_detected; control set = 6 of the 19 that succeeded,
# picked for a similar photo/PDF mix to the failure set (3 photos + 1 PDF in
# the failure set, so controls deliberately include both kinds too).
# ============================================================================
FAILED_DOCS = [
    ("indramayu", "13EQ77C6tOZoWZ-vN4HEmn6BO7qsB0hNB"),
    ("pamanukan", "1sMi1kHrlDxklqdhHneeMqt0nf3kGRsmX"),
    ("situmekar_sukabumi_pdf", "15P1XCu-7XtWGaRH1QMTtqYuieYr_nRRh"),
    ("reska_cibadak", "1NLnzLKXVlI-1lrn7WF0T1OgKWSYKHFkK"),
    ("ngawi", "1S5aLxl-MinaWO5uUPuoHiGi9Bi4GpmQd"),
    ("ciamis", "1SHKD4GFSxIYMcOct7aQDMYQp-vPo6zOx"),
]
CONTROL_DOCS = [
    ("medan_gatot_subroto", "1lCr-fbxMzZBZ2tu2yc3VaXCsbTrXaYt7"),
    ("mangga_dua_pdf", "1q14haNe1Yblxht0M5q4wpsSYR-a3HGoQ"),
    ("muntilan", "1a7X78Si8DTJCpf6Czq33IdnokoXoFRuA"),
    ("subang", "1mcuEMRteG6pJMjujoxLxuNpstTBNl87H"),
    ("jakarta_kalideres_pdf", "1H_AVgmXHQVoIuGOpfwzm7JeIcI1R2cz_"),
    ("tanjung_tabalong", "1WZuce8pnp7Z1OAIimavVYuIWRj6pLgfi"),
]
ALL_DOCS = [(n, f, "failed") for n, f in FAILED_DOCS] + [(n, f, "control") for n, f in CONTROL_DOCS]

CORE_FIELDS = ("nama", "nomor_rekening", "nominal_penempatan", "tenor_penempatan", "bentuk_reward")

_MINIMAL_FIELD_HINTS = {
    "nama": "nama - the customer's own handwritten name",
    "nomor_rekening": "nomor_rekening - account number, digits only",
    "nominal_penempatan": "nominal_penempatan - placement amount in Rupiah, digits only",
    "tenor_penempatan": ('tenor_penempatan - the ONE digit (1, 3, or 6) the customer marked on the '
                          'printed "1 / 3 / 6 Bulan" line, via strike-through of the others, a circle '
                          'around it, or bold retrace'),
    "bentuk_reward": ('bentuk_reward - "tunai" or "non_tunai" (underscore not space), the ONE the '
                       'customer marked on the printed "tunai / non tunai" line'),
}


def build_minimal_prompt(fields):
    lines = "\n".join(_MINIMAL_FIELD_HINTS[f] for f in fields)
    shape = ", ".join(f'"{f}":{{"value":null,"status":"not_detected"}}' for f in fields)
    return f"""You are given TWO images, in this order:
1. The BLANK template of this Indonesian bank form (empty, no customer data -- reference only).
2. The FILLED document -- read the field(s) below from THIS image only, using the blank template to understand where each field/box is.

Field(s) to read:
{lines}

Rules:
- Read the customer's own handwriting/mark only. Ignore printed labels, headers, and instructions.
- Use null and status "not_detected" if you cannot read it clearly. Do not guess.
- Return JSON only -- no markdown, no explanation.

Reply with EXACTLY this JSON shape:
{{{shape}}}"""


PLAIN_TEXT_PROMPT = """You are given TWO images: 1) a BLANK template of an Indonesian bank form (reference only, no customer data), 2) the FILLED document.

Look at image 2 only. Near the top of the form there is a line labeled "Nama Nasabah" (Customer Name). Is there a customer name handwritten there? If yes, write exactly what it says. If you genuinely cannot find or read any handwritten name there, say exactly: NOT VISIBLE.

Answer in one short line of plain text only. Do not use JSON. Do not add any other commentary or explanation."""


def _download_path(file_id):
    return os.path.join("downloaded_documents", file_id)


def _load_images(file_id):
    """Returns (raw_image_bgr, preprocessed_image_bgr, prep_path_label,
    raw_shape, prepped_shape). preprocessed = exactly what extractors.
    _qwen_prepare_image would send to the model today (read-only reuse)."""
    path = _download_path(file_id)
    is_pdf = prep.is_pdf_document(path)
    raw = prep.load_document(path)
    if is_pdf:
        prepped = raw
        prep_path = "pdf_render (no EXIF/rotation step exists for PDFs)"
    else:
        prepped = ex._qwen_prepare_image(path)
        rotated = raw.shape[:2] != prepped.shape[:2]
        prep_path = f"photo: exif_transpose + align_with_orientation_correction (rotated={rotated})"
    return raw, prepped, prep_path, raw.shape, prepped.shape


def _infer_instrumented(images_bgr, prompt, max_new_tokens, max_sides, gen_overrides=None):
    """Parallel of vlm._infer() that captures full diagnostic detail. Read-only
    reuse of vlm._load()/_to_pil()/_extract_json() -- does not modify vlm.py."""
    import torch
    model, processor, device = vlm._load()
    pil_images = [vlm._to_pil(img, side) for img, side in zip(images_bgr, max_sides)]

    content = [{"type": "image"} for _ in pil_images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=pil_images, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True}
    if gen_overrides:
        gen_kwargs.update(gen_overrides)

    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    elapsed = time.monotonic() - started

    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated[:, input_len:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    diag = {
        "model_path": vlm._model_path,
        "model_class": type(model).__name__,
        "device": str(device),
        "processor_class": type(processor).__name__,
        "pil_image_sizes_wh": [list(img.size) for img in pil_images],
        "input_ids_shape": list(inputs["input_ids"].shape),
        "pixel_values_shape": list(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
        "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None,
        "prompt_char_len": len(prompt),
        "templated_prompt_char_len": len(prompt_text),
        "gen_kwargs": {k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v))
                       for k, v in gen_kwargs.items()},
        "output_token_count": int(gen_ids.shape[1]),
        "inference_elapsed_s": round(elapsed, 2),
    }
    return text, diag


def run_cell(exp_name, doc_name, file_id, group, prompt, images_bgr, max_sides, gen_overrides=None,
             fields_expected=CORE_FIELDS, parse_json=True):
    """Runs ONE experiment cell, writes it to JSONL immediately, returns the record."""
    record = {
        "experiment": exp_name,
        "document": doc_name,
        "file_id": file_id,
        "group": group,  # "failed" or "control"
    }
    try:
        raw_text, diag = _infer_instrumented(
            images_bgr, prompt, max_new_tokens=400, max_sides=max_sides, gen_overrides=gen_overrides,
        )
        record["raw_response"] = raw_text
        record.update(diag)
        if parse_json:
            try:
                parsed = vlm._extract_json(raw_text)
                clean = vlm._parse_direct_semantic_json(parsed) if fields_expected == CORE_FIELDS + ("signature_nasabah", "signature_bri") else None
                if clean is None:
                    # For minimal/progressive prompts (subset of fields, no
                    # signature keys) -- parse leniently field-by-field
                    # instead of the full DIRECT_SEMANTIC contract.
                    clean = {}
                    for f in fields_expected:
                        item = parsed.get(f)
                        v = item.get("value") if isinstance(item, dict) else item
                        clean[f] = v if v not in ("", None) else None
                record["parsed"] = clean
                non_null = sum(1 for f in fields_expected if clean.get(f) is not None)
                record["non_null_count"] = non_null
                record["collapsed"] = (non_null == 0)
            except Exception as exc:
                record["parse_error"] = str(exc)
                record["collapsed"] = None
        else:
            record["collapsed"] = None  # plain-text cells: judged manually from raw_response
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["collapsed"] = None

    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tag = record.get("collapsed")
    print(f"[{exp_name}] {doc_name} ({group}): collapsed={tag} "
          f"non_null={record.get('non_null_count')} err={record.get('error') or record.get('parse_error') or ''}",
          flush=True)
    return record


def main():
    quick = "--quick" in sys.argv
    template_img = vlm._get_template_image()

    # Preload every document's raw+preprocessed image ONCE (cheap, no model
    # calls) so every experiment below reuses the same arrays.
    docs = {}
    for name, file_id, group in ALL_DOCS:
        raw, prepped, prep_path, raw_shape, prepped_shape = _load_images(file_id)
        docs[name] = {
            "file_id": file_id, "group": group, "raw": raw, "prepped": prepped,
            "prep_path": prep_path, "raw_shape": raw_shape, "prepped_shape": prepped_shape,
        }
        print(f"loaded {name} ({group}): raw={raw_shape} prepped={prepped_shape} :: {prep_path}")

    # ------------------------------------------------------------------
    # EXPERIMENT 0: baseline -- EXACT production config (full semantic
    # prompt, preprocessed image, MAX_SIDE_FULL, current generation config,
    # JSON instruction) -- on all 12 docs. This both confirms reproducibility
    # of the collapse/success pattern under this script's own instrumented
    # call path, and captures the full tensor-shape/model-identity signature
    # for later comparison.
    # ------------------------------------------------------------------
    for name, file_id, group in ALL_DOCS:
        d = docs[name]
        run_cell(
            "0_baseline_full_prompt", name, file_id, group,
            vlm.DIRECT_SEMANTIC_PROMPT, [template_img, d["prepped"]], [vlm.MAX_SIDE_FULL, vlm.MAX_SIDE_FULL],
            fields_expected=CORE_FIELDS + ("signature_nasabah", "signature_bri"),
        )

    # ------------------------------------------------------------------
    # EXPERIMENT 1: minimal prompt asking ONLY for nama -- on all 12 docs.
    # Answers: "does prompt/schema COMPLEXITY (not content) drive the
    # collapse?" If failed docs still return null even for just the name,
    # complexity/length is not the (sole) driver.
    # ------------------------------------------------------------------
    for name, file_id, group in ALL_DOCS:
        d = docs[name]
        run_cell(
            "1_minimal_name_only", name, file_id, group,
            build_minimal_prompt(["nama"]), [template_img, d["prepped"]], [vlm.MAX_SIDE_FULL, vlm.MAX_SIDE_FULL],
            fields_expected=("nama",),
        )

    if quick:
        print("--quick: stopping after experiments 0-1.")
        _write_summary()
        return

    # ------------------------------------------------------------------
    # EXPERIMENT 2: progressive field addition -- FAILED docs only.
    # name -> +nomor_rekening -> +nominal_penempatan -> +tenor_penempatan
    # -> +bentuk_reward. Finds the exact point (if any) where adding a
    # field/prompt-length flips the result back to collapsed.
    # ------------------------------------------------------------------
    progression = [
        ["nama"],
        ["nama", "nomor_rekening"],
        ["nama", "nomor_rekening", "nominal_penempatan"],
        ["nama", "nomor_rekening", "nominal_penempatan", "tenor_penempatan"],
        ["nama", "nomor_rekening", "nominal_penempatan", "tenor_penempatan", "bentuk_reward"],
    ]
    for name, file_id in FAILED_DOCS:
        d = docs[name]
        for fields in progression:
            run_cell(
                f"2_progressive_{'+'.join(fields)}", name, file_id, "failed",
                build_minimal_prompt(fields), [template_img, d["prepped"]],
                [vlm.MAX_SIDE_FULL, vlm.MAX_SIDE_FULL], fields_expected=tuple(fields),
            )

    # ------------------------------------------------------------------
    # EXPERIMENT 3: original (un-prepped) image vs preprocessed image --
    # FAILED docs only. For PDFs in the failed set this is a no-op (same
    # array either way, noted in prep_path) -- run anyway for completeness/
    # transparency rather than skipping.
    # ------------------------------------------------------------------
    for name, file_id in FAILED_DOCS:
        d = docs[name]
        run_cell(
            "3_original_unprepped_image", name, file_id, "failed",
            vlm.DIRECT_SEMANTIC_PROMPT, [template_img, d["raw"]], [vlm.MAX_SIDE_FULL, vlm.MAX_SIDE_FULL],
            fields_expected=CORE_FIELDS + ("signature_nasabah", "signature_bri"),
        )

    # ------------------------------------------------------------------
    # EXPERIMENT 4: higher, consistent resolution (2200 vs current 1600) --
    # FAILED docs only, preprocessed image.
    # ------------------------------------------------------------------
    for name, file_id in FAILED_DOCS:
        d = docs[name]
        run_cell(
            "4_higher_resolution_2200", name, file_id, "failed",
            vlm.DIRECT_SEMANTIC_PROMPT, [template_img, d["prepped"]], [2200, 2200],
            fields_expected=CORE_FIELDS + ("signature_nasabah", "signature_bri"),
        )

    # ------------------------------------------------------------------
    # EXPERIMENT 5: minimal/explicit deterministic generation config --
    # pins every sampling-related knob explicitly (in case an inherited
    # generation_config.json default interacts oddly with do_sample=False)
    # -- FAILED docs only, preprocessed image, full prompt.
    # ------------------------------------------------------------------
    minimal_gen = {
        "do_sample": False, "use_cache": True, "num_beams": 1,
        "temperature": None, "top_p": None, "top_k": None, "repetition_penalty": 1.0,
    }
    for name, file_id in FAILED_DOCS:
        d = docs[name]
        run_cell(
            "5_minimal_gen_config", name, file_id, "failed",
            vlm.DIRECT_SEMANTIC_PROMPT, [template_img, d["prepped"]], [vlm.MAX_SIDE_FULL, vlm.MAX_SIDE_FULL],
            gen_overrides=minimal_gen, fields_expected=CORE_FIELDS + ("signature_nasabah", "signature_bri"),
        )

    # ------------------------------------------------------------------
    # EXPERIMENT 6: plain-text instruction instead of JSON schema -- tests
    # whether the STRUCTURED-OUTPUT constraint itself triggers the collapse,
    # independent of prompt content. FAILED docs only. Judged manually from
    # raw_response (no JSON to parse).
    # ------------------------------------------------------------------
    for name, file_id in FAILED_DOCS:
        d = docs[name]
        run_cell(
            "6_plain_text_no_json", name, file_id, "failed",
            PLAIN_TEXT_PROMPT, [template_img, d["prepped"]], [vlm.MAX_SIDE_FULL, vlm.MAX_SIDE_FULL],
            parse_json=False,
        )

    _write_summary()


def _write_summary():
    if not JSONL_PATH.exists():
        return
    rows = [json.loads(line) for line in open(JSONL_PATH, encoding="utf-8")]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "document", "group", "collapsed", "non_null_count", "raw_response", "error"])
        for r in rows:
            w.writerow([
                r.get("experiment"), r.get("document"), r.get("group"), r.get("collapsed"),
                r.get("non_null_count"), (r.get("raw_response") or "")[:500], r.get("error") or r.get("parse_error") or "",
            ])
    print(f"Wrote summary: {SUMMARY_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
