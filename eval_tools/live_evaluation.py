"""
live_evaluation.py
===================
Read-only aggregation over an in-memory Tab 2 session (app._SESSIONS[session_id])
for the "Evaluasi" modal in the Record Table. Never processes/reprocesses a
document -- reuses match/status data already computed and stored at process
time by comparison.attach_data_entry (see app.py:_process_one_record). Called
by GET /api/sheet/evaluation/{session_id}.

Distinct from evaluation.py (the standalone 25-record ground-truth CLI
benchmark) -- that tool has its own separate normalization and is not wired
into app.py at all. This module is the live counterpart, deliberately reusing
comparison.py's already-computed match/normalization instead of duplicating
it, so results shown here can never silently diverge from what the Record
Table itself already displays for the same record.

Only 5 fields have a real ground-truth column anywhere in the spreadsheet
schema (comparison.COLUMN_FIELD_MAP) -- unit_kerja_pengelola_rekening and the
two signature fields have none. Of those 5, only tenor_penempatan and
bentuk_reward are genuinely categorical (small fixed class set) -- the only
fields where a confusion matrix / multi-class precision/recall/F1 is
mathematically valid; the other 3 (free-text/numeric) only ever get an
Accuracy figure, Precision/Recall/F1 report "N/A".

Signatures get a SEPARATE, explicitly-approximate treatment: per user
decision, the free-text "Status Verifikasi Form Pendaftaran Nasabah" column
(a human reviewer's overall verification note -- not a structured per-
signature label) is parsed conservatively for explicit signature mentions
to derive a stand-in ground truth, until a proper per-signature dataset
exists. See _derive_signature_gt_from_verification for exactly what it does
and does not infer.
"""
import re

from pipeline import comparison
from eval_tools import evaluation
import extractors.extractors as extractors

# The SAME 5-field map app.py's comparison.attach_data_entry already uses --
# sourced from comparison.py so this list can never drift out of sync with it.
GT_COMPARABLE_FIELDS = list(comparison.COLUMN_FIELD_MAP.values())

# Only these 2 fields are genuinely categorical w/ a small fixed class set --
# enforced upstream by vlm._canonicalize_tenor_penempatan/_canonicalize_
# bentuk_reward for Qwen engines, and by postprocessing's choice-group
# resolution for V18 -- so any engine's output should already land in one of
# these classes when it lands anywhere at all.
CATEGORICAL_FIELDS = {
    "tenor_penempatan": {"data_type": "tenor", "valid_classes": ("1", "3", "6")},
    "bentuk_reward": {"data_type": "choice", "valid_classes": ("tunai", "non_tunai")},
}

SIGNATURE_FIELDS = ("signature_nasabah", "signature_atasan")
SIGNATURE_PREDICTED_CLASSES = ("present", "absent", "uncertain")
_VERIF_COLUMN = "Status Verifikasi Form Pendaftaran Nasabah"


def _normalize_class_label(field, value):
    """Raw OCR/GT value -> normalized class label, or None if it falls
    outside the field's valid closed class set (missing/blank/unreadable).
    Reuses comparison._normalize_for_compare directly -- not reimplemented."""
    dtype = CATEGORICAL_FIELDS[field]["data_type"]
    norm = comparison._normalize_for_compare(value, dtype)
    if field == "bentuk_reward" and norm == "nontunai":
        # comparison.py's own "choice" bucket spells this without an
        # underscore -- fold to match vlm.py's canonical "non_tunai" so both
        # matrix axes use the same label spelling.
        norm = "non_tunai"
    valid = CATEGORICAL_FIELDS[field]["valid_classes"]
    return norm if norm in valid else None


def _field_accuracy_stats(results):
    """Per GT-comparable field: reuse each result's fields-table row's
    already-computed 'match' (set by comparison.attach_data_entry at process
    time) -- do NOT re-derive comparisons from scratch.
    match is True       -> measurable + matched
    match is False       -> measurable + mismatched
    match == "uncertain" (nama_nasabah only, fuzzy-match middle state) ->
      tracked SEPARATELY, excluded from BOTH the numerator and denominator
      of accuracy ("do not treat unavailable/non-applicable GT as wrong"
      applied here: an unresolved fuzzy call is not a confirmed wrong
      answer), reported instead as its own uncertain_rate.
    match is None        -> not measurable at all (GT or OCR value missing).
    """
    stats = {f: {"measurable": 0, "matched": 0, "mismatched": 0, "uncertain": 0}
             for f in GT_COMPARABLE_FIELDS}
    for result in results:
        by_field = {row["field"]: row for row in (result.get("fields") or [])}
        for f in GT_COMPARABLE_FIELDS:
            row = by_field.get(f)
            if not row:
                continue
            match = row.get("match")
            if match is True:
                stats[f]["measurable"] += 1
                stats[f]["matched"] += 1
            elif match is False:
                stats[f]["measurable"] += 1
                stats[f]["mismatched"] += 1
            elif match == "uncertain":
                stats[f]["uncertain"] += 1
    for s in stats.values():
        s["accuracy"] = round(s["matched"] / s["measurable"], 4) if s["measurable"] else "N/A"
        denom = s["measurable"] + s["uncertain"]
        s["uncertain_rate"] = round(s["uncertain"] / denom, 4) if denom else "N/A"
    return stats


def _build_matrix_from_pairs(pairs, classes, predicted_classes=None):
    """pairs: list of (actual_label, predicted_label), both already
    restricted to valid classes. `classes` = GT/actual row labels.
    `predicted_classes` = column labels, defaults to `classes` (tenor/reward,
    where predictions only ever come from the same closed set); signatures
    pass a wider predicted set since the model can say "uncertain", a class
    the verification-text GT can never assert."""
    pred_classes = predicted_classes or classes
    matrix = {a: {p: 0 for p in pred_classes} for a in classes}
    for actual, predicted in pairs:
        matrix[actual][predicted] += 1

    per_class = {}
    for c in classes:
        tp = matrix[c].get(c, 0)
        fp = sum(matrix[g].get(c, 0) for g in classes if g != c)
        fn = sum(v for k, v in matrix[c].items() if k != c)
        precision = round(tp / (tp + fp), 4) if (tp + fp) else "N/A"
        recall = round(tp / (tp + fn), 4) if (tp + fn) else "N/A"
        if isinstance(precision, float) and isinstance(recall, float) and (precision + recall) > 0:
            f1 = round(2 * precision * recall / (precision + recall), 4)
        else:
            f1 = "N/A"
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1,
                         "support": sum(matrix[c].values())}

    def _macro(key):
        vals = [v[key] for v in per_class.values() if isinstance(v[key], float)]
        return round(sum(vals) / len(vals), 4) if vals else "N/A"

    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[c].get(c, 0) for c in classes)
    return {
        "classes": list(classes),
        "predicted_classes": list(pred_classes),
        "matrix": matrix,
        "per_class": per_class,
        "macro_precision": _macro("precision"),
        "macro_recall": _macro("recall"),
        "macro_f1": _macro("f1"),
        "accuracy": round(correct / total, 4) if total else "N/A",
        "total_classified": total,
    }


def _confusion_matrix_categorical(results, field):
    """Real confusion matrix for tenor_penempatan/bentuk_reward, built from
    each result's ALREADY-STORED raw ocr_result/data_entry pair (no need to
    go back to session['records']). Records where either side normalizes
    outside the valid class set are excluded entirely -- never counted as a
    wrong class."""
    classes = CATEGORICAL_FIELDS[field]["valid_classes"]
    pairs, excluded = [], 0
    for result in results:
        row = next((r for r in (result.get("fields") or []) if r.get("field") == field), None)
        if not row:
            continue
        gt_label = _normalize_class_label(field, row.get("data_entry"))
        pred_label = _normalize_class_label(field, row.get("ocr_result"))
        if gt_label is None or pred_label is None:
            excluded += 1
            continue
        pairs.append((gt_label, pred_label))
    cm = _build_matrix_from_pairs(pairs, classes)
    cm["excluded_count"] = excluded
    cm["gt_source"] = "ground_truth_column"
    cm["available"] = True
    return cm


_TTD_MISSING_RE = re.compile(r"tidak ada")
_TTD_NASABAH_RE = re.compile(r"ttd\s*(pihak)?\s*nasabah|tanda\s*tangan\s*nasabah")
_TTD_BRI_RE = re.compile(r"ttd\s*(pihak)?\s*bri|tanda\s*tangan\s*(pihak)?\s*bri|ttd\s*(pihak)?\s*unit\s*kerja")


def _derive_signature_gt_from_verification(record):
    """HEURISTIC, EXPLICITLY-APPROXIMATE ground truth for signature
    presence, derived from the free-text '{_VERIF_COLUMN}' column (a human
    reviewer's overall verification note, NOT a structured per-signature
    label) -- per explicit user decision to use this as a stand-in until a
    proper per-signature dataset exists.

    Deliberately CONSERVATIVE: only ever asserts "present" for a clean "OK"
    overall status, and only ever asserts "absent" when the text EXPLICITLY
    names that specific signature as missing. A rejection reason that never
    mentions a given signature leaves that field's GT as None (excluded from
    that field's matrix for this record) -- never guessed as "present" just
    because it wasn't the cited reason, and never guessed as "absent" from
    an unrelated rejection reason (e.g. a reward-only rejection).

    Verified against the real data (assets/ocr_evaluation.xlsx) this
    column's actual distinct values reduce to exactly this handful of
    patterns: "OK, Lanjut Proses"; "Tolak, Tidak Ada Tanda Tangan Pihak
    BRI"; "Tolak, Tunai/Non Tunai? Tidak ada TTD pihak BRI"; "Tolak,
    Tunai/Non Tunai?" (no signature mention at all -> both None)."""
    text = str(record.get(_VERIF_COLUMN) or "").strip().lower()
    if not text:
        return {"signature_nasabah": None, "signature_atasan": None}
    if text.startswith("ok"):
        return {"signature_nasabah": "present", "signature_atasan": "present"}
    missing = bool(_TTD_MISSING_RE.search(text))
    return {
        "signature_nasabah": "absent" if (missing and _TTD_NASABAH_RE.search(text)) else None,
        "signature_atasan": "absent" if (missing and _TTD_BRI_RE.search(text)) else None,
    }


def _confusion_matrix_signature(results, field):
    """Confusion matrix for TTD Nasabah/TTD BRI using the heuristic
    verification-status-derived GT above. Predicted side = the already-
    computed _signature_state class ("present"/"absent"/"uncertain") already
    stored as that field row's own 'status'. Degrades to an explicitly
    unavailable ("N/A") matrix -- never fabricated -- when the uploaded
    sheet has no verification-status column at all."""
    classes = ("present", "absent")
    pairs, excluded = [], 0
    have_verif_column = False
    for result in results:
        record = result.get("record") or {}
        if _VERIF_COLUMN in record:
            have_verif_column = True
        gt_label = _derive_signature_gt_from_verification(record).get(field)
        row = next((r for r in (result.get("fields") or []) if r.get("field") == field), None)
        pred_label = (row or {}).get("status")
        if gt_label is None or pred_label not in SIGNATURE_PREDICTED_CLASSES:
            excluded += 1
            continue
        pairs.append((gt_label, pred_label))

    if not have_verif_column:
        return {
            "classes": list(classes), "predicted_classes": list(SIGNATURE_PREDICTED_CLASSES),
            "matrix": {}, "per_class": {}, "macro_precision": "N/A", "macro_recall": "N/A",
            "macro_f1": "N/A", "accuracy": "N/A", "total_classified": 0,
            "excluded_count": len(results), "gt_source": "verification_status_heuristic",
            "available": False,
        }
    cm = _build_matrix_from_pairs(pairs, classes, predicted_classes=SIGNATURE_PREDICTED_CLASSES)
    cm["excluded_count"] = excluded
    cm["gt_source"] = "verification_status_heuristic"
    cm["available"] = True
    return cm


def _decision_accuracy_stats(results):
    """Overall OK-vs-TOLAK decision accuracy, per explicit user decision
    (distinct from _field_accuracy_stats above, which is per-FIELD).
    Ground truth: reuses evaluation._map_ground_truth_decision verbatim
    (same prefix rule already proven in the standalone CLI evaluator --
    "ok"/"oke"/... -> "OK", "tolak..." -> "TOLAK", anything else/blank ->
    excluded) so this live metric can never quietly diverge from the CLI
    tool's definition of the same thing.

    Predicted: the record's OWN already-computed comparison.compute_decision
    result (session['results'][n]['decision_v9_2']['decision'] -- see
    app.py's _process_one_record), folded to binary: "OK" only for an exact
    "OK" token, everything else ("TOLAK" OR "REVIEW") folded into "TOLAK".
    Per explicit user decision: a document the system flags for manual
    Review is treated the same as a Tolak for this score (neither is an
    auto-approval) -- ground truth in the real dataset never contains a
    "Review" value itself (verified against assets/ocr_evaluation.xlsx), so
    this folding only ever affects the PREDICTED side.

    A record excluded from BOTH sides at once (no ground truth, e.g. blank
    verification-status cell) contributes to neither the numerator nor the
    denominator -- same "don't count missing GT as wrong" principle already
    used throughout this file (_field_accuracy_stats' "uncertain" handling,
    _confusion_matrix_signature's have_verif_column gate)."""
    total_with_gt = correct = false_accept = false_reject = 0
    gt_tolak = gt_ok = review_folded_into_tolak = 0
    for result in results:
        record = result.get("record") or {}
        gt = evaluation._map_ground_truth_decision(record.get(_VERIF_COLUMN))
        if gt is None:
            continue
        predicted_token = (result.get("decision_v9_2") or {}).get("decision")
        if predicted_token is None:
            continue
        if predicted_token == "REVIEW":
            review_folded_into_tolak += 1
        predicted = "OK" if predicted_token == "OK" else "TOLAK"

        total_with_gt += 1
        gt_tolak += gt == "TOLAK"
        gt_ok += gt == "OK"
        if predicted == gt:
            correct += 1
        elif gt == "TOLAK" and predicted == "OK":
            false_accept += 1
        elif gt == "OK" and predicted == "TOLAK":
            false_reject += 1

    return {
        "total_with_ground_truth": total_with_gt,
        "correct_count": correct,
        "accuracy": round(correct / total_with_gt, 4) if total_with_gt else "N/A",
        "false_accept_count": false_accept,
        "false_accept_rate": round(false_accept / gt_tolak, 4) if gt_tolak else "N/A",
        "false_reject_count": false_reject,
        "false_reject_rate": round(false_reject / gt_ok, 4) if gt_ok else "N/A",
        "review_folded_into_tolak_count": review_folded_into_tolak,
    }


def _nominal_terbilang_stats(results, digit_field="nominal_penempatan",
                              terbilang_field="nominal_penempatan_terbilang"):
    """Live counterpart of evaluation.py's nominal_terbilang_both_evidence_
    read_count/nominal_terbilang_internal_consistency_rate -- reuses
    comparison.validate_nominal_terbilang directly against each result's
    already-stored raw_results (same source app.py's own decision
    computation already reads from, see app.py's _process_one_record) so
    this can never compute a different verdict than what the record's own
    processing already determined. DIAGNOSTIC ONLY, same as the CLI
    evaluator's version -- not part of the OK/TOLAK decision.

    `digit_field`/`terbilang_field` (added alongside the reward-detail
    crop fix): same generic pair-selection comparison.validate_nominal_
    terbilang already supports, defaulting to the original nominal_
    penempatan pair so the existing call site below is unaffected. Reused
    for reward_tunai_detail/reward_tunai_terbilang via a second call site."""
    both_evidence = consistent = 0
    for result in results:
        raw_results = result.get("raw_results") or {}
        verdict, _reason, _evidence = comparison.validate_nominal_terbilang(
            raw_results, digit_field=digit_field, terbilang_field=terbilang_field,
        )
        if verdict in ("consistent", "inconsistent"):
            both_evidence += 1
            if verdict == "consistent":
                consistent += 1
    return {
        "both_evidence_read_count": both_evidence,
        "internal_consistency_rate": round(consistent / both_evidence, 4) if both_evidence else "N/A",
    }


def _runtime_stats(results):
    """ONLY over records whose stored result has processing_time_seconds --
    gracefully skips older records missing it (no zero-fill/crash)."""
    times = [r["processing_time_seconds"] for r in results
             if isinstance(r.get("processing_time_seconds"), (int, float))]
    if not times:
        return {"documents_measured": 0, "average_seconds": "N/A", "fastest_seconds": "N/A",
                "slowest_seconds": "N/A", "total_seconds": "N/A"}
    return {
        "documents_measured": len(times),
        "average_seconds": round(sum(times) / len(times), 3),
        "fastest_seconds": round(min(times), 3),
        "slowest_seconds": round(max(times), 3),
        "total_seconds": round(sum(times), 3),
    }


def _extraction_collapsed_stats(results):
    """Reuses extractors._is_extraction_collapsed's result verbatim via
    ocr_meta.extraction_collapsed -- does NOT redefine collapse."""
    total = len(results)
    count = sum(1 for r in results if (r.get("ocr_meta") or {}).get("extraction_collapsed"))
    return {"count": count, "total": total, "rate": round(count / total, 4) if total else "N/A"}


def _engine_info(results, session):
    """Sessions are normally single-engine, but session['engine'] can
    technically be overwritten by a second /api/sheet/submit with a
    different engine on the same session_id. Derive from the actual
    per-result 'engine' field (what really ran) instead of trusting
    session['engine'] blindly; report 'mixed' explicitly rather than
    silently picking one when more than one engine is present."""
    engines = sorted({r.get("engine") for r in results if r.get("engine")})
    if len(engines) == 1:
        eng = engines[0]
        return eng, extractors.ENGINE_LABELS.get(eng, eng)
    if len(engines) > 1:
        label = "Mixed (" + ", ".join(extractors.ENGINE_LABELS.get(e, e) for e in engines) + ")"
        return "mixed", label
    eng = session.get("engine")
    return eng, (extractors.ENGINE_LABELS.get(eng, eng) if eng else "N/A")


def build_session_evaluation(session):
    """Main entry point, called by GET /api/sheet/evaluation/{session_id}.
    `session`: one app._SESSIONS[session_id] dict. Every helper above
    degrades gracefully to "N/A"/0 on an empty `results` list, so a
    freshly-created session with 0 processed records returns a well-formed
    empty shape rather than a special-cased branch here."""
    results = list(session["results"].values())  # already latest-per-record, rerun-safe
    documents_evaluated = len(results)
    documents_total = len(session["records"])
    engine, engine_label = _engine_info(results, session)

    field_stats = _field_accuracy_stats(results)
    categorical_matrices = {f: _confusion_matrix_categorical(results, f) for f in CATEGORICAL_FIELDS}
    signature_matrices = {f: _confusion_matrix_signature(results, f) for f in SIGNATURE_FIELDS}

    field_accuracy = {}
    for f in GT_COMPARABLE_FIELDS:
        s = field_stats[f]
        if f in CATEGORICAL_FIELDS:
            cm = categorical_matrices[f]
            field_accuracy[f] = {**s, "precision": cm["macro_precision"],
                                  "recall": cm["macro_recall"], "f1": cm["macro_f1"]}
        else:
            field_accuracy[f] = {**s, "precision": "N/A", "recall": "N/A", "f1": "N/A"}

    # Overall accuracy = macro-avg of per-field accuracies (fields w/ >=1
    # measurable instance). Overall P/R/F1 macro = macro-avg of ONLY tenor's
    # and bentuk_reward's own macro scores -- nama/rekening/nominal's N/A
    # P/R/F1 are NEVER pooled in, and their accuracy is NEVER substituted in
    # place of precision/recall. (User-confirmed definition.)
    accuracies = [v["accuracy"] for v in field_accuracy.values() if isinstance(v["accuracy"], float)]
    overall_accuracy = round(sum(accuracies) / len(accuracies), 4) if accuracies else "N/A"

    def _macro_of(key):
        vals = [categorical_matrices[f][key] for f in CATEGORICAL_FIELDS
                if isinstance(categorical_matrices[f][key], float)]
        return round(sum(vals) / len(vals), 4) if vals else "N/A"

    overall = {
        "accuracy": overall_accuracy,
        "precision_macro": _macro_of("macro_precision"),
        "recall_macro": _macro_of("macro_recall"),
        "f1_macro": _macro_of("macro_f1"),
    }

    confusion_matrices = dict(categorical_matrices)
    confusion_matrices.update(signature_matrices)

    return {
        "engine": engine,
        "engine_label": engine_label,
        "documents_evaluated": documents_evaluated,
        "documents_total": documents_total,
        "overall": overall,
        "decision_accuracy": _decision_accuracy_stats(results),
        "nominal_terbilang": _nominal_terbilang_stats(results),
        "reward_terbilang": _nominal_terbilang_stats(
            results, digit_field="reward_tunai_detail", terbilang_field="reward_tunai_terbilang",
        ),
        "field_accuracy": field_accuracy,
        "confusion_matrices": confusion_matrices,
        "runtime": _runtime_stats(results),
        "extraction_collapsed": _extraction_collapsed_stats(results),
    }
