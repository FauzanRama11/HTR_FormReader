"""
collapse_retry_worker.py
==========================
One-shot subprocess retry worker for the "extraction collapsed"
non-determinism bug in local Qwen3-VL inference (see
extractors._is_extraction_collapsed / handover.md V20 SS9g). Confirmed
there: the SAME document/code/prompt reads correctly in one process launch
and collapses (every core field "not_detected") in another -- deterministic
WITHIN a process (do_sample=False, greedy decoding), but the outcome varies
ACROSS process launches, and VRAM pressure was ruled out as the cause. An
in-process retry would therefore just reproduce the same collapse; a
brand-new process is the only retry channel that has a chance at a
different CUDA/bitsandbytes kernel-selection outcome.

Not meant to be run manually -- invoked internally by
extractors._retry_collapsed_in_subprocess() via `subprocess.run`:

    python collapse_retry_worker.py <manifest_path> <output_json_path>

<manifest_path> is a JSON file: {"image_path": "...", "crops": [{"field":
"nama_nasabah", "path": "..."}, ...], "reward_crops": [{"field":
"reward_tunai", "path": "..."}, ...]}. "crops" (V21, optional, may be an
empty list) carries the SAME close-up crops the original (collapsed) call
used (see vlm.extract_direct_semantic_local's `extra_crops` param /
extractors.run_qwen3_vl_local_yolos) so the retry sees an IDENTICAL image
set and prompt -- not a smaller, different call. "reward_crops" (optional,
may be missing/empty) carries the SAME reward-detail follow-up crops (see
vlm.extract_direct_semantic_local's `reward_crops` param /
extractors.REWARD_DETAIL_CROP_TARGETS) for the same reason.

Reads the already-prepared images and runs vlm.extract_direct_semantic_local
on them exactly as the parent process would, and writes {"qwen_result": ...,
"raw_text": ...} as JSON to <output_json_path>. Exits 0 on success; any
failure prints to stderr and exits non-zero. The caller treats ANY non-zero
exit, timeout, or malformed output file as "retry failed" and keeps the
original (collapsed) result -- this worker never needs to be trusted to
succeed.
"""
import json
import sys
from pathlib import Path

import cv2

# Dijalankan sbg subprocess langsung (`python extractors/collapse_retry_worker.py
# ...`, lihat extractors.py `_retry_collapsed_in_subprocess`), jadi sys.path[0]
# = folder extractors/ ini sendiri, BUKAN project root -- root harus ditambah
# manual sebelum bisa `from pipeline import vlm`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import vlm


def main():
    if len(sys.argv) != 3:
        print("usage: collapse_retry_worker.py <manifest_path> <output_json_path>", file=sys.stderr)
        return 2

    manifest_path, output_path = sys.argv[1], sys.argv[2]
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    image_bgr = cv2.imread(manifest["image_path"])
    if image_bgr is None:
        print(f"failed to read image: {manifest['image_path']}", file=sys.stderr)
        return 2

    extra_crops = []
    for entry in manifest.get("crops") or []:
        crop_img = cv2.imread(entry["path"])
        if crop_img is None:
            print(f"failed to read crop: {entry['path']}", file=sys.stderr)
            return 2
        extra_crops.append((entry["field"], crop_img))

    reward_crops = {}
    for entry in manifest.get("reward_crops") or []:
        crop_img = cv2.imread(entry["path"])
        if crop_img is None:
            print(f"failed to read reward crop: {entry['path']}", file=sys.stderr)
            return 2
        reward_crops[entry["field"]] = crop_img

    qwen_result, raw_text = vlm.extract_direct_semantic_local(
        image_bgr, extra_crops=extra_crops, reward_crops=reward_crops,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"qwen_result": qwen_result, "raw_text": raw_text}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
