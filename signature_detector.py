"""
signature_detector.py
=======================
V20 -- Local signature-presence detector for the "qwen3_vl_local_yolos"
HYBRID engine ONLY. Does not affect "qwen3_vl_local" or any other engine.

Model: mdefrance/yolos-tiny-signature-detection (Hugging Face Hub) -- a
YOLOS (Vision Transformer object detector, via `transformers`) fine-tuned
for single-class ("signature") detection. Apache-2.0 licensed, NOT gated --
freely downloadable, unlike the originally-targeted Tech4Humans YOLOv8
model (tech4humans/yolov8s-signature-detector), which turned out to be a
GATED repo requiring manual access approval on huggingface.co with no
scriptable path around it (confirmed this session: hf_hub_download fails
with GatedRepoError even with a valid HF_TOKEN). Swapped per explicit user
decision rather than blocking this feature on that manual step.

Chosen specifically because it needs ZERO new pip dependencies -- this
project already requires `transformers` for Qwen, and YOLOS uses the exact
same `AutoImageProcessor`/`AutoModelForObjectDetection` + `transformers`
APIs, unlike the abandoned ultralytics/YOLOv8 approach (which also risked a
real opencv-python / opencv-python-headless dependency collision,
encountered and fixed once already this session).

Hard local-only guarantee: every function here operates on a LOCAL
directory path only. `AutoImageProcessor.from_pretrained(local_dir)` /
`AutoModelForObjectDetection.from_pretrained(local_dir)` given a local
directory never touch the network. Model setup/download is a SEPARATE,
one-time, manually-run step (see handover.md) -- this module never
auto-downloads weights at import time or at inference time.

Model-loading pattern mirrors vlm.py's `_load()`: a module-level global
cache, loaded once, reused for every subsequent call in the process
lifetime -- never reloaded per document.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Generic (not vendor-named) env var names -- the underlying model may
# change again later, same convention as vlm.py's VLM_MODEL_PATH.
SIGNATURE_DETECTOR_MODEL_PATH = os.environ.get("SIGNATURE_DETECTOR_MODEL_PATH", "").strip()

# YOLOS detection-proposal confidence threshold (NOT the same as
# extractors.SIGNATURE_CONFIDENCE_UNCERTAIN_THRESHOLD = 0.6, the separate,
# already-established present-vs-uncertain boundary applied AFTER a
# detection is found -- see extractors._signature_state, reused unchanged
# for this engine). 0.25 is a conventional low default; not yet tuned
# against a larger real-document sample from this project (only spot-
# checked on 1 real document this session, see handover.md).
SIGNATURE_DETECTOR_CONF_THRESHOLD = float(os.environ.get("SIGNATURE_DETECTOR_CONF_THRESHOLD", "0.25"))

DEFAULT_LOCAL_CANDIDATES = [
    BASE_DIR / "models" / "yolos-tiny-signature-detection",
    Path("C:/models/yolos-tiny-signature-detection"),
]

_model = None
_processor = None


def _resolve_model_path():
    if SIGNATURE_DETECTOR_MODEL_PATH:
        p = Path(SIGNATURE_DETECTOR_MODEL_PATH)
        if p.exists():
            return str(p)
        raise RuntimeError(
            f"SIGNATURE_DETECTOR_MODEL_PATH is set but the directory was not found: {SIGNATURE_DETECTOR_MODEL_PATH}"
        )
    for p in DEFAULT_LOCAL_CANDIDATES:
        if p.exists():
            return str(p)
    raise RuntimeError(
        "Local signature-detector model not found. Download "
        "mdefrance/yolos-tiny-signature-detection (Hugging Face Hub -- Apache-2.0, "
        "NOT gated) into models/yolos-tiny-signature-detection/, or set "
        "SIGNATURE_DETECTOR_MODEL_PATH."
    )


def is_ready():
    """Cheap readiness probe -- checks `transformers` is importable and the
    local model directory exists, WITHOUT loading the model (no GPU/CPU
    allocation) unless it's already cached from a prior call. Safe to call
    before every document as the hybrid engine's pre-flight gate.
    Returns (bool, reason_or_None)."""
    if _model is not None:
        return True, None
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        return False, f"transformers package not installed ({exc}). Run: pip install -r requirements.txt"
    try:
        _resolve_model_path()
    except Exception as exc:
        return False, str(exc)
    return True, None


def _load():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except ImportError as exc:
        raise RuntimeError(
            f"transformers package not installed ({exc}). Run: pip install -r requirements.txt"
        ) from exc
    model_path = _resolve_model_path()
    # from_pretrained(<local directory>) -- a local path never touches the
    # network, unlike passing a HF repo id. This is the only place in the
    # module that touches the model files.
    _processor = AutoImageProcessor.from_pretrained(model_path)
    _model = AutoModelForObjectDetection.from_pretrained(model_path).eval()
    return _model, _processor


def detect_signatures(image_bgr, conf_threshold=None):
    """Runs the YOLOS signature detector on the WHOLE prepared document
    image (no template needed). Returns a list of
    {"bbox": [x1, y1, x2, y2] (pixel coords, rounded), "confidence": float}
    -- one entry per detected box above the confidence threshold. No image
    data is stored anywhere; only these numbers are returned."""
    import torch
    import cv2
    from PIL import Image

    model, processor = _load()
    conf = conf_threshold if conf_threshold is not None else SIGNATURE_DETECTOR_CONF_THRESHOLD

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    h, w = image_bgr.shape[0], image_bgr.shape[1]

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(h, w)])
    results = processor.post_process_object_detection(outputs, threshold=conf, target_sizes=target_sizes)[0]

    detections = []
    for score, box in zip(results["scores"], results["boxes"]):
        xyxy = box.tolist()
        detections.append({"bbox": [round(v, 1) for v in xyxy], "confidence": round(float(score), 4)})
    return detections
