"""Op 5 detector selection: YOLO26n (new) or YOLOv7-tiny (fallback), both on OpenCV 5 DNN ENGINE_NEW.

    DETECTOR                 yolo26 | yolov7 (default yolov7 until the YOLO26 weights exist)
    DETECTOR_ONNX            explicit ONNX path (wins over the defaults below)
    SENTINEL_DETECTOR_ONNX   legacy name of DETECTOR_ONNX (eval scripts)
    YOLO26_ONNX              default models/yolo26/fruitfly_yolo26n.onnx (or /opt/models/yolo26/... in the image)
    YOLO26_URL / YOLO26_SHA256   fetch the YOLO26 ONNX (+ its .onnx.json) into MODEL_CACHE_DIR at cold start
                             when it is not on disk; the hash is checked, a mismatch fails closed
    MODEL_CACHE_DIR          default /tmp/models (Lambda) - where fetched weights land
    DETECTOR_CONF / DETECTOR_IOU   thresholds (default 0.25 / 0.45)

`get_detector()` builds the detector once per process; `model_info()` is the contract's `model` object.
`DetectorUnavailable` is raised when no ONNX can be found (-> the handler's `model_unavailable` error).
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repository root (lambda/vision/sentinel/vision/detector.py)
DEFAULTS = {
    "yolov7": ("yolov7-tiny/fruitfly_deploy.onnx", "yolov7-tiny"),
    "yolo26": ("yolo26/fruitfly_yolo26n.onnx", "yolo26n"),
}
MODEL_DIRS = [Path(os.environ.get("MODEL_DIR", "/opt/models")), ROOT / "models",
              Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/models"))]

_DETECTORS: dict[str, object] = {}


class DetectorUnavailable(RuntimeError):
    pass


def detector_kind() -> str:
    kind = (os.environ.get("DETECTOR") or "yolov7").strip().lower()
    if kind not in DEFAULTS:
        raise DetectorUnavailable(f"DETECTOR must be yolov7 or yolo26, not {kind!r}")
    return kind


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, sha256: str | None) -> Path:
    """Download `url` (and `url + .json` when it exists) to `dest`, hash checked. Public bucket: anonymous HTTPS."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "flybot-vision-lambda/1"})
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)
    got = _sha256(part)
    if sha256 and got != sha256.lower():
        part.unlink(missing_ok=True)
        raise DetectorUnavailable(f"model hash mismatch for {url}: {got} != {sha256}")
    part.replace(dest)
    try:
        side = urllib.request.Request(url + ".json", headers={"User-Agent": "flybot-vision-lambda/1"})
        with urllib.request.urlopen(side, timeout=30) as resp:
            dest.with_suffix(".onnx.json").write_bytes(resp.read())
    except Exception as e:  # the sidecar only carries imgsz / provenance  # noqa: BLE001
        print(f'{{"msg":"model sidecar missing","url":"{url}.json","error":"{type(e).__name__}"}}')
    print(f'{{"msg":"model fetched","url":"{url}","sha256":"{got}","ms":{round((time.perf_counter() - t0) * 1000)}}}')
    return dest


def resolve_onnx(kind: str | None = None) -> Path:
    """The ONNX file for `kind`: DETECTOR_ONNX, else the first existing default location, else a fetch."""
    kind = kind or detector_kind()
    explicit = os.environ.get("DETECTOR_ONNX") or os.environ.get("SENTINEL_DETECTOR_ONNX")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise DetectorUnavailable(f"DETECTOR_ONNX {explicit} does not exist")
    rel = DEFAULTS[kind][0]
    env_path = os.environ.get("YOLO26_ONNX") if kind == "yolo26" else os.environ.get("YOLOV7_ONNX")
    candidates = [Path(env_path)] if env_path else []
    candidates += [d / rel for d in MODEL_DIRS]
    for c in candidates:
        if c.is_file():
            return c
    url = os.environ.get("YOLO26_URL") if kind == "yolo26" else os.environ.get("YOLOV7_URL")
    if url and url.startswith("https://"):
        dest = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/models")) / rel
        return _fetch(url, dest, os.environ.get("YOLO26_SHA256" if kind == "yolo26" else "YOLOV7_SHA256"))
    raise DetectorUnavailable(f"no {kind} ONNX found (looked at {[str(c) for c in candidates]}; set DETECTOR_ONNX "
                              f"or {'YOLO26_URL' if kind == 'yolo26' else 'YOLOV7_URL'}, or run `make models`)")


def build_detector(kind: str, onnx_path: Path, engine: str = "new"):
    conf = float(os.environ.get("DETECTOR_CONF", "0.25"))
    iou = float(os.environ.get("DETECTOR_IOU", "0.45"))
    if kind == "yolo26":
        from sentinel.vision.detect_yolo26 import Yolo26Detector

        return Yolo26Detector(onnx_path, engine=engine, conf=conf, iou=iou)
    from sentinel.vision.detect_yolov7 import YoloV7Detector

    det = YoloV7Detector(onnx_path, engine=engine, conf=conf, iou=iou)
    det.name = "yolov7-tiny"
    return det


def get_detector(kind: str | None = None):
    """The process-wide detector for `kind` (default $DETECTOR). Raises DetectorUnavailable."""
    kind = kind or detector_kind()
    if kind not in _DETECTORS:
        path = resolve_onnx(kind)
        try:
            _DETECTORS[kind] = build_detector(kind, path)
        except Exception as e:  # cv2.error on an unreadable graph
            raise DetectorUnavailable(f"cannot load {path.name}: {type(e).__name__}: {e}"[:300]) from e
    return _DETECTORS[kind]


def set_detector(det, kind: str = "fake") -> None:
    """Tests: inject a detector (anything with .detect(img) -> {count, boxes[{xywh, score, class_id}]})."""
    _DETECTORS.clear()
    _DETECTORS[kind] = det
    os.environ["DETECTOR"] = kind
    DEFAULTS.setdefault(kind, (f"{kind}.onnx", kind))


def model_info(det=None) -> dict:
    """The contract's `model` object: {name, version, engine}."""
    import cv2

    det = det or get_detector()
    meta = getattr(det, "meta", {}) or {}
    version = meta.get("version") or meta.get("trained") or meta.get("exported") or (meta.get("sha256") or "")[:12] or None
    engine = getattr(det, "engine_used", None)
    try:
        engine = det.engine_used if engine is None or callable(engine) else engine
    except Exception:  # noqa: BLE001 - a fake detector has no net
        engine = "fake"
    return {"name": getattr(det, "name", type(det).__name__), "version": version,
            "engine": f"opencv-{cv2.__version__}-dnn-{engine}", "onnx": Path(str(getattr(det, "onnx_path", ""))).name or None}
