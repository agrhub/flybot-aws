"""Load the fruit-fly YOLO ONNX with OpenCV 5 DNN and run one forward pass (T0.4 smoke check).

Main project env only (OpenCV 5 + numpy, no torch):

    uv run python -m sentinel.vision.detect_smoke models/smoke-yolo11n.onnx IMAGE [--engine new]

This is a loader/pipeline check, not the detector (T1.4 `detect.py`). Detection counts from a
smoke-run model say nothing about model quality.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from sentinel import require_opencv5

ENGINES = {
    "auto": cv2.dnn.ENGINE_AUTO,  # new engine first, classic as fallback
    "new": cv2.dnn.ENGINE_NEW,
    "classic": cv2.dnn.ENGINE_CLASSIC,
}


def engine_of(net: cv2.dnn.Net) -> str:
    """Which engine actually built `net` (call after a forward pass: classic needs allocation).

    The new engine dumps its graph as a `main_graph { ... }` text block, the classic one as graphviz.
    """
    dump = net.dump()
    if dump.lstrip().startswith("digraph"):
        return "classic"
    if "main_graph" in dump:
        return "new"
    return "unknown"


def letterbox(img: np.ndarray, size: int, pad: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size (ultralytics preprocessing)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, scale, (left, top)


def blob_from(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    boxed, scale, offset = letterbox(img, size)
    blob = cv2.dnn.blobFromImage(boxed, 1 / 255.0, (size, size), swapRB=True, crop=False)
    return blob, scale, offset


def decode(out: np.ndarray, conf: float, iou: float, scale: float, offset: tuple[int, int]) -> list[dict]:
    """YOLOv8/11 head [1, 4+nc, N] (cx, cy, w, h, scores) -> boxes in source-image pixels after NMS."""
    pred = out[0].T  # N x (4+nc)
    scores = pred[:, 4:].max(axis=1)
    keep = scores >= conf
    pred, scores = pred[keep], scores[keep]
    classes = pred[:, 4:].argmax(axis=1) if len(pred) else np.zeros(0, int)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x = (cx - w / 2 - offset[0]) / scale
    y = (cy - h / 2 - offset[1]) / scale
    boxes = np.stack([x, y, w / scale, h / scale], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou) if len(boxes) else []
    return [
        {"xywh": [round(float(v), 1) for v in boxes[i]], "score": round(float(scores[i]), 4),
         "class_id": int(classes[i])}
        for i in np.array(idx).reshape(-1)
    ]


def run(onnx_path: str | Path, image_path: str | Path | None = None, engine: str = "new",
        imgsz: int | None = None, conf: float = 0.25, iou: float = 0.45,
        evidence_dir: str | Path | None = None) -> dict:
    """Load `onnx_path` with the requested DNN engine, run one forward pass, return a JSON-able dict.

    With no image a mid-grey frame is used (still exercises load + forward).
    """
    cv_version = require_opencv5()
    onnx_path = Path(onnx_path)
    meta_path = onnx_path.with_suffix(".onnx.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    size = int(imgsz or meta.get("imgsz") or 640)

    if image_path is not None:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read image {image_path}")
    else:
        img = np.full((size, size, 3), 127, dtype=np.uint8)
    blob, scale, offset = blob_from(img, size)

    t0 = time.perf_counter()
    net = cv2.dnn.readNet(str(onnx_path), "", "", ENGINES[engine])
    load_ms = (time.perf_counter() - t0) * 1000
    net.setInput(blob)
    t0 = time.perf_counter()
    out = net.forward()
    first_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    out = net.forward()
    second_ms = (time.perf_counter() - t0) * 1000

    dets = decode(out, conf, iou, scale, offset)
    evidence: dict = {"onnx": str(onnx_path), "image": str(image_path) if image_path else None}
    if evidence_dir is not None:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        overlay = img.copy()
        for d in dets:
            x, y, w, h = (round(v) for v in d["xywh"])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)
        name = f"{onnx_path.stem}__{Path(image_path).stem if image_path else 'grey'}__{engine}.jpg"
        cv2.imwrite(str(evidence_dir / name), overlay)
        evidence["overlay"] = str(evidence_dir / name)

    return {
        "opencv": cv_version,
        "engine_requested": engine,
        "engine_used": engine_of(net),
        "OPENCV_FORCE_DNN_ENGINE": os.environ.get("OPENCV_FORCE_DNN_ENGINE"),
        "input_shape": list(blob.shape),
        "output_shape": list(out.shape),
        "output_finite": bool(np.isfinite(out).all()),
        "load_ms": round(load_ms, 1),
        "forward_ms_first": round(first_ms, 1),
        "forward_ms_second": round(second_ms, 1),
        "conf": conf,
        "iou": iou,
        "detections": len(dets),
        "smoke_run_model": meta.get("smoke_run"),
        "boxes": dets,
        "evidence": evidence,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("image", nargs="?")
    ap.add_argument("--engine", choices=sorted(ENGINES), default="new")
    ap.add_argument("--imgsz", type=int, default=None, help="default: from <onnx>.json, else 640")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--evidence-dir", default="models/runs/detect_smoke")
    ap.add_argument("--no-boxes", action="store_true", help="leave the box list out of the printed JSON")
    args = ap.parse_args()
    result = run(args.onnx, args.image, args.engine, args.imgsz, args.conf, evidence_dir=args.evidence_dir)
    if args.no_boxes:
        result.pop("boxes")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
