"""YOLOv7(-tiny) ONNX on OpenCV 5 DNN: load, forward, decode, NMS (fruit-fly detector).

Main project env only (OpenCV 5 + numpy, no torch). The ONNX comes from `models/export_yolov7.py`:
input `images` [1, 3, S, S] RGB /255, output [1, N, 5 + nc] = cx, cy, w, h (input pixels), objectness,
class scores (N = 25200 at S = 640, 1 class -> 6 columns). Score = objectness * class score; NMS with
`cv2.dnn.NMSBoxes`. The `--end2end` export ([n, 7], NMS in graph) is decoded too.

    uv run python -m sentinel.vision.detect_yolov7 models/yolov7-tiny/fruitfly_deploy.onnx IMAGE [--engine new]

`resize`: "letterbox" (keep aspect, pad 114, yolov7's own inference) or "stretch" (plain resize to
S x S, like the Roboflow export the model was trained on: "Resize to 640x640 (Stretch)").
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
from sentinel.vision.detect_smoke import ENGINES, engine_of, letterbox


def preprocess(img: np.ndarray, size: int, resize: str = "letterbox") -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """BGR image -> NCHW float blob; returns (blob, (scale_x, scale_y), (pad_left, pad_top))."""
    if resize == "stretch":
        h, w = img.shape[:2]
        boxed = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        scale, offset = (size / w, size / h), (0, 0)
    elif resize == "letterbox":
        boxed, s, offset = letterbox(img, size)
        scale = (s, s)
    else:
        raise ValueError(f"resize must be 'letterbox' or 'stretch', got {resize!r}")
    blob = cv2.dnn.blobFromImage(boxed, 1 / 255.0, (size, size), swapRB=True, crop=False)
    return blob, scale, offset


def decode(out: np.ndarray, conf: float, iou: float, scale: tuple[float, float], offset: tuple[int, int]) -> list[dict]:
    """YOLOv7 output -> boxes in source-image pixels (`xywh`), after NMS.

    [1, N, 5+nc] (no NMS in graph) or [n, 7] (end2end: batch, x0, y0, x1, y1, class, score).
    """
    sx, sy = scale
    if out.ndim == 2 and out.shape[1] == 7:  # end2end: NMS already done
        rows = out[out[:, 6] >= conf]
        x0 = (rows[:, 1] - offset[0]) / sx
        y0 = (rows[:, 2] - offset[1]) / sy
        x1 = (rows[:, 3] - offset[0]) / sx
        y1 = (rows[:, 4] - offset[1]) / sy
        return [
            {"xywh": [round(float(a), 1) for a in (x0[i], y0[i], x1[i] - x0[i], y1[i] - y0[i])],
             "score": round(float(rows[i, 6]), 4), "class_id": int(rows[i, 5])}
            for i in range(len(rows))
        ]
    pred = out.reshape(-1, out.shape[-1])  # N x (5+nc)
    cls_scores = pred[:, 5:] * pred[:, 4:5]
    scores = cls_scores.max(axis=1)
    keep = scores >= conf
    pred, scores, cls_scores = pred[keep], scores[keep], cls_scores[keep]
    classes = cls_scores.argmax(axis=1) if len(pred) else np.zeros(0, int)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([(cx - w / 2 - offset[0]) / sx, (cy - h / 2 - offset[1]) / sy, w / sx, h / sy], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou) if len(boxes) else []
    return [
        {"xywh": [round(float(v), 1) for v in boxes[i]], "score": round(float(scores[i]), 4),
         "class_id": int(classes[i])}
        for i in np.array(idx).reshape(-1)
    ]


class YoloV7Detector:
    """Load once, detect many. `engine` is requested; `engine_used` says which one actually ran."""

    def __init__(self, onnx_path: str | Path, engine: str = "new", imgsz: int | None = None,
                 resize: str = "letterbox", conf: float = 0.25, iou: float = 0.45):
        self.cv_version = require_opencv5()
        self.onnx_path = Path(onnx_path)
        meta_path = self.onnx_path.with_suffix(".onnx.json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.size = int(imgsz or self.meta.get("imgsz") or 640)
        self.engine, self.resize, self.conf, self.iou = engine, resize, conf, iou
        t0 = time.perf_counter()
        self.net = cv2.dnn.readNet(str(self.onnx_path), "", "", ENGINES[engine])
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
        self._engine_used: str | None = None

    @property
    def engine_used(self) -> str:
        if self._engine_used is None:
            self._engine_used = engine_of(self.net)
        return self._engine_used

    def detect(self, img: np.ndarray) -> dict:
        blob, scale, offset = preprocess(img, self.size, self.resize)
        self.net.setInput(blob)
        t0 = time.perf_counter()
        out = self.net.forward()
        forward_ms = (time.perf_counter() - t0) * 1000
        dets = decode(out, self.conf, self.iou, scale, offset)
        return {"count": len(dets), "boxes": dets, "forward_ms": round(forward_ms, 1),
                "output_shape": list(out.shape), "output_finite": bool(np.isfinite(out).all())}


def run(onnx_path: str | Path, image_path: str | Path | None = None, engine: str = "new",
        imgsz: int | None = None, conf: float = 0.25, iou: float = 0.45, resize: str = "letterbox",
        evidence_dir: str | Path | None = None) -> dict:
    """One image (or a grey frame) through the detector; JSON-serializable dict with `evidence`."""
    det = YoloV7Detector(onnx_path, engine, imgsz, resize, conf, iou)
    if image_path is not None:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read image {image_path}")
    else:
        img = np.full((det.size, det.size, 3), 127, dtype=np.uint8)
    first = det.detect(img)
    second = det.detect(img)
    evidence: dict = {"onnx": str(onnx_path), "image": str(image_path) if image_path else None}
    if evidence_dir is not None:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        overlay = img.copy()
        for d in second["boxes"]:
            x, y, w, h = (round(v) for v in d["xywh"])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        stem = Path(image_path).stem if image_path else "grey"
        name = f"{Path(onnx_path).stem}__{stem}__{engine}__{resize}.jpg"
        cv2.imwrite(str(evidence_dir / name), overlay)
        evidence["overlay"] = str(evidence_dir / name)
    return {
        "opencv": det.cv_version,
        "engine_requested": engine,
        "engine_used": det.engine_used,
        "OPENCV_FORCE_DNN_ENGINE": os.environ.get("OPENCV_FORCE_DNN_ENGINE"),
        "imgsz": det.size,
        "resize": resize,
        "output_shape": second["output_shape"],
        "output_finite": second["output_finite"],
        "load_ms": det.load_ms,
        "forward_ms_first": first["forward_ms"],
        "forward_ms_second": second["forward_ms"],
        "conf": conf,
        "iou": iou,
        "detections": second["count"],
        "boxes": second["boxes"],
        "evidence": evidence,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("image", nargs="?")
    ap.add_argument("--engine", choices=sorted(ENGINES), default="new")
    ap.add_argument("--imgsz", type=int, default=None, help="default: from <onnx>.json, else 640")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--resize", choices=["letterbox", "stretch"], default="letterbox")
    ap.add_argument("--evidence-dir", default="models/runs/detect_yolov7")
    ap.add_argument("--no-boxes", action="store_true")
    args = ap.parse_args()
    result = run(args.onnx, args.image, args.engine, args.imgsz, args.conf, args.iou, args.resize,
                 args.evidence_dir)
    if args.no_boxes:
        result.pop("boxes")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
