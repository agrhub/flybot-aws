"""YOLO26n ONNX (Ultralytics, NMS-free end-to-end head) on OpenCV 5 DNN, new engine.

The ONNX comes from `models/colab_train_yolo26.ipynb` (opset 17, static 640, batch 1). Two graph shapes are
decoded, so the fallback export documented in `models/verify_yolo26.py` needs no code change:

    end-to-end (default YOLO26 export): output `[1, 300, 6]` = x1, y1, x2, y2, score, class, in letterboxed
        input pixels, already top-300 after the model's one-to-one head: no NMS here, rows below `conf` dropped;
    one-to-many (`end2end=False`): output `[1, 4 + nc, 8400]` = cx, cy, w, h, class scores (no objectness):
        decoded here and reduced with `cv2.dnn.NMSBoxes`, exactly like detect_yolov7.

Preprocessing is the YOLOv7 one (letterbox to S x S, pad 114, RGB, /255), so both detectors share
`detect_yolov7.preprocess`. The public interface matches `YoloV7Detector`: `detect(img) -> {count, boxes
[{xywh, score, class_id}], forward_ms, output_shape, output_finite}`, `engine_used`, `size`, `meta`.

    uv run python -m sentinel.vision.detect_yolo26 models/yolo26/fruitfly_yolo26n.onnx IMAGE [--engine new]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from sentinel import require_opencv5
from sentinel.vision.detect_smoke import ENGINES, engine_of
from sentinel.vision.detect_yolov7 import preprocess


def decode_end2end(out: np.ndarray, conf: float, scale: tuple[float, float], offset: tuple[int, int]) -> list[dict]:
    """`[1, N, 6]` (x1, y1, x2, y2, score, class) -> boxes in source-image pixels (`xywh`); no NMS needed."""
    rows = out.reshape(-1, out.shape[-1])
    rows = rows[rows[:, 4] >= conf]
    sx, sy = scale
    x0 = (rows[:, 0] - offset[0]) / sx
    y0 = (rows[:, 1] - offset[1]) / sy
    x1 = (rows[:, 2] - offset[0]) / sx
    y1 = (rows[:, 3] - offset[1]) / sy
    return [
        {"xywh": [round(float(a), 1) for a in (x0[i], y0[i], x1[i] - x0[i], y1[i] - y0[i])],
         "score": round(float(rows[i, 4]), 4), "class_id": int(rows[i, 5])}
        for i in range(len(rows))
    ]


def decode_one_to_many(out: np.ndarray, conf: float, iou: float, scale: tuple[float, float],
                       offset: tuple[int, int]) -> list[dict]:
    """`[1, 4 + nc, A]` (Ultralytics one-to-many head, no objectness) -> NMS -> boxes in source pixels."""
    pred = out.reshape(out.shape[-2], out.shape[-1]).T  # A x (4 + nc)
    cls_scores = pred[:, 4:]
    scores = cls_scores.max(axis=1)
    keep = scores >= conf
    pred, scores, cls_scores = pred[keep], scores[keep], cls_scores[keep]
    classes = cls_scores.argmax(axis=1) if len(pred) else np.zeros(0, int)
    sx, sy = scale
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([(cx - w / 2 - offset[0]) / sx, (cy - h / 2 - offset[1]) / sy, w / sx, h / sy], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou) if len(boxes) else []
    return [
        {"xywh": [round(float(v), 1) for v in boxes[i]], "score": round(float(scores[i]), 4),
         "class_id": int(classes[i])}
        for i in np.array(idx).reshape(-1)
    ]


def decode(out: np.ndarray, conf: float, iou: float, scale: tuple[float, float], offset: tuple[int, int]) -> list[dict]:
    """Pick the decoder from the output shape. `[1, N, 6]` = end-to-end; `[1, 4 + nc, A]` with A > 4 + nc = one-to-many."""
    if out.ndim == 3 and out.shape[-1] == 6 and out.shape[1] != 6:
        return decode_end2end(out, conf, scale, offset)
    if out.ndim == 2 and out.shape[-1] == 6:
        return decode_end2end(out, conf, scale, offset)
    if out.ndim == 3 and out.shape[1] < out.shape[2]:
        return decode_one_to_many(out, conf, iou, scale, offset)
    raise ValueError(f"unexpected YOLO26 output shape {list(out.shape)}: expected [1, 300, 6] or [1, 4+nc, A]")


class Yolo26Detector:
    """Load once, detect many. `engine` is requested; `engine_used` says which one actually ran."""

    name = "yolo26n"

    def __init__(self, onnx_path: str | Path, engine: str = "new", imgsz: int | None = None,
                 resize: str = "letterbox", conf: float = 0.25, iou: float = 0.45):
        self.cv_version = require_opencv5()
        self.onnx_path = Path(onnx_path)
        meta_path = self.onnx_path.with_suffix(".onnx.json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.size = int(imgsz or self.meta.get("imgsz") or 640)
        self.engine, self.resize, self.conf, self.iou = engine, resize, conf, iou
        t0 = time.perf_counter()
        self.net = cv2.dnn.readNet(str(self.onnx_path), "", "", ENGINES[engine])
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
        self._engine_used: str | None = None
        self.head: str | None = None  # "end2end" | "one_to_many", known after the first forward

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
        self.head = "end2end" if (out.ndim >= 2 and out.shape[-1] == 6 and out.shape[-2] != 6) else "one_to_many"
        dets = decode(out, self.conf, self.iou, scale, offset)
        return {"count": len(dets), "boxes": dets, "forward_ms": round(forward_ms, 1),
                "output_shape": list(out.shape), "output_finite": bool(np.isfinite(out).all()), "head": self.head}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("image")
    ap.add_argument("--engine", choices=sorted(ENGINES), default="new")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()
    det = Yolo26Detector(args.onnx, args.engine, args.imgsz, conf=args.conf, iou=args.iou)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image {args.image}")
    res = det.detect(img)
    print(json.dumps({"opencv": det.cv_version, "engine_used": det.engine_used, "load_ms": det.load_ms, **res}, indent=2))


if __name__ == "__main__":
    main()
