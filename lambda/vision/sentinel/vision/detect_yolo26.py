"""YOLO26n ONNX (Ultralytics, NMS-free end-to-end head) on OpenCV 5 DNN, new engine.

The ONNX comes from `models/colab_train_yolo26.ipynb` (opset 17, static 640, batch 1). Two graph shapes are
decoded, so the fallback export documented in `models/verify_yolo26.py` needs no code change:

    end-to-end (default YOLO26 export): output `[1, 300, 6]` = x1, y1, x2, y2, score, class, in letterboxed
        input pixels, top-300 after the model's one-to-one head; rows below `conf` dropped;
    one-to-many (`end2end=False`): output `[1, 4 + nc, 8400]` = cx, cy, w, h, class scores (no objectness).

Both heads then go through class-agnostic `cv2.dnn.NMSBoxes` at `iou`: the one-to-one head is meant to be NMS-free,
but on the valid split it still left 173 pairs of boxes on one fly at IoU >= 0.5 (`eval/results/yolo26_box_rules.json`).
`rules` (sentinel.vision.box_rules, from the DETECTOR_* variables) then drops boxes that cannot be one fruit fly
(low score, too large or small for the photo or for the photo's other boxes, too elongated); `detect()` reports
how many boxes each step removed under `postprocess`.

`tiles` (sentinel.vision.tiling, from the DETECTOR_TILES* variables; off by default) runs the same decoding on 2x2
overlapping windows so small flies are seen larger, merges the windows' boxes in full-frame pixels and then applies
the box rules; `postprocess.tiles` says whether tiles ran, why, and how many forwards the photo cost.

Preprocessing is the YOLOv7 one (letterbox to S x S, pad 114, RGB, /255), so both detectors share
`detect_yolov7.preprocess`. The public interface matches `YoloV7Detector`: `detect(img) -> {count, boxes
[{xywh, score, class_id}], forward_ms, output_shape, output_finite}`, `engine_used`, `size`, `meta`.

    uv run python -m sentinel.vision.detect_yolo26 models/yolo26/fruitfly_yolo26n.onnx IMAGE [--engine new]
        [--tiles off|on|sticky|auto] [--trap-kind sticky|lure]
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
from sentinel.vision import tiling
from sentinel.vision.box_rules import BoxRules, apply_rules, nms
from sentinel.vision.detect_smoke import ENGINES, engine_of
from sentinel.vision.detect_yolov7 import preprocess


def decode_end2end(out: np.ndarray, conf: float, scale: tuple[float, float], offset: tuple[int, int]) -> list[dict]:
    """`[1, N, 6]` (x1, y1, x2, y2, score, class) -> boxes in source-image pixels (`xywh`), before NMS."""
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
    """Pick the decoder from the output shape. `[1, N, 6]` = end-to-end; `[1, 4 + nc, A]` with A > 4 + nc = one-to-many.
    Both are reduced with class-agnostic NMS at `iou`."""
    if (out.ndim == 3 and out.shape[-1] == 6 and out.shape[1] != 6) or (out.ndim == 2 and out.shape[-1] == 6):
        return nms(decode_end2end(out, conf, scale, offset), iou)
    if out.ndim == 3 and out.shape[1] < out.shape[2]:
        return decode_one_to_many(out, conf, iou, scale, offset)
    raise ValueError(f"unexpected YOLO26 output shape {list(out.shape)}: expected [1, 300, 6] or [1, 4+nc, A]")


class Yolo26Detector:
    """Load once, detect many. `engine` is requested; `engine_used` says which one actually ran."""

    name = "yolo26n"

    def __init__(self, onnx_path: str | Path, engine: str = "new", imgsz: int | None = None,
                 resize: str = "letterbox", conf: float = 0.25, iou: float = 0.45, rules: BoxRules | None = None,
                 tiles: tiling.Tiling | None = None):
        self.cv_version = require_opencv5()
        self.onnx_path = Path(onnx_path)
        meta_path = self.onnx_path.with_suffix(".onnx.json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.size = int(imgsz or self.meta.get("imgsz") or 640)
        self.engine, self.resize, self.conf, self.iou, self.rules = engine, resize, conf, iou, rules
        self.tiling = tiles or tiling.Tiling()
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

    def forward(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float], tuple[int, int], float]:
        """Raw network output for `img`: (output, scale, offset, forward ms)."""
        blob, scale, offset = preprocess(img, self.size, self.resize)
        self.net.setInput(blob)
        t0 = time.perf_counter()
        out = self.net.forward()
        forward_ms = (time.perf_counter() - t0) * 1000
        self.head = "end2end" if (out.ndim >= 2 and out.shape[-1] == 6 and out.shape[-2] != 6) else "one_to_many"
        return out, scale, offset, forward_ms

    def _pass(self, img: np.ndarray) -> tuple[list[dict], int | None, np.ndarray, float, float]:
        """One forward: (boxes in `img` pixels after NMS, boxes NMS removed, output, input px per source px, ms)."""
        out, scale, offset, forward_ms = self.forward(img)
        raw = decode_end2end(out, self.conf, scale, offset) if self.head == "end2end" else None
        dets = decode(out, self.conf, self.iou, scale, offset)
        return dets, None if raw is None else len(raw) - len(dets), out, min(scale), forward_ms

    def detect(self, img: np.ndarray, trap_kind: str | None = None) -> dict:
        """Boxes for `img` in its pixels. `trap_kind` ("sticky" | "lure") matters only to DETECTOR_TILES=sticky."""
        t = self.tiling
        H, W = img.shape[:2]
        full, removed, out, forward_ms, forwards = None, None, None, 0.0, 0
        scale = self.size / max(W, H)
        if t.mode in ("off", "auto") or t.grid < 2 or (t.mode == "sticky" and trap_kind != "sticky"):
            full, removed, out, scale, forward_ms = self._pass(img)
            forwards = 1
        use, why = tiling.decide(t, trap_kind, full, scale, max(W, H), self.size)
        tiles = {"mode": t.mode, "used": use, "reason": why}
        dets = full
        if use:
            wins = tiling.windows(W, H, t.grid, t.overlap)
            per_win, margins = [], []
            for x0, y0, x1, y1 in wins:
                d, r, out, s, ms = self._pass(img[y0:y1, x0:x1])
                per_win.append(d)
                margins.append(t.edge_px / s)
                forward_ms += ms
                forwards += 1
                removed = None if r is None else (removed or 0) + r
            dets, cut = tiling.merge(per_win, wins, W, H, self.iou, margins, full if t.keep_full else None)
            tiles.update(grid=t.grid, overlap=t.overlap, windows=[list(w) for w in wins], cut_dropped=cut)
        tiles["forwards"] = forwards
        kept, rejected = apply_rules(dets, W, H, self.rules)
        post = {"nms_iou": self.iou, "nms_removed": removed,
                "rules": None if self.rules is None else self.rules.as_dict(), "rejected": rejected, "tiles": tiles}
        return {"count": len(kept), "boxes": kept, "forward_ms": round(forward_ms, 1),
                "output_shape": list(out.shape), "output_finite": bool(np.isfinite(out).all()), "head": self.head,
                "postprocess": post}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("image")
    ap.add_argument("--engine", choices=sorted(ENGINES), default="new")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--no-rules", action="store_true", help="NMS only, no box rules (DETECTOR_* variables otherwise)")
    ap.add_argument("--tiles", choices=tiling.MODES, default=None, help="DETECTOR_TILES (default: the variable, else off)")
    ap.add_argument("--trap-kind", choices=["sticky", "lure"], default=None)
    args = ap.parse_args()
    tiles = tiling.Tiling.from_env({**os.environ, **({"DETECTOR_TILES": args.tiles} if args.tiles else {})})
    det = Yolo26Detector(args.onnx, args.engine, args.imgsz, conf=args.conf, iou=args.iou,
                         rules=None if args.no_rules else BoxRules.from_env(), tiles=tiles)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image {args.image}")
    res = det.detect(img, trap_kind=args.trap_kind)
    print(json.dumps({"opencv": det.cv_version, "engine_used": det.engine_used, "load_ms": det.load_ms, **res}, indent=2))


if __name__ == "__main__":
    main()
