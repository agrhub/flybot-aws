"""YOLOv7 ONNX decoding (sentinel/vision/detect_yolov7.py) and, when the export exists, an OpenCV 5 load.

The ONNX is git-ignored: the load test is skipped unless models/yolov7-tiny/fruitfly_deploy.onnx
(models/export_yolov7.py) exists or SENTINEL_YOLOV7_ONNX points at one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sentinel.vision.detect_yolov7 import decode, preprocess, run

ROOT = Path(__file__).resolve().parents[3]  # repo root (lambda/vision/tests)
ONNX = Path(os.environ.get("SENTINEL_YOLOV7_ONNX", ROOT / "models" / "yolov7-tiny" / "fruitfly_deploy.onnx"))


def _row(cx, cy, w, h, obj, cls):
    return [cx, cy, w, h, obj, cls]


def test_decode_scores_nms_and_mapping():
    out = np.array([[
        _row(100, 300, 20, 20, 0.9, 0.9),   # kept
        _row(102, 301, 20, 20, 0.8, 0.9),   # overlaps the first: removed by NMS
        _row(300, 200, 10, 30, 0.9, 0.5),   # score 0.45: kept
        _row(500, 500, 10, 10, 0.9, 0.2),   # score 0.18 < 0.25: dropped
    ]], dtype=np.float32)
    # letterbox of a 1280x720 image to 640: scale 0.5, pad top 140
    dets = decode(out, conf=0.25, iou=0.45, scale=(0.5, 0.5), offset=(0, 140))
    assert len(dets) == 2
    best = max(dets, key=lambda d: d["score"])
    assert best["score"] == pytest.approx(0.81, abs=1e-4)
    assert best["xywh"] == [180.0, 300.0, 40.0, 40.0]  # (100-10)/0.5, (300-10-140)/0.5


def test_decode_end2end_rows():
    out = np.array([[0, 10, 20, 30, 60, 0, 0.9], [0, 1, 1, 2, 2, 0, 0.1]], dtype=np.float32)
    dets = decode(out, conf=0.25, iou=0.45, scale=(2.0, 2.0), offset=(0, 0))
    assert dets == [{"xywh": [5.0, 10.0, 10.0, 20.0], "score": 0.9, "class_id": 0}]


def test_preprocess_modes():
    img = np.zeros((720, 1280, 3), np.uint8)
    blob, scale, offset = preprocess(img, 640, "letterbox")
    assert blob.shape == (1, 3, 640, 640) and scale == (0.5, 0.5) and offset == (0, 140)
    blob, scale, offset = preprocess(img, 640, "stretch")
    assert scale == (0.5, 640 / 720) and offset == (0, 0)
    with pytest.raises(ValueError):
        preprocess(img, 640, "crop")


@pytest.mark.skipif(not ONNX.exists(), reason="no YOLOv7 ONNX export (git-ignored)")
def test_yolov7_onnx_runs_on_new_engine():
    r = run(ONNX, None, engine="new")
    assert r["engine_used"] == "new"
    assert r["output_shape"][-1] == 6 and r["output_finite"]
    assert r["detections"] == 0  # mid-grey frame
