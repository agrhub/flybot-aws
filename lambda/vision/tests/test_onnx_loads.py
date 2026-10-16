"""The exported fruit-fly ONNX loads in OpenCV 5 DNN (new engine) and runs a forward pass.

Skipped when no models/*.onnx and no models/yolov7-tiny/fruitfly_deploy.onnx exist (CI: the ONNX files are
git-ignored; `make models` fetches the YOLOv7-tiny one). Point SENTINEL_ONNX at a file to test a specific export.
Uses a dataset image when one is found, else a grey frame. Both output layouts are accepted: YOLOv8/11
`[1, 4 + nc, anchors]` (one anchor per cell) and YOLOv7 `[1, anchors, 5 + nc]` (three anchors per cell).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.vision.detect_smoke import ENGINES, blob_from, engine_of, run

ROOT = Path(__file__).resolve().parents[3]  # repo root (lambda/vision/tests)


def _onnx_files() -> list[Path]:
    if env := os.environ.get("SENTINEL_ONNX"):
        return [Path(env)]
    # static exports only (the classic engine cannot import YOLO11 dynamic-shape exports)
    found = sorted(p for p in (ROOT / "models").glob("*.onnx") if "dynamic" not in p.stem)
    yolov7 = ROOT / "models" / "yolov7-tiny" / "fruitfly_deploy.onnx"
    return found + ([yolov7] if yolov7.is_file() else [])


def _rows(out: np.ndarray) -> np.ndarray:
    """[anchors, values] whatever the export's layout (YOLOv7 rows vs YOLOv8/11 columns)."""
    o = out[0]
    return o if o.shape[0] > o.shape[1] else o.T


def _image() -> Path | None:
    for d in (ROOT / "models" / "datasets", ROOT.parent / "camos-ai-camera" / "models" / "fruit-fly-v3"):
        found = sorted(d.glob("**/test/images/*.jpg"))
        if found:
            return found[0]
    return None


ONNX = _onnx_files()
pytestmark = pytest.mark.skipif(not ONNX or not ONNX[0].exists(), reason="no exported .onnx under models/")


@pytest.fixture(params=ONNX, ids=[p.name for p in ONNX])
def onnx_path(request) -> Path:
    return request.param


def test_new_engine_loads_and_forwards(onnx_path: Path):
    if os.environ.get("OPENCV_FORCE_DNN_ENGINE") == "1":
        pytest.skip("OPENCV_FORCE_DNN_ENGINE=1 forces the classic engine")
    r = run(onnx_path, _image(), engine="new")
    assert r["engine_used"] == "new"
    assert r["output_finite"]
    n, a, b = r["output_shape"]
    s = r["input_shape"][2]
    cells = sum((s // stride) ** 2 for stride in (8, 16, 32))  # strides 8/16/32
    if a > b:  # YOLOv7: [1, anchors, 5 + nc], 3 anchors per cell
        assert n == 1 and b >= 6 and a == 3 * cells
    else:  # YOLOv8/11: [1, 4 + nc, anchors]
        assert n == 1 and a >= 5 and b == cells  # 4 box values + one score per class
    assert isinstance(r["evidence"], dict)


def test_new_and_classic_engines_agree(onnx_path: Path):
    if os.environ.get("OPENCV_FORCE_DNN_ENGINE") in {"1", "2"}:
        pytest.skip("engine forced by OPENCV_FORCE_DNN_ENGINE")
    img = cv2.imread(str(_image())) if _image() else np.full((640, 640, 3), 127, np.uint8)
    blob, _, _ = blob_from(img, 640)
    outs = {}
    for name in ("new", "classic"):
        net = cv2.dnn.readNet(str(onnx_path), "", "", ENGINES[name])
        net.setInput(blob)
        outs[name] = net.forward()
        assert engine_of(net) == name
    # boxes are in input pixels (0..640), scores in 0..1
    new, classic = _rows(outs["new"]), _rows(outs["classic"])
    np.testing.assert_allclose(new[:, :4], classic[:, :4], atol=0.05)
    np.testing.assert_allclose(new[:, 4:], classic[:, 4:], atol=1e-3)
