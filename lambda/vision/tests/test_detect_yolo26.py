"""YOLO26 output decoding (both graph shapes) and the detector factory, without weights."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.vision import detector
from sentinel.vision.detect_yolo26 import decode, decode_end2end, decode_one_to_many


def test_end2end_rows_map_back_through_letterbox_scale_and_offset():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [100, 50, 140, 90, 0.9, 0]  # letterboxed input pixels
    out[0, 1] = [10, 10, 20, 20, 0.1, 0]  # below conf
    dets = decode(out, conf=0.25, iou=0.45, scale=(0.5, 0.5), offset=(0, 10))
    assert len(dets) == 1
    assert dets[0] == {"xywh": [200.0, 80.0, 80.0, 80.0], "score": 0.9, "class_id": 0}
    assert decode_end2end(out, 0.25, (1.0, 1.0), (0, 0))[0]["xywh"] == [100.0, 50.0, 40.0, 40.0]


def test_one_to_many_head_is_decoded_with_nms():
    nc, anchors = 1, 8400
    out = np.zeros((1, 4 + nc, anchors), np.float32)
    out[0, :, 0] = [120, 100, 40, 40, 0.8]  # cx, cy, w, h, class score
    out[0, :, 1] = [122, 101, 40, 40, 0.7]  # overlapping duplicate: suppressed
    out[0, :, 2] = [400, 400, 20, 20, 0.6]
    dets = decode(out, conf=0.25, iou=0.45, scale=(1.0, 1.0), offset=(0, 0))
    assert [d["score"] for d in dets] == [0.8, 0.6]
    assert dets[0]["xywh"] == [100.0, 80.0, 40.0, 40.0]
    assert decode_one_to_many(out, 0.25, 0.45, (1.0, 1.0), (0, 0)) == dets


def test_unexpected_shape_is_an_error():
    with pytest.raises(ValueError):
        decode(np.zeros((1, 7, 7), np.float32), 0.25, 0.45, (1.0, 1.0), (0, 0))


def test_factory_reports_missing_weights(monkeypatch, tmp_path):
    detector._DETECTORS.clear()
    monkeypatch.setenv("DETECTOR", "yolo26")
    monkeypatch.delenv("DETECTOR_ONNX", raising=False)
    monkeypatch.delenv("SENTINEL_DETECTOR_ONNX", raising=False)
    monkeypatch.delenv("YOLO26_URL", raising=False)
    monkeypatch.setenv("YOLO26_ONNX", str(tmp_path / "missing.onnx"))
    monkeypatch.setattr(detector, "MODEL_DIRS", [tmp_path])
    with pytest.raises(detector.DetectorUnavailable) as ei:
        detector.get_detector()
    assert "YOLO26_URL" in str(ei.value)
    monkeypatch.setenv("DETECTOR", "both")
    with pytest.raises(detector.DetectorUnavailable):
        detector.detector_kind()
    detector._DETECTORS.clear()


def test_fetch_checks_the_hash(monkeypatch, tmp_path):
    import io
    import urllib.request

    payload = b"not-an-onnx"

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: Resp(payload))
    with pytest.raises(detector.DetectorUnavailable):
        detector._fetch("https://example.invalid/m.onnx", tmp_path / "m.onnx", "00" * 32)
    assert not (tmp_path / "m.onnx").exists()
    got = detector._fetch("https://example.invalid/m.onnx", tmp_path / "m.onnx", None)
    assert got.read_bytes() == payload
