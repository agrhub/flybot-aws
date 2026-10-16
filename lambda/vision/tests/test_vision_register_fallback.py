"""M4: `aliked` without usable ALIKED/LightGlue ONNX files falls back to SIFT with a recorded fallback_reason."""

import json
import logging
from pathlib import Path

import cv2

from sentinel.vision.pipeline import analyze
from sentinel.vision.register import ALIKED_MODEL, LIGHTGLUE_MODEL, register

FIX = Path(__file__).parent / "fixtures"
PREV, CUR = FIX / "vision_be4089629_prev.jpg", FIX / "vision_be4089629_cur.jpg"


def _pair():
    return cv2.imread(str(PREV)), cv2.imread(str(CUR))


def test_aliked_without_models_falls_back_to_sift(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SENTINEL_FEATURES_DIR", str(tmp_path / "empty"))
    a, b = _pair()
    for requested in ("aliked", "aliked_lightglue"):
        with caplog.at_level(logging.WARNING, logger="sentinel.vision.register"):
            r = register(a, b, method=requested)
        json.dumps(r)
        assert r["ok"] and r["method"] == "sift_flann" and r["requested_method"] == requested
        assert r["fallback_used"] is True
        assert "not found" in r["fallback_reason"] and ALIKED_MODEL in r["fallback_reason"]
        assert [x["method"] for x in r["attempts"]] == ["sift_flann"]
    assert any("not found" in m and "sift_flann" in m for m in caplog.messages)


def test_auto_without_models_has_no_fallback_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_FEATURES_DIR", str(tmp_path / "empty"))
    r = register(*_pair(), method="auto")
    assert r["method"] == "sift_flann" and r["fallback_reason"] is None and r["fallback_used"] is False


def test_unreadable_models_fall_back_to_sift(tmp_path, monkeypatch):
    for f in (ALIKED_MODEL, LIGHTGLUE_MODEL):
        (tmp_path / f).write_bytes(b"not an onnx file")
    monkeypatch.setenv("SENTINEL_FEATURES_DIR", str(tmp_path))
    r = register(*_pair(), method="aliked")
    assert r["ok"] and r["method"] == "sift_flann"
    assert r["fallback_reason"].startswith("ALIKED/LightGlue failed")
    assert [x["method"] for x in r["attempts"]] == ["aliked_lightglue", "sift_flann"]
    assert "error" in r["attempts"][0]


def test_pipeline_aliked_without_models_records_no_op_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_FEATURES_DIR", str(tmp_path / "empty"))
    r = analyze(str(CUR), str(PREV), "lure", method="aliked")
    assert not r["errors"], r["errors"]
    assert "op3_register" in r["ops_run"] and "op8_motion" in r["ops_run"]
    assert r["register"]["method"] == "sift_flann" and r["register"]["ok"]
    assert "not found" in r["register"]["fallback_reason"]
    assert r["motion"]["registration"]["method"] == "sift_flann"
    assert "not found" in r["motion"]["registration"]["fallback_reason"]
