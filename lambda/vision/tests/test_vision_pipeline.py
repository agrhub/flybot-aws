"""Per-photo pipeline (ops 1, 2, 3, 4, 6, 8) and the trap_kind meaning of op 4."""

import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.vision import pipeline, synth
from sentinel.vision.diff import diff_detections
from sentinel.vision.pipeline import analyze, analyze_photo


def _onnx_available() -> bool:
    from sentinel.vision.detector import DetectorUnavailable, resolve_onnx

    try:
        resolve_onnx()
    except DetectorUnavailable:
        return False
    return True

FIX = Path(__file__).parent / "fixtures"
PREV, CUR = FIX / "vision_be4089629_prev.jpg", FIX / "vision_be4089629_cur.jpg"
TOP_KEYS = {"schema", "image", "prev_image", "trap_kind", "width", "height", "method", "card", "card_prev", "quality",
            "register", "diff", "crowd", "motion", "boxes", "count", "privacy", "ops_run", "timings_ms", "errors",
            "evidence"}


def test_signature_is_stable():
    params = list(inspect.signature(analyze).parameters)
    assert params[:5] == ["image_path", "prev_image_path", "trap_kind", "boxes_cur", "boxes_prev"]
    assert inspect.signature(analyze).parameters["method"].default == "sift"


def test_lure_pair_end_to_end(tmp_path):
    r = analyze(str(CUR), str(PREV), "lure", out_dir=tmp_path)
    json.dumps(r)
    assert set(r) >= TOP_KEYS and r["schema"] == "sentinel.vision.pipeline/1" and not r["errors"]
    assert r["ops_run"] == ["op1_card", "op2_quality", "op3_register", "op4_diff", "op6_crowd", "op8_motion"]
    assert r["method"] == "sift_flann" and r["register"]["method"] == "sift_flann" and r["register"]["ok"]
    assert r["card"]["trap_kind"] == "lure" and r["card"]["found"]
    assert r["diff"]["status"] == "skipped" and r["count"] is None
    assert r["crowd"]["status"] == "not_applicable"
    assert r["motion"]["occluded_by_large_object"] is False and r["privacy"]["person_suspected"] is False
    assert all(k in r["timings_ms"] for k in r["ops_run"]) and r["timings_ms"]["total"] > 0
    assert all(Path(p).is_file() for p in r["evidence"].values())


def test_lure_boxes_mean_activity():
    boxes_prev = [[300, 200, 320, 215], [400, 120, 418, 136]]
    boxes_cur = [[300, 200, 320, 215], [500, 300, 518, 316]]
    r = analyze(str(CUR), str(PREV), "lure", boxes_cur, boxes_prev)
    d = r["diff"]
    assert d["status"] == "ok" and d["trap_kind"] == "lure" and d["meaning"] == "activity"
    assert d["persisted_count"] == 1 and d["new_count"] == 1 and d["disappeared_count"] == 1
    assert d["card_change_suspected"] is False and r["count"] == 2


def test_card_change_only_for_sticky():
    prev = [[i * 30.0, 10.0, i * 30.0 + 12, 22.0] for i in range(10)]
    d_sticky = diff_detections(prev, [], np.eye(3), trap_kind="sticky")
    d_lure = diff_detections(prev, [], np.eye(3), trap_kind="lure")
    assert d_sticky["card_change_suspected"] and d_sticky["meaning"] == "catches"
    assert not d_lure["card_change_suspected"] and d_lure["card_change_reasons"] == []
    assert d_lure["disappeared_count"] == 10


def test_sticky_single_photo(tmp_path):
    rng = np.random.default_rng(4)
    bg = synth.noise_background(1280, 720, rng)
    card, _ = synth.make_card(rng=rng, n_flies=12)
    img = synth.paste_card(bg, card, synth.random_quad(1280, 720, rng, coverage=(0.2, 0.3), max_tilt=0.05))
    p = tmp_path / "card.jpg"
    cv2.imwrite(str(p), img)
    r = analyze(str(p), None, "sticky", out_dir=tmp_path / "ev")
    json.dumps(r)
    assert r["ops_run"] == ["op1_card", "op2_quality", "op6_crowd"]
    assert r["card"]["found"] and r["register"] is None and r["motion"] is None and r["diff"] is None
    assert r["crowd"]["status"] == "ok" and r["crowd"]["source"] == "card" and r["crowd"]["blob_count"] >= 8


def test_occluder_triggers_privacy_pass(tmp_path):
    a, b = cv2.imread(str(PREV)), cv2.imread(str(CUR))
    h, w = b.shape[:2]
    occ = b.copy()
    cv2.ellipse(occ, (w // 2, h // 2), (w // 5, h // 4), 0, 0, 360, (120, 150, 200), -1)
    r = analyze(occ, a, "lure", out_dir=tmp_path)
    assert r["motion"]["occluded_by_large_object"] and r["privacy"]["person_suspected"]
    assert r["privacy"]["faces"] and "privacy_blur" in r["timings_ms"]


def test_bad_arguments():
    with pytest.raises(ValueError):
        analyze(str(CUR), None, "bucket")
    with pytest.raises(ValueError):
        analyze(str(CUR), None, "lure", method="orb")
    with pytest.raises(FileNotFoundError):
        analyze(str(FIX / "missing.jpg"))


def test_analyze_photo_adapter():
    r = analyze_photo(str(CUR), previous=str(PREV), trap_kind="lure", detect=False)
    assert r["trap_kind"] == "lure" and r["motion"] is not None and "saturation" in r["crowd"]
    assert r["image"] == str(CUR) and r["detect"] is None


@pytest.mark.skipif(not _onnx_available(), reason="detector ONNX not on disk (make models)")
def test_analyze_photo_runs_detector():
    r = analyze_photo(str(CUR), previous=str(PREV), trap_kind="lure")
    assert r["detect"]["engine_used"] and r["count"] == r["detect"]["count"] and "op5_detect" in r["timings_ms"]
    assert r["diff"]["status"] == "ok" and r["diff"]["meaning"] == "activity"


def test_cli(capsys):
    assert pipeline.main(["--image", str(CUR), "--prev", str(PREV), "--trap-kind", "lure"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trap_kind"] == "lure" and out["register"]["ok"]
