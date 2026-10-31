"""YOLO26 post-processing on synthetic detections: NMS on both heads and the fruit-fly box rules."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.vision import detector
from sentinel.vision.box_rules import REASONS, BoxRules, apply_rules, nms
from sentinel.vision.detect_yolo26 import decode

W, H = 1280, 960  # sqrt(W * H) = 1108.5: a 40 px box has side 0.036


def box(x, y, w, h, score=0.8):
    return {"xywh": [float(x), float(y), float(w), float(h)], "score": score, "class_id": 0}


def flies(n=5, size=40, score=0.8):
    return [box(100 + 120 * i, 300, size, size + 8, score) for i in range(n)]


def test_end2end_duplicates_of_one_fly_are_suppressed():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [100, 100, 140, 150, 0.90, 0]
    out[0, 1] = [102, 101, 141, 152, 0.60, 0]  # same fly, IoU ~0.9
    out[0, 2] = [300, 300, 340, 350, 0.80, 0]
    dets = decode(out, conf=0.25, iou=0.45, scale=(1.0, 1.0), offset=(0, 0))
    assert [d["score"] for d in dets] == [pytest.approx(0.9), pytest.approx(0.8)]


def test_adjacent_flies_survive_nms():
    a, b = box(100, 100, 40, 48, 0.9), box(130, 100, 40, 48, 0.8)  # IoU 0.14: two touching flies
    assert len(nms([a, b], 0.45)) == 2
    assert len(nms([a, box(104, 102, 40, 48, 0.7)], 0.45)) == 1


def test_huge_box_is_rejected_and_flies_kept():
    gecko = box(0, 600, 420, 140, 0.50)  # side 0.219 of the photo, aspect 3.0
    kept, rej = apply_rules(flies() + [gecko], W, H, BoxRules())
    assert len(kept) == 5 and gecko not in kept
    assert rej["too_large"] == 1


def test_lens_sized_round_box_is_rejected():
    lens = box(560, 400, 180, 170, 0.66)  # side 0.158, like the lure-trap lens on the sample photos
    kept, rej = apply_rules([lens], W, H, BoxRules())
    assert kept == [] and rej["too_large"] == 1


def test_low_score_is_rejected():
    kept, rej = apply_rules([box(500, 500, 40, 48, 0.30), box(600, 500, 40, 48, 0.36)], W, H, BoxRules())
    assert [d["score"] for d in kept] == [0.36]
    assert rej["low_score"] == 1


def test_tiny_box_is_rejected():
    kept, rej = apply_rules([box(500, 500, 3, 3, 0.9), box(600, 500, 40, 48, 0.9)], W, H, BoxRules())
    assert len(kept) == 1 and rej["too_small"] == 1
    kept, rej = apply_rules([box(500, 500, 0, 30, 0.9)], W, H, BoxRules())
    assert kept == [] and rej["too_small"] == 1


def test_elongated_box_is_rejected():
    kept, rej = apply_rules([box(500, 500, 100, 20, 0.9)], W, H, BoxRules())  # aspect 5
    assert kept == [] and rej["aspect"] == 1
    kept, _ = apply_rules([box(500, 500, 70, 20, 0.9)], W, H, BoxRules())  # aspect 3.5
    assert len(kept) == 1


def test_box_much_larger_or_smaller_than_the_photos_flies():
    big = box(800, 600, 110, 110, 0.9)  # side 0.099 of the photo (< max_side) but 3.3x the median fly (33.8 px)
    speck = box(900, 100, 7, 7, 0.9)  # 0.0063 of the photo (> min_side) but 0.21x the median fly
    kept, rej = apply_rules(flies(size=30) + [big, speck], W, H, BoxRules())
    assert len(kept) == 5
    assert rej["large_for_photo"] == 1 and rej["small_for_photo"] == 1


def test_median_rules_need_enough_boxes():
    big = box(800, 600, 120, 120, 0.9)
    kept, rej = apply_rules(flies(1) + [big], W, H, BoxRules())  # 2 boxes < median_min_boxes
    assert len(kept) == 2 and sum(rej.values()) == 0


def test_rules_off_keeps_everything_and_reports_every_reason():
    dets = [box(0, 0, 900, 900, 0.1)]
    kept, rej = apply_rules(dets, W, H, None)
    assert kept == dets and set(rej) == set(REASONS) and sum(rej.values()) == 0


def test_rules_from_env(monkeypatch):
    assert BoxRules.from_env({}) == BoxRules()
    r = BoxRules.from_env({"DETECTOR_MIN_SCORE": "0.5", "DETECTOR_MAX_SIDE": "0.2", "DETECTOR_MEDIAN_MIN_BOXES": "5",
                           "DETECTOR_MAX_ASPECT": ""})
    assert (r.min_score, r.max_side, r.median_min_boxes, r.max_aspect) == (0.5, 0.2, 5, 4.0)
    assert BoxRules.from_env({"DETECTOR_BOX_RULES": "off"}) is None
    with pytest.raises(ValueError):
        BoxRules.from_env({"DETECTOR_MAX_SIDE": "big"})


def test_factory_gives_yolo26_the_env_rules(monkeypatch, tmp_path):
    seen = {}

    class Fake:
        def __init__(self, path, engine, conf, iou, rules, tiles, regions, views):
            seen.update(conf=conf, iou=iou, rules=rules, tiles=tiles, regions=regions, views=views)

    import sentinel.vision.detect_yolo26 as y

    monkeypatch.setattr(y, "Yolo26Detector", Fake)
    monkeypatch.setenv("DETECTOR_MAX_SIDE", "0.2")
    monkeypatch.delenv("DETECTOR_BOX_RULES", raising=False)
    monkeypatch.delenv("DETECTOR_TILES", raising=False)
    for name in ("DETECTOR_TTA", "DETECTOR_TTA_LURE", "DETECTOR_TTA_STICKY"):
        monkeypatch.delenv(name, raising=False)
    detector.build_detector("yolo26", tmp_path / "m.onnx")
    assert seen["iou"] == 0.45 and seen["rules"].max_side == 0.2 and seen["rules"].min_score == 0.35
    assert seen["tiles"].mode == "off"  # tiles stay off unless DETECTOR_TILES says otherwise
    assert seen["views"].views_for("lure") == ()  # so do the test-time views
    monkeypatch.setenv("DETECTOR_TILES", "sticky")
    monkeypatch.setenv("DETECTOR_TTA_LURE", "hflip")
    detector.build_detector("yolo26", tmp_path / "m.onnx")
    assert seen["tiles"].mode == "sticky" and seen["views"].views_for("lure") == ("hflip",)
