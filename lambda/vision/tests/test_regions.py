"""Overlay masks, the trap-surface ROI and the per-kind score floor (sentinel.vision.regions, box_rules), no weights."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.vision import regions
from sentinel.vision.box_rules import BoxRules
from sentinel.vision.regions import Regions, apply_regions


def box(cx, cy, s=20, score=0.9):
    return {"xywh": [cx - s / 2, cy - s / 2, s, s], "score": score, "class_id": 0}


def test_masks_parse():
    assert regions.parse_masks(None) == () and regions.parse_masks("off") == ()
    assert regions.parse_masks("osd") == regions.OSD
    assert regions.parse_masks("0,0,0.5,0.1;0.5,0.9,1,1") == ((0, 0, 0.5, 0.1), (0.5, 0.9, 1, 1))
    with pytest.raises(ValueError):
        regions.parse_masks("0.5,0,0.2,0.1")
    with pytest.raises(ValueError):
        Regions.from_env({"DETECTOR_ROI": "floor"})
    r = Regions.from_env({"DETECTOR_MASK": "osd", "DETECTOR_ROI": "Lure", "DETECTOR_ROI_RIM": "0.03"})
    assert (r.masks, r.roi, r.rim) == (regions.OSD, "lure", 0.03)
    assert r.wants_roi("lure") and not r.wants_roi("sticky") and Regions(roi="all").wants_roi("sticky")


def test_osd_mask_drops_logo_and_date_boxes_only():
    W, H = 2304, 1296
    dets = [box(150, 60), box(400, 1250), box(1200, 600), box(2200, 1250)]  # logo, stamp, card, bottom-right
    kept, removed = apply_regions(dets, W, H, Regions(masks=regions.OSD))
    assert [d["xywh"][0] + 10 for d in kept] == [1200, 2200]
    assert removed == {"osd_mask": 2, "outside_roi": 0, "roi_rim": 0}
    assert apply_regions(dets, W, H, Regions())[0] == dets  # off: nothing dropped


def test_roi_keeps_the_floor_and_drops_the_rim_for_lure_only():
    W, H = 768, 432
    floor = [[100, 20], [660, 20], [660, 420], [100, 420]]  # 560 x 400 px: sqrt(area) = 473
    dets = [box(380, 220), box(50, 220), box(105, 220), box(650, 30)]  # centre, wall, rim, corner rim
    kept, removed = apply_regions(dets, W, H, Regions(roi="lure"), floor, "lure")
    assert len(kept) == 3 and removed["outside_roi"] == 1
    kept, removed = apply_regions(dets, W, H, Regions(roi="lure", rim=0.03), floor, "lure")  # rim 14 px
    assert len(kept) == 1 and removed == {"osd_mask": 0, "outside_roi": 1, "roi_rim": 2}
    assert len(apply_regions(dets, W, H, Regions(roi="lure"), floor, "sticky")[0]) == 4  # not a lure photo
    assert len(apply_regions(dets, W, H, Regions(roi="lure"), None, "lure")[0]) == 4  # no floor found: whole frame


def test_roi_rect_pads_and_clips():
    assert regions.roi_rect([[100, 20], [660, 20], [660, 420], [100, 420]], 768, 432) == (84, 11, 676, 429)
    assert regions.roi_rect(None, 768, 432) is None


def test_score_floor_per_trap_kind():
    r = BoxRules.from_env({"DETECTOR_MIN_SCORE_LURE": "0.45"})
    assert r.for_kind("lure").min_score == 0.45
    assert r.for_kind("sticky").min_score == 0.35 and r.for_kind(None) is r
    assert BoxRules().for_kind("lure") == BoxRules()


def test_detector_applies_regions_before_the_rules():
    from sentinel.vision.tiling import Tiling
    from tests.test_tiling import FakeYolo26

    img = np.zeros((432, 768, 3), np.uint8)
    floor = [[100, 20], [660, 20], [660, 420], [100, 420]]
    flies = [box(300 + 40 * i, 200, score=0.4) for i in range(3)]
    det = FakeYolo26(Tiling(), [[*flies, box(40, 200)]], rules=BoxRules(min_score_lure=0.45),
                     regions=Regions(roi="lure"))
    res = det.detect(img, trap_kind="lure", roi=floor)
    assert res["count"] == 0  # the wall box left the ROI, the three 0.40 flies are under the lure floor 0.45
    assert res["postprocess"]["rejected"]["outside_roi"] == 1 and res["postprocess"]["rejected"]["low_score"] == 3
    assert res["postprocess"]["regions"]["roi_used"] is True and res["postprocess"]["rules"]["min_score"] == 0.45


class RecordingDetector:
    """A YOLO26-like detector that records what the pipeline hands it (trap kind, surface polygon)."""

    name, engine_used, onnx_path = "fake-yolo26", "fake", "fake.onnx"

    def __init__(self, regions: Regions):
        self.tiling = object()  # like Yolo26Detector: the pipeline then passes trap_kind and roi
        self.regions, self.calls = regions, []

    def detect(self, img, trap_kind=None, roi=None):
        self.calls.append((trap_kind, roi))
        return {"count": 0, "boxes": [], "forward_ms": 1.0, "postprocess": None}


@pytest.mark.parametrize("roi_mode,expect_polygon", [("all", True), ("off", False)])
def test_pipeline_runs_op1_first_and_hands_the_floor_to_the_detector(monkeypatch, roi_mode, expect_polygon):
    from pathlib import Path

    from sentinel.vision import detector, pipeline

    fix = Path(__file__).parent / "fixtures"
    det = RecordingDetector(Regions(roi=roi_mode))
    monkeypatch.setenv("DETECTOR", "fake")
    detector.set_detector(det, "fake")
    try:
        res = pipeline.analyze_photo(str(fix / "vision_be4089629_cur.jpg"), str(fix / "vision_be4089629_prev.jpg"),
                                     trap_kind="lure")
    finally:
        detector._DETECTORS.clear()
        detector.DEFAULTS.pop("fake", None)
    assert [kind for kind, _ in det.calls] == ["lure", "lure"]  # this photo and the previous one
    assert all((roi is not None) == expect_polygon for _, roi in det.calls)
    assert res["card"]["found"] is True and "op1_card" in res["timings_ms"]
