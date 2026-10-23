"""Tiled detection (sentinel.vision.tiling) and its use in Yolo26Detector.detect, without weights."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.vision import tiling
from sentinel.vision.box_rules import BoxRules
from sentinel.vision.detect_yolo26 import Yolo26Detector
from sentinel.vision.regions import Regions
from sentinel.vision.tiling import Tiling


def box(x, y, w, h, score=0.9):
    return {"xywh": [x, y, w, h], "score": score, "class_id": 0}


def test_windows_cover_the_frame_with_the_overlap():
    wins = tiling.windows(2304, 1296, 2, 0.12)
    assert len(wins) == 4
    assert {(w[0], w[1]) for w in wins} == {(0, 0), (1014, 0), (0, 570), (1014, 570)}
    assert all(x1 - x0 == 1290 and y1 - y0 == 726 for x0, y0, x1, y1 in wins)
    assert all(0 <= x0 < x1 <= 2304 and 0 <= y0 < y1 <= 1296 for x0, y0, x1, y1 in wins)
    assert 1290 - 1014 == pytest.approx(0.12 * 2304, abs=1)  # neighbours share 12 % of the frame
    assert tiling.windows(640, 480, 1, 0.12) == [(0, 0, 640, 480)]


def test_a_box_on_an_inner_window_edge_is_cut_but_not_one_on_the_image_border():
    W, H = 2304, 1296
    top_left, bottom_right = (0, 0, 1290, 726), (1014, 570, 2304, 1296)
    assert tiling.cut_by_inner_edge([1270, 100, 20, 20], top_left, W, H, 2)  # reaches the right (inner) edge
    assert tiling.cut_by_inner_edge([100, 710, 20, 16], top_left, W, H, 2)  # reaches the bottom (inner) edge
    assert not tiling.cut_by_inner_edge([0, 0, 20, 20], top_left, W, H, 2)  # image corner
    assert not tiling.cut_by_inner_edge([1270, 706, 20, 20], bottom_right, W, H, 2)  # image's bottom-right corner
    assert tiling.cut_by_inner_edge([0, 300, 20, 20], bottom_right, W, H, 2)  # left edge is inside the frame
    assert not tiling.cut_by_inner_edge([500, 300, 20, 20], top_left, W, H, 2)


def test_merge_shifts_drops_cut_boxes_and_dedups_the_overlap():
    W, H = 2304, 1296
    wins = tiling.windows(W, H, 2, 0.12)
    per = [
        [box(1100, 300, 30, 30), box(1275, 400, 15, 30)],  # a fly in the overlap; a fly cut by the right edge
        [box(86, 300, 30, 30, 0.8), box(247, 400, 40, 30)],  # the same two flies seen whole from the right window
        [], [],
    ]
    merged, cut = tiling.merge(per, wins, W, H, 0.45, [2, 2, 2, 2])
    assert cut == 1
    xs = sorted(d["xywh"][0] for d in merged)
    assert xs == [1100, 1261]  # one box per fly, in full-frame pixels
    full = [box(1500, 1000, 200, 180, 0.7)]  # a big fly only the whole-frame pass saw
    merged, _ = tiling.merge(per, wins, W, H, 0.45, [2, 2, 2, 2], full=full)
    assert len(merged) == 3


def test_decide_by_mode_trap_kind_and_box_size():
    small, big = [box(0, 0, 60, 60)] * 3, [box(0, 0, 120, 120)] * 3  # at 640 / 2304: 16.7 px and 33.3 px
    s = 640 / 2304
    assert tiling.decide(Tiling(), "sticky", small, s, 2304, 640) == (False, "off")
    assert tiling.decide(Tiling(mode="on"), "lure", None, s, 2304, 640)[0]
    assert tiling.decide(Tiling(mode="sticky"), "sticky", None, s, 2304, 640)[0]
    assert not tiling.decide(Tiling(mode="sticky"), "lure", big, s, 2304, 640)[0]
    assert tiling.decide(Tiling(mode="auto"), "lure", small, s, 2304, 640) == (True, "median_px=16.7")
    assert not tiling.decide(Tiling(mode="auto"), "sticky", big, s, 2304, 640)[0]
    assert tiling.decide(Tiling(mode="auto"), None, [], s, 2304, 640) == (True, "no_box_large_frame")
    assert tiling.decide(Tiling(mode="auto"), None, [], 640 / 768, 768, 640) == (False, "no_box_small_frame")


def test_config_from_env():
    assert Tiling.from_env({}) == Tiling()
    t = Tiling.from_env({"DETECTOR_TILES": "Auto", "DETECTOR_TILE_GRID": "3", "DETECTOR_TILE_OVERLAP": "0.2",
                         "DETECTOR_TILE_AUTO_PX": "30", "DETECTOR_TILE_KEEP_FULL": "off"})
    assert (t.mode, t.grid, t.overlap, t.auto_px, t.keep_full) == ("auto", 3, 0.2, 30.0, False)
    with pytest.raises(ValueError):
        Tiling.from_env({"DETECTOR_TILES": "always"})
    with pytest.raises(ValueError):
        Tiling(mode="on", overlap=1.0)


class FakeYolo26(Yolo26Detector):
    """Yolo26Detector without a network: `_pass` answers from a script, one entry per forward."""

    def __init__(self, tiles: Tiling, script: list[list[dict]], rules: BoxRules | None = None,
                 regions: Regions | None = None):
        self.size, self.iou, self.rules, self.tiling, self.head = 640, 0.45, rules, tiles, "end2end"
        self.regions = regions or Regions()
        self.script, self.calls = list(script), []

    def _pass(self, img):
        self.calls.append(img.shape[:2])
        return self.script.pop(0), 0, np.zeros((1, 300, 6), np.float32), 640 / max(img.shape[:2]), 10.0


def test_detector_tiles_sticky_photos_only_and_reports_it():
    img = np.zeros((1296, 2304, 3), np.uint8)
    win = [[box(100, 100, 40, 40)], [box(500, 300, 40, 40)], [], [box(600, 600, 40, 40)]]
    det = FakeYolo26(Tiling(mode="sticky"), [*win])
    res = det.detect(img, trap_kind="sticky")
    assert det.calls == [(726, 1290)] * 4  # four windows, no whole-frame pass
    assert res["count"] == 3 and res["forward_ms"] == 40.0
    assert {tuple(b["xywh"][:2]) for b in res["boxes"]} == {(100, 100), (1514, 300), (1614, 1170)}
    t = res["postprocess"]["tiles"]
    assert (t["used"], t["forwards"], t["cut_dropped"], t["reason"]) == (True, 4, 0, "trap_kind=sticky")

    det = FakeYolo26(Tiling(mode="sticky"), [[box(100, 100, 200, 200)]])
    res = det.detect(img, trap_kind="lure")
    assert det.calls == [(1296, 2304)] and res["count"] == 1
    assert res["postprocess"]["tiles"] == {"mode": "sticky", "used": False, "reason": "trap_kind=lure", "forwards": 1}


def test_detector_auto_keeps_large_flies_and_tiles_small_ones():
    img = np.zeros((1296, 2304, 3), np.uint8)
    large = [box(100 + 300 * i, 200, 200, 200) for i in range(3)]  # 55 px at 640: one forward
    det = FakeYolo26(Tiling(mode="auto"), [large])
    res = det.detect(img)
    assert len(det.calls) == 1 and res["count"] == 3 and res["postprocess"]["tiles"]["used"] is False

    small = [box(1000, 1000, 50, 50)]  # 14 px at 640: tiles, and the whole-frame box joins the merge
    det = FakeYolo26(Tiling(mode="auto"), [small, [box(10, 10, 50, 50)], [], [box(1000, 430, 50, 50)], []])
    res = det.detect(img)
    assert len(det.calls) == 5 and res["postprocess"]["tiles"]["forwards"] == 5
    assert sorted(tuple(b["xywh"][:2]) for b in res["boxes"]) == [(10, 10), (1000, 1000)]


def test_box_rules_run_on_full_frame_coordinates_after_the_merge():
    img = np.zeros((1296, 2304, 3), np.uint8)
    # 30 px is 0.017 of sqrt(W*H) of the frame (kept) but would be checked against the window if rules ran per window
    flies = [[box(100 + 60 * i, 100, 30, 30) for i in range(4)], [], [], [box(300, 300, 400, 400)]]
    det = FakeYolo26(Tiling(mode="on"), flies, rules=BoxRules())
    res = det.detect(img)
    assert res["count"] == 4
    assert res["postprocess"]["rejected"]["too_large"] == 1  # 400 px = 0.23 of the frame's side > 0.12
