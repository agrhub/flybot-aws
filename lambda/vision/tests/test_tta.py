"""Test-time views (sentinel.vision.tta) and their use in Yolo26Detector, without weights."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.vision import tta
from sentinel.vision.detect_yolo26 import Yolo26Detector
from sentinel.vision.regions import Regions
from sentinel.vision.tiling import Tiling
from sentinel.vision.tta import Tta


def dark_box(img: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(img[..., 0] < 128)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())]


@pytest.mark.parametrize("name", tta.VIEWS)
def test_a_box_found_in_a_view_maps_back_to_the_photo(name):
    img = np.full((432, 768, 3), 255, np.uint8)
    img[100:130, 600:650] = 0  # x 600..650, y 100..130 on a landscape frame
    seen = dark_box(tta.view(img, name))
    back = tta.unmap([{"xywh": seen, "score": 0.5, "class_id": 0}], name, 768, 432)
    assert back[0]["xywh"] == [600.0, 100.0, 50.0, 30.0]
    assert back[0]["score"] == 0.5


def test_settings_from_the_environment():
    assert Tta.from_env({}).views_for("lure") == () and Tta.from_env({"DETECTOR_TTA": "off"}).views_for(None) == ()
    t = Tta.from_env({"DETECTOR_TTA_LURE": "hflip, vflip", "DETECTOR_TTA_STICKY": "hflip"})
    assert (t.views_for("lure"), t.views_for("sticky"), t.views_for(None)) == (("hflip", "vflip"), ("hflip",), ())
    t = Tta.from_env({"DETECTOR_TTA": "hflip", "DETECTOR_TTA_STICKY": "off", "DETECTOR_TTA_LURE": ""})
    assert (t.views_for("lure"), t.views_for("sticky"), t.views_for(None)) == (("hflip",), (), ("hflip",))
    with pytest.raises(ValueError):
        Tta.from_env({"DETECTOR_TTA": "rot45"})
    with pytest.raises(ValueError):
        Tta.from_env({"DETECTOR_TTA_LURE": "hflip,bowl"})


def box(x, y, w, h, score=0.9):
    return {"xywh": [x, y, w, h], "score": score, "class_id": 0}


class ScriptedYolo26(Yolo26Detector):
    """Yolo26Detector without a network: `forward` returns scripted end-to-end rows, one entry per forward."""

    def __init__(self, views: Tta, script: list[list[list[float]]]):
        self.size, self.iou, self.conf, self.rules, self.head, self.resize = 640, 0.45, 0.25, None, "end2end", "letterbox"
        self.tiling, self.regions, self.tta, self.meta, self.last_ms = Tiling(), Regions(), views, {}, 0.0
        self.script, self.shapes = list(script), []

    def forward(self, img):
        self.shapes.append(img.shape[:2])
        rows = np.array(self.script.pop(0) or np.zeros((0, 6)), np.float32).reshape(1, -1, 6)
        return rows, (1.0, 1.0), (0, 0), 5.0


def test_detect_joins_the_views_and_counts_a_fly_seen_twice_once():
    img = np.zeros((400, 600, 3), np.uint8)
    own = [[10, 10, 40, 30, 0.6, 0]]  # a fly at x 10..40 the photo itself shows
    flipped = [[560, 10, 590, 30, 0.8, 0], [100, 200, 130, 220, 0.5, 0]]  # the same fly mirrored + one only hflip sees
    det = ScriptedYolo26(Tta(views=("hflip",)), [own, flipped])
    res = det.detect(img, trap_kind="lure")
    assert det.shapes == [(400, 600), (400, 600)]
    assert sorted((b["xywh"], b["score"]) for b in res["boxes"]) == [([10.0, 10.0, 30.0, 20.0], 0.8),
                                                                     ([470.0, 200.0, 30.0, 20.0], 0.5)]
    assert res["postprocess"]["tiles"]["forwards"] == 2 and res["forward_ms"] == 10.0
    assert res["postprocess"]["tta"] == {"views": ["hflip"]}


def test_views_run_only_on_their_trap_kind():
    det = ScriptedYolo26(Tta(lure=("hflip",)), [[[10, 10, 40, 30, 0.6, 0]]])
    res = det.detect(np.zeros((400, 600, 3), np.uint8), trap_kind="sticky")
    assert len(det.shapes) == 1 and res["count"] == 1 and res["postprocess"]["tta"]["views"] == []
