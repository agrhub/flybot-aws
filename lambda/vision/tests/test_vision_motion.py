"""Op 8: large occluder / camera moved between two photos, and face blurring."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.vision import motion
from sentinel.vision.motion import blur_faces, detect_faces, detect_motion, yunet_path

FIX = Path(__file__).parent / "fixtures"


def _pair():
    return cv2.imread(str(FIX / "vision_be4089629_prev.jpg")), cv2.imread(str(FIX / "vision_be4089629_cur.jpg"))


def _hand(img, cx=0.45, cy=0.5, ax=0.2, ay=0.25):
    out = img.copy()
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int(cx * w), int(cy * h)), (int(ax * w), int(ay * h)), 20, 0, 360, 255, -1)
    cv2.line(m, (int(cx * w), int(cy * h)), (w - 1, int(cy * h)), 255, int(0.2 * h))  # arm from the right
    rng = np.random.default_rng(0)
    skin = np.clip(np.array([120, 150, 200]) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    out[m > 0] = cv2.GaussianBlur(skin, (9, 9), 0)[m > 0]
    return out, m


def test_real_consecutive_pair_is_quiet():
    a, b = _pair()
    r = detect_motion(a, b)
    json.dumps(r)
    assert not r["occluded_by_large_object"] and not r["camera_moved"] and not r["global_change"]
    assert r["registration"]["ok"]


def test_hand_over_the_trap_is_flagged(tmp_path):
    a, b = _pair()
    occ, m = _hand(b)
    r = detect_motion(a, occ, out_dir=tmp_path, name="h")
    assert r["occluded_by_large_object"] and r["occluder"]["in_frame"] == "cur"
    assert r["occluder"]["area_share"] >= 0.1
    x1, y1, x2, y2 = r["occluder"]["bbox"]
    ys, xs = np.nonzero(m)
    assert x1 <= xs.mean() <= x2 and y1 <= ys.mean() <= y2
    assert Path(r["evidence"]["overlay"]).is_file() and Path(r["evidence"]["fg_mask"]).is_file()


def test_object_gone_is_attributed_to_prev():
    a, b = _pair()
    occ, _ = _hand(a)
    r = detect_motion(occ, b)
    assert r["occluded_by_large_object"] and r["occluder"]["in_frame"] == "prev"


def test_small_flies_are_not_an_occluder():
    a, b = _pair()
    c = b.copy()
    rng = np.random.default_rng(1)
    for _ in range(15):
        x, y = int(rng.uniform(200, 560)), int(rng.uniform(60, 380))
        cv2.ellipse(c, (x, y), (6, 3), rng.uniform(0, 180), 0, 360, (30, 40, 50), -1)
    assert not detect_motion(a, c)["occluded_by_large_object"]


def test_lighting_change_is_not_an_occluder():
    a, b = _pair()
    h, w = b.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    spot = 1.0 + 0.6 * np.exp(-(((xx - w / 2) / (0.3 * w)) ** 2 + ((yy - h / 2) / (0.3 * h)) ** 2))  # LED hot spot
    lit = np.clip(b.astype(np.float32) * 0.7 * spot[..., None], 0, 255).astype(np.uint8)
    r = detect_motion(a, lit)
    assert not r["occluded_by_large_object"]


def test_camera_moved_by_shift():
    a, b = _pair()
    h, w = b.shape[:2]
    M = np.float32([[1, 0, 0.05 * w], [0, 1, 0.03 * h]])
    r = detect_motion(a, cv2.warpAffine(b, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    assert r["camera_moved"] and "shift" in r["moved_reasons"]
    assert not r["occluded_by_large_object"]


def test_camera_moved_when_registration_collapses():
    a, _ = _pair()
    reg = {"ok": False, "H": None, "inliers": 3, "matches": 40, "method": "sift_flann"}
    h, w = a.shape[:2]
    shifted = cv2.warpAffine(a, np.float32([[1, 0, 0.06 * w], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    sw, sh = motion._work(a)[0].shape[1], motion._work(a)[0].shape[0]
    r = detect_motion(cv2.resize(a, (sw, sh)), cv2.resize(shifted, (sw, sh)), registration=reg)
    assert r["registration"]["collapsed"] and r["camera_moved"]
    assert r["moved_reasons"] == ["registration_collapsed+phase_shift"]


def test_mog2_history_path():
    a, b = _pair()
    hist = [a, b, a, b]
    occ, _ = _hand(b)
    r = detect_motion(b, occ, history=hist)
    assert r["foreground"] == "mog2" and r["occluded_by_large_object"]


def test_blur_faces_fallbacks(monkeypatch):
    img = np.random.default_rng(0).integers(0, 255, (200, 300, 3)).astype(np.uint8)
    monkeypatch.setattr(motion, "yunet_path", lambda: None)
    out, info = blur_faces(img, fallback_boxes=[[50, 50, 150, 150]])
    assert info["method"] == "fallback_boxes"
    assert np.array_equal(out[:, 200:], img[:, 200:])  # outside the grown box untouched
    assert not np.array_equal(out[60:140, 60:140], img[60:140, 60:140])
    out2, info2 = blur_faces(img)
    assert info2["method"] == "fallback_whole_frame" and not np.array_equal(out2, img)


@pytest.mark.skipif(yunet_path() is None, reason="YuNet model not available")
def test_yunet_runs_on_a_trap_frame():
    _, b = _pair()
    faces = detect_faces(b)
    assert faces == []  # a trap interior has no face
    out, info = blur_faces(b)
    assert info["method"] == "yunet" and info["faces"] == 0 and np.array_equal(out, b)
