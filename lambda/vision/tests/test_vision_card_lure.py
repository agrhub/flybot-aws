"""Op 1 trap profiles: `lure` (white tray floor) and the unchanged `sticky` default."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.vision import synth
from sentinel.vision.card import find_card, find_lure_floor

FIX = Path(__file__).parent / "fixtures"
VOLATILE = ("time_ms", "evidence")


def _sticky_scene(seed):
    rng = np.random.default_rng(seed)
    bg = synth.noise_background(1280, 720, rng)
    card, _ = synth.make_card(rng=rng)
    q = synth.random_quad(1280, 720, rng, coverage=(0.1, 0.5), max_tilt=0.25)
    return synth.paste_card(bg, card, q)


def _tray(W=768, H=432, seed=0, night=False):
    """Synthetic lure-trap view: smooth white floor between two ribbed grey walls, tan stains, bait port."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (205, 210, 212)
    for x0 in (0, W - int(0.16 * W)):  # ribbed walls
        for x in range(x0, x0 + int(0.16 * W), 6):
            cv2.line(img, (x, 0), (x, H - 1), (95, 100, 105), 2)
            cv2.line(img, (x + 3, 0), (x + 3, H - 1), (170, 172, 175), 2)
    for _ in range(6):  # tan bait stains on the floor
        c = (int(rng.uniform(0.25, 0.75) * W), int(rng.uniform(0.1, 0.9) * H))
        cv2.ellipse(img, c, (int(rng.uniform(15, 50)), int(rng.uniform(10, 30))), rng.uniform(0, 180), 0, 360,
                    (150, 185, 210), -1)
    cv2.circle(img, (W // 2, int(0.6 * H)), 22, (225, 225, 225), -1)  # bait port nut
    cv2.circle(img, (W // 2, int(0.6 * H)), 10, (40, 60, 90), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    if night:  # IR: grey, bright centre falling off to the corners
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        fall = 1.0 - 0.45 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
        img = cv2.cvtColor(np.clip(g * fall, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img


@pytest.mark.parametrize("seed", [0, 7])
def test_sticky_default_unchanged(seed):
    img = _sticky_scene(seed)
    a, b = find_card(img), find_card(img, trap_kind="sticky")
    assert {k: v for k, v in a.items() if k not in VOLATILE} == {k: v for k, v in b.items() if k not in VOLATILE}
    # values recorded before the trap profiles were added (2026-10-15)
    expected = {0: ([[229.24, 96.0], [892.29, 294.9], [786.01, 702.95], [127.17, 411.45]], 0.2872, 0.9985),
                7: ([[367.07, 30.91], [1116.98, 149.23], [1047.78, 695.56], [320.64, 505.98]], 0.4189, 0.9982)}[seed]
    assert a["quad"] == expected[0] and a["coverage"] == expected[1] and a["fill"] == expected[2]
    assert a["trap_kind"] == "sticky" and a["roi_source"] == "card" and a["polygon"] == a["quad"]


def test_sticky_no_card_keeps_empty_roi():
    r = find_card(synth.noise_background(1280, 720, np.random.default_rng(2)))
    assert r["found"] is False and r["roi"] == [] and r["roi_source"] == "none"


def test_unknown_trap_kind_rejected():
    with pytest.raises(ValueError):
        find_card(_tray(), trap_kind="bucket")


@pytest.mark.parametrize("night", [False, True])
def test_lure_floor_between_walls(night, tmp_path):
    img = _tray(night=night)
    r = find_card(img, trap_kind="lure", out_dir=tmp_path, name="t")
    json.dumps(r)
    assert r["found"] and r["trap_kind"] == "lure" and r["roi_source"] in ("floor_quad", "floor_polygon")
    xs = [p[0] for p in r["roi"][0]]
    # floor spans x 0.16-0.84 of the frame; the walls stay out of the ROI
    assert 0.12 <= min(xs) <= 0.22 and 0.78 <= max(xs) <= 0.88
    assert 0.5 <= r["coverage"] <= 0.8
    assert Path(r["evidence"]["overlay"]).is_file()
    if r["roi_source"] == "floor_quad":
        assert r["warped_size"] and max(r["warped_size"]) == 800 and Path(r["warped_path"]).is_file()


def test_lure_rectangle_floor_gets_quad():
    r = find_card(_tray(seed=3), trap_kind="lure")
    assert r["roi_source"] == "floor_quad" and r["quad"] is not None and len(r["quad_norm"]) == 4
    assert r["floor_checks"]["quad_min_angle"] >= 78


def test_lure_falls_back_to_whole_frame():
    dark = np.full((432, 768, 3), 20, np.uint8)
    r = find_lure_floor(dark)
    assert r["found"] is False and r["roi_source"] == "whole_frame"
    assert r["roi"] == [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]] and r["quad"] is None


def test_lure_rounded_floor_polygon_only():
    img = np.full((432, 768, 3), 60, np.uint8)  # dark surround, bright round tray: no plausible 4-corner fit
    cv2.circle(img, (384, 216), 190, (210, 212, 214), -1)
    r = find_card(img, trap_kind="lure")
    assert r["found"] and r["roi_source"] == "floor_polygon" and r["quad"] is None and r["warped_path"] is None
    assert len(r["roi"][0]) >= 6


def test_real_lure_frames():
    for f in ("vision_be4089629_prev.jpg", "vision_be4089629_cur.jpg"):
        r = find_card(cv2.imread(str(FIX / f)), trap_kind="lure")
        assert r["found"] and r["roi_source"] == "floor_quad"
        assert 0.6 <= r["coverage"] <= 0.75
        xs = [p[0] for p in r["quad_norm"]]
        assert 0.1 <= min(xs) <= 0.2 and 0.8 <= max(xs) <= 0.9
