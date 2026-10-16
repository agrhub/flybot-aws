import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.vision import synth
from sentinel.vision.card import card_quad_to_roi, find_card, order_corners, warp_card

FIX = Path(__file__).parent / "fixtures"


def _scene(seed=1, coverage=(0.15, 0.4), tilt=0.1, W=1280, H=720):
    rng = np.random.default_rng(seed)
    bg = synth.noise_background(W, H, rng)
    card, _ = synth.make_card(rng=rng)
    q = synth.random_quad(W, H, rng, coverage=coverage, max_tilt=tilt)
    return synth.paste_card(bg, card, q), q


@pytest.mark.parametrize("seed", range(6))
def test_finds_card_corners(seed):
    img, q = _scene(seed, tilt=0.25)
    r = find_card(img)
    assert r["found"] and r["n_cards"] == 1 and not r["partial"]
    err = np.linalg.norm(np.asarray(r["quad"]) - q, axis=1).max()
    assert err < 8.0
    json.dumps(r)  # serializable
    assert "evidence" in r


def test_no_card_on_plain_background():
    bg = synth.noise_background(1280, 720, np.random.default_rng(2))
    r = find_card(bg)
    assert r["found"] is False and r["quad"] is None and r["roi"] == []


def test_round_yellow_disc_is_not_a_card():
    img = np.full((720, 1280, 3), 90, np.uint8)
    cv2.circle(img, (640, 360), 200, synth.CARD_BGR, -1)
    assert find_card(img)["found"] is False


def test_partial_card_flagged():
    img, q = _scene(3)
    shift = np.float32([q[:, 0].min() + 60, 0])
    bg = synth.noise_background(1280, 720, np.random.default_rng(3))
    card, _ = synth.make_card(rng=np.random.default_rng(3))
    img = synth.paste_card(bg, card, q - shift)
    r = find_card(img)
    assert r["found"] and r["partial"]


def test_two_cards():
    rng = np.random.default_rng(4)
    bg = synth.noise_background(1280, 720, rng)
    card, _ = synth.make_card(rng=rng)
    q1 = synth.random_quad(640, 720, rng, coverage=(0.2, 0.3), max_tilt=0.05)
    q2 = synth.random_quad(640, 720, rng, coverage=(0.15, 0.2), max_tilt=0.05) + np.float32([640, 0])
    r = find_card(synth.paste_card(synth.paste_card(bg, card, q1), card, q2))
    assert r["n_cards"] == 2
    assert r["cards"][0]["coverage"] >= r["cards"][1]["coverage"]


def test_warp_keeps_aspect_and_evidence(tmp_path):
    img, _ = _scene(5)
    r = find_card(img, out_dir=tmp_path, name="t")
    w, h = r["warped_size"]
    assert max(w, h) == 800
    assert Path(r["warped_path"]).is_file() and Path(r["evidence"]["overlay"]).is_file()
    assert warp_card(img, r["quad"]).shape[:2] == (h, w)


def test_card_quad_to_roi_format():
    roi = card_quad_to_roi([[0, 0], [1280, 0], [1300, 720], [-5, 720]], 1280, 720)
    assert roi == [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
    roi = card_quad_to_roi([[128, 72], [640, 72], [640, 360], [128, 360]], 1280, 720)
    assert roi == [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]]


def test_order_corners_rotated():
    pts = np.float32([[100, 0], [200, 100], [100, 200], [0, 100]])  # diamond, shuffled below
    o = order_corners(pts[[2, 0, 3, 1]])
    assert o.shape == (4, 2)
    # clockwise: signed area positive in image coordinates
    x, y = o[:, 0], o[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def test_real_export_frame_has_no_yellow_card():
    img = cv2.imread(str(FIX / "vision_be4089629_cur.jpg"))
    r = find_card(img)
    assert r["found"] is False
    assert find_card(img, profile="white")["found"] is True  # the white trap floor
