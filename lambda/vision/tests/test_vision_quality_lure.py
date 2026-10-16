"""Op 2 on lure trays: regression for the false `blurry` flags of the live AWS run (2026-10-15).

On BE4111159, 15 of 37 live photos (180 of 485 in the export) were flagged blurry although they were sharper
than the unflagged ones. Cause: the absolute sticky-card rule `card_sharpness_min` was applied to the
rectified lure floor, a smooth white tray upscaled from 768x432, so it measured floor texture, not focus.
Fixtures: real BE4111159 frames (snapshots 6acf4b9c..9a87 and 6ad01785..6208 were flagged live; 6acf60b5..9a91
was not).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.vision import synth
from sentinel.vision.card import find_card
from sentinel.vision.quality import Baseline, assess, assess_quality, load_thresholds, measure

FIX = Path(__file__).parent / "fixtures"
FLAGGED_LIVE = ("quality_be4111159_day_a.jpg", "quality_be4111159_day_b.jpg")
CLEAN = "quality_be4111159_clean.jpg"


def _scores(img):
    return measure(img, find_card(img, trap_kind="lure"))[0]


def _baseline():
    b = Baseline(20, 1)
    for f in (CLEAN, *FLAGGED_LIVE):
        s = _scores(cv2.imread(str(FIX / f)))
        b.update(s, [])
    return b.get()


@pytest.mark.parametrize("name", FLAGGED_LIVE)
def test_sharp_lure_frames_are_not_blurry(name):
    img = cv2.imread(str(FIX / name))
    r = assess_quality(img, find=True, trap_kind="lure", hour=16)
    s = r["scores"]
    assert r["scores"]["card_found"] and s["card_sharpness"] is None  # sticky-card rule not used on a floor
    assert s["floor_sharpness"] is not None and s["sharpness"] > 300
    assert "blurry" not in r["issues"]
    assert "blurry" not in assess(s, None, _baseline(), 16)  # with the camera baseline too (agent rejudge)


def test_blurred_lure_frame_is_still_blurry():
    img = cv2.imread(str(FIX / FLAGGED_LIVE[0]))
    blurred = cv2.GaussianBlur(img, (0, 0), 2.5)
    s = _scores(blurred)
    assert "blurry" in assess(s, None, _baseline(), 16)


def test_sticky_card_keeps_card_sharpness_rule():
    rng = np.random.default_rng(3)
    bg = synth.noise_background(1280, 720, rng)
    card, _ = synth.make_card(rng=rng)
    img = synth.paste_card(bg, card, synth.random_quad(1280, 720, rng, coverage=(0.2, 0.3), max_tilt=0.05))
    s = measure(img, find_card(img))[0]
    assert s["card_sharpness"] is not None and s["floor_sharpness"] is None
    t = load_thresholds()
    blurred = measure(cv2.GaussianBlur(img, (0, 0), 6), find_card(img))[0]
    assert blurred["card_sharpness"] < s["card_sharpness"]
    assert ("blurry" in assess(blurred, t)) == (blurred["card_sharpness"] < t["blurry"]["card_sharpness_min"]
                                                or blurred["sharpness"] < t["blurry"]["sharpness_min"])
