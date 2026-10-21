"""Op 2 on lure trays: regression for the false `blurry` flags of the live AWS run (2026-10-15) and for the glare
limit of the tray floor (the advisor's review, 2026-10-21; tests at the end).

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


# Glare on a lure floor (advisor review, eval/RESULTS.md section 6). Fixtures: the review sheet's AWS-12 (3.4 % of the
# floor specular, judged countable: the recapture was rejected) and AWS-09 (18 %, judged readable but unsure).
GLARE_COUNTABLE = "quality_be4111159_glare_3pct.jpg"
GLARE_STRONG = "quality_be4111159_glare_18pct.jpg"


def _reflection(img, area_share):
    """eval/failure_injection.py glare(): a soft white ellipse covering ~area_share of the frame."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(0)
    cx, cy = rng.uniform(0.35, 0.65) * w, rng.uniform(0.3, 0.6) * h
    ax = np.sqrt(area_share * w * h / np.pi * (w / h) ** 0.3)
    ay = area_share * w * h / (np.pi * ax)
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.uniform(0, 40)), 0, 360, 1.0, -1)
    m = np.clip(cv2.GaussianBlur(m, (0, 0), min(h, w) * 0.02) * 1.6, 0, 1)[..., None]
    return np.clip(img * (1 - m) + 255 * m, 0, 255).astype(np.uint8)


def test_small_glare_on_lure_floor_is_countable():
    r = assess_quality(cv2.imread(str(FIX / GLARE_COUNTABLE)), find=True, trap_kind="lure", hour=14)
    s, t = r["scores"], load_thresholds()
    assert s["glare_region"] == "floor"
    assert t["glare"]["ratio_max"] < s["glare_ratio"] <= t["glare"]["floor_ratio_max"]
    assert "glare" not in r["issues"]


def test_large_glare_on_lure_floor_still_flags():
    r = assess_quality(cv2.imread(str(FIX / GLARE_STRONG)), find=True, trap_kind="lure", hour=13)
    assert r["scores"]["glare_region"] == "floor" and "glare" in r["issues"]
    img = _reflection(cv2.imread(str(FIX / CLEAN)), 0.06)  # the failure-injection reflection
    r = assess_quality(img, find=True, trap_kind="lure", hour=13)
    assert r["scores"]["glare_region"] == "floor" and "glare" in r["issues"]


def test_card_and_frame_glare_keep_the_strict_limit():
    s = _scores(cv2.imread(str(FIX / GLARE_COUNTABLE)))
    for region in ("card", "frame", None):
        scores = {**s, "glare_region": region} if region else {k: v for k, v in s.items() if k != "glare_region"}
        assert "glare" in assess(scores, None, None, 14)
