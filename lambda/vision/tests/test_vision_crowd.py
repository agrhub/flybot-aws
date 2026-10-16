"""Op 6: sticky-card saturation, watershed split of touching flies, lure not applicable."""

import json
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision import synth
from sentinel.vision.crowd import estimate_crowd, measure_crowd
from sentinel.vision.quality import load_thresholds


def _card_with(flies, w=800, h=500, touching_pairs=0, rng=None):
    """Yellow card (no grid) with `flies` separate dark ellipses and `touching_pairs` pairs of touching ones."""
    rng = rng or np.random.default_rng(0)
    card = np.zeros((h, w, 3), np.uint8)
    card[:] = synth.CARD_BGR
    card = np.clip(card.astype(np.float32) + rng.normal(0, 4, card.shape), 0, 255).astype(np.uint8)
    pts = []
    while len(pts) < flies + touching_pairs:
        p = rng.uniform([40, 40], [w - 40, h - 40])
        if all(np.hypot(*(p - q)) > 60 for q in pts):
            pts.append(p)
    for i, (x, y) in enumerate(pts):
        if i < flies:
            cv2.ellipse(card, (int(x), int(y)), (14, 8), rng.uniform(0, 180), 0, 360, (25, 40, 60), -1)
        else:  # two flies side by side, bodies overlapping a little
            cv2.circle(card, (int(x - 9), int(y)), 11, (25, 40, 60), -1)
            cv2.circle(card, (int(x + 9), int(y)), 11, (25, 40, 60), -1)
    return card


def test_counts_separate_flies():
    m = measure_crowd(_card_with(25))
    assert m["component_count"] == 25 and m["blob_count"] == 25 and m["split_count"] == 0


def test_watershed_splits_touching_flies():
    m = measure_crowd(_card_with(20, touching_pairs=5, rng=np.random.default_rng(3)))
    assert m["component_count"] == 25
    assert m["blob_count"] == 30 and m["split_count"] == 5


def test_saturation_tracks_covered_area():
    lo = measure_crowd(_card_with(10))["saturation_pct"]
    hi = measure_crowd(_card_with(40, rng=np.random.default_rng(1)))["saturation_pct"]
    expected_hi = 40 * np.pi * 14 * 8 / (800 * 500) * 100  # ellipse area share
    assert hi > 3 * lo
    assert abs(hi - expected_hi) < 0.35 * expected_hi


def test_card_full_threshold_from_yaml():
    thr = load_thresholds()["crowd"]["card_full_pct"]
    card = np.zeros((500, 800, 3), np.uint8)
    card[:] = synth.CARD_BGR
    cv2.rectangle(card, (100, 100), (700, 400), (25, 40, 60), -1)  # nearly black card: dark everywhere
    for x in range(110, 700, 40):  # keep it blob-like for the adaptive threshold
        cv2.line(card, (x, 100), (x, 400), synth.CARD_BGR, 3)
    m = measure_crowd(card)
    assert m["card_full_threshold_pct"] == thr
    assert m["card_full"] == (m["saturation_pct"] >= thr)
    assert not measure_crowd(_card_with(5))["card_full"]


def test_estimate_on_rectified_card(tmp_path):
    rng = np.random.default_rng(2)
    bg = synth.noise_background(1280, 720, rng)
    card = _card_with(15, w=400, h=250, rng=rng)
    q = synth.random_quad(1280, 720, rng, coverage=(0.25, 0.35), max_tilt=0.1)
    r = estimate_crowd(synth.paste_card(bg, card, q), trap_kind="sticky", out_dir=tmp_path, name="c")
    json.dumps(r)
    assert r["status"] == "ok" and r["source"] == "card"
    assert 13 <= r["blob_count"] <= 17
    assert all(Path(p).is_file() for p in r["evidence"].values())


def test_lure_not_applicable():
    r = estimate_crowd(_card_with(10), trap_kind="lure")
    json.dumps(r)
    assert r["status"] == "not_applicable" and r["saturation_pct"] is None and r["blob_count"] is None
