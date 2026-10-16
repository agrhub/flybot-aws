import json
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision import synth
from sentinel.vision.card import find_card
from sentinel.vision.quality import Baseline, assess_quality, load_thresholds

FIX = Path(__file__).parent / "fixtures"


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    bg = synth.noise_background(1280, 720, rng)
    card, _ = synth.make_card(rng=rng, n_flies=30)
    return synth.paste_card(bg, card, synth.random_quad(1280, 720, rng, coverage=(0.25, 0.4)))


def test_good_photo_ok():
    img = _scene()
    r = assess_quality(img, find=True)
    assert r["ok"], r["issues"]
    json.dumps(r)
    assert "evidence" in r


def test_dark_overexposed():
    img = _scene()
    assert "dark" in assess_quality((img * 0.08).astype(np.uint8))["issues"]
    assert "dark" not in assess_quality((img * 0.08).astype(np.uint8), hour=22)["issues"]  # night: not a defect
    assert "overexposed" in assess_quality(np.clip(img.astype(int) + 200, 0, 255).astype(np.uint8))["issues"]


def test_blurry_against_baseline():
    img = _scene(1)
    base = Baseline(min_count=1)
    r = assess_quality(img)
    base.update(r["scores"], r["issues"])
    blur = cv2.GaussianBlur(img, (0, 0), 6)
    assert "blurry" in assess_quality(blur, baseline=base.get())["issues"]


def test_low_contrast_against_baseline():
    img = _scene(2)
    base = Baseline(min_count=1)
    r = assess_quality(img)
    base.update(r["scores"], r["issues"])
    hazy = cv2.addWeighted(img, 0.3, np.full_like(img, 150), 0.7, 0)
    assert "low_contrast" in assess_quality(hazy, baseline=base.get())["issues"]


def test_glare_and_ir():
    img = _scene(3)
    card = find_card(img)
    glare = img.copy()
    c = np.mean(card["quad"], axis=0).astype(int)
    cv2.circle(glare, tuple(int(v) for v in c), 60, (255, 255, 255), -1)
    assert "glare" in assess_quality(glare, card=find_card(glare))["issues"]
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert "ir" in assess_quality(gray)["issues"]


def test_no_card_and_partial():
    bg = synth.noise_background(1280, 720, np.random.default_rng(4))
    assert "no_card" in assess_quality(bg, find=True)["issues"]
    card, _ = synth.make_card(rng=np.random.default_rng(4))
    q = np.float32([[-150, 200], [300, 180], [320, 500], [-160, 520]])
    assert "partial_card" in assess_quality(synth.paste_card(bg, card, q), find=True)["issues"]


def test_thresholds_yaml_loads():
    t = load_thresholds()
    assert t["low_contrast"]["ratio_contrast"] > 0 and t["sample_width"] == 320


def test_real_frame_scores(tmp_path):
    img = cv2.imread(str(FIX / "vision_be4089629_prev.jpg"))
    r = assess_quality(img, out_dir=tmp_path, name="real")
    assert r["scores"]["brightness"] > 35
    assert Path(r["evidence"]["glare_mask"]).is_file()
