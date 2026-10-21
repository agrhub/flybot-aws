"""Op 2: photo quality gate v2 (blur, exposure, glare, dirt/droplets, IR night mode, card visibility).

    uv run python -m sentinel.vision.quality IMAGE [--card] [--out DIR]

`assess_quality(img, card=find_card(img))` returns a JSON-serializable dict:

    ok, issues (subset of ISSUES, most serious first), scores {...}, baseline_used, time_ms, evidence

Thresholds live in quality_thresholds.yaml. Scene-dependent rules (blur, haze, dirt) are judged
against the camera's own recent good photos (`Baseline`), as FlyBot production does
(ezviz-manager server/src/services/imageQuality.js); absolute floors apply without a baseline.
Issues map to camera actions (dark/overexposed/low_contrast/glare -> set_image,
blurry -> capture again, dirty -> remind to clean, ir -> postpone, no_card/partial_card -> ROI / visit).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

from sentinel.vision.card import find_card, roi_polygon_px, warp_card
from sentinel.vision.evidence import save_jpeg

ISSUES = ["no_card", "partial_card", "dark", "overexposed", "ir", "blurry", "low_contrast", "glare", "dirty"]
THRESHOLDS_PATH = Path(__file__).with_name("quality_thresholds.yaml")


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_thresholds(path: str | Path | None = None) -> dict:
    return _load(str(path or THRESHOLDS_PATH))


def _gray_sample(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return img


def _lapvar(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var())


def _flat_share(gray: np.ndarray, window: int, flat_ratio: float) -> tuple[float, np.ndarray]:
    """Share of pixels whose local std is far below the image's median local std, and that mask."""
    g = gray.astype(np.float32)
    mu = cv2.boxFilter(g, -1, (window, window))
    mu2 = cv2.boxFilter(g * g, -1, (window, window))
    std = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    med = float(np.median(std))
    if med <= 1e-6:
        return 1.0, np.full(gray.shape, 255, np.uint8)
    flat = (std < flat_ratio * med).astype(np.uint8) * 255
    return float(np.count_nonzero(flat)) / flat.size, flat


def measure(img: np.ndarray, card: dict | None = None, t: dict | None = None) -> tuple[dict, dict]:
    """Quality scores of a BGR frame (and of its card when `card` is a find_card result).

    Returns (scores, masks) where masks are uint8 images for evidence.
    """
    t = t or load_thresholds()
    small = _gray_sample(img, int(t["sample_width"]))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    n = gray.size
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = _lapvar(gray)
    clip_high = float(np.count_nonzero(gray >= 250)) / n
    clip_low = float(np.count_nonzero(gray <= 5)) / n
    sat_mean = float(hsv[..., 1].mean())

    spec = ((hsv[..., 2] >= t["glare"]["v_min"]) & (hsv[..., 1] <= t["glare"]["s_max"])).astype(np.uint8) * 255
    region = None
    poly = roi_polygon_px(card)  # sticky card quad or lure floor polygon
    if poly is not None:
        sx = small.shape[1] / float(img.shape[1])
        region = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(region, np.round(np.asarray(poly) * sx).astype(np.int32), 255)  # both are convex
    lure = card is not None and card.get("trap_kind") == "lure"
    if region is not None and np.count_nonzero(region):
        glare_ratio = float(np.count_nonzero(spec & region)) / np.count_nonzero(region)
        glare_region = "floor" if lure else "card"
    else:
        glare_ratio = float(np.count_nonzero(spec)) / n
        glare_region = "frame"
    flat, flat_mask = _flat_share(gray, int(t["dirty"]["window"]), float(t["dirty"]["flat_ratio"]))

    scores = {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "edge_ratio": round(sharpness / (contrast * contrast), 5) if contrast > 0 else 0.0,
        "clip_low": round(clip_low, 4),
        "clip_high": round(clip_high, 4),
        "saturation_mean": round(sat_mean, 2),
        "glare_ratio": round(glare_ratio, 4),
        "glare_region": glare_region,  # what glare_ratio is a share of: "card", "floor" (lure tray) or "frame"
        "flat_share": round(flat, 4),
        "card_found": None,
        "card_coverage": None,
        "card_occlusion": None,
        "card_partial": None,
        "card_sharpness": None,
        "card_brightness": None,
        "card_contrast": None,
        "floor_sharpness": None,
    }
    if card is not None:
        scores["card_found"] = bool(card.get("found"))
        if lure and card.get("found") and region is not None and np.count_nonzero(region):
            # Lure tray floor: smooth white plastic, so its Laplacian variance measures how much is ON the floor
            # (stains, flies, IR noise), not focus. Informational only, at the 320-wide copy (no upscaling).
            # As card_sharpness (rectified 800-px crop, upscaled ~1.5x from 768x432 frames) it made 180 of 485
            # sharp BE4111159 photos "blurry" (live AWS run 2026-10-15, test_vision_quality_lure.py).
            inner = cv2.erode(region, np.ones((5, 5), np.uint8))
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
            scores["floor_sharpness"] = round(float(lap[inner > 0].var()), 2) if np.count_nonzero(inner) else None
        if card.get("found") and card.get("quad") is None:  # lure floor without a 4-corner fit
            scores.update(card_coverage=card.get("coverage"), card_occlusion=card.get("occlusion"),
                          card_partial=bool(card.get("partial")))
        elif card.get("found"):
            warped = warp_card(img, card["quad"])
            wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            scores.update(
                card_coverage=card.get("coverage"),
                card_occlusion=card.get("occlusion"),
                card_partial=bool(card.get("partial")),
                card_sharpness=None if lure else round(_lapvar(wg), 2),  # sticky card only (see floor_sharpness)
                card_brightness=round(float(wg.mean()), 2),
                card_contrast=round(float(wg.std()), 2),
            )
    return scores, {"glare": spec, "flat": flat_mask}


def assess(scores: dict, t: dict | None = None, baseline: dict | None = None, hour: int | None = None) -> list[str]:
    """Issues from scores (most serious first). `hour` is the farm-local hour, if known."""
    t = t or load_thresholds()
    issues: set[str] = set()
    if scores.get("card_found") is False:
        issues.add("no_card")
    elif scores.get("card_found"):
        if (scores.get("card_coverage") or 0) < t["card"]["min_coverage"]:
            issues.add("no_card")
        elif scores.get("card_partial") or (scores.get("card_occlusion") or 0) > t["card"]["occlusion_max"]:
            issues.add("partial_card")

    daylight = hour is None or 7 <= hour < 17
    if scores["brightness"] < t["dark"]["brightness_max"]:
        if daylight or not t["dark"]["daylight_only"]:
            issues.add("dark")
    elif scores["brightness"] > t["overexposed"]["brightness_min"] or scores["clip_high"] > t["overexposed"]["clip_high_max"]:
        issues.add("overexposed")
    else:
        if scores["sharpness"] < t["blurry"]["sharpness_min"]:
            issues.add("blurry")
        if scores.get("card_sharpness") is not None and scores["card_sharpness"] < t["blurry"]["card_sharpness_min"]:
            issues.add("blurry")
        if scores["contrast"] < t["low_contrast"]["contrast_min"]:
            issues.add("low_contrast")
        if baseline:
            if baseline.get("edge_ratio", 0) > 0 and scores["edge_ratio"] < baseline["edge_ratio"] * t["blurry"]["ratio_edge"]:
                issues.add("blurry")
            if baseline.get("contrast", 0) > 0 and scores["contrast"] < baseline["contrast"] * t["low_contrast"]["ratio_contrast"]:
                issues.add("low_contrast")
            if scores["flat_share"] > baseline.get("flat_share", 0) + t["dirty"]["delta_baseline"]:
                issues.add("dirty")
        elif scores["flat_share"] > t["dirty"]["ratio_max"]:
            issues.add("dirty")
    if scores["saturation_mean"] < t["ir"]["saturation_max"]:
        issues.add("ir")
    # A lure tray floor tolerates more glare than a sticky card (quality_thresholds.yaml glare.floor_ratio_max);
    # scores without glare_region (older analyzers, fakes) and thresholds without the floor limit get the strict one
    glare_max = t["glare"]["ratio_max"]
    if scores.get("glare_region") == "floor":
        glare_max = t["glare"].get("floor_ratio_max", glare_max)
    if scores["glare_ratio"] > glare_max:
        issues.add("glare")
    return [i for i in ISSUES if i in issues]


class Baseline:
    """A camera's usual scores: medians over its last `size` photos without issues."""

    KEYS = ("contrast", "edge_ratio", "flat_share", "sharpness")

    def __init__(self, size: int = 20, min_count: int = 5):
        self.size, self.min_count = size, min_count
        self._rows: deque = deque(maxlen=size)

    def update(self, scores: dict, issues: list[str]) -> None:
        # Card visibility is not a photo defect; it must not keep a photo out of the baseline
        if not [i for i in issues if i not in ("no_card", "partial_card")]:
            self._rows.append({k: scores[k] for k in self.KEYS})

    def get(self) -> dict | None:
        if len(self._rows) < self.min_count:
            return None
        return {k: float(np.median([r[k] for r in self._rows])) for k in self.KEYS} | {"size": len(self._rows)}


def assess_quality(
    img: np.ndarray,
    *,
    card: dict | None = None,
    find: bool = False,
    thresholds: dict | None = None,
    baseline: dict | None = None,
    hour: int | None = None,
    out_dir: str | Path | None = None,
    name: str = "quality",
    trap_kind: str = "sticky",
) -> dict:
    """Quality gate of one BGR frame. `card`: a find_card result (or `find=True` to run it for `trap_kind`).

    For a `lure` trap, `no_card` means the tray floor was not found and the card_* scores describe the floor;
    card_sharpness stays None (the absolute card blur floor is a sticky-card rule) and `floor_sharpness` is
    reported for information. Blur on a lure trap is judged on the whole frame (absolute floor + edge ratio
    against the camera baseline); glare on the tray floor against `glare.floor_ratio_max` (`scores.glare_region`).
    """
    t0 = time.perf_counter()
    t = thresholds or load_thresholds()
    if card is None and find:
        card = find_card(img, trap_kind=trap_kind)
    scores, masks = measure(img, card, t)
    issues = assess(scores, t, baseline, hour)
    res = {
        "ok": not issues,
        "issues": issues,
        "scores": scores,
        "baseline_used": baseline,
        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "evidence": {},
    }
    if out_dir is not None:
        out = Path(out_dir)
        res["evidence"]["glare_mask"] = save_jpeg(masks["glare"], out / f"{name}_glare.jpg")
        res["evidence"]["flat_mask"] = save_jpeg(masks["flat"], out / f"{name}_flat.jpg")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("--card", action="store_true", help="run find_card first and check the card too")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    img = cv2.imread(a.image)
    if img is None:
        print(f"cannot read {a.image}", file=sys.stderr)
        return 2
    print(json.dumps(assess_quality(img, find=a.card, out_dir=a.out, name=Path(a.image).stem), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
