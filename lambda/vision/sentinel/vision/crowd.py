"""Op 6: how full is a sticky card (dark-blob coverage + watershed split of touching flies).

    uv run python -m sentinel.vision.crowd IMAGE [--trap-kind sticky|lure] [--out DIR]

`estimate_crowd(img, trap_kind=..., card=find_card(img))` runs on the rectified card crop (the whole
frame when no card was found) and returns a JSON-serializable dict:

    status ("ok" | "not_applicable"), trap_kind, saturation_pct (dark area / surface area x 100),
    saturation (same as a 0-1 share), blob_count (segments after the watershed), component_count
    (connected dark regions before it), split_count (blob_count - component_count: extra flies found by
    splitting touching blobs), count_by_area, fly_radius_px, single_area_px, card_full (bool),
    card_full_threshold_pct, source
    ("card" | "full_frame"), time_ms, evidence {mask, segments}

Saturation is a sticky-card notion: flies accumulate on a card until it is replaced. A `lure` trap is not
sticky (a photo shows the flies present at that moment), so for `trap_kind="lure"` the result is
`status: "not_applicable"` with null numbers.

Method: dark pixels = adaptive mean threshold (local contrast; a large bait pad or stain is not darker
than its own neighbourhood), opened, closed (one fly's body, wings and legs become one blob), holes
filled, specks dropped. Blobs larger than `split_area` (1.6) x the median blob area are clusters of
touching flies: Euclidean distance transform, one marker per distance peak (local maxima about 1.5 fly
radii apart, the radius = median inscribed radius of the image's own blobs, most of which are single
flies), `cv2.watershed`. Single-size blobs are kept whole (splitting them cuts elongated flies in two:
on the camos valid split, watershed on every blob over-counted by +30 flies per picture).
`count_by_area` (sum of round(blob area / median area)) is reported as a cross-check. Parameters in
`DEFAULTS`, overridable from quality_thresholds.yaml (`crowd:`).

Validation (crowd_eval.py, docs/evidence/vision/crowd_eval.json): the camos fruit-fly pictures show flies
on a white bait disc, not on a yellow card; parameters were chosen on the valid split, numbers are
reported on the test split. Saturation is the area of the dark bodies; translucent wings mostly stay out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.card import CANONICAL_LONG, find_card, warp_card
from sentinel.vision.evidence import save_jpeg
from sentinel.vision.quality import load_thresholds

DEFAULTS = {
    "work_long": 800,         # crop is measured at this long side (canonical card size)
    # dark pixels: adaptive MEAN threshold, block = 6 % of the long side (odd, >= 15 px), C grey levels below
    # the local mean; then open 3, close 5, holes filled, specks < min_area_share of the surface dropped.
    # Chosen on the camos `valid` split (93 images; component count vs labels r 0.988, bias +0.02), never on test
    "block_share": 0.06,
    "adaptive_c": 30,
    "close": 5,
    "min_area_share": 0.0004,
    # watershed markers: distance peaks >= peak_rel x fly radius, local max over a disc of peak_sep x radius
    "split_area": 1.6,   # two touching flies overlap, so a pair is < 2x one fly (1.9x in the tests); valid split:
                         # 1.6-2.5 all give blob r 0.986-0.988
    "peak_rel": 0.5,
    "peak_sep": 1.5,
    "min_seg": 0.25,
    "card_full_pct": 10.0,    # suggestion threshold, see quality_thresholds.yaml crowd.card_full_pct
}


def _params(overrides: dict | None = None) -> dict:
    p = dict(DEFAULTS)
    try:
        p.update(load_thresholds().get("crowd") or {})
    except (OSError, KeyError):
        pass
    p.update(overrides or {})
    return p


def _ell(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def dark_mask(gray: np.ndarray, surface: np.ndarray, p: dict) -> np.ndarray:
    """uint8 mask of dark (insect) pixels inside `surface`: adaptive threshold, cleaned, holes filled."""
    h, w = gray.shape
    block = max(15, round(p["block_share"] * max(h, w)) | 1)
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, p["adaptive_c"])
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ell(3))
    if p["close"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ell(int(p["close"])))  # joins a fly's body, wings and legs
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cs, -1, 255, -1)
    m = cv2.bitwise_and(m, surface)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = p["min_area_share"] * float(max(1, np.count_nonzero(surface)))
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab].astype(np.uint8) * 255


def split_blobs(img_bgr: np.ndarray, mask: np.ndarray, p: dict) -> dict:
    """Distance transform + watershed on the oversized blobs of `mask`.

    A blob whose area exceeds `split_area` x the median blob area (most blobs are single flies) is a
    cluster of touching flies: one marker per distance-transform peak inside it, then `cv2.watershed`.
    Blobs of single-fly size are kept whole (splitting them only cuts elongated flies in two).
    Returns counts, the fly radius, the median single area and the label image (0 = background).
    """
    n, comp, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    n_comp = n - 1
    if n_comp <= 0:
        return {"blob_count": 0, "component_count": 0, "split_count": 0, "fly_radius_px": None,
                "single_area_px": None, "count_by_area": 0, "labels": np.zeros(mask.shape, np.int32)}
    areas = st[1:, cv2.CC_STAT_AREA].astype(np.float64)
    single = float(np.median(areas))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    comp_max = np.zeros(n, np.float32)
    np.maximum.at(comp_max, comp.ravel(), dist.ravel())
    r = max(2.0, float(np.median(comp_max[1:])))
    big = np.zeros(n, bool)
    big[1:] = areas > p["split_area"] * single
    count_by_area = int(sum(max(1, round(a / single)) for a in areas))

    # markers: 1 = background, whole single-size blobs keep their own label, peaks inside big blobs
    markers = np.ones(mask.shape, np.int32)
    small_px = (mask > 0) & ~big[comp]
    markers[small_px] = comp[small_px] + 1
    nxt = n + 1
    big_px = (mask > 0) & big[comp]
    if big_px.any():
        k = max(3, round(2 * p["peak_sep"] * r) | 1)
        local_max = dist >= cv2.dilate(dist, _ell(k)) - 1e-3
        peaks = (local_max & (dist >= p["peak_rel"] * r) & big_px).astype(np.uint8)
        n_pk, pk = cv2.connectedComponents(peaks, connectivity=8)
        markers[big_px] = 0  # unknown, flooded by watershed
        markers[pk > 0] = pk[pk > 0] + nxt - 1
        nxt += n_pk
        has = np.zeros(n, bool)
        has[comp[pk > 0]] = True
        for c in np.flatnonzero(big & ~has):  # a big blob without a peak still gets one marker
            ys, xs = np.nonzero(comp == c)
            j = int(np.argmax(dist[ys, xs]))
            markers[ys[j], xs[j]] = nxt
            nxt += 1
        cv2.watershed(img_bgr, markers)
    seg = markers.copy()
    seg[(seg <= 1) | (mask == 0)] = 0
    _, cnt = np.unique(seg[seg > 0], return_counts=True)
    # watershed slivers smaller than min_seg x a single fly are not flies
    blob_count = int(np.count_nonzero(cnt >= p["min_seg"] * single))
    return {"blob_count": blob_count, "component_count": int(n_comp), "split_count": int(max(0, blob_count - n_comp)),
            "fly_radius_px": round(r, 2), "single_area_px": round(single, 1), "count_by_area": count_by_area,
            "labels": seg}


def measure_crowd(crop: np.ndarray, surface: np.ndarray | None = None, params: dict | None = None) -> dict:
    """Saturation and blob counts of a card crop (BGR). `surface`: uint8 mask of the card area (None = all)."""
    p = _params(params)
    h, w = crop.shape[:2]
    s = min(1.0, p["work_long"] / float(max(h, w)))
    if s < 1.0:
        crop = cv2.resize(crop, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        if surface is not None:
            surface = cv2.resize(surface, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if surface is None:
        surface = np.full(gray.shape, 255, np.uint8)
    mask = dark_mask(gray, surface, p)
    sp = split_blobs(crop, mask, p)
    area = float(max(1, np.count_nonzero(surface)))
    sat = float(np.count_nonzero(mask)) / area
    return {
        "saturation_pct": round(100.0 * sat, 3),
        "saturation": round(sat, 5),
        "blob_count": sp["blob_count"],
        "component_count": sp["component_count"],
        "split_count": sp["split_count"],
        "count_by_area": sp["count_by_area"],
        "fly_radius_px": sp["fly_radius_px"],
        "single_area_px": sp["single_area_px"],
        "card_full": bool(100.0 * sat >= p["card_full_pct"]),
        "card_full_threshold_pct": p["card_full_pct"],
        "_mask": mask,
        "_labels": sp["labels"],
        "_crop": crop,
    }


def segments_image(crop: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Crop with every watershed segment in a random colour and its boundary in red."""
    rng = np.random.default_rng(0)
    lut = rng.integers(60, 255, (int(labels.max()) + 1, 3)).astype(np.uint8)
    lut[0] = 0
    color = lut[labels]
    vis = cv2.addWeighted(crop, 0.5, color, 0.5, 0)
    vis[labels == 0] = crop[labels == 0]
    edges = cv2.morphologyEx((labels > 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    inner = cv2.Laplacian(labels.astype(np.float32), cv2.CV_32F) != 0
    vis[(edges > 0) | (inner & (labels > 0))] = (0, 0, 255)
    return vis


def estimate_crowd(
    img: np.ndarray,
    *,
    trap_kind: str = "sticky",
    card: dict | None = None,
    params: dict | None = None,
    out_dir: str | Path | None = None,
    name: str = "crowd",
) -> dict:
    """Op 6 on one BGR frame. `card`: a find_card result (computed when None, sticky only)."""
    t0 = time.perf_counter()
    if trap_kind == "lure":
        return {"status": "not_applicable", "trap_kind": "lure",
                "reason": "lure traps are not sticky: flies do not accumulate, saturation has no meaning",
                "saturation_pct": None, "saturation": None, "blob_count": None, "component_count": None,
                "split_count": None, "fly_radius_px": None, "card_full": None, "card_full_threshold_pct": None,
                "source": None, "time_ms": round((time.perf_counter() - t0) * 1000.0, 2), "evidence": {}}
    if trap_kind != "sticky":
        raise ValueError(f"unknown trap_kind {trap_kind!r}")
    card = card if card is not None else find_card(img, trap_kind="sticky")
    if card.get("found") and card.get("quad") is not None:
        crop, source = warp_card(img, card["quad"], CANONICAL_LONG), "card"
    else:
        crop, source = img, "full_frame"
    m = measure_crowd(crop, None, params)
    res = {"status": "ok", "trap_kind": "sticky", "source": source,
           **{k: v for k, v in m.items() if not k.startswith("_")}, "evidence": {}}
    if out_dir is not None:
        out = Path(out_dir)
        res["evidence"]["mask"] = save_jpeg(m["_mask"], out / f"{name}_crowd_mask.jpg")
        seg = segments_image(m["_crop"], m["_labels"])
        if source == "full_frame":  # no card: the frame may show people, keep only the dark blobs readable
            from sentinel.vision.evidence import pixelate

            seg[m["_mask"] == 0] = pixelate(seg)[m["_mask"] == 0]
        res["evidence"]["segments"] = save_jpeg(seg, out / f"{name}_crowd_segments.jpg")
    res["time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("--trap-kind", default="sticky", choices=["sticky", "lure"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    img = cv2.imread(a.image)
    if img is None:
        print(f"cannot read {a.image}", file=sys.stderr)
        return 2
    print(json.dumps(estimate_crowd(img, trap_kind=a.trap_kind, out_dir=a.out, name=Path(a.image).stem), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
