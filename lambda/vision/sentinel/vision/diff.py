"""Op 4: new catches between two photos of a trap (box matching after registration + absdiff heatmap).

    uv run python -m sentinel.vision.diff PREV CUR [--out DIR]

`diff_detections(prev_boxes, cur_boxes, H)` matches the previous photo's boxes, moved into the
current photo by the registration homography H, to the current boxes with the Hungarian algorithm
(scipy linear_sum_assignment) on centroid distance; a pair counts only within
`dist_scale x mean box size`. Returns a JSON-serializable dict:

    trap_kind, meaning, new, persisted, disappeared (lists), new_count, persisted_count, disappeared_count,
    card_change_suspected, card_change_reasons, change_ratio, evidence {heatmap}

`trap_kind` changes what the numbers mean (`meaning`): on a `sticky` card flies stay, so `new` are
**catches** since the previous photo and a mass disappearance means the card was replaced
(`card_change_suspected`). On a `lure` trap nothing is glued: `new` / `disappeared` are **arrivals /
departures** (activity between two moments), never catches, and `card_change_suspected` is always false.

`compare_photos(prev_img, cur_img, trap_kind=...)` is ops 1+3+4 end to end: find the card (sticky) or
the tray floor (lure) in both frames (whole frames when there is no 4-corner surface), rectify, register
(SIFT by default, PHASE1_STATUS finding 4), move the boxes into the canonical frame, diff.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.vision.blobs import detect_dark_blobs
from sentinel.vision.card import CANONICAL_LONG, card_homography, find_card
from sentinel.vision.evidence import pixelate, save_jpeg
from sentinel.vision.register import register

DIST_SCALE = 0.6  # match radius = 0.6 x mean sqrt(box area)
MIN_DIST_PX = 4.0
CHANGE_PIXEL = 30  # grey-level absdiff counted as changed
# Card change: most old detections vanished, or the registration failed and the picture changed a lot
CARD_CHANGE_MIN_PREV = 5
CARD_CHANGE_RATIO = 2.0
CARD_CHANGE_APPEARANCE = 0.35


def transform_boxes(boxes, M: np.ndarray | None) -> list[list[float]]:
    """Boxes xyxy through a 3x3 transform: the axis-aligned box of the 4 moved corners."""
    if not boxes:
        return []
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    if M is None:
        return b.tolist()
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, np.asarray(M, np.float64)).reshape(-1, 4, 2)
    return np.concatenate([moved.min(axis=1), moved.max(axis=1)], axis=1).tolist()


def _centres_sizes(boxes) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    s = np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None))
    return c, s


def match_boxes(prev_boxes, cur_boxes, dist_scale: float = DIST_SCALE, min_px: float = MIN_DIST_PX):
    """Hungarian matching of two box lists in the same frame -> (pairs [(i, j, dist)], unmatched_prev, unmatched_cur)."""
    n, m = len(prev_boxes), len(cur_boxes)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    cp, sp = _centres_sizes(prev_boxes)
    cc, sc = _centres_sizes(cur_boxes)
    dist = np.linalg.norm(cp[:, None, :] - cc[None, :, :], axis=2)
    thr = np.maximum(min_px, dist_scale * (sp[:, None] + sc[None, :]) / 2)
    big = 1e6
    cost = np.where(dist <= thr, dist, big)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(dist[i, j])) for i, j in zip(rows, cols) if cost[i, j] < big]
    mp = {i for i, _, _ in pairs}
    mc = {j for _, j, _ in pairs}
    return pairs, [i for i in range(n) if i not in mp], [j for j in range(m) if j not in mc]


def change_heatmap(prev: np.ndarray, cur: np.ndarray, H: np.ndarray | None) -> tuple[float, np.ndarray, np.ndarray]:
    """Absdiff of the aligned pair: (changed share of the overlap, heat 0-255, valid-overlap mask)."""
    h, w = cur.shape[:2]
    g1 = prev if prev.ndim == 2 else cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g2 = cur if cur.ndim == 2 else cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    if H is not None:
        aligned = cv2.warpPerspective(g1, np.asarray(H, np.float64), (w, h))
        valid = cv2.warpPerspective(np.full(g1.shape, 255, np.uint8), np.asarray(H, np.float64), (w, h))
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    else:
        aligned = cv2.resize(g1, (w, h))
        valid = np.full((h, w), 255, np.uint8)
    a = cv2.GaussianBlur(aligned, (5, 5), 0)
    b = cv2.GaussianBlur(g2, (5, 5), 0)
    # Global exposure change is not a change of the trap: match means first
    a = cv2.convertScaleAbs(a, alpha=1.0, beta=float(b[valid > 0].mean() - a[valid > 0].mean()) if valid.any() else 0.0)
    heat = cv2.absdiff(a, b)
    heat[valid == 0] = 0
    n_valid = int(np.count_nonzero(valid))
    ratio = float(np.count_nonzero((heat >= CHANGE_PIXEL) & (valid > 0))) / n_valid if n_valid else 1.0
    return ratio, heat, valid


def diff_detections(
    prev_boxes,
    cur_boxes,
    H=None,
    *,
    registration_ok: bool = True,
    prev_img: np.ndarray | None = None,
    cur_img: np.ndarray | None = None,
    dist_scale: float = DIST_SCALE,
    out_dir: str | Path | None = None,
    name: str = "diff",
    pixelate_evidence: bool = False,
    trap_kind: str = "sticky",
) -> dict:
    """New / persisted / disappeared detections between two photos (boxes xyxy in each photo's frame)."""
    t0 = time.perf_counter()
    Hm = np.asarray(H, np.float64) if H is not None and registration_ok else None
    moved = transform_boxes(prev_boxes, Hm)
    pairs, gone, fresh = match_boxes(moved, cur_boxes, dist_scale)
    r2 = lambda b: [round(float(v), 1) for v in b]
    if trap_kind not in ("sticky", "lure"):
        raise ValueError(f"unknown trap_kind {trap_kind!r}")
    res = {
        "trap_kind": trap_kind,
        "meaning": "catches" if trap_kind == "sticky" else "activity",
        "new": [r2(cur_boxes[j]) for j in fresh],
        "persisted": [{"prev": i, "cur": j, "dist_px": round(d, 2)} for i, j, d in pairs],
        "disappeared": [r2(prev_boxes[i]) for i in gone],
        "new_count": len(fresh),
        "persisted_count": len(pairs),
        "disappeared_count": len(gone),
        "prev_count": len(prev_boxes),
        "cur_count": len(cur_boxes),
        "registration_ok": bool(registration_ok),
        "change_ratio": None,
        "card_change_suspected": False,
        "card_change_reasons": [],
        "evidence": {},
    }
    if prev_img is not None and cur_img is not None:
        ratio, heat, _ = change_heatmap(prev_img, cur_img, Hm)
        res["change_ratio"] = round(ratio, 4)
        if out_dir is not None:
            base = pixelate(cur_img) if pixelate_evidence else cur_img
            base = base if base.ndim == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            color = cv2.applyColorMap(cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
            over = cv2.addWeighted(base, 0.55, color, 0.45, 0)
            for b in res["new"]:
                cv2.rectangle(over, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            for b in transform_boxes(res["disappeared"], Hm):
                cv2.rectangle(over, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            res["evidence"]["heatmap"] = save_jpeg(over, Path(out_dir) / f"{name}_heatmap.jpg")
    reasons = []
    if trap_kind != "sticky":
        pass  # a lure trap has no card: flies leaving is activity, not a replacement
    elif len(prev_boxes) >= CARD_CHANGE_MIN_PREV and len(gone) >= max(CARD_CHANGE_MIN_PREV, CARD_CHANGE_RATIO * len(pairs)):
        reasons.append("disappeared>>persisted")
    if trap_kind == "sticky" and not registration_ok and res["change_ratio"] is not None and res["change_ratio"] >= CARD_CHANGE_APPEARANCE:
        reasons.append("registration_failed+appearance_change")
    res["card_change_suspected"] = bool(reasons)
    res["card_change_reasons"] = reasons
    res["time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return res


def _canonical(img: np.ndarray, card: dict, long_side: int) -> tuple[np.ndarray, np.ndarray, str]:
    """(canonical image, frame->canonical 3x3, source): the rectified card, else the scaled frame."""
    if card.get("found") and card.get("quad") is not None:
        M, size = card_homography(card["quad"], long_side)
        src = "floor" if card.get("trap_kind") == "lure" else "card"
        return cv2.warpPerspective(img, M, size, borderMode=cv2.BORDER_REPLICATE), M, src
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), M, "full_frame"


def compare_photos(
    prev_img: np.ndarray,
    cur_img: np.ndarray,
    prev_boxes=None,
    cur_boxes=None,
    *,
    profile: str = "yellow",
    method: str = "sift_flann",
    long_side: int = CANONICAL_LONG,
    out_dir: str | Path | None = None,
    name: str = "pair",
    features: dict | None = None,
    trap_kind: str = "sticky",
) -> dict:
    """Ops 1+3+4 on a photo pair. Boxes are xyxy in each original frame; None = interim blob boxes."""
    t0 = time.perf_counter()
    cp = find_card(prev_img, trap_kind=trap_kind, profile=profile)
    cc = find_card(cur_img, trap_kind=trap_kind, profile=profile)
    a, Ma, src_a = _canonical(prev_img, cp, long_side)
    b, Mb, src_b = _canonical(cur_img, cc, long_side)
    full = "full_frame" in (src_a, src_b)
    reg = register(a, b, method=method, out_dir=out_dir, name=name, pixelate_evidence=full, features=features)
    if prev_boxes is None or cur_boxes is None:
        box_source = "classic_blobs"
        pb, cb = detect_dark_blobs(a)["boxes"], detect_dark_blobs(b)["boxes"]
    else:
        box_source = "given"
        pb, cb = transform_boxes(prev_boxes, Ma), transform_boxes(cur_boxes, Mb)
    d = diff_detections(pb, cb, reg["H"], registration_ok=reg["ok"], prev_img=a, cur_img=b,
                        out_dir=out_dir, name=name, pixelate_evidence=full, trap_kind=trap_kind)
    ev = {**reg["evidence"], **d["evidence"]}
    return {
        "trap_kind": trap_kind,
        "frame_source": [src_a, src_b],
        "card": {"prev": {k: cp[k] for k in ("found", "quad_norm", "coverage")},
                 "cur": {k: cc[k] for k in ("found", "quad_norm", "coverage")}},
        "register": {k: v for k, v in reg.items() if k != "evidence"},
        "diff": {k: v for k, v in d.items() if k != "evidence"},
        "box_source": box_source,
        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "evidence": ev,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prev")
    ap.add_argument("cur")
    ap.add_argument("--method", default="sift_flann", choices=["auto", "aliked_lightglue", "sift_flann"])
    ap.add_argument("--trap-kind", default="sticky", choices=["sticky", "lure"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    p, c = cv2.imread(a.prev), cv2.imread(a.cur)
    if p is None or c is None:
        print("cannot read input images", file=sys.stderr)
        return 2
    print(json.dumps(compare_photos(p, c, method=a.method, out_dir=a.out, trap_kind=a.trap_kind), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
