"""Op 1: find the trap surface and rectify it, for both FlyBot trap kinds.

    uv run python -m sentinel.vision.card IMAGE [--trap-kind sticky|lure] [--out DIR] [--profile yellow|white|auto]

FlyBot's `Camera.trapKind` has exactly two values, and `find_card(img, trap_kind=...)` handles both:

* `sticky` (default): a yellow sticky card (counts accumulate until the card is replaced). HSV mask ->
  contour -> approxPolyDP -> side-line refinement -> warpPerspective. Profiles: `yellow` (default),
  `white` (bright unsaturated surface, opt-in only), `auto` (yellow, then white).
* `lure`: a bait trap that is not sticky (a photo shows the flies present at that moment). The
  production trap cameras look down into a white tray: `find_lure_floor` segments the bright,
  smooth, low-saturation tray floor (texture + brightness + saturation mask, morphological
  open/close, largest region, holes filled) and returns its polygon as the ROI, a rectified crop when
  a 4-corner fit is plausible, otherwise the polygon only; when no floor is found the ROI is the
  whole frame.

Returned JSON-serializable dict (both kinds):

    found, trap_kind, profile, roi_source ("card" | "floor_quad" | "floor_polygon" | "whole_frame" | "none"),
    quad (4 [x, y] pixel corners TL, TR, BR, BL, or None), quad_norm (same, 0-1),
    polygon (pixel polygon of the ROI, or None), roi (camos shadow `detections[].roi`: a list of
    polygons of normalized [x, y]), coverage (ROI area / frame area), fill, occlusion, partial,
    n_cards, cards, warped_size, warped_path, time_ms, evidence {overlay, mask, warped}

`found` semantics:
* sticky: a plausible yellow card quad was found; when False, `roi` is `[]` and ops 3-4 work on the
  whole frame .
* lure: a plausible tray floor region was found (coverage 0.15-0.97, solidity >= 0.85, bright and
  unsaturated); `quad` is set only when the floor also fits 4 corners (region fills >= 0.9 of the
  quad), so `warped_path` exists only then. When False, `roi` is the whole frame
  (`roi_source: "whole_frame"`), never `[]`, because a lure camera always looks into its trap.

Canonical size keeps the surface's aspect ratio, long side `canonical_long` pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.evidence import pixelate_outside, save_jpeg

WORK_LONG = 640  # masks and contours are computed on a copy with this long side
CANONICAL_LONG = 800

PROFILES: dict[str, dict] = {
    # OpenCV hue is 0-180: yellow glue cards sit around 22-34. max_overflow = blob area / quad area:
    # a card fills its 4-corner quad (~1.0); a round bait cap or disc overflows it (~1.3-1.6)
    # min_fill = mask share inside the quad; min_solidity = blob area / convex hull area (bait stains
    # and streaks are concave); interior angles of a card seen in perspective stay within min_angle..180-min_angle
    "yellow": {"h": (18, 40), "s": (70, 255), "v": (80, 255), "min_area": 0.04, "min_fill": 0.8, "max_overflow": 1.12,
               "min_solidity": 0.85, "min_angle": 40.0,
               # tan bait stains next to a card are yellow but less saturated: when the merged blob fails
               # the shape tests, retry with a stricter saturation floor
               "s_retry": (120, 160)},
    "white": {"h": (0, 180), "s": (0, 60), "v": (140, 255), "min_area": 0.08, "min_fill": 0.6, "max_overflow": 9.0,
              "min_solidity": 0.0, "min_angle": 20.0},
}
MAX_ASPECT = 8.0
BORDER_MARGIN = 2  # px on the work image: contour points this close to the edge lie on the frame border
# A card cut by the frame runs along the border: partial when this share of its outline is on it
PARTIAL_BORDER_SHARE = 0.03


def _mask(hsv: np.ndarray, prof: dict) -> np.ndarray:
    lo = np.array([prof["h"][0], prof["s"][0], prof["v"][0]], np.uint8)
    hi = np.array([prof["h"][1], prof["s"][1], prof["v"][1]], np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    # Closing fills the dark insects on the card so the card stays one blob
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL (clockwise in image coordinates, TL = smallest x + y)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise on screen (y down)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _min_interior_angle(q: np.ndarray) -> float:
    angles = []
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        cosv = float(np.dot(a, b) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b)))
        ang = float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
        angles.append(min(ang, 180.0 - ang))
    return min(angles)


def _approx_quad(hull: np.ndarray) -> tuple[np.ndarray, str]:
    """4-corner approximation of a convex hull: approxPolyDP with a growing epsilon, else minAreaRect."""
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.12, 23):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32), "approx_poly"
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32), "min_area_rect"


def _line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    den = vx1 * vy2 - vy1 * vx2
    if abs(den) < 1e-9:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / den
    return np.array([x1 + t * vx1, y1 + t * vy1], np.float32)


def refine_quad(quad: np.ndarray, contour: np.ndarray, band: float = 0.04) -> np.ndarray:
    """Fit a line to the contour points along each side and intersect neighbours.

    approxPolyDP keeps contour vertices, which cut off acute corners and rounded card corners;
    the side lines are better estimated than the corners themselves.
    """
    q = order_corners(quad)
    pts = contour.reshape(-1, 2).astype(np.float32)
    diag = float(np.linalg.norm(q[2] - q[0]))
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1:
            return q
        n = np.array([-d[1], d[0]]) / L
        rel = pts - a
        along = rel @ (d / L)
        dist = np.abs(rel @ n)
        sel = pts[(dist < band * diag) & (along > 0.15 * L) & (along < 0.85 * L)]
        if len(sel) < 5:
            return q
        lines.append(cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4))
    out = []
    for i in range(4):
        p = _line_intersection(lines[(i - 1) % 4], lines[i])
        if p is None or np.linalg.norm(p - q[i]) > 0.2 * diag:
            return q
        out.append(p)
    return order_corners(np.array(out, np.float32))


def _candidates(mask: np.ndarray, prof: dict) -> list[dict]:
    h, w = mask.shape
    frame = float(h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < prof["min_area"] * frame:
            continue
        hull = cv2.convexHull(cnt)
        solidity = area / max(1.0, cv2.contourArea(hull))
        if solidity < prof["min_solidity"]:
            continue
        quad, method = _approx_quad(hull)
        if method == "approx_poly":
            quad = refine_quad(quad, cnt)
        quad_area = abs(cv2.contourArea(quad))
        if quad_area <= 0:
            continue
        overflow = area / quad_area
        if overflow > prof["max_overflow"]:
            continue
        # Share of the quad the color mask actually covers: low = occluded or not a card
        qmask = np.zeros_like(mask)
        cv2.fillConvexPoly(qmask, np.round(quad).astype(np.int32), 255)
        inside = cv2.countNonZero(cv2.bitwise_and(mask, qmask))
        fill = min(1.0, inside / max(1.0, float(cv2.countNonZero(qmask))))
        q = order_corners(quad)
        sides = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
        if sides.min() < 8 or sides.max() / max(1e-6, sides.min()) > MAX_ASPECT * 1.5:
            continue
        width = max(sides[0], sides[2])
        height = max(sides[1], sides[3])
        if max(width, height) / max(1e-6, min(width, height)) > MAX_ASPECT:
            continue
        if fill < prof["min_fill"]:
            continue
        if _min_interior_angle(q) < prof["min_angle"]:
            continue
        pts = cnt.reshape(-1, 2)
        on_border = (pts[:, 0] <= BORDER_MARGIN) | (pts[:, 1] <= BORDER_MARGIN) | (pts[:, 0] >= w - 1 - BORDER_MARGIN) | (
            pts[:, 1] >= h - 1 - BORDER_MARGIN
        )
        partial = float(on_border.mean()) >= PARTIAL_BORDER_SHARE
        out.append(
            {
                "quad_work": q,
                "method": method,
                "coverage": quad_area / frame,
                "fill": fill,
                "overflow": float(overflow),
                "solidity": float(solidity),
                "partial": bool(partial),
            }
        )
    out.sort(key=lambda c: c["coverage"], reverse=True)
    return out


def card_quad_to_roi(quad, w: int, h: int, ndigits: int = 4) -> list[list[list[float]]]:
    """Pixel quad -> camos protocol ROI: a list of polygons of normalized [x, y] (0-1, clipped).

    device-protocol.md v1: detection `roi` is `[[[x, y], ...], ...]`, coordinates normalized 0-1 on
    the original frame (e.g. `[[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]]`).
    """
    pts = np.asarray(quad, np.float64).reshape(-1, 2)
    poly = [[round(float(np.clip(x / w, 0.0, 1.0)), ndigits), round(float(np.clip(y / h, 0.0, 1.0)), ndigits)] for x, y in pts]
    return [poly]


def canonical_size(quad, long_side: int = CANONICAL_LONG) -> tuple[int, int]:
    """(width, height) of the rectified card: the quad's own aspect ratio, long side `long_side`."""
    q = order_corners(np.asarray(quad, np.float32))
    width = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    height = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    s = long_side / max(1e-6, max(width, height))
    return max(1, round(width * s)), max(1, round(height * s))


def card_homography(quad, long_side: int = CANONICAL_LONG) -> tuple[np.ndarray, tuple[int, int]]:
    """Homography frame -> canonical card frame, and the canonical (width, height)."""
    q = order_corners(np.asarray(quad, np.float32))
    cw, ch = canonical_size(q, long_side)
    dst = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
    return cv2.getPerspectiveTransform(q, dst), (cw, ch)


def warp_card(img: np.ndarray, quad, long_side: int = CANONICAL_LONG) -> np.ndarray:
    """Rectified card image (aspect kept, long side `long_side`)."""
    M, size = card_homography(quad, long_side)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


TRAP_KINDS = ("sticky", "lure")

# Lure tray floor (find_lure_floor). Tuned on 768x432 / 2304x1296 frames of BE4089629 and BE4111159,
# measured on a copy with long side WORK_LONG: the floor is smooth (mean Sobel magnitude over an 11 px box
# below tex_max, or below the frame median when that is higher), at least v_rel of the frame's
# 95th-percentile brightness (night IR frames fall off towards the corners), and not strongly saturated
# (tan bait stains reach S ~ 100). The ribbed side walls are textured and drop out.
LURE: dict = {"tex_max": 12.0, "v_rel": 0.40, "s_max": 110, "open": 9, "close": 31,
              "min_coverage": 0.15, "max_coverage": 0.97, "min_solidity": 0.85,
              "min_v": 70, "max_s_median": 70,
              # 4-corner fit: region fills >= quad_fill of the quad, the quad covers >= quad_cover of the
              # region (a slanted side must not cut a floor corner off), quad area / hull area <= quad_overflow,
              # corners at most quad_margin (share of the frame side) outside the frame. The lure cameras look
              # straight down at a rectangular floor, so every interior angle must be within 90 +- 12 deg
              # (quad_min_angle 78): on the 50-photo hand-check every fit with a slanted side that cut a floor
              # corner off or stuck out of the floor had a corner angle of 63-75 deg, every good fit >= 81.4
              # (docs/evidence/vision/card_check_lure.md). quad_dev (outline distance to the hull) is reported only:
              # the fisheye floor sides bulge, so good fits reach 0.064 of the diagonal
              "quad_fill": 0.9, "quad_cover": 0.95, "quad_overflow": 1.12, "quad_min_angle": 78.0,
              "quad_margin": 0.02, "poly_eps": 0.01}
WHOLE_FRAME_ROI = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]


def _empty_result(trap_kind: str, profile: str, tried: list[str], W: int, H: int) -> dict:
    return {
        "found": False,
        "trap_kind": trap_kind,
        "profile": profile,
        "profiles_tried": tried,
        "frame_size": [W, H],
        "roi_source": "none",
        "quad": None,
        "quad_norm": None,
        "polygon": None,
        "roi": [],
        "coverage": 0.0,
        "fill": None,
        "occlusion": None,
        "partial": False,
        "n_cards": 0,
        "cards": [],
        "quad_method": None,
        "warped_size": None,
        "warped_path": None,
        "time_ms": None,
        "evidence": {},
    }


def find_card(
    img: np.ndarray,
    *,
    trap_kind: str = "sticky",
    profile: str = "yellow",
    out_dir: str | Path | None = None,
    name: str = "card",
    canonical_long: int = CANONICAL_LONG,
) -> dict:
    """Find the trap surface in a BGR frame for `trap_kind` ("sticky" | "lure"). See module docstring."""
    if trap_kind == "lure":
        return find_lure_floor(img, out_dir=out_dir, name=name, canonical_long=canonical_long)
    if trap_kind != "sticky":
        raise ValueError(f"unknown trap_kind {trap_kind!r} (expected one of {TRAP_KINDS})")
    return _find_sticky(img, profile=profile, out_dir=out_dir, name=name, canonical_long=canonical_long)


def _work_copy(img: np.ndarray) -> tuple[np.ndarray, float]:
    if img is None or img.ndim != 3:
        raise ValueError("find_card expects a BGR image")
    H, W = img.shape[:2]
    scale = min(1.0, WORK_LONG / float(max(H, W)))
    work = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
    return work, scale


def _find_sticky(img: np.ndarray, *, profile: str, out_dir, name: str, canonical_long: int) -> dict:
    t0 = time.perf_counter()
    work, scale = _work_copy(img)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    tried = ["yellow", "white"] if profile == "auto" else [profile]
    used, cands, mask = tried[-1], [], None
    for p in tried:
        if p not in PROFILES:
            raise ValueError(f"unknown profile {p!r}")
        prof = PROFILES[p]
        for s_min in (prof["s"][0], *prof.get("s_retry", ())):
            step = {**prof, "s": (s_min, prof["s"][1])}
            mask = _mask(hsv, step)
            cands = _candidates(mask, step)
            if cands:
                break
        used = p
        if cands:
            break

    cards = []
    for c in cands:
        quad = c["quad_work"] / scale
        cards.append(
            {
                "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in quad],
                "quad_norm": card_quad_to_roi(quad, W, H)[0],
                "coverage": round(float(c["coverage"]), 4),
                "fill": round(float(c["fill"]), 4),
                "overflow": round(c["overflow"], 4),
                "solidity": round(c["solidity"], 4),
                "partial": c["partial"],
                "method": c["method"],
            }
        )

    res = _empty_result("sticky", used, tried, W, H)
    res["n_cards"], res["cards"] = len(cards), cards
    warped = None
    if cards:
        best = cards[0]
        quad = np.asarray(best["quad"], np.float32)
        res.update(
            found=True,
            roi_source="card",
            quad=best["quad"],
            quad_norm=best["quad_norm"],
            polygon=best["quad"],
            roi=card_quad_to_roi(quad, W, H),
            coverage=best["coverage"],
            fill=best["fill"],
            occlusion=round(1.0 - best["fill"], 4),
            partial=best["partial"],
            quad_method=best["method"],
        )
        warped = warp_card(img, quad, canonical_long)
        res["warped_size"] = [int(warped.shape[1]), int(warped.shape[0])]
    res["time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    _write_evidence(img, res, mask, warped, out_dir, name)
    return res


def lure_floor_mask(work: np.ndarray, prm: dict | None = None) -> np.ndarray:
    """uint8 mask of smooth, bright, unsaturated pixels (tray floor candidates) on a work-size BGR image."""
    prm = prm or LURE
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    g = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    tex = cv2.boxFilter(mag, -1, (11, 11))
    v = cv2.GaussianBlur(hsv[..., 2], (9, 9), 0)
    tex_t = max(float(prm["tex_max"]), float(np.median(tex)))
    vref = float(np.percentile(v, 95))
    m = ((tex < tex_t) & (v >= prm["v_rel"] * vref) & (hsv[..., 1] <= prm["s_max"])).astype(np.uint8) * 255

    def ell(n: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell(int(prm["open"])))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(int(prm["close"])))


def find_lure_floor(
    img: np.ndarray,
    *,
    out_dir: str | Path | None = None,
    name: str = "card",
    canonical_long: int = CANONICAL_LONG,
    prm: dict | None = None,
) -> dict:
    """Lure trap profile: segment the tray floor; polygon ROI, rectified crop when 4 corners fit."""
    t0 = time.perf_counter()
    prm = prm or LURE
    work, scale = _work_copy(img)
    H, W = img.shape[:2]
    h, w = work.shape[:2]
    mask = lure_floor_mask(work, prm)
    res = _empty_result("lure", "lure_floor", ["lure_floor"], W, H)
    res.update(roi_source="whole_frame", roi=[[list(p) for p in WHOLE_FRAME_ROI[0]]], coverage=1.0)
    checks: dict = {}
    region = None
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cs, _ = cv2.findContours((lab == i).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cs, key=cv2.contourArea)
        region = np.zeros_like(mask)
        cv2.drawContours(region, [cnt], -1, 255, -1)  # holes (bait port, flies, stains) filled
        area = float(cv2.countNonZero(region))
        hull = cv2.convexHull(cnt)
        hull_area = max(1.0, cv2.contourArea(hull))
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        inside = region > 0
        checks = {
            "coverage": round(area / float(h * w), 4),
            "solidity": round(area / hull_area, 4),
            "v_median": float(np.median(hsv[..., 2][inside])),
            "s_median": float(np.median(hsv[..., 1][inside])),
        }
        ok = (prm["min_coverage"] <= checks["coverage"] <= prm["max_coverage"]
              and checks["solidity"] >= prm["min_solidity"]
              and checks["v_median"] >= prm["min_v"] and checks["s_median"] <= prm["max_s_median"])
        checks["plausible"] = bool(ok)
        if ok:
            poly = cv2.approxPolyDP(hull, prm["poly_eps"] * cv2.arcLength(hull, True), True).reshape(-1, 2)
            poly_full = poly.astype(np.float64) / scale
            res.update(
                found=True,
                roi_source="floor_polygon",
                polygon=[[round(float(x), 2), round(float(y), 2)] for x, y in poly_full],
                roi=card_quad_to_roi(poly_full, W, H),
                coverage=checks["coverage"],
                fill=checks["solidity"],
            )
            quad, method = _approx_quad(hull)
            if method == "approx_poly":
                quad = refine_quad(quad, cnt)
            q = order_corners(quad)
            qmask = np.zeros_like(mask)
            cv2.fillConvexPoly(qmask, np.round(q).astype(np.int32), 255)
            q_area = float(max(1, cv2.countNonZero(qmask)))
            both = cv2.countNonZero(cv2.bitwise_and(region, qmask))
            q_fill, q_cover = both / q_area, both / max(1.0, area)
            q_over = abs(cv2.contourArea(q)) / hull_area
            mx, my = prm["quad_margin"] * w, prm["quad_margin"] * h
            inside_frame = bool(((q[:, 0] >= -mx) & (q[:, 0] <= w - 1 + mx) & (q[:, 1] >= -my) & (q[:, 1] <= h - 1 + my)).all())
            q32 = q.reshape(-1, 1, 2).astype(np.float32)
            hull32 = hull.reshape(-1, 1, 2).astype(np.float32)
            dev = max(max(abs(cv2.pointPolygonTest(q32, (float(x), float(y)), True)) for x, y in hull.reshape(-1, 2)),
                      max(abs(cv2.pointPolygonTest(hull32, (float(x), float(y)), True)) for x, y in q))
            dev_share = dev / float(np.hypot(w, h))
            checks.update(quad_fill=round(q_fill, 4), quad_cover=round(q_cover, 4), quad_overflow=round(q_over, 4),
                          quad_min_angle=round(_min_interior_angle(q), 2), quad_in_frame=inside_frame,
                          quad_dev=round(dev_share, 4), quad_method=method)
            if (method == "approx_poly" and q_fill >= prm["quad_fill"] and q_cover >= prm["quad_cover"]
                    and q_over <= prm["quad_overflow"] and inside_frame
                    and checks["quad_min_angle"] >= prm["quad_min_angle"]):
                qf = q / scale
                quad_l = [[round(float(x), 2), round(float(y), 2)] for x, y in qf]
                qn = card_quad_to_roi(qf, W, H)[0]
                res.update(roi_source="floor_quad", quad=quad_l, quad_norm=qn, quad_method=method, n_cards=1,
                           cards=[{"quad": quad_l, "quad_norm": qn, "coverage": checks["coverage"],
                                   "fill": round(q_fill, 4), "partial": False, "method": method}])
    res["floor_checks"] = checks
    warped = None
    if res["quad"] is not None:
        warped = warp_card(img, res["quad"], canonical_long)
        res["warped_size"] = [int(warped.shape[1]), int(warped.shape[0])]
    res["time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    _write_evidence(img, res, region if region is not None else mask, warped, out_dir, name)
    return res


def roi_polygon_px(card: dict | None) -> list | None:
    """Pixel polygon of the ROI a find_card result keeps (card quad or lure floor polygon); None = whole frame."""
    if not card or not card.get("found"):
        return None
    return card.get("polygon") or card.get("quad")


def _write_evidence(img, res: dict, mask, warped, out_dir, name: str) -> None:
    if out_dir is None:
        return
    out = Path(out_dir)
    overlay = draw_card(img, res)
    res["evidence"]["overlay"] = save_jpeg(pixelate_outside(overlay, roi_polygon_px(res)), out / f"{name}_overlay.jpg")
    if mask is not None:
        res["evidence"]["mask"] = save_jpeg(mask, out / f"{name}_mask.jpg")
    if warped is not None:
        res["warped_path"] = save_jpeg(warped, out / f"{name}_warped.jpg")
        res["evidence"]["warped"] = res["warped_path"]


def draw_card(img: np.ndarray, res: dict) -> np.ndarray:
    """Copy of `img` with the ROI drawn: card quads (primary green, others orange, partial red), lure floor cyan."""
    vis = img.copy()
    t = max(2, round(max(img.shape[:2]) / 300))
    if res.get("trap_kind") == "lure" and res.get("polygon") is not None:
        cv2.polylines(vis, [np.round(np.asarray(res["polygon"])).astype(np.int32)], True, (255, 255, 0), t)
    for i, c in enumerate(res.get("cards", [])):
        pts = np.round(np.asarray(c["quad"])).astype(np.int32)
        color = (0, 0, 255) if c["partial"] else ((0, 220, 0) if i == 0 else (0, 160, 255))
        cv2.polylines(vis, [pts], True, color, t)
        cv2.circle(vis, tuple(int(v) for v in pts[0]), 3 * t, (255, 0, 255), -1)  # TL marker
    label = (f"{res.get('profile')} found={res.get('found')} {res.get('roi_source', '')} "
             f"cov={res.get('coverage', 0):.2f} n={res.get('n_cards', 0)}")
    cv2.putText(vis, label, (10, 10 + 12 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * t, (0, 0, 255), max(1, t // 2))
    return vis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trap-kind", default="sticky", choices=list(TRAP_KINDS))
    ap.add_argument("--profile", default="yellow", choices=["yellow", "white", "auto"])
    a = ap.parse_args(argv)
    img = cv2.imread(a.image)
    if img is None:
        print(f"cannot read {a.image}", file=sys.stderr)
        return 2
    res = find_card(img, trap_kind=a.trap_kind, profile=a.profile, out_dir=a.out, name=Path(a.image).stem)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
