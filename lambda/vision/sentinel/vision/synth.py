"""Synthetic yellow sticky cards for tests and the card-finding benchmark.

The 2026-10 FlyBot export has no photo of a yellow card (docs/evidence/vision/card_check.md), so the
card finder is measured on known-geometry composites: a yellow card with dark "flies" pasted in
perspective onto a background (a real export frame or noise). Ground truth corners are exact.
"""

from __future__ import annotations

import cv2
import numpy as np

CARD_BGR = (40, 215, 235)  # saturated yellow (H ~ 27 in OpenCV HSV)


def make_card(w: int = 400, h: int = 250, n_flies: int = 20, rng: np.random.Generator | None = None,
              grid: bool = True) -> tuple[np.ndarray, list[list[float]]]:
    """A flat yellow card with `n_flies` dark elliptic blobs; returns (image, fly boxes xyxy)."""
    rng = rng or np.random.default_rng(0)
    card = np.zeros((h, w, 3), np.uint8)
    card[:] = CARD_BGR
    noise = rng.normal(0, 4, card.shape)
    card = np.clip(card.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if grid:  # printed grid lines of real cards give texture for registration
        for x in range(0, w, max(8, w // 10)):
            cv2.line(card, (x, 0), (x, h - 1), (30, 170, 190), 1)
        for y in range(0, h, max(8, h // 6)):
            cv2.line(card, (0, y), (w - 1, y), (30, 170, 190), 1)
    boxes = []
    for _ in range(n_flies):
        cx, cy = rng.uniform(15, w - 15), rng.uniform(15, h - 15)
        a, b = rng.uniform(4, 9), rng.uniform(2.5, 5)
        ang = rng.uniform(0, 180)
        cv2.ellipse(card, (int(cx), int(cy)), (int(a), int(b)), ang, 0, 360, (25, 40, 60), -1)
        cv2.ellipse(card, (int(cx + 2), int(cy)), (int(a * 0.7), int(b * 1.6)), ang + 90, 0, 360, (90, 120, 130), 1)
        r = max(a, b) + 2
        boxes.append([cx - r, cy - r, cx + r, cy + r])
    return card, boxes


def paste_card(background: np.ndarray, card: np.ndarray, quad) -> np.ndarray:
    """Warp `card` onto `background` so its corners (TL, TR, BR, BL) land on `quad` (pixels)."""
    h, w = card.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    H, W = background.shape[:2]
    warped = cv2.warpPerspective(card, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    out = background.copy()
    out[mask > 127] = warped[mask > 127]
    return out


def random_quad(W: int, H: int, rng: np.random.Generator, coverage=(0.12, 0.5), max_tilt: float = 0.18,
                aspect: float = 1.6) -> np.ndarray:
    """A random perspective quad fully inside a W x H frame (TL, TR, BR, BL)."""
    for _ in range(200):
        cov = rng.uniform(*coverage)
        area = cov * W * H
        ch = np.sqrt(area / aspect)
        cw = ch * aspect
        if cw >= W * 0.95 or ch >= H * 0.95:
            continue
        cx = rng.uniform(cw / 2 + 5, W - cw / 2 - 5)
        cy = rng.uniform(ch / 2 + 5, H - ch / 2 - 5)
        rot = rng.uniform(-35, 35)
        base = cv2.boxPoints(((cx, cy), (cw, ch), rot))  # BL, TL, TR, BR order from OpenCV
        jitter = rng.uniform(-max_tilt, max_tilt, (4, 2)) * np.array([cw, ch]) * 0.5
        q = (base + jitter).astype(np.float32)
        inside = (q[:, 0] > 2).all() and (q[:, 0] < W - 3).all() and (q[:, 1] > 2).all() and (q[:, 1] < H - 3).all()
        if inside and cv2.isContourConvex(q.reshape(-1, 1, 2)):
            from sentinel.vision.card import order_corners

            return order_corners(q)
    raise RuntimeError("could not place a quad")


def noise_background(W: int = 1280, H: int = 720, rng: np.random.Generator | None = None) -> np.ndarray:
    """Grey-green textured background (leaves-like), no yellow."""
    rng = rng or np.random.default_rng(0)
    small = rng.integers(30, 140, (H // 16 + 1, W // 16 + 1, 3)).astype(np.uint8)
    small[..., 2] = (small[..., 2] * 0.6).astype(np.uint8)  # less red -> no yellow hue
    bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(bg, (5, 5), 0)
