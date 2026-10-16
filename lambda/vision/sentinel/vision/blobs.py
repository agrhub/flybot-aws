"""Interim box source: dark compact blobs on a bright trap surface (classic, no model).

The 2026-10 FlyBot export carries counts only, not the production model's boxes, and op 5
(`detect.py`, T1.4) is not built yet. Op 4 needs boxes to match, so until then this black-hat
blob finder stands in. It is a stand-in, not a detector: it finds any small dark thing (flies,
debris, stain specks) and is never reported as an insect count.
"""

from __future__ import annotations

import cv2
import numpy as np

# Blob area as a share of the image: flies in the Roboflow camos set are ~33x44 px on 640x640 (~0.35 %)
MIN_AREA_SHARE = 0.0003
MAX_AREA_SHARE = 0.02
MIN_CONTRAST = 25  # black-hat response (grey levels) below this is background texture


def detect_dark_blobs(img: np.ndarray, min_area_share: float = MIN_AREA_SHARE, max_area_share: float = MAX_AREA_SHARE,
                      min_contrast: int = MIN_CONTRAST) -> dict:
    """Boxes [x1, y1, x2, y2] (pixels of `img`) of dark compact blobs; JSON-serializable dict."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(9, round(0.06 * max(h, w)) | 1)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    otsu, _ = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(min_contrast), float(otsu))
    _, bw = cv2.threshold(blackhat, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    area_img = float(h * w)
    boxes = []
    for i in range(1, n):
        x, y, bw_, bh_, a = (int(v) for v in stats[i])
        if not (min_area_share * area_img <= a <= max_area_share * area_img):
            continue
        if max(bw_, bh_) / max(1, min(bw_, bh_)) > 5:  # thin lines: grid, cracks, card edges
            continue
        boxes.append([x, y, x + bw_, y + bh_])
    return {"boxes": boxes, "n": len(boxes), "threshold": round(thr, 1), "kernel": k, "source": "classic_blobs", "evidence": {}}
