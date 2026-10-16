"""Evidence images for the vision ops: small JPEGs, privacy-safe outside the trap.

Every op writes its evidence through `save_jpeg` (long side <= 800 px, quality 80) so the
evidence folder stays small. Photos may show people (one export camera looks at a building),
and no face detector is wired in yet, so `pixelate_outside` coarsens everything outside the trap
quad before the image is stored: the trap stays sharp, a face at a window does not.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 800
JPEG_QUALITY = 80
# Pixelation block, as a share of the long side (16 px on an 800 px image)
PIXEL_BLOCK_SHARE = 0.02


def fit(img: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    """Downscale so the long side is at most `max_side`; returns (image, scale applied)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return img, scale


def save_jpeg(img: np.ndarray, path: str | Path, max_side: int = MAX_SIDE, quality: int = JPEG_QUALITY) -> str:
    """Write `img` as a small JPEG and return the path as a POSIX string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    small, _ = fit(img, max_side)
    if not cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
        raise OSError(f"could not write {path}")
    return path.as_posix()


def pixelate(img: np.ndarray, block_share: float = PIXEL_BLOCK_SHARE) -> np.ndarray:
    """Coarse mosaic of the whole image (block = `block_share` of the long side)."""
    h, w = img.shape[:2]
    block = max(4, round(max(h, w) * block_share))
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def pixelate_outside(img: np.ndarray, quad: list | np.ndarray | None) -> np.ndarray:
    """Pixelate everything outside `quad` (pixel corners); the whole image when quad is None."""
    coarse = pixelate(img)
    if quad is None:
        return coarse
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(np.asarray(quad, np.float32)).astype(np.int32), 255)
    out = coarse.copy()
    out[mask > 0] = img[mask > 0]
    return out
