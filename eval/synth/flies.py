"""Fly bank: single fruit flies cut out of the camos/fruit-fly v3 dataset (Roboflow Universe, CC BY 4.0).

The dataset photographs flies on a white disc, so each fly is cut out as a *transmittance* map: the crop divided by
the local background colour. Pasting multiplies the target floor by that map under a soft alpha mask, which keeps
the fly's own shading, translucent wings and soft edges and takes the target's lighting for free.

Only isolated, fully visible flies on a clean bright background are kept: no other labelled box within a margin
around the fly, box not touching the photo edge, background ring bright, unsaturated and even. By default the
crops come from the `test` and `valid` splits, which the detector never trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "camos-ai-camera" / "models" / "fruit-fly-v3"


@dataclass
class FlyCrop:
    """One fly: `ratio` HxWx3 float32 transmittance (1 = background), `alpha` HxW float32 in [0, 1]."""

    id: str  # "<split>/<image file>#<label row>"
    ratio: np.ndarray
    alpha: np.ndarray
    box: tuple[float, float, float, float]  # labelled box inside the crop, xyxy pixels
    bg_bgr: tuple[float, float, float] = field(default=(255.0, 255.0, 255.0))

    @property
    def long_side(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(x1 - x0, y1 - y0)


def read_labels(path: Path, w: int, h: int) -> list[list[float]]:
    """YOLO label rows -> xyxy pixel boxes. Polygon rows (31 train files) become their bounding box."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        v = [float(t) for t in line.split()]
        if len(v) == 5:
            _, cx, cy, bw, bh = v
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        elif len(v) >= 7:
            xs, ys = v[1::2], v[2::2]
            boxes.append([min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h])
    return boxes


def _expand(b, pad: float) -> list[float]:
    w, h = b[2] - b[0], b[3] - b[1]
    return [b[0] - pad * w, b[1] - pad * h, b[2] + pad * w, b[3] + pad * h]


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def cut_out(img: np.ndarray, box, pad: float = 0.35, t_lo: float = 10.0, t_hi: float = 38.0) -> FlyCrop | None:
    """Cut the fly in `box` (xyxy) out of `img` as transmittance + soft alpha; None when the ring is not clean."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _expand(box, pad)
    x0, y0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
    x1, y1 = int(min(W, np.ceil(x1))), int(min(H, np.ceil(y1)))
    crop = img[y0:y1, x0:x1].astype(np.float32)
    if crop.shape[0] < 12 or crop.shape[1] < 12:
        return None
    ring = np.ones(crop.shape[:2], bool)
    bx0, by0, bx1, by1 = (round(box[0] - x0), round(box[1] - y0),
                          round(box[2] - x0), round(box[3] - y0))
    ring[max(0, by0):by1, max(0, bx0):bx1] = False
    if ring.sum() < 40:
        return None
    ring_px = crop[ring]
    bg = np.median(ring_px, axis=0)
    hsv_bg = cv2.cvtColor(np.uint8([[np.clip(bg, 0, 255)]]), cv2.COLOR_BGR2HSV)[0, 0]
    ring_std = float(np.mean(np.abs(ring_px - bg)))
    if hsv_bg[2] < 150 or hsv_bg[1] > 70 or ring_std > 14:
        return None  # dark, coloured or busy surroundings: the cut-out would carry them along
    lab = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(np.uint8([[np.clip(bg, 0, 255)]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    dist = np.linalg.norm(lab - lab_bg, axis=2)
    dist = np.maximum(dist, (lab_bg[0] - lab[..., 0]) * 1.2)  # darker than the disc counts double
    alpha = np.clip((dist - t_lo) / (t_hi - t_lo), 0, 1)
    core = (alpha > 0.5).astype(np.uint8)
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n <= 1:
        return None
    inside = np.zeros_like(core)
    inside[max(0, by0):by1, max(0, bx0):bx1] = 1
    keep = np.zeros_like(core)
    for i in range(1, n):  # components that touch the labelled box are the fly; the rest is dirt or a neighbour
        comp = lab_cc == i
        if (comp & (inside > 0)).any() and st[i, cv2.CC_STAT_AREA] >= 6:
            keep[comp] = 1
    if keep.sum() < 15:
        return None
    if keep[0].any() or keep[-1].any() or keep[:, 0].any() or keep[:, -1].any():
        return None  # the "fly" runs into the crop edge: bait pad, disc rim or a neighbour came along
    if float(alpha[max(0, by0):by1, max(0, bx0):bx1].mean()) > 0.8:
        return None  # a fly never fills its box: this is a dark patch, not a fly
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = alpha * keep
    alpha[~(inside > 0) & (cv2.dilate(inside, np.ones((7, 7), np.uint8)) == 0)] = 0  # nothing far outside the box
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    ratio = np.clip(crop / np.maximum(bg, 1.0), 0.0, 1.0)  # a fly only darkens the floor
    core_px = ratio.mean(axis=2)[alpha > 0.5]
    if core_px.size < 10 or float(np.percentile(core_px, 5)) > 0.45:
        return None  # faint: a fly in flight (motion blur) or out of focus
    return FlyCrop(id="", ratio=ratio.astype(np.float32), alpha=alpha.astype(np.float32),
                   box=(box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0), bg_bgr=tuple(float(c) for c in bg))


def load_bank(dataset: Path = DEFAULT_DATASET, splits=("test", "valid"), max_crops: int = 1200, seed: int = 7,
              min_side: float = 18.0, max_side: float = 90.0, isolation: float = 0.35) -> list[FlyCrop]:
    """Deterministic list of clean single-fly crops (`max_crops` at most, sampled with `seed`)."""
    dataset = Path(dataset)
    rng = np.random.default_rng(seed)
    cands: list[tuple[str, Path, int]] = []
    per_image: dict[Path, list[list[float]]] = {}
    for split in splits:
        img_dir, lab_dir = dataset / split / "images", dataset / split / "labels"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"dataset split not found: {img_dir} (models/dataset.md says where it lives)")
        for p in sorted(img_dir.glob("*.jpg")):
            boxes = read_labels(lab_dir / (p.stem + ".txt"), 640, 640)
            per_image[p] = boxes
            for i, b in enumerate(boxes):
                side = max(b[2] - b[0], b[3] - b[1])
                if not (min_side <= side <= max_side) or min(b[0], b[1]) < 3 or max(b[2], b[3]) > 637:
                    continue
                eb = _expand(b, isolation)
                if any(_overlap(eb, _expand(o, isolation)) for j, o in enumerate(boxes) if j != i):
                    continue
                cands.append((f"{split}/{p.name}#{i}", p, i))
    clean: list[FlyCrop] = []
    img, img_path = None, None
    for cid, p, i in cands:  # grouped by image: each photo is decoded once
        if p != img_path:
            img, img_path = cv2.imread(str(p), cv2.IMREAD_COLOR), p
        fc = cut_out(img, per_image[p][i])
        if fc is not None:
            fc.id = cid
            clean.append(fc)
    if not clean:
        raise RuntimeError(f"no clean fly crop found in {dataset} {splits}")
    return [clean[k] for k in rng.permutation(len(clean))[:max_crops]]
