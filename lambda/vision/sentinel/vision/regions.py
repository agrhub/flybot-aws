"""Where a detector box may count: overlay masks and the trap surface ROI, applied after NMS and before the box rules.

    DETECTOR_MASK       off (default) | osd | x0,y0,x1,y1[;x0,y0,x1,y1...]
                        drop boxes whose centre lies in one of these rectangles (fractions of the frame). `osd` =
                        the camera overlays: the EZVIZ logo (top-left) and the date / time text along the bottom-left,
                        where the synthetic photos also carry their SIMULATED stamp
    DETECTOR_ROI        off (default) | lure | all
                        drop boxes whose centre lies outside the trap surface op 1 found (sentinel.vision.card: the
                        lure tray floor polygon, the sticky card quad); `lure` = lure traps only. No surface found ->
                        no ROI filter (the whole frame counts, as before)
    DETECTOR_ROI_RIM    0      also drop boxes whose centre is inside the surface but closer to its edge than this
                        fraction of sqrt(surface area) (the tray's rim and wall)

Boxes are `{xywh, score, class_id}` in frame pixels.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# The EZVIZ logo sits in the top-left 12 % x 8 % of the frame; its date overlay along the bottom-left (bottom 10 %,
# left 30 %), and the synthetic photos' SIMULATED stamp covers up to 42 % of the width there (eval/synth/composite.py)
OSD = ((0.0, 0.0, 0.15, 0.09), (0.0, 0.92, 0.45, 1.0))
ROI_MODES = ("off", "lure", "all")
REASONS = ("osd_mask", "outside_roi", "roi_rim")


@dataclass(frozen=True)
class Regions:
    masks: tuple[tuple[float, float, float, float], ...] = ()
    roi: str = "off"
    rim: float = 0.0

    def __post_init__(self):
        if self.roi not in ROI_MODES:
            raise ValueError(f"DETECTOR_ROI must be one of {ROI_MODES}, not {self.roi!r}")

    @classmethod
    def from_env(cls, env: dict | None = None) -> Regions:
        env = os.environ if env is None else env
        return cls(masks=parse_masks(env.get("DETECTOR_MASK")), roi=str(env.get("DETECTOR_ROI") or "off").strip().lower(),
                   rim=float(env.get("DETECTOR_ROI_RIM") or 0.0))

    def wants_roi(self, trap_kind: str | None) -> bool:
        return self.roi == "all" or (self.roi == "lure" and trap_kind == "lure")

    def as_dict(self) -> dict:
        return asdict(self)


def parse_masks(raw: str | None) -> tuple[tuple[float, float, float, float], ...]:
    """`off` / empty -> none, `osd` -> OSD, else `x0,y0,x1,y1;...` fractions of the frame."""
    raw = (raw or "").strip().lower()
    if raw in ("", "off", "0", "false", "no", "none"):
        return ()
    if raw == "osd":
        return OSD
    out = []
    for part in raw.split(";"):
        vals = tuple(float(v) for v in part.split(","))
        if len(vals) != 4 or not (0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1):
            raise ValueError(f"DETECTOR_MASK rectangle must be x0,y0,x1,y1 fractions with x0 < x1, y0 < y1: {part!r}")
        out.append(vals)
    return tuple(out)


def _centre(d: dict) -> tuple[float, float]:
    x, y, w, h = d["xywh"]
    return x + w / 2.0, y + h / 2.0


def apply_regions(dets: list[dict], width: int, height: int, regions: Regions | None, polygon=None,
                  trap_kind: str | None = None) -> tuple[list[dict], dict]:
    """Drop boxes in a mask, outside the surface `polygon` ([[x, y], ...] frame px; None = none found) or on its rim.
    Returns (kept, {reason: count}) with every reason in REASONS."""
    removed = dict.fromkeys(REASONS, 0)
    if regions is None:
        return list(dets), removed
    poly = None
    if polygon is not None and len(polygon) >= 3 and regions.wants_roi(trap_kind):
        poly = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
        rim_px = regions.rim * float(abs(cv2.contourArea(poly))) ** 0.5
    kept = []
    for d in dets:
        cx, cy = _centre(d)
        fx, fy = cx / max(width, 1), cy / max(height, 1)
        if any(x0 <= fx <= x1 and y0 <= fy <= y1 for x0, y0, x1, y1 in regions.masks):
            removed["osd_mask"] += 1
            continue
        if poly is not None:
            dist = cv2.pointPolygonTest(poly, (float(cx), float(cy)), True)  # > 0 inside, distance to the edge
            if dist < 0:
                removed["outside_roi"] += 1
                continue
            if dist < rim_px:
                removed["roi_rim"] += 1
                continue
        kept.append(d)
    return kept, removed


def roi_rect(polygon, width: int, height: int, pad: float = 0.02) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of the surface polygon, padded by `pad` x frame, clipped; None without one."""
    if polygon is None or len(polygon) < 3:
        return None
    a = np.asarray(polygon, np.float32).reshape(-1, 2)
    px, py = pad * width, pad * height
    x0, y0 = max(0, int(a[:, 0].min() - px)), max(0, int(a[:, 1].min() - py))
    x1, y1 = min(width, int(np.ceil(a[:, 0].max() + px))), min(height, int(np.ceil(a[:, 1].max() + py)))
    return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None
