"""Tiled detection for small flies: the photo is cut into `grid` x `grid` overlapping windows, each letterboxed to the
detector's input, so a fly is seen about `grid` times larger than on the whole frame.

YOLO26n letterboxed whole-frame to 640 finds almost no fly whose long side is under ~20 px at its input
(eval/SYNTHETIC_RESULTS.md, recall by fly size). On the ESP32 sticky card (15 x 25 cm across a 2304 px frame) flies
are ~22 px at 640 and the whole-frame recall is a few percent; on 2x2 tiles most are found. On lure traps the flies
are large and tiles add seam duplicates, so tiling is conditional:

    DETECTOR_TILES          off (default) | on | sticky | auto
                            on: always tile; sticky: tile photos of sticky traps (FlyBot Camera.trapKind);
                            auto: a whole-frame pass first, then tile when its boxes are small at the detector input
                            (median long side < DETECTOR_TILE_AUTO_PX) or when it found none on a frame larger than
                            DETECTOR_TILE_AUTO_MIN_SIDE x the input
    DETECTOR_TILE_GRID      2      windows per side (grid^2 forwards, plus the whole-frame pass for auto)
    DETECTOR_TILE_OVERLAP   0.12   overlap between neighbouring windows, as a fraction of the frame / grid
    DETECTOR_TILE_EDGE_PX   2      a box within this many detector-input pixels of an inner window edge is cut by it
                            and dropped (the neighbouring window sees the whole fly); image borders are not inner edges
    DETECTOR_TILE_AUTO_PX   24     auto: median long side (detector-input px of the whole-frame pass) below which tiles run
    DETECTOR_TILE_AUTO_MIN_SIDE 2  auto with no whole-frame box: tile when the frame's long side >= this x the input
    DETECTOR_TILE_KEEP_FULL on     when tiles ran after a whole-frame pass (auto), its boxes join the merge (big
                            flies cut by every window survive)

Window boxes are mapped to full-frame pixels, merged with class-agnostic NMS at the detector's IoU, and the box rules
then run on full-frame coordinates, as without tiles.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

from sentinel.vision.box_rules import nms

MODES = ("off", "on", "sticky", "auto")


@dataclass(frozen=True)
class Tiling:
    mode: str = "off"
    grid: int = 2
    overlap: float = 0.12
    edge_px: float = 2.0
    auto_px: float = 24.0
    auto_min_side: float = 2.0
    keep_full: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"DETECTOR_TILES must be one of {MODES}, not {self.mode!r}")
        if self.grid < 1 or not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"tile grid must be >= 1 and overlap in [0, 1): {self.grid}, {self.overlap}")

    @classmethod
    def from_env(cls, env: dict | None = None) -> Tiling:
        env = os.environ if env is None else env
        d = cls()

        def num(name: str, default, kind=float):
            raw = env.get(name)
            return default if raw in (None, "") else kind(raw)

        mode = str(env.get("DETECTOR_TILES") or d.mode).strip().lower()
        keep = str(env.get("DETECTOR_TILE_KEEP_FULL", "on")).strip().lower() not in ("off", "0", "false", "no")
        return cls(mode=mode, grid=num("DETECTOR_TILE_GRID", d.grid, int), overlap=num("DETECTOR_TILE_OVERLAP", d.overlap),
                   edge_px=num("DETECTOR_TILE_EDGE_PX", d.edge_px), auto_px=num("DETECTOR_TILE_AUTO_PX", d.auto_px),
                   auto_min_side=num("DETECTOR_TILE_AUTO_MIN_SIDE", d.auto_min_side), keep_full=keep)

    def as_dict(self) -> dict:
        return asdict(self)


def windows(width: int, height: int, grid: int, overlap: float,
            rect: tuple[int, int, int, int] | None = None) -> list[tuple[int, int, int, int]]:
    """`grid` x `grid` windows (x0, y0, x1, y1, frame pixels) covering the frame, or only `rect` (x0, y0, x1, y1)
    when given (e.g. the bounding box of the trap floor); neighbours share `overlap` x area / grid."""
    rx0, ry0, rx1, ry1 = rect or (0, 0, width, height)
    rw, rh = rx1 - rx0, ry1 - ry0
    tw = min(rw, round(rw / grid * (1 + overlap)))
    th = min(rh, round(rh / grid * (1 + overlap)))
    out = []
    for gy in range(grid):
        for gx in range(grid):
            x0 = min(rw - tw, max(0, round(gx * rw / grid - overlap * rw / grid / 2)))
            y0 = min(rh - th, max(0, round(gy * rh / grid - overlap * rh / grid / 2)))
            out.append((rx0 + x0, ry0 + y0, rx0 + x0 + tw, ry0 + y0 + th))
    return out


def cut_by_inner_edge(xywh: list[float], win: tuple[int, int, int, int], width: int, height: int, margin: float,
                      rect: tuple[int, int, int, int] | None = None) -> bool:
    """True when the box (window pixels) reaches an edge of `win` that is inside the tiled area (the frame, or
    `rect`), i.e. the fly may go on into the neighbouring window."""
    x, y, w, h = xywh
    x0, y0, x1, y1 = win
    rx0, ry0, rx1, ry1 = rect or (0, 0, width, height)
    return ((x0 > rx0 and x <= margin) or (y0 > ry0 and y <= margin)
            or (x1 < rx1 and x + w >= (x1 - x0) - margin) or (y1 < ry1 and y + h >= (y1 - y0) - margin))


def merge(window_dets: list[list[dict]], wins: list[tuple[int, int, int, int]], width: int, height: int, iou: float,
          margins: list[float], full: list[dict] | None = None,
          rect: tuple[int, int, int, int] | None = None) -> tuple[list[dict], int]:
    """Window boxes (window pixels) -> full-frame boxes: drop the ones cut by an inner window edge, shift, add the
    whole-frame boxes when given, class-agnostic NMS. `rect` = the tiled area when it is not the frame.
    Returns (boxes, number dropped as cut)."""
    dets, cut = list(full or []), 0
    for boxes, win, margin in zip(window_dets, wins, margins, strict=True):
        for d in boxes:
            if cut_by_inner_edge(d["xywh"], win, width, height, margin, rect):
                cut += 1
                continue
            x, y, w, h = d["xywh"]
            dets.append({**d, "xywh": [round(x + win[0], 1), round(y + win[1], 1), w, h]})
    return nms(dets, iou), cut


def median_long_side(dets: list[dict], scale: float) -> float | None:
    """Median long side of the boxes at the detector input (`scale` = input px per source px); None without boxes."""
    if not dets:
        return None
    return float(np.median([max(d["xywh"][2], d["xywh"][3]) for d in dets])) * scale


def decide(t: Tiling, trap_kind: str | None, full: list[dict] | None, scale: float, long_side: int,
           size: int) -> tuple[bool, str]:
    """Whether to tile this photo, and why. `full` is the whole-frame pass (auto only), `scale` its input/source ratio."""
    if t.mode == "off" or t.grid < 2:
        return False, "off"
    if t.mode == "on":
        return True, "on"
    if t.mode == "sticky":
        return trap_kind == "sticky", f"trap_kind={trap_kind}"
    med = median_long_side(full or [], scale)
    if med is None:
        big = long_side >= t.auto_min_side * size
        return big, "no_box_large_frame" if big else "no_box_small_frame"
    return med < t.auto_px, f"median_px={med:.1f}"
