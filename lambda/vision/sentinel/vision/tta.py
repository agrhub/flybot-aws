"""Test-time views for YOLO26: the same photo mirrored, run through the model again, boxes mapped back.

Most flies the detector misses on the synthetic lure and sticky photos get no box, or one under the score floor, in
the photo as it is, and many of them get one in a mirrored copy (eval/results/detector_ab.json, `tta_*` variants;
eval/results/detector_misses.json). Each view costs one forward per pass (the whole frame, or each tile window); the
views' boxes join the photo's own before NMS, so a fly seen in two views keeps its best score and counts once.

    DETECTOR_TTA            off (default) | a comma list of views for every trap kind: hflip, vflip, rot90
    DETECTOR_TTA_LURE, DETECTOR_TTA_STICKY
                            the views for one trap kind instead of DETECTOR_TTA (unset = the same; off = none).
                            Sticky photos are tiled (DETECTOR_TILES=sticky), so a view there costs one forward per window

Measured (v4 weights, production settings otherwise): lure hflip,vflip: synthetic recall 0.819 -> 0.876 at the same
floor, no false fly on 320 real empty-trap photos; rot90 adds 72 there, so it is not recommended. Sticky hflip: 0.800
-> 0.852 (8 forwards a photo instead of 4).

Boxes are `{xywh, score, class_id}` in the pixels of the image the view was made from.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

VIEWS = ("hflip", "vflip", "rot90")


def parse_views(raw: str | None) -> tuple[str, ...]:
    raw = (raw or "").strip().lower()
    if raw in ("", "off", "0", "false", "no", "none"):
        return ()
    views = tuple(v.strip() for v in raw.split(",") if v.strip())
    bad = [v for v in views if v not in VIEWS]
    if bad:
        raise ValueError(f"DETECTOR_TTA views must be in {VIEWS}, not {bad}")
    return views


@dataclass(frozen=True)
class Tta:
    views: tuple[str, ...] = ()
    lure: tuple[str, ...] | None = None
    sticky: tuple[str, ...] | None = None

    @classmethod
    def from_env(cls, env: dict | None = None) -> Tta:
        env = os.environ if env is None else env

        def kind(name: str) -> tuple[str, ...] | None:
            raw = env.get(name)
            return None if raw is None or str(raw).strip() == "" else parse_views(raw)

        return cls(views=parse_views(env.get("DETECTOR_TTA")), lure=kind("DETECTOR_TTA_LURE"),
                   sticky=kind("DETECTOR_TTA_STICKY"))

    def views_for(self, trap_kind: str | None) -> tuple[str, ...]:
        own = {"lure": self.lure, "sticky": self.sticky}.get(trap_kind or "")
        return self.views if own is None else own

    def as_dict(self) -> dict:
        return asdict(self)


def view(img: np.ndarray, name: str) -> np.ndarray:
    """The image as the model sees it in view `name`."""
    if name == "hflip":
        return np.ascontiguousarray(img[:, ::-1])
    if name == "vflip":
        return np.ascontiguousarray(img[::-1])
    if name == "rot90":  # clockwise
        return np.ascontiguousarray(np.rot90(img, k=-1))
    raise ValueError(f"unknown view {name!r}")


def unmap(dets: list[dict], name: str, width: int, height: int) -> list[dict]:
    """Boxes found in view `name` of a `width` x `height` image -> that image's pixels."""
    out = []
    for d in dets:
        x, y, w, h = d["xywh"]
        if name == "hflip":
            box = [width - x - w, y, w, h]
        elif name == "vflip":
            box = [x, height - y - h, w, h]
        elif name == "rot90":  # the source point (sx, sy) is at (height - sy, sx) in the clockwise view
            box = [y, height - x - w, h, w]
        else:
            raise ValueError(f"unknown view {name!r}")
        out.append({**d, "xywh": [round(float(v), 1) for v in box]})
    return out
