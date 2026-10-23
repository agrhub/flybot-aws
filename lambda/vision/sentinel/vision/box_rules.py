"""Detector post-processing shared by both YOLO26 heads: class-agnostic NMS and the fruit-fly box rules.

The rules reject boxes that cannot be one fruit fly without retraining the model. Defaults and their evidence
(`eval/results/yolo26_box_rules.json`, produced by `eval/detector_box_rules.py`; side = sqrt(w * h) of a box
divided by sqrt(W * H) of the photo):

    DETECTOR_IOU                0.45  class-agnostic NMS for both heads: 26 of 73,980 labelled train flies overlap a
                                      neighbour at IoU >= 0.45, while the end-to-end head left duplicate pairs
    DETECTOR_MIN_SCORE          0.35  valid split after NMS: F1 0.9526 (0.30: 0.9527, 0.25: 0.9523) with higher
                                      precision; 2.4 % of the valid true positives score lower
    DETECTOR_MAX_SIDE           0.12  167 of 73,980 train flies are larger (close-up pictures); the round lens of
                                      the lure trap (BE4111159, bundled sample) measures 0.149-0.158 and scores 0.43-0.66
    DETECTOR_MIN_SIDE           0.004 no labelled fly is smaller (train minimum 0.0086)
    DETECTOR_MAX_MEDIAN_RATIO   3.0   side / median side of the photo's boxes: 11 of 73,834 train flies are above
    DETECTOR_MIN_MEDIAN_RATIO   0.25  24 of 73,834 train flies are below
    DETECTOR_MEDIAN_MIN_BOXES   3     the median rules need at least this many boxes left in the photo
    DETECTOR_MAX_ASPECT         4.0   long side / short side: 165 of 73,980 train flies are above; best F1 on valid
    DETECTOR_MIN_SCORE_LURE, DETECTOR_MIN_SCORE_STICKY
                                -     score floor for one trap kind instead of DETECTOR_MIN_SCORE (unset = the same)
    DETECTOR_BOX_RULES          on    "off" keeps NMS and drops every rule (score floor = DETECTOR_CONF)

Boxes are `{xywh, score, class_id}` in source-image pixels, as the detectors return them.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

REASONS = ("low_score", "too_large", "too_small", "aspect", "large_for_photo", "small_for_photo")


@dataclass(frozen=True)
class BoxRules:
    min_score: float = 0.35
    max_side: float = 0.12
    min_side: float = 0.004
    max_median_ratio: float = 3.0
    min_median_ratio: float = 0.25
    median_min_boxes: int = 3
    max_aspect: float = 4.0
    min_score_lure: float | None = None
    min_score_sticky: float | None = None

    @classmethod
    def from_env(cls, env: dict | None = None) -> BoxRules | None:
        """The rules from the DETECTOR_* variables (defaults above); None when DETECTOR_BOX_RULES=off.
        A value that is not a number raises ValueError (the detector then fails to load, like a bad DETECTOR_CONF)."""
        env = os.environ if env is None else env
        if str(env.get("DETECTOR_BOX_RULES", "on")).strip().lower() in ("off", "0", "false", "no"):
            return None
        d = cls()

        def num(name: str, default, kind=float):
            raw = env.get(name)
            return default if raw in (None, "") else kind(raw)

        return cls(min_score=num("DETECTOR_MIN_SCORE", d.min_score), max_side=num("DETECTOR_MAX_SIDE", d.max_side),
                   min_side=num("DETECTOR_MIN_SIDE", d.min_side),
                   max_median_ratio=num("DETECTOR_MAX_MEDIAN_RATIO", d.max_median_ratio),
                   min_median_ratio=num("DETECTOR_MIN_MEDIAN_RATIO", d.min_median_ratio),
                   median_min_boxes=num("DETECTOR_MEDIAN_MIN_BOXES", d.median_min_boxes, int),
                   max_aspect=num("DETECTOR_MAX_ASPECT", d.max_aspect),
                   min_score_lure=num("DETECTOR_MIN_SCORE_LURE", None), min_score_sticky=num("DETECTOR_MIN_SCORE_STICKY", None))

    def for_kind(self, trap_kind: str | None) -> BoxRules:
        """These rules with the score floor of `trap_kind` ("lure" | "sticky") when one is set."""
        floor = {"lure": self.min_score_lure, "sticky": self.min_score_sticky}.get(trap_kind or "")
        return self if floor is None else replace(self, min_score=floor)

    def as_dict(self) -> dict:
        return asdict(self)


def nms(dets: list[dict], iou: float) -> list[dict]:
    """Class-agnostic NMS (`cv2.dnn.NMSBoxes`) on `{xywh, score}` boxes; kept boxes in descending score order."""
    if len(dets) < 2:
        return list(dets)
    idx = cv2.dnn.NMSBoxes([list(map(float, d["xywh"])) for d in dets], [float(d["score"]) for d in dets], 0.0, float(iou))
    return [dets[i] for i in np.array(idx, dtype=int).reshape(-1)]


def _side(d: dict) -> float:
    w, h = d["xywh"][2], d["xywh"][3]
    return float(max(w, 0.0) * max(h, 0.0)) ** 0.5


def apply_rules(dets: list[dict], width: int, height: int, rules: BoxRules | None) -> tuple[list[dict], dict]:
    """Drop boxes that cannot be one fruit fly. Returns (kept, {reason: count}) with every reason in REASONS.

    Per box: score, side relative to the photo, aspect ratio. Then, when at least `median_min_boxes` boxes are left,
    side relative to the median side of those boxes (a lizard among flies, a speck among flies)."""
    rejected = dict.fromkeys(REASONS, 0)
    if rules is None:
        return list(dets), rejected
    root = float(width * height) ** 0.5 or 1.0
    kept = []
    for d in dets:
        w, h = float(d["xywh"][2]), float(d["xywh"][3])
        side = _side(d) / root
        if d["score"] < rules.min_score:
            rejected["low_score"] += 1
        elif side > rules.max_side:
            rejected["too_large"] += 1
        elif side < rules.min_side or min(w, h) <= 0:
            rejected["too_small"] += 1
        elif max(w, h) / min(w, h) > rules.max_aspect:
            rejected["aspect"] += 1
        else:
            kept.append(d)
    if len(kept) >= max(1, rules.median_min_boxes):
        sides = [_side(d) for d in kept]
        med = float(np.median(sides))
        out = []
        for d, s in zip(kept, sides, strict=True):
            if s > rules.max_median_ratio * med:
                rejected["large_for_photo"] += 1
            elif s < rules.min_median_ratio * med:
                rejected["small_for_photo"] += 1
            else:
                out.append(d)
        kept = out
    return kept, rejected
