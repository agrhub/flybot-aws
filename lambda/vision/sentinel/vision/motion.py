"""Op 8: what changed in front of the trap between two photos of one camera, and face blurring.

    uv run python -m sentinel.vision.motion PREV CUR [--history H1 H2 ...] [--out DIR]

`detect_motion(prev, cur)` returns a JSON-serializable dict:

    occluded_by_large_object (bool), occlusion_undetermined (bool), occluder {bbox (pixels of cur), bbox_norm, area_share, in_frame
    ("cur" | "prev"), solidity}, camera_moved (bool), moved_reasons, shift {dx, dy, px, share, rotation_deg,
    scale, source}, registration {ok, inliers, matches, method}, global_change (bool), changed_share,
    foreground ("ratio+chroma" | "mog2"), time_ms, evidence {fg_mask, overlay}

* Foreground: the previous photo is aligned onto the current one with the registration homography
  (SIFT, op 3) when it is trustworthy; each frame is divided by its own heavy Gaussian blur (removes
  the IR LED hot spot, vignetting and smooth sun gradients) and the absolute difference of these ratio
  images is thresholded (`RATIO_T`), OR'd with a Lab chroma difference (`CHROMA_T`, skipped in IR
  mode). With `history` (>= 3 earlier frames of the same camera) a
  `cv2.createBackgroundSubtractorMOG2` model is learnt on them and the current frame is classified with
  learning rate 0 instead.
* `occluded_by_large_object`: the largest foreground blob (after morphological close/open) covers at
  least `LARGE_SHARE` of the frame and is compact (solidity >= `MIN_SOLIDITY`): a hand, a person, an
  animal, a leaf. Flies are far below that size. When most of the frame changed (`GLOBAL_SHARE`), it is a
  lighting / IR switch (`global_change`), not an object. The blob is attributed to the frame where it
  contrasts more with a ring of its surroundings (`in_frame`): the object may be in the previous photo.
  When the registration collapsed and the camera moved, occlusion is `occlusion_undetermined`.
* `camera_moved`: the registration is good and the frame centre moved more than `MOVE_SHARE` of the
  diagonal (or rotated > `MOVE_DEG`, or scaled > `MOVE_SCALE`); or the registration collapsed (inliers
  below `MIN_INLIERS`, or below `COLLAPSE_RATIO` x the camera's usual inliers when given) while
  `cv2.phaseCorrelate` still finds a confident global shift larger than `MOVE_SHARE`.

`blur_faces(img)` detects faces with OpenCV's YuNet (`cv2.FaceDetectorYN`, model
face_detection_yunet_2023mar.onnx, opencv_zoo, MIT) and blurs each (box grown 25 %). Without the model it
falls back to blurring the given foreground boxes, or pixelating the whole image when none are given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.evidence import pixelate, save_jpeg
from sentinel.vision.register import register

WORK_LONG = 640
RATIO_T = 0.3          # local reflectance change counted as foreground (30 %)
CHROMA_T = 15.0        # Lab chroma distance counted as foreground
SIGMA_SHARE = 0.08     # illumination blur sigma, share of the long side
LARGE_SHARE = 0.04     # largest blob >= 4 % of the frame = large object
MIN_SOLIDITY = 0.5
GLOBAL_SHARE = 0.6     # >= 60 % of the frame changed = lighting / IR switch
MIN_INLIERS = 15
COLLAPSE_RATIO = 0.25
MOVE_SHARE = 0.02      # frame-centre shift > 2 % of the diagonal
MOVE_DEG = 2.0
MOVE_SCALE = 0.05
PHASE_MIN_RESPONSE = 0.1

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
REPO = Path(__file__).resolve().parents[2]


def _work(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, WORK_LONG / float(max(h, w)))
    return (cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img), s


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def motion_from_h(H: np.ndarray, size: tuple[int, int]) -> dict:
    """Frame-centre shift, rotation and scale of homography H (prev -> cur) on a frame of `size` (w, h)."""
    w, h = size
    c = np.array([[[w / 2.0, h / 2.0]]], np.float64)
    m = cv2.perspectiveTransform(c, H)[0, 0] - c[0, 0]
    a = H[:2, :2] / H[2, 2]
    rot = float(np.degrees(np.arctan2(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1])))
    scale = float(np.sqrt(abs(np.linalg.det(a))))
    px = float(np.hypot(*m))
    return {"dx": round(float(m[0]), 2), "dy": round(float(m[1]), 2), "px": round(px, 2),
            "share": round(px / float(np.hypot(w, h)), 4), "rotation_deg": round(rot, 3), "scale": round(scale, 4)}


def phase_shift(a: np.ndarray, b: np.ndarray) -> dict:
    """Global translation a -> b with cv2.phaseCorrelate (Hanning window) and its response (0-1)."""
    ga, gb = np.float32(_gray(a)), np.float32(_gray(b))
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    px = float(np.hypot(dx, dy))
    return {"dx": round(float(dx), 2), "dy": round(float(dy), 2), "px": round(px, 2),
            "share": round(px / float(np.hypot(ga.shape[1], ga.shape[0])), 4), "response": round(float(resp), 4)}


def _ratio(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Grey image divided by its own heavy blur: removes smooth illumination (IR LED hot spot, sun gradient)."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    return g / (cv2.GaussianBlur(g, (0, 0), sigma) + 8.0)


def is_ir(img: np.ndarray) -> bool:
    """Near-grayscale frame (IR night mode): mean HSV saturation below 8 (quality_thresholds.yaml ir)."""
    return img.ndim == 3 and float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()) < 8.0


def foreground_diff(prev: np.ndarray, cur: np.ndarray, H: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(foreground mask, valid-overlap mask) of an aligned pair (work-size BGR images), illumination-robust.

    Foreground = local reflectance changed by more than RATIO_T (each frame divided by its own blur,
    sigma SIGMA_SHARE of the long side) OR the chroma (Lab a, b) moved by more than CHROMA_T; the chroma
    test is skipped when either frame is in IR mode. A plain absdiff flagged 14 of 200 real consecutive
    lure-trap pairs (day/night switch, sun patches); this flagged 0 at the same synthetic recall or better.
    """
    h, w = cur.shape[:2]
    if H is not None:
        a = cv2.warpPerspective(prev, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.erode(cv2.warpPerspective(np.full(prev.shape[:2], 255, np.uint8), H, (w, h)),
                          np.ones((5, 5), np.uint8))
    else:
        a = cv2.resize(prev, (w, h)) if prev.shape[:2] != cur.shape[:2] else prev
        valid = np.full((h, w), 255, np.uint8)
    sigma = SIGMA_SHARE * max(h, w)
    fg = np.abs(_ratio(_gray(a), sigma) - _ratio(_gray(cur), sigma)) > RATIO_T
    if a.ndim == 3 and cur.ndim == 3 and not (is_ir(a) or is_ir(cur)):
        la = cv2.cvtColor(cv2.GaussianBlur(a, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        lb = cv2.cvtColor(cv2.GaussianBlur(cur, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        fg |= np.hypot(la[..., 1] - lb[..., 1], la[..., 2] - lb[..., 2]) > CHROMA_T
    fg = fg.astype(np.uint8) * 255
    fg[valid == 0] = 0
    return fg, valid


def foreground_mog2(history: list[np.ndarray], cur: np.ndarray) -> np.ndarray:
    """MOG2 background model learnt on `history` (same size as cur), foreground of `cur` (learning rate 0)."""
    mog = cv2.createBackgroundSubtractorMOG2(history=max(3, len(history)), varThreshold=25, detectShadows=False)
    for f in history:
        mog.apply(cv2.GaussianBlur(f, (5, 5), 0), learningRate=-1)
    return mog.apply(cv2.GaussianBlur(cur, (5, 5), 0), learningRate=0)


def largest_blob(fg: np.ndarray) -> tuple[dict | None, np.ndarray]:
    """Clean the mask and describe its largest blob (bbox, area share, solidity) on the work frame."""
    k = max(3, round(0.02 * max(fg.shape)) | 1)
    m = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None, m
    c = max(cs, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    hull = max(1.0, cv2.contourArea(cv2.convexHull(c)))
    x, y, bw, bh = cv2.boundingRect(c)
    blob = np.zeros_like(m)
    cv2.drawContours(blob, [c], -1, 255, -1)
    return {"bbox": [x, y, x + bw, y + bh], "area_share": area / float(m.size), "solidity": area / hull,
            "_mask": blob}, m


def _which_frame(prev: np.ndarray, cur: np.ndarray, blob: np.ndarray) -> str:
    """The frame holding the object: where the blob contrasts more with a ring of its surroundings.

    Where the object is, its colour differs from the trap around it; in the other frame the blob area is
    the trap surface itself, which looks like its surroundings.
    """
    k = max(5, round(0.04 * max(blob.shape)) | 1)
    ring = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) & ~blob
    inside, around = blob > 0, ring > 0
    if not around.any():
        return "cur"
    contrast = []
    for f in (prev, cur):
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        contrast.append(float(np.linalg.norm(lab[inside].mean(axis=0) - lab[around].mean(axis=0))))
    return "cur" if contrast[1] >= contrast[0] else "prev"


def detect_motion(
    prev: np.ndarray,
    cur: np.ndarray,
    *,
    history: list[np.ndarray] | None = None,
    registration: dict | None = None,
    usual_inliers: float | None = None,
    method: str = "sift_flann",
    out_dir: str | Path | None = None,
    name: str = "motion",
) -> dict:
    """Op 8 on two BGR photos of the same camera (prev earlier). See module docstring.

    `registration`: a register() result on these two full frames at work size (reused if given).
    `usual_inliers`: the camera's typical inlier count, to detect a collapse relative to it.
    """
    t0 = time.perf_counter()
    a, _ = _work(prev)
    b, s = _work(cur)
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    reg = registration or register(a, b, method=method)
    H = np.asarray(reg["H"], np.float64) if reg.get("H") is not None else None
    size = (b.shape[1], b.shape[0])

    reasons: list[str] = []
    shift = None
    if reg.get("ok") and H is not None:
        shift = {**motion_from_h(H, size), "source": "registration"}
        if shift["share"] > MOVE_SHARE:
            reasons.append("shift")
        if abs(shift["rotation_deg"]) > MOVE_DEG:
            reasons.append("rotation")
        if abs(shift["scale"] - 1.0) > MOVE_SCALE:
            reasons.append("scale")
    collapsed = reg.get("inliers", 0) < MIN_INLIERS or (
        usual_inliers is not None and reg.get("inliers", 0) < COLLAPSE_RATIO * usual_inliers)
    ph = phase_shift(a, b)
    if collapsed:
        shift = shift or {**{k: ph[k] for k in ("dx", "dy", "px", "share")}, "rotation_deg": None, "scale": None,
                          "source": "phase_correlation"}
        if ph["response"] >= PHASE_MIN_RESPONSE and ph["share"] > MOVE_SHARE:
            reasons.append("registration_collapsed+phase_shift")

    if history and len(history) >= 3:
        hist = [cv2.resize(_work(f)[0], size, interpolation=cv2.INTER_AREA) for f in history]
        fg = foreground_mog2(hist, b)
        valid = np.full(fg.shape, 255, np.uint8)
        fg_kind = "mog2"
    else:
        use_h = H if reg.get("ok") else None  # a good homography also aligns a moved camera
        fg, valid = foreground_diff(a, b, use_h)
        fg_kind = "ratio+chroma"
    n_valid = max(1, int(np.count_nonzero(valid)))
    changed = float(np.count_nonzero(fg & valid)) / n_valid
    global_change = changed >= GLOBAL_SHARE
    blob, clean = largest_blob(fg)
    # a collapsed registration with a camera move cannot be told apart from an object: undetermined
    undetermined = bool(collapsed and reasons)
    occluded = bool(blob and not global_change and not undetermined and blob["area_share"] >= LARGE_SHARE
                    and blob["solidity"] >= MIN_SOLIDITY)
    occluder = None
    if blob:
        x1, y1, x2, y2 = blob["bbox"]
        occluder = {
            "bbox": [round(v / s, 1) for v in (x1, y1, x2, y2)],
            "bbox_norm": [round(x1 / size[0], 4), round(y1 / size[1], 4), round(x2 / size[0], 4), round(y2 / size[1], 4)],
            "area_share": round(blob["area_share"], 4),
            "solidity": round(blob["solidity"], 4),
            "in_frame": _which_frame(a, b, blob["_mask"]) if occluded else None,
        }
    res = {
        "occluded_by_large_object": occluded,
        "occlusion_undetermined": undetermined,
        "occluder": occluder,
        "camera_moved": bool(reasons),
        "moved_reasons": reasons,
        "shift": shift,
        "phase_correlation": ph,
        "registration": {"ok": bool(reg.get("ok")), "inliers": int(reg.get("inliers", 0)),
                         "matches": int(reg.get("matches", 0)), "method": reg.get("method"),
                         "fallback_reason": reg.get("fallback_reason"), "collapsed": bool(collapsed)},
        "global_change": bool(global_change),
        "changed_share": round(changed, 4),
        "foreground": fg_kind,
        "time_ms": None,
        "evidence": {},
    }
    if out_dir is not None:
        out = Path(out_dir)
        res["evidence"]["fg_mask"] = save_jpeg(clean, out / f"{name}_fg.jpg")
        over = pixelate(b)  # whole frames may show people: never stored sharp
        color = np.zeros_like(over)
        color[clean > 0] = (0, 0, 255)
        over = cv2.addWeighted(over, 1.0, color, 0.5, 0)
        if occluder:
            x1, y1, x2, y2 = blob["bbox"]
            cv2.rectangle(over, (x1, y1), (x2, y2), (0, 255, 255) if occluded else (200, 200, 200), 2)
        cv2.putText(over, f"occluded={occluded} moved={bool(reasons)} changed={changed:.2f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        res["evidence"]["overlay"] = save_jpeg(over, out / f"{name}_overlay.jpg")
    res["time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return res


# ---------------------------------------------------------------------------- faces


def yunet_path() -> Path | None:
    """The YuNet model: $SENTINEL_FACE_MODEL, models/features/, or the copy next to this module."""
    cands = [os.environ.get("SENTINEL_FACE_MODEL"), REPO / "models" / "features" / YUNET_FILE,
             Path(__file__).with_name("models") / YUNET_FILE]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


@lru_cache(maxsize=1)
def _yunet_checked(path: str) -> bool:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() == YUNET_SHA256


@lru_cache(maxsize=4)
def _yunet(path: str, w: int, h: int):
    return cv2.FaceDetectorYN.create(path, "", (w, h), 0.6, 0.3, 5000)


def detect_faces(img: np.ndarray, score: float = 0.6) -> list[dict] | None:
    """Faces [{bbox xyxy, score}] with YuNet, or None when the model is unavailable or fails its hash."""
    p = yunet_path()
    if p is None or not _yunet_checked(str(p)):
        return None
    h, w = img.shape[:2]
    s = min(1.0, 960.0 / max(h, w))  # YuNet is fast on <= 960 px; faces of a few px are not identifiable anyway
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    det = _yunet(str(p), small.shape[1], small.shape[0])
    det.setScoreThreshold(score)
    _, faces = det.detect(small)
    out = []
    for f in faces if faces is not None else []:
        x, y, fw, fh = (float(v) / s for v in f[:4])
        out.append({"bbox": [round(x, 1), round(y, 1), round(x + fw, 1), round(y + fh, 1)], "score": round(float(f[14]), 4)})
    return out


def _blur_box(img: np.ndarray, box, grow: float = 0.25) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    gx, gy = (x2 - x1) * grow / 2, (y2 - y1) * grow / 2
    x1, y1 = max(0, int(x1 - gx)), max(0, int(y1 - gy))
    x2, y2 = min(w, int(np.ceil(x2 + gx))), min(h, int(np.ceil(y2 + gy)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    k = max(3, (max(x2 - x1, y2 - y1) // 3) | 1)
    img[y1:y2, x1:x2] = pixelate(cv2.GaussianBlur(roi, (k, k), 0), block_share=0.15)


def blur_faces(img: np.ndarray, fallback_boxes: list | None = None) -> tuple[np.ndarray, dict]:
    """(copy of img with faces blurred, info {method, faces, boxes_blurred}).

    method "yunet": every YuNet face blurred. Without the model: "fallback_boxes" blurs the given boxes
    (e.g. op 8's large foreground blob, xyxy pixels, grown 25 %), else "fallback_whole_frame" pixelates all.
    """
    out = img.copy()
    faces = detect_faces(img)
    if faces is not None:
        for f in faces:
            _blur_box(out, f["bbox"])
        return out, {"method": "yunet", "faces": len(faces), "boxes_blurred": [f["bbox"] for f in faces],
                     "model": YUNET_FILE}
    if fallback_boxes:
        for b in fallback_boxes:
            _blur_box(out, b, grow=0.25)
        return out, {"method": "fallback_boxes", "faces": None, "boxes_blurred": [list(b) for b in fallback_boxes]}
    return pixelate(out), {"method": "fallback_whole_frame", "faces": None, "boxes_blurred": []}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prev")
    ap.add_argument("cur")
    ap.add_argument("--history", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    p, c = cv2.imread(a.prev), cv2.imread(a.cur)
    if p is None or c is None:
        print("cannot read input images", file=sys.stderr)
        return 2
    hist = [cv2.imread(h) for h in a.history] if a.history else None
    print(json.dumps(detect_motion(p, c, history=hist, out_dir=a.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
