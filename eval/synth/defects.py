"""Deliberately damaged synthetic photos: one defect per photo, with what the agent policy should do about it.

The image functions are the failure-injection harness's (eval/failure_injection.py: blur, glare, darkness, trap out of
view; copied here so this package does not import the agent harness) and op 8's own synthetic occluder
(sentinel.vision.motion_eval). `EXPECTED` is written from
lambda/vision/sentinel/agent/policy.yaml and the RulePlanner *before* any result is looked at; eval.synth.check
compares the Lambda's answer with it.

    expected.flag     the response's quality flag the defect should raise (`quality.flags`, handler.quality_block)
    expected.rule     the RulePlanner decision rule prefix the policy prescribes
    expected.actions  proposal actions that rule produces through the Lambda contract (sentinel/agent/proposals.py)
    expected.counted  False: the photo must not be counted (policy: occlusion, trap missing)
    needsPrev         True: only detectable against the previous photo (op 8), so the check sends prevPhoto
"""

from __future__ import annotations

import cv2
import numpy as np

TYPES = ("blur", "dark", "glare", "dirty_lens", "occluded_hand", "occluded_leaf", "camera_moved", "no_trap")

# platform "camos" (ESP32 / AI camera) can change its image settings; "ezviz" cannot (policy quality.isp_fixes)
EXPECTED = {
    "blur": {"flag": "blurry", "rule": "quality.recapture", "actions": ["capture_now"], "counted": True},
    "dark": {"flag": "dark", "rule": {"ezviz": "quality.recapture", "camos": "quality.isp_fix"},
             "actions": {"ezviz": ["capture_now"], "camos": ["set_isp"]}, "counted": True},
    "glare": {"flag": "overexposed", "rule": {"ezviz": "quality.recapture", "camos": "quality.isp_fix"},
              "actions": {"ezviz": ["capture_now"], "camos": ["set_isp"]}, "counted": True},
    "dirty_lens": {"flag": "dirty", "rule": "quality.clean_lens", "actions": ["review_request"], "counted": True},
    "occluded_hand": {"flag": "occluded", "rule": "occlusion.recapture", "actions": ["capture_now"], "counted": False},
    "occluded_leaf": {"flag": "occluded", "rule": "occlusion.recapture", "actions": ["capture_now"], "counted": False},
    "camera_moved": {"flag": "camera_moved", "rule": "motion.camera_moved", "actions": ["review_request"],
                     "counted": True},
    "no_trap": {"flag": "trap_missing", "rule": "trap_missing.recapture", "actions": ["capture_now"], "counted": False},
}
NEEDS_PREV = {"occluded_hand", "occluded_leaf", "camera_moved"}


def blur(img: np.ndarray, rel_sigma: float) -> np.ndarray:
    """Gaussian blur, sigma = rel_sigma x image width (eval/failure_injection.py)."""
    return cv2.GaussianBlur(img, (0, 0), max(0.1, rel_sigma * img.shape[1]))


def glare(img: np.ndarray, area_share: float = 0.06, seed: int = 0) -> np.ndarray:
    """A soft white specular blob, ~area_share of the frame (eval/failure_injection.py)."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(0.35, 0.65) * w, rng.uniform(0.3, 0.6) * h
    ax = np.sqrt(area_share * w * h / np.pi * (w / h) ** 0.3)
    ay = area_share * w * h / (np.pi * ax)
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.uniform(0, 40)), 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), min(h, w) * 0.02)
    m = np.clip(m * 1.6, 0, 1)[..., None]
    return np.clip(img * (1 - m) + 255 * m, 0, 255).astype(np.uint8)


def darken(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma > 1 darkens: out = 255 x (in/255)^gamma (eval/failure_injection.py)."""
    return np.clip(255.0 * (img / 255.0) ** gamma, 0, 255).astype(np.uint8)


def tile_strip(img: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """The frame's strip between x0 and x1 (fractions of the width) tiled across it: housing only, no trap in view
    (eval/failure_injection.py crop_trap_away uses the right 12 %)."""
    w = img.shape[1]
    strip = img[:, int(w * x0): max(int(w * x0) + 1, int(w * x1))]
    return np.tile(strip, (1, int(np.ceil(w / strip.shape[1])), 1))[:, :w]


def expected_for(kind: str, platform: str) -> dict:
    e = EXPECTED[kind]
    pick = (lambda v: v[platform] if isinstance(v, dict) else v)
    return {"flag": e["flag"], "rule": pick(e["rule"]), "actions": list(pick(e["actions"])), "counted": e["counted"],
            "needsPrev": kind in NEEDS_PREV}


def _leaf(rng: np.random.Generator, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """A fallen leaf lying over the trap: pointed ellipse, green-to-yellow, midrib and veins; (patch BGR, mask)."""
    L = rng.uniform(0.45, 0.7) * min(W, H) * 2.0
    Wd = L * rng.uniform(0.32, 0.45)
    cx, cy, ang = rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H, rng.uniform(0, 180)
    t = np.linspace(-1, 1, 80)
    half = Wd / 2 * np.sqrt(np.clip(1 - t ** 2, 0, 1)) * (1 - 0.25 * t)  # wider near the stalk
    pts = np.concatenate([np.stack([t * L / 2, half], 1), np.stack([t[::-1] * L / 2, -half[::-1]], 1)])
    R = cv2.getRotationMatrix2D((0, 0), ang, 1.0)[:, :2]
    pts = (pts @ R.T + [cx, cy]).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 255)
    base = np.array([rng.uniform(30, 60), rng.uniform(100, 150), rng.uniform(60, 120)])  # BGR green / olive
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = ((xx - cx) * np.cos(np.radians(ang)) + (yy - cy) * np.sin(np.radians(ang))) / (L / 2)
    patch = base + np.stack([0 * grad, 18 * grad, 40 * np.clip(grad, 0, 1)], -1) + rng.normal(0, 5, (H, W, 3))
    p0 = (int(cx - np.cos(np.radians(ang)) * L / 2), int(cy - np.sin(np.radians(ang)) * L / 2))
    p1 = (int(cx + np.cos(np.radians(ang)) * L / 2), int(cy + np.sin(np.radians(ang)) * L / 2))
    vein = np.zeros((H, W), np.uint8)
    cv2.line(vein, p0, p1, 255, max(2, int(Wd * 0.03)))
    for k in np.linspace(0.15, 0.85, 6):
        q = (int(p0[0] + (p1[0] - p0[0]) * k), int(p0[1] + (p1[1] - p0[1]) * k))
        for s in (-1, 1):
            a = np.radians(ang + s * 50)
            cv2.line(vein, q, (int(q[0] + np.cos(a) * Wd * 0.45), int(q[1] + np.sin(a) * Wd * 0.45)), 255,
                     max(1, int(Wd * 0.015)))
    patch[vein > 0] = patch[vein > 0] * 0.75 + 40
    patch = cv2.GaussianBlur(np.clip(patch, 0, 255).astype(np.uint8), (5, 5), 0)
    return patch, m


def _dirty(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Smudges and dried droplets on the lens: soft hazy blotches (local blur + veil) and small ring-edged spots."""
    H, W = img.shape[:2]
    veil = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(4, 9))):
        c = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        ax = (int(rng.uniform(0.08, 0.25) * W), int(rng.uniform(0.08, 0.25) * H))
        cv2.ellipse(veil, c, ax, rng.uniform(0, 180), 0, 360, float(rng.uniform(0.5, 1.0)), -1)
    veil = cv2.GaussianBlur(veil, (0, 0), W * 0.03)
    veil = np.clip(veil * strength, 0, 0.95)[..., None]
    hazy = cv2.GaussianBlur(img, (0, 0), W * 0.006).astype(np.float32)
    milk = np.full_like(hazy, 200.0)
    out = img.astype(np.float32) * (1 - veil) + (hazy * 0.6 + milk * 0.4) * veil
    for _ in range(int(rng.integers(15, 40))):  # droplets
        c = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        r = int(rng.uniform(0.004, 0.015) * W) + 1
        cv2.circle(out, c, r, (175, 175, 175), -1, cv2.LINE_AA)
        cv2.circle(out, c, r, (110, 110, 110), max(1, r // 4), cv2.LINE_AA)
    return np.clip(cv2.GaussianBlur(out, (3, 3), 0), 0, 255).astype(np.uint8)


def apply(img: np.ndarray, kind: str, rng: np.random.Generator, trap_kind: str = "lure") -> tuple[
        np.ndarray, dict, np.ndarray | None, np.ndarray | None]:
    """Damage uint8 BGR `img`. Returns (image, params, hidden mask or None, 2x3 affine of the frame or None).

    The hidden mask marks pixels where flies can no longer be seen (occluder, no trap); the affine moves the boxes
    of a camera that moved.
    """
    H, W = img.shape[:2]
    if kind == "blur":
        s = float(rng.uniform(0.006, 0.016))
        return blur(img, s), {"severity": round(s, 4), "unit": "sigma / width"}, None, None
    if kind == "dark":
        g = float(rng.uniform(5.0, 6.5))
        return darken(img, g), {"severity": round(g, 2), "unit": "gamma"}, None, None
    if kind == "glare":
        a = float(rng.uniform(0.04, 0.09))
        return glare(img, a, seed=int(rng.integers(1 << 30))), {"severity": round(a, 3), "unit": "blob area / frame"}, \
            None, None
    if kind == "dirty_lens":
        s = float(rng.uniform(0.7, 1.0))
        return _dirty(img, rng, s), {"severity": round(s, 2), "unit": "smudge opacity"}, None, None
    if kind == "occluded_hand":
        from sentinel.vision.motion_eval import occluder, paste

        for _ in range(30):
            patch, m, k = occluder(rng, W, H)
            if k == "hand":
                break
        share = float((m > 127).mean())
        return paste(img, patch, m), {"severity": round(share, 3), "unit": "frame share covered"}, m > 127, None
    if kind == "occluded_leaf":
        from sentinel.vision.motion_eval import paste

        patch, m = _leaf(rng, W, H)
        share = float((m > 127).mean())
        return paste(img, patch, m), {"severity": round(share, 3), "unit": "frame share covered"}, m > 127, None
    if kind == "camera_moved":
        share = float(rng.uniform(0.04, 0.09))
        ang = float(rng.uniform(0, 2 * np.pi))
        d = share * float(np.hypot(W, H))
        M = cv2.getRotationMatrix2D((W / 2, H / 2), float(rng.uniform(-5, 5)), 1.0)
        M[:, 2] += (d * np.cos(ang), d * np.sin(ang))
        return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT), \
            {"severity": round(share, 3), "unit": "shift / diagonal"}, None, M
    if kind == "no_trap":
        # the right rim strip: housing only (the sticky card ends before it; the left edge carries the camera logo)
        out = tile_strip(img, 0.95, 1.0) if trap_kind == "sticky" else tile_strip(img, 0.88, 1.0)
        return out, {"severity": 1.0, "unit": "trap out of view"}, np.ones((H, W), bool), None
    raise ValueError(f"unknown defect {kind!r}")


def move_box(px: list[float], M: np.ndarray) -> list[float]:
    """Axis-aligned box (x, y, w, h px) after the 2x3 affine `M`."""
    x, y, w, h = px
    pts = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]], np.float64) @ M.T
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return [round(float(x0), 1), round(float(y0), 1), round(float(x1 - x0), 1), round(float(y1 - y0), 1)]
