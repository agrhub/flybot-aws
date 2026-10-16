"""Op 3: register two photos of the same trap (OpenCV 5 ALIKED + LightGlueMatcher, fallback SIFT+FLANN).

    uv run python -m sentinel.vision.register PREV CUR [--method auto|aliked_lightglue|sift_flann] [--out DIR]

`register(prev, cur)` returns a JSON-serializable dict:

    ok, H (3x3, maps prev pixels -> cur pixels), inliers, matches, inlier_ratio, method,
    requested_method, fallback_used, fallback_reason (why SIFT ran instead of the requested ALIKED, else None),
    attempts (every method tried, with its numbers), keypoints {prev, cur}, time_ms, evidence
    {aligned_pair, matches}

Models (OpenCV 5 `features` module, ONNX): ALIKED `aliked-n16rot-top1k-640.onnx` and its LightGlue
`aliked_lightglue.onnx`, the files OpenCV's own tests use (opencv_extra testdata/dnn/download_models.py);
looked up in $SENTINEL_FEATURES_DIR or models/features/ (`make models` fetches them, SHA-256 checked).
Without them (or when OpenCV cannot load them), `auto` and `aliked_lightglue` both fall back to SIFT+FLANN and
record a `fallback_reason` (also logged as a warning) instead of failing the op.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.evidence import pixelate, save_jpeg

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
ALIKED_MODEL = "aliked-n16rot-top1k-640.onnx"
LIGHTGLUE_MODEL = "aliked_lightglue.onnx"
MIN_INLIERS = 15
RANSAC_PX = 4.0
MAX_KEYPOINTS = 512  # LightGlue cost grows ~quadratically with keypoints (attention)
METHOD_ALIASES = {"aliked": "aliked_lightglue", "sift": "sift_flann"}


def features_dir() -> Path:
    return Path(os.environ.get("SENTINEL_FEATURES_DIR", REPO / "models" / "features"))


def models_available() -> dict:
    d = features_dir()
    return {"aliked": (d / ALIKED_MODEL).is_file(), "lightglue": (d / LIGHTGLUE_MODEL).is_file(), "dir": d.as_posix()}


def _aliked():
    return _aliked_at(str(features_dir() / ALIKED_MODEL))


@lru_cache(maxsize=2)
def _aliked_at(path: str):
    p = cv2.ALIKED_Params()  # inputSize 640x640 is fixed by this ONNX export; engine 2 = new DNN engine
    return cv2.ALIKED_create(path, p)


def _lightglue():
    return _lightglue_at(str(features_dir() / LIGHTGLUE_MODEL))


@lru_cache(maxsize=2)
def _lightglue_at(path: str):
    return cv2.LightGlueMatcher_create(path)


@lru_cache(maxsize=1)
def _sift():
    return cv2.SIFT_create(nfeatures=2000)


def extract(img: np.ndarray, method: str, max_keypoints: int = MAX_KEYPOINTS) -> dict:
    """Keypoints + descriptors of one image for `method`; reusable across pairs."""
    t0 = time.perf_counter()
    if method == "aliked_lightglue":
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        kps, desc = _aliked().detectAndCompute(bgr, None)
    elif method == "sift_flann":
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = _sift().detectAndCompute(gray, None)
    else:
        raise ValueError(f"unknown method {method!r}")
    kps = list(kps)
    if desc is None or not kps:
        pts = np.zeros((0, 2), np.float32)
        desc = np.zeros((0, 128), np.float32)
    else:
        order = np.argsort([-k.response for k in kps])[:max_keypoints]
        pts = np.array([kps[i].pt for i in order], np.float32).reshape(-1, 2)
        desc = np.ascontiguousarray(desc[order], dtype=np.float32)
    return {
        "method": method,
        "pts": pts,
        "desc": desc,
        "size": (int(img.shape[1]), int(img.shape[0])),
        "time_ms": (time.perf_counter() - t0) * 1000.0,
    }


def _match(fa: dict, fb: dict) -> list[tuple[int, int]]:
    if len(fa["pts"]) < 4 or len(fb["pts"]) < 4:
        return []
    if fa["method"] == "aliked_lightglue":
        m = _lightglue()
        m.setPairInfo(fa["pts"], fb["pts"], fa["size"], fb["size"])
        try:
            ms = m.match(fa["desc"], fb["desc"])
        finally:
            m.clearPairInfo()
        return [(d.queryIdx, d.trainIdx) for d in ms]
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    knn = flann.knnMatch(fa["desc"], fb["desc"], k=2)
    return [(p[0].queryIdx, p[0].trainIdx) for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]


def homography_sane(H: np.ndarray | None, size_prev: tuple[int, int]) -> bool:
    """Reject degenerate homographies: flips, extreme scale, strong perspective, non-convex image."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    det = float(np.linalg.det(H[:2, :2]))
    if not 0.25 < det < 4.0:
        return False
    w, h = size_prev
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H)
    return bool(cv2.isContourConvex(proj.astype(np.float32)))


def register_features(fa: dict, fb: dict, min_inliers: int = MIN_INLIERS, ransac_px: float = RANSAC_PX) -> dict:
    """Match two feature sets and fit a homography prev -> cur with USAC MAGSAC++."""
    t0 = time.perf_counter()
    pairs = _match(fa, fb)
    t_match = (time.perf_counter() - t0) * 1000.0
    out = {"method": fa["method"], "matches": len(pairs), "inliers": 0, "H": None, "ok": False,
           "match_ms": round(t_match, 2), "inlier_pairs": []}
    if len(pairs) >= 4:
        src = np.float32([fa["pts"][i] for i, _ in pairs])
        dst = np.float32([fb["pts"][j] for _, j in pairs])
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_px, maxIters=5000, confidence=0.999)
        if H is not None and mask is not None:
            inl = mask.ravel().astype(bool)
            out["inliers"] = int(inl.sum())
            out["H"] = H
            out["inlier_pairs"] = [p for p, k in zip(pairs, inl) if k]
            if out["inliers"] >= 4:
                err = np.linalg.norm(cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2) - dst[inl], axis=1)
                out["reproj_px"] = round(float(np.median(err)), 3)
            out["ok"] = out["inliers"] >= min_inliers and homography_sane(H, fa["size"])
    out["inlier_ratio"] = round(out["inliers"] / out["matches"], 4) if out["matches"] else 0.0
    return out


def _missing_models_reason() -> str | None:
    av = models_available()
    missing = [f for f, ok in ((ALIKED_MODEL, av["aliked"]), (LIGHTGLUE_MODEL, av["lightglue"])) if not ok]
    if not missing:
        return None
    return f"ALIKED/LightGlue ONNX not found in {av['dir']}: {', '.join(missing)} (run `make models`)"


def _methods(method: str) -> tuple[list[str], str | None]:
    """(methods to try in order, fallback_reason when the requested ALIKED cannot run at all)."""
    method = METHOD_ALIASES.get(method, method)
    if method == "auto":
        return (["aliked_lightglue", "sift_flann"] if _missing_models_reason() is None else ["sift_flann"]), None
    if method == "aliked_lightglue":
        reason = _missing_models_reason()
        return (["sift_flann"], reason) if reason else (["aliked_lightglue"], None)
    if method == "sift_flann":
        return [method], None
    raise ValueError(f"unknown method {method!r}")


def register(
    prev: np.ndarray,
    cur: np.ndarray,
    *,
    method: str = "auto",
    max_keypoints: int = MAX_KEYPOINTS,
    min_inliers: int = MIN_INLIERS,
    out_dir: str | Path | None = None,
    name: str = "reg",
    pixelate_evidence: bool = False,
    features: dict | None = None,
) -> dict:
    """Register `prev` onto `cur` (rectified cards, or whole frames when no card was found).

    `features` may carry precomputed {"prev": extract(...), "cur": extract(...)} for one method.
    `pixelate_evidence`: coarsen the evidence images (whole frames may show people).
    """
    t0 = time.perf_counter()
    attempts = []
    best = None
    feats = {}
    methods, fallback_reason = _methods(method)
    i = 0
    while i < len(methods):
        m = methods[i]
        i += 1
        try:
            if features and features.get("prev", {}).get("method") == m:
                fa, fb = features["prev"], features["cur"]
            else:
                fa, fb = extract(prev, m, max_keypoints), extract(cur, m, max_keypoints)
            r = register_features(fa, fb, min_inliers)
        except cv2.error as e:
            if m != "aliked_lightglue":
                raise
            # model present but unusable (corrupt file, OpenCV build without the new engine, ...)
            fallback_reason = f"ALIKED/LightGlue failed: {type(e).__name__}: {str(e).strip()[:200]}"
            attempts.append({"method": m, "ok": False, "error": fallback_reason})
            if "sift_flann" not in methods:
                methods.append("sift_flann")
            continue
        attempts.append({
            "method": m, "ok": r["ok"], "matches": r["matches"], "inliers": r["inliers"],
            "keypoints": [len(fa["pts"]), len(fb["pts"])],
            "extract_ms": [round(fa["time_ms"], 2), round(fb["time_ms"], 2)], "match_ms": r["match_ms"],
        })
        feats[m] = (fa, fb)
        if best is None or r["ok"] or (not best["ok"] and r["inliers"] > best["inliers"]):
            best = r
        if r["ok"]:
            break
    if fallback_reason:
        log.warning("register: %s -> sift_flann (requested %s)", fallback_reason, method)
    fa, fb = feats[best["method"]]
    H = best["H"]
    res = {
        "ok": bool(best["ok"]),
        "H": H.tolist() if H is not None else None,
        "inliers": best["inliers"],
        "matches": best["matches"],
        "inlier_ratio": best["inlier_ratio"],
        "reproj_px": best.get("reproj_px"),
        "method": best["method"],
        "requested_method": method,
        "fallback_used": len(attempts) > 1 or fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "attempts": attempts,
        "keypoints": {"prev": len(fa["pts"]), "cur": len(fb["pts"])},
        "sizes": {"prev": list(fa["size"]), "cur": list(fb["size"])},
        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "evidence": {},
    }
    if out_dir is not None:
        res["evidence"] = write_evidence(prev, cur, H, fa, fb, best["inlier_pairs"], Path(out_dir), name, pixelate_evidence)
    return res


def write_evidence(prev, cur, H, fa, fb, inlier_pairs, out: Path, name: str, pix: bool) -> dict:
    ev = {}
    h, w = cur.shape[:2]
    aligned = cv2.warpPerspective(prev, H, (w, h)) if H is not None else cv2.resize(prev, (w, h))
    blend = cv2.addWeighted(aligned, 0.5, cur, 0.5, 0)
    tiles = [aligned, cur, blend]
    if pix:
        tiles = [pixelate(t) for t in tiles]
    for t, lbl in zip(tiles, ("prev aligned", "cur", "blend")):
        cv2.putText(t, lbl, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    ev["aligned_pair"] = save_jpeg(np.hstack(tiles), out / f"{name}_aligned.jpg", max_side=1200)
    a, b = (pixelate(prev), pixelate(cur)) if pix else (prev, cur)
    k1 = [cv2.KeyPoint(float(x), float(y), 3) for x, y in fa["pts"]]
    k2 = [cv2.KeyPoint(float(x), float(y), 3) for x, y in fb["pts"]]
    step = max(1, len(inlier_pairs) // 150)
    dm = [cv2.DMatch(i, j, 0.0) for i, j in inlier_pairs[::step]]
    viz = cv2.drawMatches(a, k1, b, k2, dm, None, matchColor=(0, 255, 0), singlePointColor=(0, 140, 255))
    ev["matches"] = save_jpeg(viz, out / f"{name}_matches.jpg", max_side=1200)
    return ev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prev")
    ap.add_argument("cur")
    ap.add_argument("--method", default="auto", choices=["auto", "aliked_lightglue", "sift_flann", "aliked", "sift"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p, c = cv2.imread(a.prev), cv2.imread(a.cur)
    if p is None or c is None:
        print("cannot read input images", file=sys.stderr)
        return 2
    print(json.dumps(register(p, c, method=a.method, out_dir=a.out, pixelate_evidence=True), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
