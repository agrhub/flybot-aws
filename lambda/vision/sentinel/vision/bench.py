"""Op 1-4 benchmark: synthetic card accuracy and 1080p timings.

    uv run python -m sentinel.vision.bench [--n 200] [--reps 30] [--aliked-reps 3]

Writes docs/evidence/vision/card_synth.json (corner error, hit rate per case) and
docs/evidence/vision/timings.json (p50/p95 ms per op on 1920x1080 frames). Backgrounds are real
export frames resized to 1920x1080 (the export has no 1080p frame: sizes are 1280x720, 768x432,
2304x1296, 640x640) plus textured noise. The machine, CPU count and OpenCV threads are recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision import synth
from sentinel.vision.blobs import detect_dark_blobs
from sentinel.vision.card import find_card, warp_card
from sentinel.vision.quality import assess_quality
from sentinel.vision.register import extract, models_available, register_features

W, H = 1920, 1080


def backgrounds(manifest: Path, k: int, rng: random.Random) -> list[np.ndarray]:
    with open(manifest, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rng.shuffle(rows)
    out = []
    for r in rows:
        img = cv2.imread(str(manifest.parent / r["image_path"]))
        if img is not None and img.shape[0] != img.shape[1]:
            out.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR))
        if len(out) >= k:
            break
    return out


def _err(found_quad, gt) -> float:
    return float(np.linalg.norm(np.asarray(found_quad, np.float32) - np.asarray(gt, np.float32), axis=1).max())


def synthetic(bgs: list[np.ndarray], n: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    cases = {"normal": [], "angled": [], "partial": [], "two_cards": [], "no_card": []}
    for i in range(n):
        bg = bgs[i % len(bgs)] if i % 4 else synth.noise_background(W, H, rng)
        case = list(cases)[i % len(cases)]
        card, _ = synth.make_card(480, 300, int(rng.integers(0, 80)), rng)
        if case == "no_card":
            r = find_card(bg)
            cases[case].append({"ok": not r["found"], "found": r["found"]})
            continue
        if case == "two_cards":
            q1 = synth.random_quad(W // 2, H, rng, coverage=(0.15, 0.4), max_tilt=0.1)
            q2 = synth.random_quad(W // 2, H, rng, coverage=(0.15, 0.4), max_tilt=0.1) + np.float32([W // 2, 0])
            img = synth.paste_card(synth.paste_card(bg, card, q1), card, q2)
            r = find_card(img)
            ok = r["n_cards"] == 2
            errs = [min(_err(c["quad"], q) for c in r["cards"]) for q in (q1, q2)] if r["cards"] else [None]
            cases[case].append({"ok": ok, "n_cards": r["n_cards"], "err_px": max(e for e in errs if e is not None) if r["cards"] else None})
            continue
        tilt = 0.05 if case == "normal" else 0.3
        q = synth.random_quad(W, H, rng, coverage=(0.1, 0.45), max_tilt=tilt)
        if case == "partial":
            # push the card over the left or bottom edge by 15-35 % of its width
            shift = rng.uniform(0.15, 0.35) * (q[:, 0].max() - q[:, 0].min())
            q = q - np.float32([q[:, 0].min() + shift, 0])
        img = synth.paste_card(bg, card, q)
        r = find_card(img)
        if case == "partial":
            ok = r["found"] and r["partial"]
            cases[case].append({"ok": bool(ok), "found": r["found"], "partial": r["partial"]})
        else:
            e = _err(r["quad"], q) if r["found"] else None
            cases[case].append({"ok": bool(r["found"] and e is not None and e <= 0.02 * np.hypot(W, H)), "err_px": e})
    out = {}
    for k, v in cases.items():
        errs = [x["err_px"] for x in v if x.get("err_px") is not None]
        out[k] = {
            "n": len(v),
            "ok": sum(x["ok"] for x in v),
            "hit_rate": round(sum(x["ok"] for x in v) / len(v), 4) if v else None,
            "corner_err_px_p50": round(float(np.median(errs)), 2) if errs else None,
            "corner_err_px_p95": round(float(np.percentile(errs, 95)), 2) if errs else None,
        }
    out["criteria"] = {"normal/angled": "found and max corner error <= 2 % of the frame diagonal (44 px at 1080p)",
                       "partial": "found and partial=True", "two_cards": "n_cards == 2", "no_card": "found=False"}
    return out


def _stats(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": round(float(np.median(ms)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
            "max_ms": round(float(max(ms)), 2)}


def timings(bgs: list[np.ndarray], reps: int, aliked_reps: int) -> dict:
    rng = np.random.default_rng(3)
    frames = []
    for i in range(reps):
        card, _ = synth.make_card(480, 300, 40, rng)
        q = synth.random_quad(W, H, rng, coverage=(0.15, 0.4))
        frames.append(synth.paste_card(bgs[i % len(bgs)], card, q))
    t = {"card": [], "quality_with_card": [], "quality_frame_only": [], "blobs_warped": [], "sift_register_warped": []}
    for f in frames:
        t0 = time.perf_counter()
        c = find_card(f)
        t["card"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        assess_quality(f, card=c)
        t["quality_with_card"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        assess_quality(f)
        t["quality_frame_only"].append((time.perf_counter() - t0) * 1000)
        w = warp_card(f, c["quad"]) if c["found"] else cv2.resize(f, (800, 450))
        t0 = time.perf_counter()
        detect_dark_blobs(w)
        t["blobs_warped"].append((time.perf_counter() - t0) * 1000)
        shifted = cv2.warpAffine(w, np.float32([[1, 0, 6], [0, 1, -4]]), (w.shape[1], w.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        t0 = time.perf_counter()
        register_features(extract(w, "sift_flann", 2000), extract(shifted, "sift_flann", 2000))
        t["sift_register_warped"].append((time.perf_counter() - t0) * 1000)
    out = {k: _stats(v) for k, v in t.items()}
    out["ops1_2_total_p50_ms"] = round(out["card"]["p50_ms"] + out["quality_with_card"]["p50_ms"], 2)
    av = models_available()
    if aliked_reps and av["aliked"] and av["lightglue"]:
        ext, mat = [], []
        extract(frames[0], "aliked_lightglue")  # first call builds the network
        for f in frames[:aliked_reps]:
            c = find_card(f)
            w = warp_card(f, c["quad"])
            shifted = cv2.warpAffine(w, np.float32([[1, 0, 6], [0, 1, -4]]), (w.shape[1], w.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            fa, fb = extract(w, "aliked_lightglue"), extract(shifted, "aliked_lightglue")
            ext += [fa["time_ms"], fb["time_ms"]]
            r = register_features(fa, fb)
            mat.append(r["match_ms"])
        out["aliked_extract_warped"] = _stats(ext)
        out["lightglue_match_512kp"] = _stats(mat)
    else:
        out["aliked"] = "not measured (models missing)"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--aliked-reps", type=int, default=3)
    ap.add_argument("--out-dir", default="docs/evidence/vision")
    a = ap.parse_args(argv)
    bgs = backgrounds(Path(a.manifest), 24, random.Random(5))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = {"machine": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
           "cv2": cv2.__version__, "cv2_threads": cv2.getNumThreads(), "frame": [W, H],
           "note": "shared dev laptop; other agents' jobs ran at the same time, so numbers are upper bounds"}
    syn = {"env": env, "backgrounds": "real export frames resized to 1920x1080 (3 of 4) + textured noise (1 of 4)",
           **synthetic(bgs, a.n)}
    (out / "card_synth.json").write_text(json.dumps(syn, indent=1), encoding="utf-8")
    tim = {"env": env, **timings(bgs, a.reps, a.aliked_reps)}
    (out / "timings.json").write_text(json.dumps(tim, indent=1), encoding="utf-8")
    print(json.dumps({"synthetic": syn, "timings": tim}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
