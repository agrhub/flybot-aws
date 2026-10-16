"""Per-op timings of the per-photo pipeline on real consecutive lure-trap pairs (no evidence files).

    uv run python -m sentinel.vision.pipeline_bench [--n 20] [--method sift] [--out docs/evidence/vision/pipeline_timings.json]

Pairs: consecutive photos of BE4089629 / BE4111159 (768x432 and 2304x1296 camera frames, 640x640 uploads
left out), every k-th pair so the set spans the days. Timings are wall-clock on whatever machine runs
this; the JSON records the machine (platform, CPU count, OpenCV version).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.pipeline import analyze

LURE_CAMERAS = ("BE4089629", "BE4111159")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--method", default="sift", choices=["sift", "aliked"])
    ap.add_argument("--out", default="docs/evidence/vision/pipeline_timings.json")
    a = ap.parse_args(argv)
    mpath = Path(a.manifest)
    with open(mpath, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["image_path"] and r["device_serial"] in LURE_CAMERAS]
    rows.sort(key=lambda r: (r["device_serial"], r["captured_at"]))
    def camera_frame(r: dict) -> bool:  # 640x640 = uploaded picture, not a camera frame (1/8 decode to check)
        im = cv2.imread(str(mpath.parent / r["image_path"]), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        return im is not None and im.shape[:2] != (80, 80)

    ok = {r["snapshot_id"]: camera_frame(r) for r in rows}
    pairs = [(p, c) for p, c in itertools.pairwise(rows)
             if p["device_serial"] == c["device_serial"] and ok[p["snapshot_id"]] and ok[c["snapshot_id"]]]
    step = max(1, len(pairs) // a.n)
    sel = [(p, c, cv2.imread(str(mpath.parent / p["image_path"])), cv2.imread(str(mpath.parent / c["image_path"])))
           for p, c in pairs[::step][: a.n]]
    per_op: dict[str, list[float]] = {}
    sizes, flags = [], {"floor_found": 0, "registered": 0, "occluded": 0, "camera_moved": 0, "errors": 0}
    for p, c, ip, ic in sel:
        t0 = time.perf_counter()
        r = analyze(ic, ip, "lure", method=a.method)
        per_op.setdefault("wall_total", []).append((time.perf_counter() - t0) * 1000)
        for k, v in r["timings_ms"].items():
            per_op.setdefault(k, []).append(v)
        sizes.append(f"{ic.shape[1]}x{ic.shape[0]}")
        flags["floor_found"] += bool(r["card"] and r["card"]["found"])
        flags["registered"] += bool(r["register"] and r["register"]["ok"])
        flags["occluded"] += bool(r["motion"] and r["motion"]["occluded_by_large_object"])
        flags["camera_moved"] += bool(r["motion"] and r["motion"]["camera_moved"])
        flags["errors"] += bool(r["errors"])
    summary = {
        "n_pairs": len(sel), "method": a.method, "trap_kind": "lure",
        "frame_sizes": {s: sizes.count(s) for s in sorted(set(sizes))},
        "ms": {k: {"p50": round(float(np.median(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
               for k, v in per_op.items()},
        "flags": flags,
        "machine": {"platform": platform.platform(), "cpus": os.cpu_count(), "opencv": cv2.__version__,
                    "processor": platform.processor()},
        "note": "image decode not included (arrays passed in); no evidence files written",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
