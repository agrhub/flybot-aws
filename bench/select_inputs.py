"""pick the fixed benchmark input set and package it (run once on the dev machine).

    uv run python bench/select_inputs.py [--n 500] [--manifest data/manifest.csv]

Selection (deterministic, no randomness): consecutive photo pairs (previous, current) of the two lure-trap
cameras BE4089629 / BE4111159 from the FlyBot export, same camera, both real camera frames (the 640x640
uploads are left out, as in sentinel/vision/pipeline_bench.py), sorted by (camera, captured_at), then every
k-th pair so the set spans all days of both cameras. Writes:

    bench/inputs/pairs.csv           the 500 pairs: idx, camera, snapshot ids, relative paths, sizes, sha256
    bench/build/bench_inputs.tar     images/<camera>/<snapshot>.jpg (current + previous) + pairs.csv
                                     + model/fruitfly_deploy.onnx(.json)   (git-ignored, goes to S3)
    bench/build/sentinel_src.tar     the repo's sentinel/ package (no __pycache__), what the instances import
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import tarfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
LURE_CAMERAS = ("BE4089629", "BE4111159")
ONNX = ROOT / "models" / "yolov7-tiny" / "fruitfly_deploy.onnx"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    mpath = Path(a.manifest)
    with open(mpath, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["image_path"] and r["device_serial"] in LURE_CAMERAS]
    rows.sort(key=lambda r: (r["device_serial"], r["captured_at"]))
    size = {}
    for r in rows:
        im = cv2.imread(str(mpath.parent / r["image_path"]))
        size[r["snapshot_id"]] = None if im is None else (im.shape[1], im.shape[0])
    ok = {k: v is not None and v != (640, 640) for k, v in size.items()}
    pairs = [(p, c) for p, c in itertools.pairwise(rows)
             if p["device_serial"] == c["device_serial"] and ok[p["snapshot_id"]] and ok[c["snapshot_id"]]]
    if len(pairs) < a.n:
        raise SystemExit(f"only {len(pairs)} eligible pairs, need {a.n}")
    step = len(pairs) / a.n
    sel = [pairs[int(i * step)] for i in range(a.n)]

    out_csv = ROOT / "bench" / "inputs" / "pairs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    build = ROOT / "bench" / "build"
    build.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "camera", "prev_snapshot", "cur_snapshot", "prev_captured_at", "cur_captured_at",
                    "prev_path", "cur_path", "prev_size", "cur_size", "prev_sha256", "cur_sha256"])
        for i, (p, c) in enumerate(sel):
            rel = {}
            for tag, r in (("prev", p), ("cur", c)):
                src = mpath.parent / r["image_path"]
                rel[tag] = f"images/{r['device_serial']}/{r['snapshot_id']}.jpg"
                files[rel[tag]] = src
            w.writerow([i, c["device_serial"], p["snapshot_id"], c["snapshot_id"], p["captured_at"], c["captured_at"],
                        rel["prev"], rel["cur"], "x".join(map(str, size[p["snapshot_id"]])),
                        "x".join(map(str, size[c["snapshot_id"]])), sha256(files[rel["prev"]]),
                        sha256(files[rel["cur"]])])

    with tarfile.open(build / "bench_inputs.tar", "w") as tar:
        tar.add(out_csv, arcname="pairs.csv")
        for arc, src in sorted(files.items()):
            tar.add(src, arcname=arc)
        tar.add(ONNX, arcname="model/fruitfly_deploy.onnx")
        meta = ONNX.with_suffix(".onnx.json")
        if meta.exists():
            tar.add(meta, arcname="model/fruitfly_deploy.onnx.json")
    with tarfile.open(build / "sentinel_src.tar", "w") as tar:
        tar.add(ROOT / "sentinel", arcname="sentinel",
                filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t)
    sizes = {}
    for s in [size[p["snapshot_id"]] for p, _ in sel] + [size[c["snapshot_id"]] for _, c in sel]:
        sizes["x".join(map(str, s))] = sizes.get("x".join(map(str, s)), 0) + 1
    cams = {cam: sum(1 for _, c in sel if c["device_serial"] == cam) for cam in LURE_CAMERAS}
    print(f"eligible pairs {len(pairs)}, selected {len(sel)}, unique images {len(files)}, cameras {cams}, "
          f"frame sizes (prev+cur) {sizes}")
    print(f"onnx sha256 {sha256(ONNX)}")
    for f in ("bench_inputs.tar", "sentinel_src.tar"):
        print(f, (build / f).stat().st_size, sha256(build / f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
