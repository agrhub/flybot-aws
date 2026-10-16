"""Hand-check set for op 1: 50 export photos spread over the 3 cameras and their days.

    uv run python -m sentinel.vision.card_check [--n 50] [--out docs/evidence/vision/card_check]
        [--sheet SCRATCH.jpg]
    uv run python -m sentinel.vision.card_check --trap-kind lure --out docs/evidence/vision/card_check_lure

`--trap-kind lure` checks the lure profile instead: 50 real camera frames of the two lure-trap cameras
(BE4089629, BE4111159; the 640x640 uploaded pictures are left out), overlay = tray floor polygon
(cyan) and 4-corner fit (green), everything outside the polygon pixelated.

Runs `find_card` with the `yellow` profile (the plan's glue card) and the `auto` profile (yellow,
then the white trap surface), writes one privacy-safe overlay per photo (everything outside the
detected quad pixelated) plus results.json. Verdicts are given by a person looking at the photos
and written to docs/evidence/vision/card_check.md. `--sheet` writes an unpixelated contact sheet for
that person to a scratch path (not for git).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.card import draw_card, find_card
from sentinel.vision.evidence import pixelate_outside, save_jpeg


def pick(rows: list[dict], n: int) -> list[dict]:
    """Round-robin over cameras, then spread over each camera's farm days and hours."""
    by_cam = defaultdict(lambda: defaultdict(list))
    for r in rows:
        day = (datetime.fromisoformat(r["captured_at"]) + timedelta(hours=7)).date().isoformat()
        by_cam[r["device_serial"]][day].append(r)
    cams = sorted(by_cam)
    per_cam = [n // len(cams) + (1 if i < n % len(cams) else 0) for i in range(len(cams))]
    out = []
    for cam, k in zip(cams, per_cam):
        days = sorted(by_cam[cam])
        idx = np.linspace(0, len(days) - 1, k).round().astype(int)
        for j, di in enumerate(idx):
            lst = sorted(by_cam[cam][days[di]], key=lambda r: r["captured_at"])
            out.append(lst[(j * 7) % len(lst)])  # vary the hour
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="docs/evidence/vision/card_check")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--trap-kind", default="sticky", choices=["sticky", "lure"])
    a = ap.parse_args(argv)
    mpath = Path(a.manifest)
    with open(mpath, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if a.trap_kind == "lure":
        return check_lure(rows, mpath, a)
    sel = pick(rows, a.n)
    out = Path(a.out)
    results, tiles = [], []
    for i, r in enumerate(sel, 1):
        img = cv2.imread(str(mpath.parent / r["image_path"]))
        y = find_card(img, profile="yellow")
        w = find_card(img, profile="auto")
        vis = draw_card(img, w if w["found"] else y)
        name = f"{i:02d}_{r['device_serial']}_{r['snapshot_id']}.jpg"
        path = save_jpeg(pixelate_outside(vis, w["quad"]), out / name, max_side=640)
        results.append({
            "i": i, "snapshot_id": r["snapshot_id"], "camera": r["device_serial"], "captured_at": r["captured_at"],
            "size": [img.shape[1], img.shape[0]], "ai_count": r["ai_count"],
            "yellow": {k: y[k] for k in ("found", "coverage", "n_cards", "partial")},
            "auto": {k: w[k] for k in ("found", "profile", "coverage", "n_cards", "partial", "quad_norm")},
            "evidence": path,
        })
        if a.sheet:
            t = cv2.resize(vis, (320, int(320 * img.shape[0] / img.shape[1])) if img.shape[1] >= img.shape[0] else (int(320 * img.shape[1] / img.shape[0]), 320))
            c = np.zeros((320, 320, 3), np.uint8)
            c[: t.shape[0], : t.shape[1]] = t
            cv2.putText(c, f"#{i} {r['device_serial'][-4:]} {r['captured_at'][5:16]}", (3, 312), 0, 0.45, (0, 0, 255), 1)
            tiles.append(c)
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    if a.sheet and tiles:
        while len(tiles) % 10:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[k : k + 10]) for k in range(0, len(tiles), 10)])
        cv2.imwrite(a.sheet, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps({"n": len(results), "yellow_found": sum(r["yellow"]["found"] for r in results),
                      "auto_found": sum(r["auto"]["found"] for r in results)}))
    return 0


LURE_CAMERAS = ("BE4089629", "BE4111159")


def check_lure(rows: list[dict], mpath: Path, a) -> int:
    """Lure profile on real lure-trap camera frames (not the 640x640 uploaded pictures)."""
    frames = []
    for r in rows:
        if r["device_serial"] not in LURE_CAMERAS:
            continue
        img = cv2.imread(str(mpath.parent / r["image_path"]))
        if img is not None and img.shape[:2] != (640, 640):
            frames.append(r)
    sel = pick(frames, a.n)
    out = Path(a.out)
    results, tiles = [], []
    for i, r in enumerate(sel, 1):
        img = cv2.imread(str(mpath.parent / r["image_path"]))
        res = find_card(img, trap_kind="lure")
        vis = draw_card(img, res)
        name = f"{i:02d}_{r['device_serial']}_{r['snapshot_id']}.jpg"
        keep = res["polygon"] if res["found"] else None
        path = save_jpeg(pixelate_outside(vis, keep) if keep else vis, out / name, max_side=640)
        results.append({
            "i": i, "snapshot_id": r["snapshot_id"], "camera": r["device_serial"], "captured_at": r["captured_at"],
            "size": [img.shape[1], img.shape[0]],
            "lure": {k: res[k] for k in ("found", "roi_source", "coverage", "fill", "quad_norm", "roi")},
            "floor_checks": res["floor_checks"], "time_ms": res["time_ms"], "evidence": path,
        })
        if a.sheet:
            t = cv2.resize(vis, (384, 216))
            cv2.putText(t, f"#{i} {r['device_serial'][-4:]} {r['captured_at'][5:16]}", (3, 208), 0, 0.45, (0, 0, 255), 1)
            tiles.append(t)
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    if a.sheet and tiles:
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[k : k + 5]) for k in range(0, len(tiles), 5)])
        cv2.imwrite(a.sheet, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    src = [r["lure"]["roi_source"] for r in results]
    print(json.dumps({"n": len(results), "found": sum(r["lure"]["found"] for r in results),
                      "floor_quad": src.count("floor_quad"), "floor_polygon": src.count("floor_polygon"),
                      "whole_frame": src.count("whole_frame"),
                      "time_ms_median": float(np.median([r["time_ms"] for r in results]))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
