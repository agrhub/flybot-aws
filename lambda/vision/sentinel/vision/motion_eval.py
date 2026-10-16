"""Evaluate op 8 on real consecutive lure-trap pairs and synthetic composites.

    uv run python -m sentinel.vision.motion_eval [--n-real 200] [--n-synth 100]
        [--out docs/evidence/vision/motion_eval.json] [--evidence docs/evidence/vision/motion]

* Real pairs: consecutive photos of BE4089629 / BE4111159 (real camera frames, 640x640 uploads left
  out), shuffled with seed 5. Pairs 1-200 of that order were used to choose RATIO_T / CHROMA_T
  (scratch experiment); this evaluation uses the **next** `n_real` pairs, so it is held out. Nobody put
  a hand in these traps, so every `occluded_by_large_object` here is a false positive unless a person
  confirms otherwise (each flagged pair is listed for a look). `camera_moved` flags are listed the same way.
* Synthetic occluders: on held-out real pairs, a large object (skin-toned hand/arm, dark animal/leaf,
  coloured cloth texture; 4-35 % of the frame) is pasted into the current photo -> recall and IoU of the
  occluder box/mask.
* Synthetic camera moves: the current photo shifted 3-8 % of the diagonal and rotated up to 4 deg
  -> recall of `camera_moved`.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.evidence import pixelate, save_jpeg
from sentinel.vision.motion import detect_motion

LURE_CAMERAS = ("BE4089629", "BE4111159")
WORK = (640, 360)


def _load(mdir: Path, r: dict) -> np.ndarray | None:
    im = cv2.imread(str(mdir / r["image_path"])) if r["image_path"] else None
    if im is None or im.shape[:2] == (640, 640):
        return None
    return cv2.resize(im, WORK, interpolation=cv2.INTER_AREA)


def real_pairs(manifest: Path, skip: int, n: int) -> list[tuple]:
    with open(manifest, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["image_path"]]
    lure = sorted([r for r in rows if r["device_serial"] in LURE_CAMERAS], key=lambda r: (r["device_serial"], r["captured_at"]))
    pairs = [(a, b) for a, b in itertools.pairwise(lure) if a["device_serial"] == b["device_serial"]]
    random.seed(5)
    random.shuffle(pairs)
    out, valid = [], 0
    for a, b in pairs:
        ia, ib = _load(manifest.parent, a), _load(manifest.parent, b)
        if ia is None or ib is None:
            continue
        valid += 1
        if valid <= skip:
            continue
        out.append((ia, ib, a, b))
        if len(out) == n:
            break
    return out


def occluder(rng: np.random.Generator, W: int = WORK[0], H: int = WORK[1]) -> tuple[np.ndarray, np.ndarray, str]:
    """(patch BGR, mask, kind) of a random large object."""
    kind = ("hand", "animal", "cloth")[int(rng.integers(0, 3))]
    m = np.zeros((H, W), np.uint8)
    cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H
    ax, ay = rng.uniform(0.12, 0.3) * W, rng.uniform(0.12, 0.3) * H
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), rng.uniform(0, 180), 0, 360, 255, -1)
    if rng.random() < 0.5:  # an arm entering from the side
        cv2.line(m, (int(cx), int(cy)), (int(rng.choice([0, W])), int(cy + rng.uniform(-50, 50))), 255, int(ay * 0.8))
    if kind == "hand":
        base = np.array([rng.uniform(90, 160), rng.uniform(120, 180), rng.uniform(160, 230)])
        patch = cv2.GaussianBlur(np.clip(base + rng.normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8), (9, 9), 0)
        shade = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 25)
        patch = np.clip(patch * (0.6 + 0.4 * shade[..., None]), 0, 255).astype(np.uint8)
    elif kind == "animal":
        base = np.array([rng.uniform(30, 90), rng.uniform(40, 110), rng.uniform(30, 90)])
        patch = cv2.GaussianBlur(np.clip(base + rng.normal(0, 12, (H, W, 3)), 0, 255).astype(np.uint8), (5, 5), 0)
    else:
        yy, xx = np.mgrid[0:H, 0:W]
        per = rng.uniform(8, 30)
        c1, c2 = rng.uniform(20, 230, 3), rng.uniform(20, 230, 3)
        t = ((np.sin(xx / per) > 0) ^ (np.sin(yy / per * 1.3) > 0))[..., None]
        patch = np.clip(np.where(t, c1, c2) + rng.normal(0, 8, (H, W, 3)), 0, 255).astype(np.uint8)
    return patch, m, kind


def paste(img: np.ndarray, patch: np.ndarray, m: np.ndarray) -> np.ndarray:
    a = cv2.GaussianBlur(m, (7, 7), 0).astype(np.float32)[..., None] / 255
    return np.clip(img * (1 - a) + patch * a, 0, 255).astype(np.uint8)


def moved(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    share = rng.uniform(0.03, 0.08)
    ang = rng.uniform(0, 2 * np.pi)
    d = share * np.hypot(w, h)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4, 4), 1.0)
    M[:, 2] += (d * np.cos(ang), d * np.sin(ang))
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), share


def _iou_box(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--n-real", type=int, default=200)
    ap.add_argument("--n-synth", type=int, default=100)
    ap.add_argument("--skip", type=int, default=200, help="pairs used for tuning (left out)")
    ap.add_argument("--out", default="docs/evidence/vision/motion_eval.json")
    ap.add_argument("--evidence", default="docs/evidence/vision/motion")
    a = ap.parse_args(argv)
    pairs = real_pairs(Path(a.manifest), a.skip, a.n_real)
    ev = Path(a.evidence)
    real_rows, times = [], []
    for ia, ib, ra, rb in pairs:
        t0 = time.perf_counter()
        m = detect_motion(ia, ib)
        times.append((time.perf_counter() - t0) * 1000)
        real_rows.append({"prev": ra["snapshot_id"], "cur": rb["snapshot_id"], "camera": rb["device_serial"],
                          "prev_at": ra["captured_at"], "cur_at": rb["captured_at"],
                          "occluded": m["occluded_by_large_object"], "camera_moved": m["camera_moved"],
                          "moved_reasons": m["moved_reasons"], "global_change": m["global_change"],
                          "changed_share": m["changed_share"], "inliers": m["registration"]["inliers"]})
    rng = np.random.default_rng(20261015)
    occ_rows, move_rows = [], []
    for k, (ia, ib, _, rb) in enumerate(pairs[: a.n_synth]):
        patch, mask, kind = occluder(rng)
        cur = paste(ib, patch, mask)
        m = detect_motion(ia, cur)
        ys, xs = np.nonzero(mask)
        gt_box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        o = m["occluder"] or {}
        occ_rows.append({"cur": rb["snapshot_id"], "kind": kind, "area_share": round(float(mask.mean() / 255), 4),
                         "detected": m["occluded_by_large_object"], "in_frame": o.get("in_frame"),
                         "box_iou": round(_iou_box(o["bbox"], gt_box), 4) if o.get("bbox") else 0.0,
                         "camera_moved": m["camera_moved"]})
        if k < 3:
            vis = pixelate(cur)
            if o.get("bbox"):
                x1, y1, x2, y2 = (int(v) for v in o["bbox"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis, f"synthetic {kind}: occluded={m['occluded_by_large_object']}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            occ_rows[-1]["evidence"] = save_jpeg(vis, ev / f"synth_occluder_{k}.jpg", max_side=640)
        mv, share = moved(ib, rng)
        m2 = detect_motion(ib, mv)
        move_rows.append({"cur": rb["snapshot_id"], "shift_share": round(share, 4), "detected": m2["camera_moved"],
                          "reasons": m2["moved_reasons"], "measured_share": (m2["shift"] or {}).get("share"),
                          "occluded": m2["occluded_by_large_object"]})
    n_occ = len(occ_rows)
    det = [r for r in occ_rows if r["detected"]]
    summary = {
        "real_pairs": len(real_rows),
        "real_occluded_flags": sum(r["occluded"] for r in real_rows),
        "real_camera_moved_flags": sum(r["camera_moved"] for r in real_rows),
        "real_global_change": sum(r["global_change"] for r in real_rows),
        "synthetic_occluders": n_occ,
        "occluder_recall": round(len(det) / n_occ, 4) if n_occ else None,
        "occluder_recall_by_kind": {k: f"{sum(r['detected'] for r in occ_rows if r['kind'] == k)}/"
                                       f"{sum(r['kind'] == k for r in occ_rows)}" for k in ("hand", "animal", "cloth")},
        "occluder_box_iou_mean": round(float(np.mean([r["box_iou"] for r in det])), 4) if det else None,
        "occluder_in_frame_cur": sum(r["in_frame"] == "cur" for r in det),
        "synthetic_moves": len(move_rows),
        "camera_moved_recall": round(sum(r["detected"] for r in move_rows) / len(move_rows), 4) if move_rows else None,
        "moves_flagged_as_occlusion": sum(r["occluded"] for r in move_rows),
        "time_ms_real": {"p50": round(float(np.median(times)), 1), "p95": round(float(np.percentile(times, 95)), 1)},
        "frame_size": list(WORK),
        "tuning": f"real pairs 1-{a.skip} (seed-5 order) chose RATIO_T/CHROMA_T; these are pairs {a.skip + 1}-"
                  f"{a.skip + len(real_rows)}",
    }
    out = {"summary": summary, "real": real_rows, "synthetic_occluders": occ_rows, "synthetic_moves": move_rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
