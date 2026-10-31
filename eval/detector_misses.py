"""Which synthetic flies the YOLO26 detector misses, and what they have in common.

    uv run --project lambda/vision python -m eval.detector_misses [--variant production] [--floor-lure 0.35]
        [--out eval/results/detector_misses.json]

Reads the synthetic cache of eval.detector_ab (`cache --only synth`; built with the decoder floor CACHE_CONF, so
boxes under the Lambda's 0.25 are there) and the generator's manifests (data/synth/demo-30d, data/synth/pool). For each
pasted fly (ground truth) under one detector_ab variant: counted or missed (greedy one-to-one at IoU 0.3, as
detector_ab), and for a missed fly why, from the boxes the cache holds at its place (IoU >= 0.3):

    below_floor     a fruit_fly box is there but scores under the variant's floor (>= CACHE_CONF)
    other_insect    the best box there is the other_insect class
    dropped         a fruit_fly box above the floor is there but the trap ROI or a box rule removed it
    taken           a counted box is there but the matching gave it to a neighbouring fly
    unseen          no box at all (the model does not see it at this size / scale)

and its attributes: long side at the detector input, its box's aspect ratio, angle, the nearest other fly (IoU and
centre distance in fly lengths), distance to the edge of the trap floor op 1 found (fly lengths, > 0 inside), distance
to the floor's centre (share of the floor's radius), local contrast (mean grey of the ring around the box minus the
box, 0-255), and which fly-bank crop was pasted. Miss rates per bin of each attribute follow. Pixel statistics are
computed from the files; no photo is looked at.

`carry` estimates the rule "keep a fly seen in the previous photo" on sticky cards (the synthetic lure photos are
independent: their flies do not stay): with the nearest earlier cached photo of the same card, a
current box under the floor (>= --carry-min) is counted when the previous photo counted a box at IoU >= 0.5 there.
Recall gained, and false flies added on the synthetic photos, per trap kind. All rows are SIMULATED data.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lambda" / "vision"))
RESULTS = ROOT / "eval" / "results" / "detector_misses.json"


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def manifest_rows() -> dict[str, dict]:
    from eval.detector_ab import SYNTH_SETS

    rows = {}
    for d in SYNTH_SETS:
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        rows.update({p["photoId"]: p for p in m["photos"]})
    return rows


def candidates(rec: dict, v: dict) -> list[dict]:
    """Every box the variant's detector pass produced (tiles merged as the variant does), before ROI and rules."""
    from sentinel.vision import tiling

    mode = v.get("tiles", {}).get(rec["kind"], "off")
    t = None if mode == "off" else rec["frame_tiles"] if mode == "frame" else rec.get("roi_tiles")
    if t is None:
        return rec["full"]
    full = rec["full"] if mode == "roi+full" else None
    return tiling.merge(t["per"], t["wins"], rec["W"], rec["H"], 0.45, t["margins"], full, t["rect"])[0]


def contrast(gray: np.ndarray, b: list[float]) -> float | None:
    H, W = gray.shape
    x0, y0, x1, y1 = (round(c) for c in b)
    pad = max(3, int(0.5 * max(x1 - x0, y1 - y0)))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    outer = gray[Y0:Y1, X0:X1].astype(np.float64)
    inner = gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)].astype(np.float64)
    ring = (outer.sum() - inner.sum()) / max(1, outer.size - inner.size)
    return round(float(ring - inner.mean()), 1)


def analyse(recs: list[dict], rows: dict, v: dict, input_px: int = 640) -> list[dict]:
    import cv2
    from eval.detector_ab import GRID, OVERLAP, boxes_for, xyxy
    from eval.synth.check import gt_boxes, match_pairs

    floor = v.get("floor", {})
    out = []
    for r in recs:
        p = rows.get(r["id"])
        if p is None or not r["truth"]:
            continue
        kind, W, H = r["kind"], r["W"], r["H"]
        kept, _ = boxes_for(r, v)
        kept_xy = [xyxy(d) for d in kept]
        gts = gt_boxes(p)
        meta = [b for b in p["boxes"] if not b.get("hidden")]
        hit = match_pairs(gts, kept_xy, 0.3)
        cands = [(xyxy(d), d) for d in candidates(r, v)]
        tiled = v.get("tiles", {}).get(kind, "off") != "off"
        scale = input_px / max(W, H) * ((GRID / (1 + OVERLAP)) if tiled else 1.0)
        poly = np.array(r["roi"], np.float32).reshape(-1, 1, 2) if r.get("roi") else None
        if poly is not None:
            m = cv2.moments(poly)
            cxy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else None
            radius = (cv2.contourArea(poly) / np.pi) ** 0.5
        gray = cv2.imread(r["file"], cv2.IMREAD_GRAYSCALE)
        f = floor.get(kind, 0.35)
        for i, (g, b) in enumerate(zip(gts, meta, strict=True)):
            gw, gh = g[2] - g[0], g[3] - g[1]
            L = max(gw, gh)
            cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            others = [o for j, o in enumerate(gts) if j != i]
            nn_iou = max((iou(g, o) for o in others), default=0.0)
            nn_d = min((np.hypot(cx - (o[0] + o[2]) / 2, cy - (o[1] + o[3]) / 2) for o in others), default=None)
            here = [(iou(g, q), d) for q, d in cands if iou(g, q) >= 0.3]
            best = max(here, key=lambda t: t[1]["score"])[1] if here else None
            row = {"id": r["id"], "trap": r["trap"], "kind": kind, "hit": i in hit,
                   "input_px": round(b["longSidePx"] * scale, 1), "aspect": round(max(gw, gh) / max(1e-6, min(gw, gh)), 2),
                   "angle": round(b["angle"] % 180, 1), "nn_iou": round(nn_iou, 3),
                   "nn_dist": None if nn_d is None else round(float(nn_d) / L, 2), "crop": b["flyCrop"],
                   "contrast": None if gray is None else contrast(gray, g),
                   "best_score": None if best is None else best["score"],
                   "best_class": None if best is None else int(best.get("class_id", 0)), "flies": len(gts)}
            if poly is not None:
                row["edge"] = round(float(cv2.pointPolygonTest(poly, (float(cx), float(cy)), True)) / L, 2)
                if cxy:
                    row["centre"] = round(float(np.hypot(cx - cxy[0], cy - cxy[1])) / radius, 2)
            if not row["hit"]:
                if best is None:
                    why = "unseen"
                elif best.get("class_id", 0) != 0:
                    why = "other_insect"
                elif best["score"] < f:
                    why = "below_floor"
                elif any(iou(g, q) >= 0.3 for q in kept_xy):
                    why = "taken"
                else:
                    why = "dropped"
                row["why"] = why
            out.append(row)
    return out


BINS = {
    "input_px": [0, 12, 16, 20, 24, 28, 32, 40, 1e9],
    "aspect": [0, 1.25, 1.5, 2.0, 1e9],
    "angle": [0, 30, 60, 90, 120, 150, 180.1],
    "nn_iou": [0, 1e-9, 0.02, 0.05, 0.1, 1.01],
    "nn_dist": [0, 1.0, 1.5, 2.0, 3.0, 1e9],
    "edge": [-1e9, 0, 0.5, 1.0, 2.0, 1e9],
    "centre": [0, 0.2, 0.4, 0.6, 0.8, 1e9],
    "contrast": [-1e9, 10, 20, 30, 40, 60, 1e9],
    "flies": [0, 5, 10, 20, 40, 1e9],
}


def binned(rows: list[dict], key: str) -> list[dict]:
    edges, out = BINS[key], []
    for lo, hi in pairwise(edges):
        rs = [r for r in rows if r.get(key) is not None and lo <= r[key] < hi]
        if rs:
            miss = sum(not r["hit"] for r in rs)
            out.append({"bin": [lo if lo > -1e8 else None, hi if hi < 1e8 else None], "flies": len(rs), "missed": miss,
                        "miss_rate": round(miss / len(rs), 3)})
    return out


def summary(rows: list[dict]) -> dict:
    missed = [r for r in rows if not r["hit"]]
    out = {"flies": len(rows), "missed": len(missed), "recall": round(1 - len(missed) / len(rows), 4) if rows else None,
           "why": dict(Counter(r["why"] for r in missed).most_common()),
           "best_score_of_below_floor": _quant([r["best_score"] for r in missed if r["why"] == "below_floor"]),
           "median": {k: {"hit": _med([r.get(k) for r in rows if r["hit"]]), "missed": _med([r.get(k) for r in missed])}
                      for k in BINS},
           "by": {k: binned(rows, k) for k in BINS}}
    crops = defaultdict(list)
    for r in rows:
        crops[r["crop"]].append(r["hit"])
    rates = sorted(((sum(not h for h in hs), len(hs), c) for c, hs in crops.items() if len(hs) >= 5), reverse=True)
    if rates:
        top = rates[: max(1, len(rates) // 10)]
        out["crops"] = {"crops_seen_5_times": len(rates),
                        "misses_from_worst_10pct_crops": sum(m for m, _, _ in top),
                        "worst": [{"crop": c, "missed": m, "of": n} for m, n, c in rates[:10]],
                        "crops_never_counted": sum(m == n for m, n, _ in rates)}
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def _quant(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"n": len(xs), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "at_or_above_0.25": sum(x >= 0.25 for x in xs),
            "at_or_above_0.20": sum(x >= 0.20 for x in xs), "at_or_above_0.15": sum(x >= 0.15 for x in xs)}


def carry(recs: list[dict], rows: dict, v: dict, carry_min: float) -> dict:
    """Recall and false flies with "keep a box under the floor where the previous photo counted one"."""
    from eval.detector_ab import boxes_for, xyxy
    from eval.synth.check import gt_boxes, match

    by_id = {r["id"]: r for r in recs}

    def previous(p: dict) -> dict | None:
        """The nearest earlier photo of the same card that the cache holds (it may sample every N-th photo)."""
        for _ in range(4):
            if p.get("cardChanged") or not p.get("previousPhotoId"):
                return None
            if p["previousPhotoId"] in by_id:
                return by_id[p["previousPhotoId"]]
            p = rows.get(p["previousPhotoId"]) or {}
        return None

    agg = defaultdict(lambda: Counter())
    for r in recs:
        p = rows.get(r["id"])
        prev = previous(p) if p and p["trapKind"] == "sticky" else None  # lure photos are independent (eval/synth)
        if prev is None:
            continue
        kept, _ = boxes_for(r, v)
        prev_kept = [xyxy(d) for d in boxes_for(prev, v)[0]]
        kept_xy = [xyxy(d) for d in kept]
        low = {**v, "floor": {r["kind"]: carry_min}}
        extra = [xyxy(d) for d in boxes_for(r, low)[0]
                 if d["score"] < v.get("floor", {}).get(r["kind"], 0.35)
                 and any(iou(xyxy(d), q) >= 0.5 for q in prev_kept) and all(iou(xyxy(d), k) < 0.45 for k in kept_xy)]
        gts = gt_boxes(p)
        a = agg[r["kind"]]
        a["photos"] += 1
        a["true"] += len(gts)
        a["tp"] += match(gts, kept_xy, 0.3)
        a["det"] += len(kept_xy)
        a["tp_carry"] += match(gts, kept_xy + extra, 0.3)
        a["det_carry"] += len(kept_xy) + len(extra)
    out = {}
    for k, a in agg.items():
        out[k] = {"photos_with_previous": a["photos"], "true": a["true"],
                  "recall": round(a["tp"] / a["true"], 4) if a["true"] else None,
                  "precision": round(a["tp"] / a["det"], 4) if a["det"] else None,
                  "recall_carry": round(a["tp_carry"] / a["true"], 4) if a["true"] else None,
                  "precision_carry": round(a["tp_carry"] / a["det_carry"], 4) if a["det_carry"] else None,
                  "flies_added": a["tp_carry"] - a["tp"], "false_added": (a["det_carry"] - a["det"]) - (a["tp_carry"] - a["tp"])}
    return out


def main(argv: list[str] | None = None) -> int:
    from eval.detector_ab import load, variants

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="production", help="a detector_ab variant name")
    ap.add_argument("--floor-lure", type=float, default=None)
    ap.add_argument("--floor-sticky", type=float, default=None)
    ap.add_argument("--carry-min", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    data = load("synth")
    if data is None:
        raise SystemExit("no synthetic cache: run `python -m eval.detector_ab cache --only synth` first")
    v = dict(variants()[a.variant])
    floor = {**v.get("floor", {})}
    if a.floor_lure is not None:
        floor["lure"] = a.floor_lure
    if a.floor_sticky is not None:
        floor["sticky"] = a.floor_sticky
    v["floor"] = floor
    rows = manifest_rows()
    flies = analyse(data["records"], rows, v)
    out = {"kind": "missed synthetic flies of the YOLO26 detector step; SIMULATED data",
           "generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "cache": data["meta"],
           "variant": a.variant, "settings": v, "notes": __doc__.split("\n\n", 2)[2].strip(),
           "by_kind": {k: summary([r for r in flies if r["kind"] == k]) for k in ("lure", "sticky")},
           "by_trap": {t: {"flies": len(rs), "recall": round(sum(r["hit"] for r in rs) / len(rs), 4),
                           "why": dict(Counter(r["why"] for r in rs if not r["hit"]).most_common())}
                       for t in sorted({r["trap"] for r in flies}) for rs in [[r for r in flies if r["trap"] == t]]},
           "carry": {"min_score": a.carry_min, **carry(data["records"], rows, v, a.carry_min)}}
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    for k, s in out["by_kind"].items():
        print(k, s["flies"], "recall", s["recall"], s["why"])
        print("  median hit/missed", s["median"])
    print("carry", json.dumps(out["carry"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
