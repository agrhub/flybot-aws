"""Validate op 6 on the camos fruit-fly test split: blob_count and saturation_pct vs labelled fly count.

    uv run python -m sentinel.vision.crowd_eval [--dataset ../camos-ai-camera/models/fruit-fly-v3/test]
        [--out docs/evidence/vision/crowd_eval.json] [--evidence docs/evidence/vision/crowd]

Ground truth = number of YOLO boxes per label file (Roboflow camos/fruit-fly-wmcqi v3, CC BY 4.0).
These 76 pictures show flies on a **white bait disc**, not on a yellow sticky card (the dataset has no
yellow card), so no card is found and op 6 runs in two surface modes:

* `full_frame`: the whole 640x640 picture (what `estimate_crowd` does when no card is found);
* `disc`: the bright surface found by the lure-floor segmentation (`card.find_lure_floor`), so the dark
  background outside the disc is not counted.

Reports Pearson r (and Spearman rho) of blob_count and saturation_pct against the label count, the
blob-count error, and the saturation distribution used to set `crowd.card_full_pct`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

from sentinel.vision.card import find_lure_floor
from sentinel.vision.crowd import measure_crowd, segments_image
from sentinel.vision.evidence import save_jpeg

DATASET = Path(__file__).resolve().parents[3] / "camos-ai-camera" / "models" / "fruit-fly-v3" / "test"


def label_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if len(line.split()) >= 5)


def surface_mask(img: np.ndarray) -> np.ndarray | None:
    f = find_lure_floor(img)
    if not f["found"]:
        return None
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(np.asarray(f["polygon"])).astype(np.int32), 255)
    return m


def _corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    return {"pearson_r": round(float(r), 4), "pearson_p": float(f"{p:.3g}"),
            "spearman_rho": round(float(rho), 4), "spearman_p": float(f"{prho:.3g}")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--out", default="docs/evidence/vision/crowd_eval.json")
    ap.add_argument("--evidence", default="docs/evidence/vision/crowd")
    ap.add_argument("--n-evidence", type=int, default=6)
    a = ap.parse_args(argv)
    d = Path(a.dataset)
    images = sorted((d / "images").glob("*.jpg"))
    rows = []
    for p in images:
        img = cv2.imread(str(p))
        gt = label_count(d / "labels" / (p.stem + ".txt"))
        row = {"image": p.name, "labels": gt}
        surf = surface_mask(img)
        for mode, s in (("full_frame", None), ("disc", surf)):
            m = measure_crowd(img, s)
            row[mode] = {k: m[k] for k in ("saturation_pct", "blob_count", "component_count", "split_count",
                                           "fly_radius_px")}
            row[mode]["_m"] = m
        row["disc_found"] = surf is not None
        rows.append(row)

    summary: dict = {"dataset": d.as_posix(), "n_images": len(rows),
                     "label_count": {"min": min(r["labels"] for r in rows), "median": float(np.median([r["labels"] for r in rows])),
                                     "max": max(r["labels"] for r in rows)},
                     "disc_found": sum(r["disc_found"] for r in rows), "modes": {}}
    gt = [r["labels"] for r in rows]
    for mode in ("full_frame", "disc"):
        bc = [r[mode]["blob_count"] for r in rows]
        cc = [r[mode]["component_count"] for r in rows]
        sat = [r[mode]["saturation_pct"] for r in rows]
        err = np.asarray(bc, float) - np.asarray(gt, float)
        summary["modes"][mode] = {
            "blob_count_vs_labels": _corr(bc, gt),
            "component_count_vs_labels": _corr(cc, gt),
            "saturation_pct_vs_labels": _corr(sat, gt),
            "blob_count_mae": round(float(np.abs(err).mean()), 3),
            "blob_count_bias": round(float(err.mean()), 3),
            "split_count_total": int(sum(r[mode]["split_count"] for r in rows)),
            "saturation_pct": {q: round(float(np.percentile(sat, v)), 3)
                               for q, v in (("min", 0), ("p50", 50), ("p90", 90), ("p95", 95), ("max", 100))},
            # least-squares line labels ~ a * saturation_pct + b: flies per % of covered surface
            "labels_per_saturation_pct": [round(float(v), 4) for v in np.polyfit(sat, gt, 1)],
        }
    # evidence: the densest and sparsest pictures, disc mode
    order = np.argsort(gt)
    n_hi = a.n_evidence - a.n_evidence // 2
    pick = list(order[: a.n_evidence // 2]) + (list(order[-n_hi:]) if n_hi else [])
    ev = Path(a.evidence)
    for i in pick:
        r = rows[i]
        m = r["disc"]["_m"]
        vis = segments_image(m["_crop"], m["_labels"])
        cv2.putText(vis, f"labels {r['labels']} blobs {m['blob_count']} sat {m['saturation_pct']:.1f}%", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        r["evidence"] = save_jpeg(vis, ev / f"{Path(r['image']).stem[:11]}_segments.jpg", max_side=640)
    for r in rows:
        for mode in ("full_frame", "disc"):
            r[mode].pop("_m", None)
    out = {"summary": summary, "rows": rows,
           "credit": "camos/fruit-fly-wmcqi v3 test split, Roboflow Universe, CC BY 4.0"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
