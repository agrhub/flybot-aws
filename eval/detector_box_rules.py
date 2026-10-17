"""YOLO26n post-processing evaluation: the previous decoding (end-to-end head, score >= 0.25, no NMS) against NMS and
the fruit-fly box rules (sentinel.vision.box_rules) on the OpenCV 5 DNN new engine.

    cd lambda/vision && uv run python ../../eval/detector_box_rules.py [--dataset ../../../camos-ai-camera/models/fruit-fly-v3]

Writes eval/results/yolo26_box_rules.json:
  labels     box statistics of the camos/fruit-fly v3 labels (train / valid / test), the evidence for the size,
             median-ratio, aspect and NMS defaults;
  splits     precision / recall / F1 at IoU 0.5 on the valid split (used to choose the score and aspect limits) and
             the test split (held out), for: before (previous decoding), nms (NMS only), after (NMS + rules);
             duplicate pairs (IoU >= 0.5) left by the previous decoding; score quantiles of true positives;
             a sweep of the score and aspect limits on both splits;
  samples    the 24 bundled lure-trap photos (data/samples): count per photo before / after, the boxes the rules
             removed (with their side relative to the photo), and the count error against FlyBot's production
             count (Roboflow RF-DETR, not human ground truth).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lambda" / "vision"))
sys.path.insert(0, str(ROOT))

from eval.run_eval import iou_xyxy, match_image, roboflow_split

FLOOR = 0.25  # the previous score threshold (DETECTOR_CONF default)


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def q(values, ps=(0, 0.1, 1, 5, 50, 95, 99, 99.9, 100)) -> dict:
    a = np.asarray(values, dtype=float)
    return {str(p): round(float(np.percentile(a, p)), 4) for p in ps} if len(a) else {}


def label_stats(split_dir: Path, rules) -> dict:
    """Side = sqrt(w h) as a fraction of sqrt(W H) (labels are normalised, so of the image), aspect, ratio to the photo's
    median side (photos with >= median_min_boxes boxes), best IoU with another labelled fly."""
    sides, aspects, ratios, neighbour = [], [], [], []
    for lab in sorted((split_dir / "labels").glob("*.txt")):
        boxes = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                _, cx, cy, w, h = map(float, p)
                boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, w, h))
        if not boxes:
            continue
        s = [(b[4] * b[5]) ** 0.5 for b in boxes]
        sides += s
        aspects += [max(b[4], b[5]) / max(1e-9, min(b[4], b[5])) for b in boxes]
        if len(s) >= rules.median_min_boxes:
            m = statistics.median(s)
            ratios += [v / m for v in s]
        for i, a in enumerate(boxes):
            neighbour.append(max([iou_xyxy(a[:4], b[:4]) for j, b in enumerate(boxes) if j != i] or [0.0]))
    sides, aspects, ratios, neighbour = map(np.asarray, (sides, aspects, ratios, neighbour))
    return {"boxes": len(sides), "side": q(sides), "aspect": q(aspects), "median_ratio": q(ratios),
            "neighbour_iou": q(neighbour),
            "above_max_side": int((sides > rules.max_side).sum()), "below_min_side": int((sides < rules.min_side).sum()),
            "above_max_aspect": int((aspects > rules.max_aspect).sum()),
            "above_max_median_ratio": int((ratios > rules.max_median_ratio).sum()),
            "below_min_median_ratio": int((ratios < rules.min_median_ratio).sum()), "median_ratio_boxes": len(ratios),
            "neighbour_iou_ge_nms": int((neighbour >= 0.45).sum()), "neighbour_iou_ge_0.5": int((neighbour >= 0.5).sum())}


def xyxy(d: dict) -> list[float]:
    x, y, w, h = d["xywh"]
    return [x, y, x + w, y + h, d["score"]]


def prf(images: list[tuple[list[dict], list[list[float]]]]) -> dict:
    tp = fp = 0
    n_gt = sum(len(g) for _, g in images)
    for dets, gts in images:
        for _, t in match_image([xyxy(d) for d in dets], gts, 0.5):
            tp, fp = tp + t, fp + (not t)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": n_gt - tp}


def dup_pairs(dets: list[dict], thr: float = 0.5) -> int:
    b = [xyxy(d) for d in dets]
    return sum(1 for i in range(len(b)) for j in range(i + 1, len(b)) if iou_xyxy(b[i][:4], b[j][:4]) >= thr)


def run_split(det, split_dir: Path) -> dict:
    import cv2

    from sentinel.vision.box_rules import BoxRules, apply_rules
    from sentinel.vision.detect_yolo26 import decode, decode_end2end

    rules = det.rules or BoxRules()
    items = roboflow_split(split_dir)
    per = []  # (before, nms_low, gts, w, h)
    for path, gts in items:
        img = cv2.imread(str(path))
        out, scale, offset, _ = det.forward(img)
        if det.head != "end2end":
            raise SystemExit(f"{det.onnx_path.name} is not the end-to-end graph")
        before = decode_end2end(out, FLOOR, scale, offset)
        low = decode(out, 0.05, det.iou, scale, offset)  # NMS'd, low floor for the score study and sweeps
        per.append((before, low, gts, img.shape[1], img.shape[0]))

    def with_rules(r) -> dict:
        return prf([(apply_rules([d for d in low if d["score"] >= FLOOR], w, h, r)[0], g) for _, low, g, w, h in per])

    after_rej: Counter = Counter()
    after_imgs = []
    for _, low, g, w, h in per:
        kept, rej = apply_rules([d for d in low if d["score"] >= FLOOR], w, h, rules)
        after_rej.update(rej)
        after_imgs.append((kept, g))
    tp_scores = []
    for _, low, g, _, _ in per:
        dets = [xyxy(d) for d in low]
        tp_scores += [s for s, t in match_image(dets, g, 0.5) if t]
    return {
        "images": len(items), "gt_boxes": sum(len(g) for _, _, g, _, _ in per),
        "before": prf([(b, g) for b, _, g, _, _ in per]),
        "before_duplicate_pairs_iou_0.5": sum(dup_pairs(b) for b, *_ in per),
        "nms": prf([([d for d in low if d["score"] >= FLOOR], g) for _, low, g, _, _ in per]),
        "after": prf(after_imgs), "after_rejected": dict(after_rej),
        "tp_score_quantiles_after_nms": q(tp_scores), "tp_n": len(tp_scores),
        "tp_below_min_score": round(float(np.mean(np.asarray(tp_scores) < rules.min_score)), 4) if tp_scores else None,
        "sweep_min_score": {f"{s:.2f}": with_rules(replace(rules, min_score=s)) for s in (0.25, 0.30, 0.35, 0.40, 0.45)},
        "sweep_max_aspect": {f"{a:.1f}": with_rules(replace(rules, max_aspect=a)) for a in (3.0, 4.0, 5.0, 1e9)},
        "sweep_max_side": {f"{m:.2f}": with_rules(replace(rules, max_side=m)) for m in (0.10, 0.12, 0.15, 0.20)},
    }


def run_samples(det, manifest: Path) -> dict:
    import cv2

    from sentinel.vision.box_rules import apply_rules
    from sentinel.vision.detect_yolo26 import decode, decode_end2end

    with open(manifest, encoding="utf-8", newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: (r["device_serial"], r["captured_at"]))
    photos, pairs_before, pairs_after, removed = [], [], [], []
    for r in rows:
        img = cv2.imread(str(manifest.parent / r["image_path"]))
        h, w = img.shape[:2]
        out, scale, offset, _ = det.forward(img)
        before = decode_end2end(out, FLOOR, scale, offset)
        kept, rej = apply_rules(decode(out, FLOOR, det.iou, scale, offset), w, h, det.rules)
        root = (w * h) ** 0.5
        for d in decode(out, FLOOR, det.iou, scale, offset):
            if d not in kept:
                bw, bh = d["xywh"][2], d["xywh"][3]
                removed.append({"photo": r["snapshot_id"], "camera": r["device_serial"], "score": d["score"],
                                "side": round((bw * bh) ** 0.5 / root, 4), "aspect": round(max(bw, bh) / max(1e-6, min(bw, bh)), 2),
                                "centre": [round((d["xywh"][0] + bw / 2) / w, 2), round((d["xywh"][1] + bh / 2) / h, 2)]})
        prod = int(r["ai_count"]) if r.get("ai_count") not in (None, "") else None
        photos.append({"snapshot_id": r["snapshot_id"], "camera": r["device_serial"], "size": [w, h], "production": prod,
                       "before": len(before), "after": len(kept), "rejected": {k: v for k, v in rej.items() if v}})
        if prod is not None:
            pairs_before.append(abs(len(before) - prod))
            pairs_after.append(abs(len(kept) - prod))
    by_cam = {}
    for cam in sorted({p["camera"] for p in photos}):
        ps = [p for p in photos if p["camera"] == cam]
        by_cam[cam] = {"photos": len(ps), "sum_before": sum(p["before"] for p in ps), "sum_after": sum(p["after"] for p in ps),
                       "sum_production": sum(p["production"] or 0 for p in ps)}
    return {"manifest": rel(manifest), "photos": len(photos), "sum_before": sum(p["before"] for p in photos),
            "sum_after": sum(p["after"] for p in photos),
            "mae_vs_production_before": round(statistics.fmean(pairs_before), 3) if pairs_before else None,
            "mae_vs_production_after": round(statistics.fmean(pairs_after), 3) if pairs_after else None,
            "by_camera": by_cam, "removed_boxes": removed, "per_photo": photos,
            "note": "production count = FlyBot's Roboflow RF-DETR at capture time, not human ground truth"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", type=Path, default=ROOT / "models" / "yolo26" / "fruitfly_yolo26n.onnx")
    ap.add_argument("--dataset", type=Path, default=ROOT.parent / "camos-ai-camera" / "models" / "fruit-fly-v3")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "samples" / "samples_manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "yolo26_box_rules.json")
    a = ap.parse_args(argv)

    import cv2

    from sentinel.vision.box_rules import BoxRules
    from sentinel.vision.detect_yolo26 import Yolo26Detector

    rules = BoxRules.from_env() or BoxRules()
    det = Yolo26Detector(a.onnx, "new", conf=FLOOR, iou=0.45, rules=rules)
    res = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "opencv": cv2.__version__,
           "engine_used": det.engine_used, "onnx": rel(a.onnx), "resize": det.resize, "imgsz": det.size,
           "before": f"end-to-end rows with score >= {FLOOR}, no NMS, no rules (decoding until this change)",
           "after": {"nms_iou": det.iou, "rules": rules.as_dict()},
           "dataset": "camos/fruit-fly-wmcqi v3 (Roboflow Universe, CC BY 4.0)", "match_iou": 0.5,
           "labels": {s: label_stats(a.dataset / s, rules) for s in ("train", "valid", "test")},
           "splits": {s: run_split(det, a.dataset / s) for s in ("valid", "test")},
           "samples": run_samples(det, a.manifest)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for s, v in res["splits"].items():
        print(s, "before", v["before"], "\n   nms", v["nms"], "\n after", v["after"])
    sm = res["samples"]
    print("samples: flies", sm["sum_before"], "->", sm["sum_after"], "MAE vs production", sm["mae_vs_production_before"],
          "->", sm["mae_vs_production_after"])
    print("wrote", rel(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
