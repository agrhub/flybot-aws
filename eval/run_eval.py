"""Evaluation run: detection, vision ops, agent task success, failure injection -> eval/results/*.json + eval/RESULTS.md.

    uv run python -m eval.run_eval --quick         # a few minutes on a laptop (warm caches): every section, small N
    uv run python -m eval.run_eval                 # full: 3 photo pairs per image perturbation, 40-photo op 2 sweep,
                                                   #       all three approval policies on the 30-day replay
    uv run python -m eval.run_eval --only detection vision      # subset: detection | vision | agent | failure

Every number in eval/RESULTS.md comes from a JSON file in eval/results/ (written by this run): the Markdown is rendered
from those files only (`render_results_md`). Values that were not measured say so (`not measured`), they are never
filled in. Sections:

1. detection.json
   a. YOLOv7-tiny on OpenCV 5 ENGINE_NEW vs the Roboflow test split boxes (camos/fruit-fly v3 test, 76 images):
      AP@50, AP@50-95, P and R at the deployed threshold 0.25 and at best F1. The PyTorch test.py numbers
      (eval/results/yolov7_tiny_test_metrics.csv) are shown next to them.
   b. counts vs FlyBot's production count (Roboflow RF-DETR at capture time; not human ground truth): MAE and bias per camera
      over the 30-day replay window and over the 93-photo sample of eval/results/count_vs_production__*.json.
   c. counts vs human counts from data/labels/counts.csv (labeler `agent-test` rows are tool checks, ignored; `unusable`
      rows left out). With 0 labels the section says "0 labels, not measured".
2. vision_ops.json: one table of the vision evidence files (read, not recomputed).
3. agent.json: the 9-step scenario on fakes (expected vs actual per step), replay 7 and 30 days (real photos, real counts),
   a STAGED replay with a synthetic surge injected into the count fields and a simulated operator (approval latency),
   and the A/B table "FlyBot today vs Sentinel".
4. failure_injection.json (eval/failure_injection.py).
5. aws_live.json: per-photo traces of the deployed stack (data/eval/aws_traces.jsonl, pulled by `eval.aws_traces pull`; this run
   only re-summarizes the file, it never calls AWS).
6. expert_review.json: policy-correct rate once an agronomist filled docs/evidence/eval/expert_review.csv
   (eval/score_expert_review.py; 0 verdicts -> "not measured").
Then the Limitations list. `--render-only` re-renders eval/RESULTS.md from the JSON files without recomputing anything heavy.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "eval" / "results"
RESULTS_MD = ROOT / "eval" / "RESULTS.md"
EVAL_DATA = ROOT / "data" / "eval"
FARM_TZ = timedelta(hours=7)
SECTIONS = ("detection", "vision", "agent", "failure")


def rel(p: Path | str) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def jload(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def jdump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


# =========================================================================== metric functions (pure, unit-tested)

def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (p in 0..100); None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def count_error_metrics(pairs: list[tuple[float, float]]) -> dict:
    """pairs = [(predicted, reference)]. MAE, bias = mean(predicted - reference), median |err|, within 2, sums."""
    if not pairs:
        return {"n": 0}
    err = [p - r for p, r in pairs]
    return {"n": len(pairs), "mae": round(statistics.fmean(abs(e) for e in err), 3),
            "bias": round(statistics.fmean(err), 3), "median_abs_err": round(statistics.median(abs(e) for e in err), 3),
            "max_abs_err": round(max(abs(e) for e in err), 3), "within_2": sum(abs(e) <= 2 for e in err),
            "sum_pred": round(sum(p for p, _ in pairs), 3), "sum_ref": round(sum(r for _, r in pairs), 3)}


def iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_image(dets: list[list[float]], gts: list[list[float]], thr: float = 0.5) -> list[tuple[float, bool]]:
    """Greedy matching by descending score. dets = [[x1,y1,x2,y2,score]], gts = [[x1,y1,x2,y2]].
    Returns [(score, is_true_positive)] for every detection; a ground-truth box is matched at most once."""
    used: set[int] = set()
    out = []
    for d in sorted(dets, key=lambda d: -d[4]):
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou_xyxy(d[:4], g)
            if v > best:
                best, best_j = v, j
        if best >= thr and best_j >= 0:
            used.add(best_j)
            out.append((d[4], True))
        else:
            out.append((d[4], False))
    return out


def average_precision(scores: list[float], tps: list[bool], n_gt: int) -> float:
    """AP with the precision envelope and 101-point interpolation (the YOLOv7 test.py convention, so numbers compare with its
    output; a perfect detector scores 0.995, not 1.0)."""
    if n_gt == 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tps, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-16)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def detection_metrics(images: list[tuple[list[list[float]], list[list[float]]]], conf: float = 0.25) -> dict:
    """images = [(dets [[x1,y1,x2,y2,score]], gts [[x1,y1,x2,y2]])]. AP@50, AP@50-95, P/R at `conf`, best-F1 point."""
    n_gt = sum(len(g) for _, g in images)
    aps = []
    for thr in [0.5 + 0.05 * i for i in range(10)]:
        sc, tp = [], []
        for dets, gts in images:
            for s, t in match_image(dets, gts, thr):
                sc.append(s)
                tp.append(t)
        aps.append(average_precision(sc, tp, n_gt))
    sc50, tp50 = [], []
    for dets, gts in images:
        for s, t in match_image(dets, gts, 0.5):
            sc50.append(s)
            tp50.append(t)

    def at(c: float) -> dict:
        tp = sum(1 for s, t in zip(sc50, tp50) if s >= c and t)
        fp = sum(1 for s, t in zip(sc50, tp50) if s >= c and not t)
        fn = n_gt - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"conf": round(c, 4), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    grid = sorted({round(c, 3) for c in np.arange(0.05, 0.96, 0.01)})
    best = max((at(c) for c in grid), key=lambda d: d["f1"])
    return {"n_images": len(images), "n_gt_boxes": n_gt, "n_detections_at_floor": len(sc50),
            "ap50": round(aps[0], 4), "ap50_95": round(float(np.mean(aps)), 4), "at_conf_0.25": at(conf), "best_f1": best}


def policy_correct_rate(steps: list[dict]) -> dict:
    ok = sum(1 for s in steps if s["pass"])
    return {"steps": len(steps), "policy_correct": ok, "rate": round(ok / len(steps), 4) if steps else None}


def latency_distribution(hours: list[float]) -> dict:
    if not hours:
        return {"n": 0}
    return {"n": len(hours), "p50_h": round(percentile(hours, 50), 3), "p90_h": round(percentile(hours, 90), 3),
            "max_h": round(max(hours), 3), "mean_h": round(statistics.fmean(hours), 3)}


def farm_day(iso: str) -> str:
    return (datetime.fromisoformat(iso).astimezone(UTC) + FARM_TZ).date().isoformat()


def flybot_today_alerts(photos: list[dict], *, trap_kind: str, threshold: int = 15, card_replaced: dict[str, str] | None = None,
                        policy=None) -> dict:
    """What FlyBot's own rules would have raised on these photos, using production counts.

    Rules copied from ezviz-manager: notificationService.checkPestThreshold (sticky: the photo's count >= threshold; lure: the day's
    average per photo so far >= threshold; 120 min cooldown per camera), surgeService.detectSurge (today's PEAK >= 5, rise >= 3,
    >= 1.5 x the peak 3 days earlier; skipped when a card was replaced in the last 4 days; one per camera per day), imageQuality
    reminder (a photo with quality issues, once a day per camera). photos: dicts with camera, captured_at, count_production,
    quality_issues, ordered by time.
    """
    card_replaced = card_replaced or {}
    by_cam: dict[str, list[dict]] = defaultdict(list)
    for p in photos:
        by_cam[p["camera"]].append(p)
    out: dict = {"trap_kind": trap_kind, "threshold": threshold, "per_camera": {}, "threshold_alerts": 0, "threshold_alert_days": 0,
                 "surge_alerts": 0, "quality_alerts": 0, "first_surge_alert_at": {}}
    for cam, lst in sorted(by_cam.items()):
        day_counts: dict[str, list[float]] = defaultdict(list)
        last_thr: datetime | None = None
        surge_days: set[str] = set()
        q_days: set[str] = set()
        thr_days: set[str] = set()
        thr = surge = qual = 0
        for p in sorted(lst, key=lambda x: x["captured_at"]):
            t = datetime.fromisoformat(p["captured_at"])
            day = farm_day(p["captured_at"])
            c = p.get("count_production")
            if p.get("quality_issues") and day not in q_days:
                q_days.add(day)
                qual += 1
            if c is None:
                continue
            day_counts[day].append(float(c))
            figure = float(c) if trap_kind == "sticky" else statistics.fmean(day_counts[day])
            if figure >= threshold and (last_thr is None or t - last_thr >= timedelta(minutes=120)):
                last_thr = t
                thr += 1
                thr_days.add(day)
            if day not in surge_days:
                today = max(day_counts[day])
                base_day = (date.fromisoformat(day) - timedelta(days=3)).isoformat()
                base = max(day_counts[base_day]) if day_counts.get(base_day) else None
                replaced = card_replaced.get(cam)
                skip = bool(replaced and (date.fromisoformat(day) - date.fromisoformat(farm_day(replaced))).days in range(5))
                if base is not None and not skip and today >= 5 and today - base >= 3 and today >= base * 1.5:
                    surge_days.add(day)
                    surge += 1
                    out["first_surge_alert_at"].setdefault(cam, p["captured_at"])
        out["per_camera"][cam] = {"photos": len(lst), "threshold_alerts": thr, "threshold_alert_days": len(thr_days),
                                  "surge_alerts": surge, "quality_alerts": qual}
        out["threshold_alerts"] += thr
        out["threshold_alert_days"] += len(thr_days)
        out["surge_alerts"] += surge
        out["quality_alerts"] += qual
    return out


def ab_summary(flybot: dict, sentinel: dict) -> dict:
    """Side-by-side counts. sentinel = replay analysis (decisions, reviews, proposals, extra captures)."""
    return {"flybot_alerts": flybot["threshold_alerts"] + flybot["surge_alerts"] + flybot["quality_alerts"],
            "flybot_threshold_alerts": flybot["threshold_alerts"], "flybot_threshold_alert_days": flybot["threshold_alert_days"],
            "flybot_surge_alerts": flybot["surge_alerts"], "flybot_quality_alerts": flybot["quality_alerts"],
            "sentinel_human_asks": sentinel["reviews"] + sentinel["proposals"],
            "sentinel_reviews": sentinel["reviews"], "sentinel_proposals": sentinel["proposals"],
            "sentinel_extra_captures": sentinel["extra_captures"], "sentinel_no_ops": sentinel["no_ops"]}


# =========================================================================== 1. detection

def roboflow_split(split_dir: Path) -> list[tuple[Path, list[list[float]]]]:
    """[(image path, [[x1,y1,x2,y2] px])] from a Roboflow YOLO export (images/ + labels/ with `cls cx cy w h` normalised)."""
    import cv2

    out = []
    for img in sorted((split_dir / "images").glob("*.jpg")):
        lab = split_dir / "labels" / (img.stem + ".txt")
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        gts = []
        if lab.exists():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    _, cx, cy, bw, bh = (float(v) for v in parts[:5])
                    gts.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        out.append((img, gts))
    return out


def detect_split(onnx: Path, items: list[tuple[Path, list]], cache: Path) -> tuple[list[list[list[float]]], dict]:
    """Detections (conf >= 0.001, NMS iou 0.65 like test.py) per image on OpenCV 5 ENGINE_NEW, cached on (onnx, image list)."""
    import cv2

    from sentinel.vision.detect_yolov7 import YoloV7Detector

    st = onnx.stat()
    sig = f"{onnx.name}:{st.st_size}:{len(items)}:{items[0][0].name if items else ''}:new:stretch:0.001:0.65"
    if cache.exists():
        try:
            c = jload(cache)
            if c.get("sig") == sig:
                return c["dets"], {**c["meta"], "cache": "hit"}
        except (OSError, ValueError, KeyError):
            pass
    det = YoloV7Detector(onnx, "new", resize="stretch", conf=0.001, iou=0.65)
    t0 = time.perf_counter()
    dets, fwd = [], []
    for img_path, _ in items:
        img = cv2.imread(str(img_path))
        r = det.detect(img)
        dets.append([[b["xywh"][0], b["xywh"][1], b["xywh"][0] + b["xywh"][2], b["xywh"][1] + b["xywh"][3], b["score"]] for b in r["boxes"]])
        fwd.append(r["forward_ms"])
    meta = {"engine_used": det.engine_used, "opencv": cv2.__version__, "seconds": round(time.perf_counter() - t0, 1),
            "forward_ms_median": round(statistics.median(fwd), 1) if fwd else None,
            "note": "forward time measured on a shared dev laptop while other jobs ran; not a benchmark"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"sig": sig, "dets": dets, "meta": meta}), encoding="utf-8")
    return dets, {**meta, "cache": "miss"}


def reference_pytorch_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else None


def detection_vs_roboflow(onnx: Path, dataset: Path, cache_dir: Path) -> dict:
    split = dataset / "test"
    ref = reference_pytorch_row(RESULTS / "yolov7_tiny_test_metrics.csv")
    ref_out = None if ref is None else {k: ref[k] for k in ("model", "weights_sha256", "split", "images", "boxes", "precision", "recall",
                                                              "mAP50", "mAP50_95", "conf", "iou", "device", "tool")} | {"source": rel(RESULTS / "yolov7_tiny_test_metrics.csv")}
    if not (split / "images").is_dir() or not onnx.exists():
        return {"status": "not measured: Roboflow test split or ONNX missing", "dataset": str(dataset), "reference_pytorch": ref_out}
    items = roboflow_split(split)
    dets, meta = detect_split(onnx, items, cache_dir / "roboflow_test_dets.json")
    m = detection_metrics([(d, g) for d, (_, g) in zip(dets, items, strict=True)])
    return {"status": "measured", "dataset": "camos/fruit-fly-wmcqi v3 test split (Roboflow Universe, CC BY 4.0)",
            "dataset_path": str(dataset / "test"), "onnx": rel(onnx), "resize": "stretch (as the training export)",
            "floor_conf": 0.001, "nms_iou": 0.65, "match_iou": 0.5, **m, **meta, "reference_pytorch": ref_out}


def trace_photos(trace_path: Path) -> list[dict]:
    out = []
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") == "photo":
                out.append(r)
    return out


def production_count_metrics(photos: list[dict], source: str) -> dict:
    by_cam: dict[str, list] = defaultdict(list)
    allp = []
    for p in photos:
        if p.get("count_production") is None or p.get("count_yolov7") is None:
            continue
        pair = (p["count_yolov7"], p["count_production"])
        by_cam[p["camera"]].append(pair)
        allp.append(pair)
    return {"source": source, "reference": "FlyBot production ai_count (Roboflow RF-DETR at capture time), not human ground truth",
            "all": count_error_metrics(allp), "by_camera": {c: count_error_metrics(v) for c, v in sorted(by_cam.items())}}


def sample_file_metrics() -> dict | None:
    p = RESULTS / "count_vs_production__fruitfly_deploy__trap-cameras.json"
    if not p.exists():
        return None
    j = jload(p)
    lb = (j.get("by_resize") or {}).get("letterbox")
    if not lb:
        return None
    return {"source": rel(p), "n": j.get("n"), "engine_used": j.get("engine_used"), "conf": j.get("conf"),
            "all": lb["all"], "by_camera": lb["by_camera"], "by_count_bucket": lb.get("by_count_bucket")}


def read_human_labels(path: Path) -> dict:
    """{snapshot_id: {count (mean over labelers), labelers, flags}} + counters of what was skipped."""
    info: dict = {"labels": {}, "ignored_agent_test_rows": 0, "ignored_unusable": 0, "rows": 0}
    if not path.exists():
        info["file_missing"] = True
        return info
    per: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            info["rows"] += 1
            if str(r.get("labeler", "")).startswith("agent-test"):
                info["ignored_agent_test_rows"] += 1
                continue
            if r.get("true_count") in (None, ""):
                continue
            if str(r.get("unusable", "0")) == "1":
                info["ignored_unusable"] += 1
                continue
            per[r["snapshot_id"]].append(r)
    for sid, rows in per.items():
        info["labels"][sid] = {"count": statistics.fmean(int(r["true_count"]) for r in rows), "labelers": len(rows),
                               "camera": rows[0].get("device_serial"),
                               "flags": sorted({f for r in rows for f in ("card_full", "glare", "person", "no_trap", "staged") if str(r.get(f, "0")) == "1"}),
                               "production": int(rows[0]["ai_count"]) if rows[0].get("ai_count") not in (None, "") else None}
    return info


def human_count_metrics(labels_path: Path, det_count_of) -> dict:
    """Detector vs human counts. `det_count_of(snapshot_id) -> int | None`. 0 labels -> 'not measured'."""
    info = read_human_labels(labels_path)
    labs = info["labels"]
    base = {"source": rel(labels_path), "rows_in_file": info["rows"], "ignored_agent_test_rows": info["ignored_agent_test_rows"],
            "ignored_unusable": info["ignored_unusable"]}
    if not labs:
        return {**base, "n_labels": 0, "status": "0 labels, not measured"}
    pairs, prod_pairs, by_cam = [], [], defaultdict(list)
    missing = 0
    for sid, v in labs.items():
        d = det_count_of(sid)
        if d is None:
            missing += 1
            continue
        pairs.append((d, v["count"]))
        by_cam[v["camera"]].append((d, v["count"]))
        if v["production"] is not None:
            prod_pairs.append((v["production"], v["count"]))
    out = {**base, "n_labels": len(labs), "status": "measured" if pairs else "labels present but no detector counts",
           "no_detector_count": missing, "double_labeled": sum(1 for v in labs.values() if v["labelers"] > 1),
           "yolov7_vs_human": count_error_metrics(pairs), "production_vs_human": count_error_metrics(prod_pairs),
           "by_camera": {c: count_error_metrics(v) for c, v in sorted(by_cam.items())}}
    return out


def detection_section(quick: bool, onnx: Path, dataset: Path, replay30: dict | None) -> dict:
    cache_dir = ROOT / "data" / "cache" / "eval"
    res: dict = {"opencv_engine": "ENGINE_NEW", "model": rel(onnx)}
    res["roboflow_test"] = detection_vs_roboflow(onnx, dataset, cache_dir)
    if replay30 is not None:
        res["production_counts"] = production_count_metrics(replay30["photos"], replay30["trace"])
        res["production_counts"]["window"] = replay30["window"]
    else:
        res["production_counts"] = {"status": "not measured: replay not run in this invocation"}
    res["production_counts_sample"] = sample_file_metrics()

    def det_count(sid: str) -> int | None:
        f = ROOT / "data" / "cache" / "yolov7" / f"{sid}.json"
        if f.exists():
            try:
                return int(jload(f)["count"])
            except (OSError, ValueError, KeyError):
                return None
        return None

    res["human_counts"] = human_count_metrics(ROOT / "data" / "labels" / "counts.csv", det_count)
    res["not_measured"] = ["Detection accuracy on real yellow-sticky-card photos: none exist yet (all real photos are lure traps).",
                           "Detection accuracy on real lure-trap photos against human counts: " + res["human_counts"]["status"] + "."]
    return res


# =========================================================================== 2. vision ops (read the evidence files)

def _g(d: Any, *path, default=None):
    for k in path:
        if isinstance(d, dict) and k in d or isinstance(d, list) and isinstance(k, int) and -len(d) <= k < len(d):
            d = d[k]
        else:
            return default
    return d


def _md_number(path: Path, pattern: str) -> str | None:
    import re

    if not path.exists():
        return None
    m = re.search(pattern, path.read_text(encoding="utf-8"))
    return m.group(1).strip() if m else None


def vision_ops_section() -> dict:
    V = ROOT / "docs" / "evidence" / "vision"
    rows: list[dict] = []

    def add(op, metric, value, n, dataset, src, note=""):
        rows.append({"op": op, "metric": metric, "value": value, "n": n, "dataset": dataset, "source": rel(src) if isinstance(src, Path) else src,
                     "note": note})

    def need(p: Path) -> Any | None:
        return jload(p) if p.exists() else None

    cs = need(V / "card_synth.json")
    for k in ("normal", "angled", "partial", "two_cards", "no_card"):
        if cs and k in cs:
            add("1 card (sticky)", f"hit rate, {k}", cs[k]["hit_rate"], cs[k]["n"], "synthetic cards pasted on real frames",
                V / "card_synth.json", (f"corner error p50/p95 px: {cs[k]['corner_err_px_p50']}/{cs[k]['corner_err_px_p95']}"
                 if cs[k].get("corner_err_px_p50") is not None else "no corner reference for this case"))
    md = V / "card_check.md"
    add("1 card (sticky)", "real photos with a yellow card", _md_number(md, r"\*\*Yellow sticky card\*\* \| \*\*(\d+)\*\*") or "not found in file", 3000,
        "real export", md, "no real sticky card photo exists: sticky profile only measured on synthetic composites")
    lr = need(V / "card_check_lure" / "results.json")
    hr = need(V / "card_check_lure" / "holdout_results.json")
    if lr:
        add("1 floor (lure)", "floor found", f"{sum(1 for r in lr if _g(r, 'lure', 'found'))}/{len(lr)}", len(lr), "real lure photos (tuning set)",
            V / "card_check_lure" / "results.json", "roi_source: " + json.dumps(dict(Counter(_g(r, "lure", "roi_source") for r in lr))))
    if hr:
        add("1 floor (lure)", "floor_quad kept (held out)", f"{sum(1 for r in hr if r.get('roi_source') == 'floor_quad')}/{len(hr)}", len(hr),
            "real lure photos (hold-out, seed 20261015)", V / "card_check_lure" / "holdout_results.json",
            "roi_source: " + json.dumps(dict(Counter(r.get("roi_source") for r in hr))))
    mdl = V / "card_check_lure.md"
    ok_poly = _md_number(mdl, r"floor polygon ROI correct \| (\d+ / \d+)")
    add("1 floor (lure)", "floor polygon ROI correct (hand verdict)", ok_poly or "not found in file", 50, "real lure photos (tuning set)", mdl,
        "verdicts by eye, not machine-readable; hold-out: all 50 polygons correct by eye (same file)")
    q = need(V / "quality_calibration.json")
    if q:
        add("2 quality", "exact agreement with FlyBot imageQuality flags", _g(q, "all_photos", "exact_agreement"), _g(q, "all_photos", "photos"),
            "all export photos", V / "quality_calibration.json", "FlyBot flags only `low_contrast` in this export")
        add("2 quality", "exact agreement (photos since 2026-10-11)", _g(q, "checked_window", "exact_agreement"), _g(q, "checked_window", "photos"),
            "export, 2026-10-11 onward", V / "quality_calibration.json")
        lc = _g(q, "all_photos", "per_issue", "low_contrast", default={})
        add("2 quality", "low_contrast precision / recall vs FlyBot", f"{lc.get('precision')} / {lc.get('recall')}", _g(q, "all_photos", "photos"),
            "all export photos", V / "quality_calibration.json", "no human labels: FlyBot flags are the reference, not ground truth")
    de = need(V / "diff_eval.json")
    if de:
        cc = _g(de, "native_pairs", "card_change", default={})
        add("4 diff", "card-change recall (events)", f"{cc.get('recall')} ({cc.get('events')} event)", cc.get("pairs"), "native-resolution consecutive pairs",
            V / "diff_eval.json", f"false flags {cc.get('false_flags')} (interim blob boxes, not YOLO boxes)")
        al = de.get("aliked_sample", {})
        add("3 register", "inliers p50 ALIKED+LightGlue vs SIFT", f"{al.get('aliked_inliers_p50')} vs {al.get('sift_inliers_p50')}", al.get("pairs"),
            "6 real pairs", V / "diff_eval.json", f"ms/pair p50 {round(al.get('aliked_ms_per_pair_p50', 0))} vs {round(al.get('sift_ms_per_pair_p50', 0))}")
    dy = need(RESULTS / "diff_with_yolo_boxes.json")
    if dy:
        sec = dy.get("since_2026-10-07", {})
        cc = sec.get("card_change", {})
        add("4 diff", "card-change recall with YOLOv7 boxes", f"{cc.get('recall')} ({cc.get('events')} event)", cc.get("pairs"),
            "pairs since 2026-10-07", RESULTS / "diff_with_yolo_boxes.json", f"false flags {cc.get('false_flags')}")
        for cam, v in (sec.get("per_camera") or {}).items():
            add("4 diff", f"new catches vs FlyBot daily rise, MAE ({cam})", v.get("mae"), v.get("days_compared"),
                "days since 2026-10-07 (traps empty; production rise = 0)", RESULTS / "diff_with_yolo_boxes.json",
                f"sum new {v.get('sum_new')} vs sum daily rise {v.get('sum_daily_rise')}: no ground truth")
    ce = need(V / "crowd_eval.json")
    cv = need(V / "crowd_eval_valid.json")
    for label, c, p in (("test split", ce, V / "crowd_eval.json"), ("valid split", cv, V / "crowd_eval_valid.json")):
        if c:
            m = _g(c, "summary", "modes", "full_frame", default={})
            add("6 crowd", f"Pearson r blob count vs labels ({label})", _g(m, "blob_count_vs_labels", "pearson_r"), _g(c, "summary", "n_images"),
                "camos fruit-fly v3 (white-disc pictures, not sticky cards)", p,
                f"saturation r {_g(m, 'saturation_pct_vs_labels', 'pearson_r')}; blob MAE {m.get('blob_count_mae')}, bias {m.get('blob_count_bias')}")
    add("7 density", "validation", "not measured", None, "synthetic trap layouts only", V / "density" / "density_synthetic.json",
        "no real multi-trap data: the map is shown on synthetic layouts")
    me = need(V / "motion_eval.json")
    if me:
        s = me["summary"]
        add("8 motion", "occluder recall (synthetic, held-out pairs)", s.get("occluder_recall"), s.get("synthetic_occluders"),
            "real pairs 201-400 + pasted occluders", V / "motion_eval.json", f"by kind {json.dumps(s.get('occluder_recall_by_kind'))}; box IoU {s.get('occluder_box_iou_mean')}")
        add("8 motion", "camera-moved recall (synthetic)", s.get("camera_moved_recall"), s.get("synthetic_moves"), "same pairs, shifted + rotated",
            V / "motion_eval.json")
        add("8 motion", "false occluder flags on real pairs", f"{s.get('real_occluded_flags')}/{s.get('real_pairs')}", s.get("real_pairs"),
            "real consecutive lure pairs (nobody touched the traps)", V / "motion_eval.json", f"false camera-moved {s.get('real_camera_moved_flags')}")
        add("8 motion", "time per pair p50 / p95 ms", f"{_g(s, 'time_ms_real', 'p50')} / {_g(s, 'time_ms_real', 'p95')}", s.get("real_pairs"),
            "640x360 working size", V / "motion_eval.json")
    pt = need(V / "pipeline_timings.json")
    if pt:
        add("pipeline", "wall time per lure pair p50 / p95 ms", f"{_g(pt, 'ms', 'wall_total', 'p50')} / {_g(pt, 'ms', 'wall_total', 'p95')}", pt.get("n_pairs"),
            "real lure pairs, ops 1,2,3,4,6,8", V / "pipeline_timings.json", "4-core laptop shared with other jobs; image decode excluded")
    return {"rows": rows, "not_measured": [
        "Op 1/6 on real sticky yellow cards (none exist in the export).",
        "Op 7 density map on a real multi-trap farm (synthetic layouts only).",
        "Op 8 on real people / hands (no such photo in the export; occluders are synthetic composites).",
        "Face detection accuracy (no photo with a person)."]}


# =========================================================================== 3. agent

SCENARIO_EXPECT: dict[str, tuple[str, Any]] = {}


def _scn(key: str, expected: str):
    def deco(fn):
        SCENARIO_EXPECT[key] = (expected, fn)
        return fn
    return deco


def _names(tr, status=None):
    return [c.name for c in tr.tool_calls if status is None or c.status == status]


def _rules(tr):
    return list(dict.fromkeys(d.rule for d in tr.decisions))


@_scn("1_glare_isp_fix", "glare -> set_image(brightness -8, wdr 2 from the reported 8) -> automatic verification capture -> quality ok, change kept")
def _s1(res, rt, demo):
    t = res["1_glare_isp_fix"]
    bench = rt.devices.get(demo.BENCH)
    return (names4(t) == ["analyze_trap_photo", "set_image", "capture", "analyze_trap_photo"] and ("set_image", {"brightness": -8, "wdr": 2}) in bench.calls
            and any(e.kind == "isp_verified" and e.detail["result"] == "kept" for e in t.events) and "quality.fixed" in _rules(t))


def names4(t):
    return _names(t)[:4]


@_scn("2_surge_neighbours_treatment", "surge on lure trap A -> neighbour captures B, C (<= 3) -> B confirms -> propose_treatment with evidence, pending a person")
def _s2(res, rt, demo):
    t = res["2_surge_neighbours_treatment"]
    caps = [c for c in t.tool_calls if c.name == "capture"]
    prop = t.tool_calls[-1]
    return ([c.args["camera"] for c in caps] == [demo.B, demo.C] and len(caps) <= 3 and prop.name == "propose_treatment" and prop.status == "ok"
            and bool(prop.args.get("evidence")) and t.approval is not None and t.approval.status in ("pending", "expired") and "surge.confirmed" in _rules(t))


@_scn("3_approval_timeout", "nobody confirms within 24 h -> approval expires, no action is taken (no_op)")
def _s3(res, rt, demo):
    t = res["3_approval_timeout"]
    daily_acts = [c.name for d in res["3_daily_traces"] if d.loop == "daily_per_plot" for c in d.tool_calls
                  if c.name in ("capture", "set_image", "propose_treatment")]
    return (t.approval is not None and t.approval.status == "expired" and t.outcome.endswith("approval_expired_no_op")
            and len(t.tool_calls) == len(res["2_surge_neighbours_treatment"].tool_calls) and not daily_acts)


@_scn("4_disagreement_review", "NPU 30 vs OpenCV 18 -> 4 MP capture -> still apart -> human review")
def _s4(res, rt, demo):
    t = res["4_disagreement_review"]
    cap = next((c for c in t.tool_calls if c.name == "capture"), None)
    rv = next((c for c in t.tool_calls if c.name == "request_human_review"), None)
    return bool(cap and cap.args.get("resolution") == "max" and rv and rv.status == "ok" and "disagreement.review" in _rules(t))


@_scn("5_propose_without_evidence", "LLM proposes a treatment with no evidence -> blocked by the guardrail, no approval")
def _s5(res, rt, demo):
    t = res["5_propose_without_evidence"]
    c = next((c for c in t.tool_calls if c.name == "propose_treatment"), None)
    return bool(c and c.status == "blocked" and (c.result or {}).get("code") == "evidence" and t.approval is None)


@_scn("6_llm_error_fallback", "LLM provider error (throttling) -> planner_fallback traced, the rule planner decides the same run")
def _s6(res, rt, demo):
    t = res["6_llm_error_fallback"]
    return t.planner.fallback and any(e.kind == "planner_fallback" for e in t.events) and _names(t)[:1] == ["analyze_trap_photo"] and bool(t.decisions)


@_scn("7_kill_switch", "kill switch on -> analysis only, no action, no FlyBot write")
def _s7(res, rt, demo):
    t = res["7_kill_switch"]
    return t.kill_switch and t.outcome == "kill_switch_no_action" and _names(t) == ["analyze_trap_photo"]


@_scn("8_battery_low_budget_halved", "battery 35 % -> extra-capture budget halved 4 -> 2, the ISP fix is still allowed")
def _s8(res, rt, demo):
    t = res["8_battery_low_budget_halved"]
    ev = next((e for e in t.events if e.kind == "battery_budget"), None)
    return bool(ev and ev.detail["limit"] == 2 and _names(t, "ok")[:2] == ["analyze_trap_photo", "set_image"])


@_scn("9_battery_critical_no_extra", "battery 15 % -> no extra capture and no ISP change; the camera's schedule continues")
def _s9(res, rt, demo):
    t = res["9_battery_critical_no_extra"]
    return _names(t) == ["analyze_trap_photo"] and any(d.rule == "battery.critical" for d in t.decisions)


def agent_scenario() -> dict:
    from eval.failure_injection import false_actuations
    from sentinel.agent import demo

    t0 = time.perf_counter()
    res = demo.run_scenario()
    rt = res["_runtime"]
    steps = []
    for key, (expected, fn) in SCENARIO_EXPECT.items():
        tr = res[key]
        try:
            ok = bool(fn(res, rt, demo))
            err = None
        except Exception as e:  # noqa: BLE001 - a broken predicate is a failed step
            ok, err = False, f"{type(e).__name__}: {e}"
        steps.append({"step": key, "expected": expected, "observed": f"{tr.state}; {tr.outcome}", "tools": [f"{c.name}:{c.status}" for c in tr.tool_calls],
                      "pass": ok, **({"error": err} if err else {}), "tool_ms": round(sum(c.ms or 0 for c in tr.tool_calls), 1)})
    traces = rt.traces.all()
    devices = list(rt.devices._devices.values())
    viol = false_actuations(rt, traces, devices, rt.flybot)
    ok_caps = [c for t in traces for c in t.tool_calls if c.name == "capture" and c.status == "ok"]
    esc = [s for s in steps if any(t in ("propose_treatment:ok", "request_human_review:ok", "propose_card_change:ok") for t in s["tools"])]
    return {"source": "sentinel/agent/demo.py on fakes (scripted counts, no AWS / FlyBot / LLM): exercises the logic, measures nothing about vision",
            "step_results": steps, **policy_correct_rate(steps), "false_actuations": len(viol), "false_actuation_detail": viol,
            "escalation_steps": len(esc), "escalation_rate": round(len(esc) / len(steps), 4),
            "extra_captures": len(ok_caps), "proposals": sum(1 for t in traces for c in t.tool_calls if c.name == "propose_treatment" and c.status == "ok"),
            "planner_fallbacks": sum(1 for t in traces if t.planner.fallback), "traces": len(traces), "seconds": round(time.perf_counter() - t0, 2)}


# ----- replay

def _quiet(dt: datetime) -> bool:
    from sentinel.agent.policy import in_quiet_hours, load_policy

    return in_quiet_hours(dt, load_policy())


def run_replay_main(argv: list[str], run_id: str, out_dir: Path, patches: list | None = None, agent_traces: bool = True) -> dict:
    """Run eval.replay.main with the given arguments (stdout swallowed), optionally with module attributes patched.
    Returns {"summary", "photos", "trace", "agent_traces"}."""
    from eval import replay

    out_dir.mkdir(parents=True, exist_ok=True)
    agent_jsonl = out_dir / run_id / "agent_traces.jsonl"
    if agent_jsonl.exists():
        agent_jsonl.unlink()
    env = {"SENTINEL_TRACE_JSONL": str(agent_jsonl)} if agent_traces else {}
    with contextlib.ExitStack() as stack:
        for p in patches or []:
            stack.enter_context(p)
        stack.enter_context(mock.patch.dict(os.environ, env))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = replay.main([*argv, "--run-id", run_id, "--out-dir", str(out_dir)])
    if rc != 0:
        raise RuntimeError(f"replay {run_id} failed (exit {rc})")
    run_dir = out_dir / run_id
    summary = jload(run_dir / "summary.json")
    return {"summary": summary, "photos": trace_photos(run_dir / "trace.jsonl"), "trace": rel(run_dir / "trace.jsonl"),
            "summary_path": rel(run_dir / "summary.json"), "agent_traces": agent_jsonl if agent_jsonl.exists() else None,
            "window": summary.get("window")}


def analyze_replay(run: dict) -> dict:
    """Task-success metrics of one replay: decisions, escalation, extra captures / proposals per day, approval latency,
    fallbacks, false actuations (captures in quiet hours), review reasons."""
    s = run["summary"]
    trace_path = ROOT / run["trace"]
    photos, reviews, caps, props, approvals = [], [], [], {}, {}
    set_images = 0
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            k = r["kind"]
            if k == "photo":
                photos.append(r)
            elif k == "review":
                reviews.append(r)
            elif k == "capture":
                caps.append(r)
            elif k == "set_image":
                set_images += 1
            elif k == "proposal":
                props[r["proposal_id"]] = r
            elif k == "approval":
                approvals[r["proposal_id"]] = r
    days = sorted({farm_day(p["captured_at"]) for p in photos})
    n_days = max(1, len(days))
    decisions = Counter(t for p in photos for t in p["decision"])
    ok_caps = [c for c in caps if c.get("ok")]
    quiet_viol = [c for c in ok_caps if _quiet(datetime.fromisoformat(c["at"]))]
    lat = []
    for pid, a in approvals.items():
        p = props.get(pid)
        if p and a.get("decided_at"):
            t1 = datetime.fromisoformat(p["at"])
            t2 = datetime.fromisoformat(a["decided_at"])
            lat.append((t2 - t1).total_seconds() / 3600.0)
    esc_photos = sum(1 for p in photos if "request_human_review" in p["decision"] or "propose_treatment" in p["decision"]
                     or "propose_card_change" in p["decision"])
    review_reasons: Counter = Counter()
    for r in reviews:
        txt = r.get("reason", "")
        review_reasons["count disagreement (YOLOv7 vs production)" if "disagree" in txt else txt[:80]] += 1
    review_by_day: Counter = Counter()
    for r in reviews:
        review_by_day[farm_day(r["at"])] += 1
    by_kind = Counter(p["proposal_kind"] for p in props.values())
    return {
        "run_id": s["run_id"], "policy": s["policy"], "agent": s["agent"], "window": s.get("window"), "photos": s["photos_processed"],
        "photos_per_camera": s["photos_per_camera"], "farm_days": n_days, "decisions": dict(sorted(decisions.items())),
        "no_ops": decisions.get("no_op", 0), "reviews": len(reviews), "review_reasons": dict(review_reasons),
        "review_days": dict(sorted(review_by_day.items())),
        "escalation_rate": round(esc_photos / len(photos), 4) if photos else None, "escalated_photos": esc_photos,
        "extra_captures": len(ok_caps), "extra_captures_failed": len(caps) - len(ok_caps),
        "extra_captures_per_day": round(len(ok_caps) / n_days, 3), "set_image_calls": set_images,
        "proposals": len(props), "proposals_by_kind": dict(by_kind), "proposals_per_day": round(len(props) / n_days, 3),
        "approvals": len(approvals), "approval_latency_h": latency_distribution(lat), "pending_at_end": s["proposals"]["pending_at_end"],
        "planner_fallbacks": s["fallbacks"], "agent_errors": s["agent_errors"],
        "false_actuations": len(quiet_viol), "false_actuation_detail": quiet_viol[:5],
        "yolov7_vs_production_mae": _g(s, "detector_vs_production", "mae"), "trace_sha256": s["trace_sha256"], "trace": run["trace"],
        "summary": run["summary_path"], "runtime_s": _g(s, "runtime", "total_s"),
    }


class _Operator:
    """Simulated operator: answers a proposal after a log-normal delay (median 2 h, sigma 0.8); 10 % never answer.
    ASSUMED, not measured: there is no real approval-latency data yet."""

    MEDIAN_H, SIGMA, P_NO_RESPONSE = 2.0, 0.8, 0.10


def operator_context_class():
    from eval import replay

    class OperatorContext(replay.ReplayContext):
        def propose(self, kind, payload=None, camera=None, reason=""):
            pid = f"prop-{len(self.proposals) + 1:04d}"
            responds = self.rng.random() >= _Operator.P_NO_RESPONSE
            lat = self.rng.lognormvariate(math.log(_Operator.MEDIAN_H), _Operator.SIGMA) if responds else None
            due = replay.iso_ms(self.now + timedelta(hours=lat)) if lat is not None else None
            p = {"proposal_id": pid, "kind": kind, "camera": camera, "payload": replay._jsonable(payload or {}), "reason": reason,
                 "created_at": replay.iso_ms(self.now), "status": "pending", "decided_at": None, "due_at": due}
            self.proposals[pid] = p
            self.log("proposal", proposal_id=pid, proposal_kind=kind, camera=camera, payload=p["payload"], reason=reason,
                     status="pending", due_at=due, simulated_latency_h=None if lat is None else round(lat, 3))
            return {"proposal_id": pid, "status": "pending"}

        def _resolve_due(self, until):
            for p in self.proposals.values():
                if p["status"] == "pending" and p["due_at"] and replay.parse_ts(p["due_at"]) <= until:
                    p["status"], p["decided_at"] = "approved", p["due_at"]
                    self.log("approval", proposal_id=p["proposal_id"], proposal_kind=p["kind"], decided_at=p["decided_at"],
                             policy="simulated-operator")
                    self._notify_approved(p)

    return OperatorContext


STRESS = {"cameras": ["BE4089629", "BE4111159"], "from": "2026-10-12", "hours": (8, 16), "extra_flies_by_day": {"2026-10-12": 10, "2026-10-13": 16,
          "2026-10-14": 24, "2026-10-15": 30}}


def owner_lure_serials() -> set[str]:
    """Cameras confirmed on site as lure traps (sentinel/agent/cameras.json, trap_kind_authoritative)."""
    from sentinel.agent.data import AgentData

    return {c["serial"] for c in AgentData.seed_from_env() if c.get("trap_kind") == "lure" and c.get("trap_kind_authoritative")}


def build_patches(*, lure: bool, inject: bool, operator: bool) -> list:
    """Module patches for eval.replay.main.
    lure:     the export still says trapKind=sticky for both orchard cameras (FlyBot not yet corrected); apply the confirmed
              `lure` to the manifest rows and the camera list, as FlyBot reports once corrected. (replay_adapter only
              trusts the event/export trap kind, so without this the agent runs the sticky rules on lure traps.)
    inject:   synthetic surge in the count fields (production ai_count + YOLOv7 boxes) of the two lure cameras, daylight hours,
              last days. Pixels untouched (the replay agent never reads pixels). STAGED, not real.
    operator: simulated approvals (log-normal delay, 10 % never answer)."""
    from eval import replay

    orig_load, orig_cams = replay.load_manifest, replay.load_cameras
    lures = owner_lure_serials()

    def load_manifest(path):
        rows = orig_load(path)
        for r in rows:
            if lure and r["device_serial"] in lures:
                r["trap_kind"] = "lure"
            if inject and r["device_serial"] in STRESS["cameras"] and r.get("captured_at"):
                day = farm_day(r["captured_at"])
                hour = (datetime.fromisoformat(r["captured_at"]).astimezone(UTC) + FARM_TZ).hour
                k = STRESS["extra_flies_by_day"].get(day)
                if k and STRESS["hours"][0] <= hour <= STRESS["hours"][1] and r.get("ai_count") not in (None, ""):
                    r["_inject_k"] = k
                    r["ai_count"] = str(int(float(r["ai_count"])) + k)
        return rows

    def load_cameras(path, rows, serials):
        cams = orig_cams(path, rows, serials)
        for c in cams:
            if lure and c["serial"] in lures:
                c["trapKind"] = "lure"
        return cams

    class InjectedSource(replay.Yolov7Source):
        def detect(self, row, image_path):
            det = super().detect(row, image_path)
            k = int(row.get("_inject_k") or 0)
            if not k:
                return det
            det = dict(det)
            boxes = list(det["boxes"])
            size = statistics.median([b["xywh"][2] for b in boxes] or [14.0])
            rng = random.Random(row["snapshot_id"])
            w, h = det.get("width") or 768, det.get("height") or 432
            for _ in range(k):
                boxes.append({"xywh": [round(rng.uniform(0.1, 0.8) * w, 1), round(rng.uniform(0.1, 0.8) * h, 1), size, size],
                              "score": round(rng.uniform(0.4, 0.9), 4), "class_id": 0})
            det["boxes"], det["count"] = boxes, len(boxes)
            return det

    out = []
    if lure or inject:
        out += [mock.patch.object(replay, "load_manifest", load_manifest), mock.patch.object(replay, "load_cameras", load_cameras)]
    if inject:
        out.append(mock.patch.object(replay, "Yolov7Source", InjectedSource))
    if operator:
        out.append(mock.patch.object(replay, "ReplayContext", operator_context_class()))
    return out


def agent_replay_section(quick: bool, out_dir: Path) -> dict:
    from sentinel.agent.data import AgentData

    t0 = time.perf_counter()
    runs: dict[str, dict] = {}
    full = {}
    plan = [("replay_7d", ["--days", "7", "--approve-after-hours", "2"], "7d_approve-after-2h", True),
            ("replay_30d", ["--days", "30", "--approve-after-hours", "2"], "30d_approve-after-2h", True),
            ("replay_30d_as_exported", ["--days", "30", "--approve-after-hours", "2"], "30d_as_exported_sticky", False)]
    if not quick:
        plan += [("replay_30d_approve_none", ["--days", "30", "--approve-none"], "30d_approve-none", True),
                 ("replay_30d_approve_all", ["--days", "30", "--approve-all"], "30d_approve-all", True)]
    for key, argv, rid, lure in plan:
        r = run_replay_main(argv, rid, out_dir, patches=build_patches(lure=lure, inject=False, operator=False), agent_traces=False)
        full[key] = r
        runs[key] = analyze_replay(r)
        runs[key]["trap_kind_config"] = "owner-confirmed lure" if lure else "as exported (sticky)"
        print(f"  {key}: {runs[key]['photos']} photos, reviews {runs[key]['reviews']}, proposals {runs[key]['proposals']}, "
              f"false actuations {runs[key]['false_actuations']}, {runs[key]['runtime_s']} s", flush=True)
    # staged stress replay: synthetic surge + simulated operator
    sr = run_replay_main(["--days", "7", "--approve-none"], "stress_7d_synthetic_surge", out_dir,
                         patches=build_patches(lure=True, inject=True, operator=True), agent_traces=False)
    full["stress"] = sr
    runs["stress_7d"] = analyze_replay(sr)
    runs["stress_7d"]["staged"] = {"what": "synthetic surge injected into production ai_count and YOLOv7 boxes of both lure cameras, daylight hours",
                                   "extra_flies_by_day": STRESS["extra_flies_by_day"], "hours": list(STRESS["hours"]),
                                   "operator": {"median_h": _Operator.MEDIAN_H, "sigma": _Operator.SIGMA, "p_no_response": _Operator.P_NO_RESPONSE,
                                                "note": "assumed log-normal delay; not measured"}}
    runs["stress_7d"]["trap_kind_config"] = "owner-confirmed lure"
    runs["stress_7d"]["policy"] = "simulated operator (log-normal, median 2 h, 10 % never answer)"
    print(f"  stress: proposals {runs['stress_7d']['proposals']}, approvals {runs['stress_7d']['approvals']}, "
          f"false actuations {runs['stress_7d']['false_actuations']}", flush=True)

    # A/B: FlyBot today vs Sentinel, on the 30-day replay and on the staged one
    cams = jload(ROOT / "data" / "export" / "cameras.json") if (ROOT / "data" / "export" / "cameras.json").exists() else []
    {c["serial"]: c for c in cams}
    threshold = int(next((c.get("alertThreshold") for c in cams if c.get("alertThreshold")), 15))
    card_replaced = {c["serial"]: c["cardReplacedAt"] for c in cams if c.get("cardReplacedAt")}
    export_kind = next((c.get("trapKind") for c in cams if c.get("trapKind")), "sticky")
    owner_kinds = {c["serial"]: c.get("trap_kind") for c in AgentData.seed_from_env() if c.get("trap_kind")}
    owner_kind = next((k for k in owner_kinds.values() if k == "lure"), "lure")

    def ab(run_key: str, full_key: str) -> dict:
        photos = full[full_key]["photos"]
        a = runs[run_key]
        out = {"window": a["window"], "photos": a["photos"]}
        for label, kind in (("flybot_as_configured_today", export_kind), ("flybot_trapkind_lure_owner_confirmed", owner_kind)):
            fb = flybot_today_alerts(photos, trap_kind=kind, threshold=threshold, card_replaced=card_replaced)
            out[label] = {**fb, "summary": ab_summary(fb, {"reviews": a["reviews"], "proposals": a["proposals"], "extra_captures": a["extra_captures"],
                                                           "no_ops": a["no_ops"]})}
        out["sentinel"] = {k: a[k] for k in ("reviews", "review_reasons", "proposals", "extra_captures", "set_image_calls", "no_ops", "decisions",
                                              "escalation_rate", "escalated_photos")}
        return out

    ab_out = {"replay_30d": ab("replay_30d", "replay_30d"), "stress_7d": ab("stress_7d", "stress")}
    # stress: lead time between FlyBot's surge alert and Sentinel's first proposal
    st = ab_out["stress_7d"]
    first_prop: dict[str, str] = {}
    with open(ROOT / full["stress"]["trace"], encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["kind"] == "proposal" and r.get("camera"):
                first_prop.setdefault(r["camera"], r["at"])
    st["first_proposal_at"] = first_prop
    def _t(x):
        return datetime.fromisoformat(x)

    lead = {}
    fb_first = st["flybot_trapkind_lure_owner_confirmed"]["first_surge_alert_at"]
    if first_prop and fb_first:
        t_fb, t_se = min(_t(x) for x in fb_first.values()), min(_t(x) for x in first_prop.values())
        lead = {"flybot_first_surge_alert": min(fb_first.values()), "sentinel_first_proposal": min(first_prop.values()),
                "sentinel_minus_flybot_h": round((t_se - t_fb).total_seconds() / 3600, 2)}
    st["sentinel_proposal_minus_flybot_surge_alert_h"] = lead

    reviews30 = runs["replay_30d"]
    return {
        "runs": runs, "ab": ab_out,
        "ab_notes": [
            ("The traps have been EMPTY since 2026-10-07 (production counts ~0 on both cameras): there is no real surge, new catch, card change or "
            "treatment decision in the real replay, so the A/B on real data cannot show the 'trap'-driven benefits (earlier detection, "
            "fewer false alarms on real infestations)."),
            ("What the real replay does show: Sentinel's only human-facing action is the count-disagreement review (YOLOv7 vs production) on "
            f"{len(reviews30['review_days'])} days; those photos are the first labeling queue. Neither system could be scored for correctness "
            "without human labels (0 labels)."),
            ("The staged replay (stress_7d) injects a synthetic surge into counts, so it exercises the surge -> neighbour -> proposal path and the "
            "approval latency; its numbers measure the logic and the budgets, not detection quality."),
            "FlyBot's alerts are emulated from its own rules on the production counts (flybot_today_alerts), not read from its notification log.",
        ],
        "config": {"alert_threshold": threshold, "flybot_trap_kind_today": export_kind, "owner_confirmed_trap_kind": owner_kind,
                   "card_replaced": card_replaced},
        "files": {k: {"trace": v["trace"], "summary": v["summary_path"], "agent_traces": rel(v["agent_traces"]) if v.get("agent_traces") else None}
                  for k, v in full.items()},
        "seconds": round(time.perf_counter() - t0, 1),
    }, full


# =========================================================================== markdown (from the JSON only)

def _fmt(v, nd=3) -> str:
    if v is None:
        return "not measured"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


def _table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |")
    return "\n".join(out)


def aws_live_section(results_dir: Path) -> dict | None:
    """Re-summarize data/eval/aws_traces.jsonl (offline); fall back to the last aws_live.json."""
    try:
        from eval import aws_traces
    except ImportError:  # the Sentinel-era trace puller is not in this repo: keep the saved summary
        aws_traces = None

    if aws_traces is not None and aws_traces.JSONL.exists():
        res = aws_traces.summarize(aws_traces.JSONL)
        pull = aws_traces.JSONL.parent / "aws_traces.pull.json"
        if pull.exists():
            res["pull"] = jload(pull)
        jdump(results_dir / "aws_live.json", res)
        return res
    p = results_dir / "aws_live.json"
    return jload(p) if p.exists() else None


def expert_section(results_dir: Path) -> dict | None:
    from eval import score_expert_review as ser

    if not ser.CSV_PATH.exists():
        return None
    res = ser.score(ser.CSV_PATH)
    jdump(results_dir / "expert_review.json", res)
    return res


def _top(d: dict, n: int = 8) -> str:
    return ", ".join(f"{k}: {v}" for k, v in list(d.items())[:n]) or "none"


def render_aws(aws: dict, results_dir: Path) -> list[str]:
    L = ["## 5. Live on AWS (15 Oct)", ""]
    if aws.get("status") != "measured":
        return L + [f"not measured ({aws.get('status')}).", ""]
    lat = aws["latency"]
    pull = aws.get("pull") or {}
    L += [(f"{aws['photos']} per-photo traces of the deployed stack (stage dev, Sydney), {aws['window']['from']} .. {aws['window']['to']}; planner: "
           f"{json.dumps(aws['planner']['mode'])}, fallbacks {aws['planner']['fallbacks']}; {json.dumps(aws['by_source'])}. Read-only scan of DynamoDB `sentinel-traces` "
           f"({pull.get('scanned', 'n/a')} items, of which {pull.get('skipped', {}).get('excluded_serial', 0)} from excluded serials {', '.join(aws['excluded_serials'])}; "
           f"{json.dumps(pull.get('other_loops_after_exclusion', {}))} are not per-photo traces and are not analysed here). "
           f"Source: `{rel(results_dir / 'aws_live.json')}` (from `data/eval/aws_traces.jsonl`, git-ignored; `eval/aws_traces.py`)."), "",
          "### 5a. Decisions", "",
          _table(["camera", "photos", "recapture decisions"], [[c, v.get("photos"), v.get("recapture_decisions", 0)] for c, v in aws["by_camera"].items()]), "",
          _table(["decision action", "count"], [[k, v] for k, v in aws["decisions_by_action"].items()]), "",
          _table(["decision rule", "count"], [[k, v] for k, v in aws["decisions_by_rule"].items()]), "",
          ("Op 2 `blurry` flag vs the photo's own Laplacian sharpness (median, higher = sharper): " +
           "; ".join(f"{c}: flagged {v['flagged_median']} (n={v['flagged_n']}) vs unflagged {v['unflagged_median']} (n={v['unflagged_n']})"
                     for c, v in aws.get("blurry_flag_vs_laplacian_sharpness", {}).items()) +
           ". A real blur flag would sit on the LOWER sharpness; whether these flags are false alarms needs a human look (expert sheet rows AWS-*)."), "",
          f"Outcomes: {_top(aws['outcomes'])}. Photos with a quality issue (op 2): {_top(aws['photos_with_quality_issue'])}. Proposals created: {aws['proposals_created']}. Kill switch traces: {aws['kill_switch_traces']}.", "",
          "### 5b. Recaptures and follow-ups", "",
          _table(["recapture reason (photo issues)", "decisions"], [[k, v] for k, v in aws["recapture"]["by_reason"].items()]), "",
          (f"Recapture decisions {aws['recapture']['decisions']} (rules: {_top(aws['recapture']['by_rule'])}); capture tool calls: {_top(aws['recapture']['capture_tool_calls'])}; "
           f"guardrail blocks: {_top(aws['guardrails']['guardrail_block_events'])}. Follow-ups scheduled: {aws['followups']['scheduled_tool_calls_ok']} "
           f"in {aws['followups']['traces_with_followup']} traces (kinds: {_top(aws['followups']['by_kind'])}). None of the recaptures reached a camera "
           "(FlyBot actions are not wired to production yet), so whether a recapture fixes the photo is not measured live."), "",
          "### 5c. Latency per photo (from the trace)", ""]
    rows = []
    for key, label in (("agent_s", "agent run, s (finished_at - created_at)"), ("vision_ms", "OpenCV pipeline, ms (vision_ms)"), ("tool_call_ms", "tool call, ms")):
        d = lat[key]
        rows.append([label, *[d.get(k) for k in ("n", "p50", "p90", "p95", "max")]])
    ing = lat.get("ingest_delay_s")
    if isinstance(ing, dict):
        for k, label in (("live_le_1h", "photo -> trace, s (captured <= 1 h before the trace)"), ("backfill_gt_1h", "photo -> trace, s (older photos: backfill, NOT a latency)")):
            d = ing[k]
            rows.append([label, *[d.get(x) for x in ("n", "p50", "p90", "p95", "max")]])
    L += [_table(["metric", "n", "p50", "p90", "p95", "max"], rows), "",
          f"{lat['note']} Source: `{rel(results_dir / 'aws_live.json')}` (`latency`).", ""]
    return L


def render_expert(ex: dict | None) -> list[str]:
    L = ["## 6. Expert review of agent decisions", ""]
    if ex is None:
        return L + ["not measured: `docs/evidence/eval/expert_review.csv` does not exist yet.", ""]
    src = f"`{rel(RESULTS / 'expert_review.json')}` (from `{ex.get('source')}` via `eval/score_expert_review.py`)"
    if ex.get("status") != "measured":
        return L + [(f"**{ex.get('status')}.** Sheet: {ex.get('rows')} rows ({json.dumps(ex.get('rows_per_source'))} to review + {ex.get('agent_example_rows_excluded')} "
                     "worked examples written by the agent, excluded from scoring). Instructions: `eval/EXPERT_REVIEW.md`. "
                     f"Source: {src}."), ""]
    o = ex["overall"]
    rows = [["all", *[o[k] for k in ("correct", "incorrect", "unsure")], o["policy_correct_rate"], json.dumps(o["wilson95"]), o["worst_case_rate_unsure_as_incorrect"]]]
    for group in ("by_source", "by_decision"):
        rows += [[f"{group[3:]}: {k}", *[v[x] for x in ("correct", "incorrect", "unsure")], v["policy_correct_rate"], json.dumps(v["wilson95"]),
                  v["worst_case_rate_unsure_as_incorrect"]] for k, v in ex[group].items()]
    L += [_table(["group", "correct", "incorrect", "unsure", "policy-correct rate", "Wilson 95 %", "worst case (unsure = incorrect)"], rows), "",
          (f"{ex['verdicts']} verdicts; {ex['agent_example_rows_excluded']} agent-example rows excluded; {ex['pending']} rows pending. Source: {src}."), ""]
    for key, title in (("incorrect_rows", "Rows judged incorrect"), ("unsure_rows", "Rows judged unsure")):
        listed = ex.get(key) or []
        if listed:
            L += [f"{title} (the reviewer's note in English, translated afterwards by the team; his own words are in `expert_note`):", ""]
            L += [f"- {r['id']} ({r['source']}, `{r['decision']}`): {r.get('note_en') or 'no English translation'}" for r in listed] + [""]
    return L


def render_results_md(meta: dict, det: dict | None, vis: dict | None, agent: dict | None, fail: dict | None,
                      aws: dict | None = None, expert: dict | None = None, results_dir: Path = RESULTS) -> str:
    L: list[str] = []
    L += ["# Evaluation results", "",
          ("Generated by `uv run python -m eval.run_eval` from the JSON files in `eval/results/`. **Do not edit by hand**: every number below is read from "
          "one of those files (source named in each table). Anything not measured says so."), "",
          (f"Run: {meta['started']} | mode: {'quick' if meta['quick'] else 'full'} | OpenCV {meta['opencv']} | {meta['machine']} | "
          f"sections: {', '.join(meta['sections'])} | {meta['seconds']} s"), ""]
    if det:
        L += ["## 1. Detection: YOLOv7-tiny on OpenCV 5 (ENGINE_NEW)", ""]
        rb = det["roboflow_test"]
        L += ["### 1a. Roboflow test split boxes", ""]
        if rb.get("status") == "measured":
            rows = [["OpenCV 5 ENGINE_NEW (this run)", rb["n_images"], rb["n_gt_boxes"], rb["ap50"], rb["ap50_95"],
                     f"{rb['at_conf_0.25']['precision']} / {rb['at_conf_0.25']['recall']} (conf 0.25)",
                     f"{rb['best_f1']['precision']} / {rb['best_f1']['recall']} (conf {rb['best_f1']['conf']})"]]
            ref = rb.get("reference_pytorch")
            if ref:
                rows.append(["PyTorch test.py (reference)", ref["images"], ref["boxes"], ref["mAP50"], ref["mAP50_95"], f"{ref['precision']} / {ref['recall']}", "test.py max-F1 point"])
            L += [_table(["pipeline", "images", "boxes", "AP@50", "AP@50-95", "P / R at deployed threshold", "P / R at best F1"], rows), "",
                  (f"Source: `{rel(RESULTS / 'detection.json')}` (`roboflow_test`); dataset {rb['dataset']}; detections cached at conf >= {rb['floor_conf']}, "
                  f"NMS IoU {rb['nms_iou']}, matching IoU {rb['match_iou']}; engine used {rb['engine_used']}, OpenCV {rb['opencv']}; "
                  f"forward median {_fmt(rb.get('forward_ms_median'))} ms ({rb['note']})."), ""]
        else:
            L += [rb.get("status", "not measured"), ""]
        pc = det.get("production_counts") or {}
        L += ["### 1b. Counts vs FlyBot production count (Roboflow RF-DETR; not human ground truth)", ""]
        if pc.get("all"):
            rows = [["all", *[_fmt(pc["all"].get(k)) for k in ("n", "mae", "bias", "median_abs_err", "within_2")]]]
            rows += [[c, *[_fmt(v.get(k)) for k in ("n", "mae", "bias", "median_abs_err", "within_2")]] for c, v in pc["by_camera"].items()]
            L += [_table(["camera", "photos", "MAE", "bias (YOLOv7 - production)", "median abs err", "within 2"], rows), "",
                  f"Source: `{rel(RESULTS / 'detection.json')}` (`production_counts`), computed from `{pc['source']}` ({pc['window']['from']} .. {pc['window']['to']}, 30-day replay window, 640x640 test pictures and AC6527557 excluded)."]
        smp = det.get("production_counts_sample")
        if smp:
            L += ["", _table(["camera (93-photo stratified sample)", "photos", "MAE", "bias", "within 2"],
                             [["all", *[_fmt(smp["all"].get(k)) for k in ("n", "mae", "bias_mean_ours_minus_prod", "within_2")]]] +
                             [[c, *[_fmt(v.get(k)) for k in ("n", "mae", "bias_mean_ours_minus_prod", "within_2")]] for c, v in smp["by_camera"].items()]), "",
                  f"Source: `{rel(RESULTS / 'detection.json')}` (`production_counts_sample`), copied from `{smp['source']}` (letterbox, conf {smp['conf']}, engine {smp['engine_used']})."]
        L += [""]
        hc = det["human_counts"]
        L += ["### 1c. Counts vs human counts", ""]
        if hc.get("n_labels", 0) == 0:
            L += [(f"**{hc['status']}.** YOLOv7 / production / NPU vs human: not measured. Source `{rel(RESULTS / 'detection.json')}` (`human_counts`), "
                  f"computed from `{hc['source']}`: {hc['rows_in_file']} rows, {hc['ignored_agent_test_rows']} are `agent-test` tool checks "
                  "(not ground truth) and are ignored. See `eval/GROUND_TRUTH.md` to label the 150-photo sample."), ""]
        else:
            rows = [["YOLOv7-tiny vs human", *[_fmt(hc["yolov7_vs_human"].get(k)) for k in ("n", "mae", "bias", "within_2")]],
                    ["production vs human", *[_fmt(hc["production_vs_human"].get(k)) for k in ("n", "mae", "bias", "within_2")]]]
            L += [_table(["comparison", "photos", "MAE", "bias (pred - human)", "within 2"], rows), "",
                  (f"NPU vs human: not measured (counts.csv has no NPU count; no AI-camera photo has a human label). Source: `{rel(RESULTS / 'detection.json')}` (`human_counts`), "
                   f"computed from `{hc['source']}`; {hc['n_labels']} labeled photos, "
                   f"{hc['double_labeled']} double-labeled (labeler agreement {'not measured' if not hc['double_labeled'] else 'see file'})."), ""]
    if vis:
        L += ["## 2. Vision operations (evidence files, not recomputed)", "",
              _table(["op", "metric", "value", "n", "data", "source", "note"],
                     [[r["op"], r["metric"], _fmt(r["value"]), _fmt(r["n"]), r["dataset"], f"`{r['source']}`", r["note"]] for r in vis["rows"]]), "",
              f"Table collected into `{rel(RESULTS / 'vision_ops.json')}`; each row's own source file is in the `source` column.", ""]
    if agent:
        sc, runs, ab = agent["scenario"], agent["replay"]["runs"], agent["replay"]["ab"]
        L += ["## 3. Agent task success", "", "### 3a. Scenario suite on fakes (expected vs actual per step)", "",
              _table(["step", "expected", "observed", "pass"], [[s["step"], s["expected"], s["observed"], "PASS" if s["pass"] else "FAIL"] for s in sc["step_results"]]), "",
              (f"Policy-correct decisions on the scripted scenarios (an agent-run check against the expected behaviour in the table, not an expert judgement): **{sc['policy_correct']}/{sc['steps']}** (rate {sc['rate']}); false actuations **{sc['false_actuations']}**; "
              f"escalation steps {sc['escalation_steps']}/{sc['steps']} (rate {sc['escalation_rate']}); extra captures {sc['extra_captures']}; "
              f"proposals {sc['proposals']}; planner fallbacks {sc['planner_fallbacks']} (step 6, injected on purpose). {sc['source']}. Source: `{rel(RESULTS / 'agent.json')}`."), "",
              "### 3b. Replay on the real export (7 and 30 days) and the staged surge replay", ""]
        order = [("replay_7d", "real 7 d (lure)"), ("replay_30d", "real 30 d (lure)"), ("replay_30d_as_exported", "real 30 d (as exported: sticky)"), ("stress_7d", "STAGED 7 d (synthetic surge)")]
        order += [(k, k) for k in runs if k not in dict(order)]
        head = ["metric"] + [lab for k, lab in order if k in runs]
        keys = [("trap kind given to the agent", lambda r: r.get("trap_kind_config", "")), ("window", lambda r: f"{r['window']['from']}..{r['window']['to']}"), ("photos", lambda r: r["photos"]), ("approval policy", lambda r: r["policy"]),
                ("decisions", lambda r: json.dumps(r["decisions"])), ("escalation rate (photos with a review or proposal)", lambda r: r["escalation_rate"]),
                ("human reviews", lambda r: r["reviews"]), ("extra captures (ok)", lambda r: r["extra_captures"]),
                ("extra captures per farm day", lambda r: r["extra_captures_per_day"]), ("set_image calls", lambda r: r["set_image_calls"]),
                ("proposals", lambda r: f"{r['proposals']} {json.dumps(r['proposals_by_kind'])}"), ("proposals per farm day", lambda r: r["proposals_per_day"]),
                ("approvals / pending at end", lambda r: f"{r['approvals']} / {r['pending_at_end']}"),
                ("approval latency h (p50 / p90 / max)", lambda r: "not measured (no proposals)" if not r["approval_latency_h"].get("n") else
                 f"{r['approval_latency_h']['p50_h']} / {r['approval_latency_h']['p90_h']} / {r['approval_latency_h']['max_h']} (n={r['approval_latency_h']['n']}, "
                 f"{'simulated' if 'simulated' in r['policy'] else 'constant by policy'})"),
                ("planner fallbacks / agent errors", lambda r: f"{r['planner_fallbacks']} / {r['agent_errors']}"),
                ("false actuations (captures in quiet hours)", lambda r: r["false_actuations"]),
                ("YOLOv7 vs production MAE", lambda r: r["yolov7_vs_production_mae"]), ("trace sha256", lambda r: r["trace_sha256"][:12])]
        L += [_table(head, [[name, *[fn(runs[k]) for k, _ in order if k in runs]] for name, fn in keys]), "",
              f"Source: `{rel(RESULTS / 'agent.json')}` (`replay.runs`), summarizing " + "; ".join(f"`{v['summary']}`" for v in [{"summary": r["summary"]} for r in runs.values()]) + ".", ""]
        r30 = runs["replay_30d"]
        L += ["Why the real replay has reviews: " + "; ".join(f"{k}: {v}" for k, v in r30["review_reasons"].items()) +
              f" (on farm days {', '.join(r30['review_days'])}).", ""]
        L += ["### 3c. A/B: FlyBot today vs Sentinel (replay, same photos)", ""]
        for key, label in (("replay_30d", "Real 30-day replay"), ("stress_7d", "STAGED 7-day replay (synthetic surge)")):
            a = ab[key]
            rows = []
            for cfg, lab in (("flybot_as_configured_today", f"FlyBot as configured today (trapKind={agent['replay']['config']['flybot_trap_kind_today']})"),
                             ("flybot_trapkind_lure_owner_confirmed", f"FlyBot with trapKind={agent['replay']['config']['owner_confirmed_trap_kind']} (owner-confirmed)")):
                s = a[cfg]["summary"]
                rows.append([lab, s["flybot_threshold_alerts"], s["flybot_threshold_alert_days"], s["flybot_surge_alerts"], s["flybot_quality_alerts"], s["flybot_alerts"]])
            se = a["sentinel"]
            L += [f"**{label}** ({a['window']['from']}..{a['window']['to']}, {a['photos']} photos)", "",
                  _table(["system", "threshold alerts", "alert days", "surge alerts", "quality reminders", "total alerts"], rows), "",
                  _table(["system", "human asks (reviews + proposals)", "reviews", "proposals", "extra captures", "no-op photos"],
                         [["Sentinel (agent)", a["flybot_as_configured_today"]["summary"]["sentinel_human_asks"], se["reviews"], se["proposals"], se["extra_captures"], se["no_ops"]]]), ""]
            if key == "stress_7d":
                L += [(f"First FlyBot surge alert vs first Sentinel proposal: {json.dumps(a.get('sentinel_proposal_minus_flybot_surge_alert_h'))} "
                      "(positive = Sentinel later). Both start from the same surge rule on counts; Sentinel adds a neighbour confirmation and a person's approval "
                      "before anything is proposed."), ""]
        L += ["Notes: " + " ".join(agent["replay"]["ab_notes"]), "", f"Source: `{rel(RESULTS / 'agent.json')}` (`replay.ab`), traces under `data/eval/replay/` (git-ignored).", ""]
    if fail:
        L += ["## 4. Failure injection", "",
              (f"{fail['passed']}/{fail['total']} case runs passed; false actuations {fail['false_actuations_total']}. Real photos "
              f"({', '.join(p['cur'] + ' ' + p['camera'] for p in fail['photos'])}) are perturbed; the real OpenCV 5 pipeline and planners run on them. "
              f"Simulated: {fail['simulated']}. Source: `{rel(RESULTS / 'failure_injection.json')}`."), ""]
        rows = []
        for c in fail["cases"]:
            if c.get("photo") != fail["photos"][0]["cur"]:
                continue
            rows.append([c["case"], c["perturbation"], c["expected"], c["observed"][:300], "PASS" if c["pass"] else "FAIL", c.get("gap") or "",
                         f"`{c['evidence_image']}`" if c.get("evidence_image") else ""])
        L += [_table(["case", "perturbation", "expected", "observed", "result", "gap found", "evidence image"], rows), ""]
        extra = [c for c in fail["cases"] if c.get("photo") != fail["photos"][0]["cur"]]
        if extra:
            by = defaultdict(lambda: [0, 0])
            for c in fail["cases"]:
                if c["group"] == "image":
                    by[c["case"]][0] += c["pass"]
                    by[c["case"]][1] += 1
            L += ["Image perturbations over all photo pairs: " + "; ".join(f"{k} {v[0]}/{v[1]}" for k, v in by.items()) + ".", ""]
        sw = fail.get("sweep")
        if sw:
            L += [f"### Op 2 flag rate by perturbation strength ({sw['n_photos']} real clean photos, {json.dumps(sw['cameras'])}; baseline = median of 10 earlier frames of the camera)", "",
                  (f"Clean photos flagged with any issue: no baseline {sw['clean_false_positive']['no_baseline']['any_issue']}/{sw['clean_false_positive']['no_baseline']['n']}, "
                  f"with baseline {sw['clean_false_positive']['with_baseline']['any_issue']}/{sw['clean_false_positive']['with_baseline']['n']}. "
                  "The agent's path passes **no baseline**."), ""]
            rows = []
            for name, s in sw["sweeps"].items():
                for lv in s["levels"]:
                    rows.append([name, f"{lv['level']} {lv['unit']}", s["issue"], lv["no_baseline"]["flag_rate"], lv["with_baseline"]["flag_rate"],
                                 json.dumps(lv["no_baseline"]["other_issues"])])
            L += [_table(["perturbation", "strength", "expected flag", "flag rate (agent path, no baseline)", "flag rate (with baseline)", "other issues flagged (no baseline)"], rows), ""]
    if aws is not None:
        L += render_aws(aws, results_dir)
    L += render_expert(expert)
    L += ["## Limitations (everything not measured, and what limits what was)", ""]
    nm: list[str] = []
    for d in (det, vis):
        if d:
            nm += [x for x in d.get("not_measured", []) if not x.startswith("Detection accuracy on real lure-trap photos against human counts")]
    hc = (det or {}).get("human_counts") or {}
    nm += [
        (f"Human ground truth for counts: {hc.get('status', '0 labels, not measured')}. So there is no MAE / bias against people for YOLOv7, production (Roboflow RF-DETR) "
        "or the NPU, no precision / recall on real lure-trap photos, and every \"vs production\" number is agreement between two models, not accuracy."),
        "NPU (AI camera) counts vs human or vs OpenCV 5 on real photos: not measured (no real AI-camera photo with a count and a label; firmware 0.25.0 not flashed).",
    ]
    if agent:
        nm += [("Real-data task success of the agent: the traps are empty (no surge, new catch, card change, treatment decision on real photos), so those paths are "
                "exercised only by the scenario and the staged replay (synthetic)."),
               "Approval latency of real people: simulated (assumed log-normal), not measured; no real proposal has been approved or rejected yet.",
               "The scenario suite checks the agent against behaviour its author wrote down (9 steps on fakes): it is a regression test, not an independent judgement.",
               ("Replays use cached YOLOv7 counts and the stored production counts, emulate FlyBot's own alerts from its rules (not from its notification log), and "
                "cannot change pixels: a recapture or an ISP change returns the stored frame or nothing."),
               "Real ISP fix on a real AI camera (waits for firmware 0.25.0 OTA): not measured.",
               "Live AI-camera loop (IoT Core -> worker -> trace) with a physical camera: only a simulated thing was verified.",
               "The LLM planner (Bedrock) on real photos: no live trace uses it (planner mode `rule`); LLM behaviour is exercised only with scripted fake models.",
               ("Cost per 1,000 photos: see bench/REPORT_COOL.md, not part of this run. Lambda duration, cold starts and queue delay: not in the traces, not measured here "
               "(CloudWatch).")]
    if fail:
        nm += [("Failure injection perturbs 3 photo pairs from one camera (BE4089629) with synthetic effects: ISP effects are not simulated in pixels, a recapture "
                "returns an unperturbed frame, devices are fakes; it shows how the code reacts, not how a real camera or a real person behaves."),
               ("Op 2 `blurry` on the agent path: flag rate by blur strength is in section 4; whether the live `blurry` flags in section 5 are real blur or false alarms "
               "is not measured (no human looked at all of them; some are in the expert sheet).")]
    if aws:
        nm += [(f"Live AWS traces: {aws.get('photos')} photos (traces created within one 35-minute poller sweep on 15 Oct, photos captured 14-15 Oct), 2 EZVIZ cameras, rule planner, FlyBot actions not wired (every recapture blocked `unknown_camera`): "
               "no live proposal, approval, ISP change or surge; the sample says the loop runs and how fast, not that its decisions are right.")]
    if expert is None or expert.get("status") != "measured":
        nm += [("Policy-correct decision rate judged by an AgriHub agronomist on 50 cases: the sheet is ready (`docs/evidence/eval/expert_review.csv`, "
               "`eval/EXPERT_REVIEW.md`), 0 verdicts, not measured.")]
    else:
        nm += [f"Expert review: one reviewer panel, {expert['verdicts']} verdicts; confidence interval in section 6; inter-rater agreement not measured."]
    nm += ["Sample sizes and scope: 2 cameras, 1 region, 30 days, traps empty since 2026-10-07; no seasonal or multi-farm variation; no yellow sticky cards.",
           "Timings come from a shared 4-core dev laptop and a dev stack: they are not benchmarks (see bench/)."]
    L += [f"- {x}" for x in nm] + [""]
    return "\n".join(L)


# =========================================================================== driver

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="small N everywhere (a few minutes with warm caches)")
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=None)
    ap.add_argument("--onnx", type=Path, default=ROOT / "models" / "yolov7-tiny" / "fruitfly_deploy.onnx")
    ap.add_argument("--roboflow-dir", type=Path, default=ROOT.parent / "camos-ai-camera" / "models" / "fruit-fly-v3")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--results-md", type=Path, default=RESULTS_MD)
    ap.add_argument("--replay-dir", type=Path, default=EVAL_DATA / "replay")
    ap.add_argument("--render-only", action="store_true", help="recompute nothing heavy: re-summarize AWS traces / expert sheet and re-render the Markdown")
    a = ap.parse_args(argv)
    sections = set() if a.render_only else set(a.only or SECTIONS)
    t0 = time.perf_counter()
    started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    import cv2

    from sentinel import require_opencv5
    require_opencv5()
    rdir: Path = a.results_dir
    rdir.mkdir(parents=True, exist_ok=True)

    def load_or(name: str):
        p = rdir / name
        return jload(p) if p.exists() else None

    agent = det = vis = fail = None
    replay30 = None
    if "agent" in sections:
        print("[agent] scenario", flush=True)
        scenario = agent_scenario()
        print(f"  scenario {scenario['policy_correct']}/{scenario['steps']} policy-correct, false actuations {scenario['false_actuations']}", flush=True)
        print("[agent] replays", flush=True)
        replay_res, full = agent_replay_section(a.quick, a.replay_dir)
        agent = {"scenario": scenario, "replay": replay_res}
        jdump(rdir / "agent.json", agent)
        replay30 = {"photos": full["replay_30d"]["photos"], "trace": full["replay_30d"]["trace"], "window": full["replay_30d"]["window"]}
    if "detection" in sections:
        print("[detection]", flush=True)
        if replay30 is None:
            tp = a.replay_dir / "30d_approve-after-2h" / "trace.jsonl"
            if tp.exists():
                s = jload(tp.parent / "summary.json")
                replay30 = {"photos": trace_photos(tp), "trace": rel(tp), "window": s.get("window")}
        det = detection_section(a.quick, a.onnx, a.roboflow_dir, replay30)
        jdump(rdir / "detection.json", det)
        rb = det["roboflow_test"]
        print(f"  roboflow test: {rb.get('status')} AP50 {rb.get('ap50')} | human labels: {det['human_counts']['status']}", flush=True)
    if "vision" in sections:
        print("[vision]", flush=True)
        vis = vision_ops_section()
        jdump(rdir / "vision_ops.json", vis)
        print(f"  {len(vis['rows'])} rows", flush=True)
    if "failure" in sections:
        print("[failure injection]", flush=True)
        from eval import failure_injection as fi

        fail = fi.run_all(photos=1 if a.quick else 3, sweep_n=12 if a.quick else 40)
        jdump(rdir / "failure_injection.json", fail)
        print(f"  {fail['passed']}/{fail['total']} passed, false actuations {fail['false_actuations_total']}", flush=True)
    det = det or load_or("detection.json")
    vis = vis or load_or("vision_ops.json")
    agent = agent or load_or("agent.json")
    fail = fail or load_or("failure_injection.json")
    aws = aws_live_section(rdir)
    expert = expert_section(rdir)
    if a.render_only and (rdir / "eval_run.json").exists():
        prev = jload(rdir / "eval_run.json")
        started, quick_flag, sec = prev.get("started", started), prev.get("quick", a.quick), prev.get("sections", [])
        t0 = time.perf_counter() - prev.get("seconds", 0)  # show the duration of the run that produced the JSON files
    else:
        quick_flag, sec = a.quick, sorted(sections)
    meta = {"started": started, "quick": quick_flag, "sections": sec, "opencv": cv2.__version__,
            "machine": f"{platform.system()} {platform.release()} {platform.processor() or platform.machine()}, {os.cpu_count()} CPUs",
            "seconds": round(time.perf_counter() - t0, 1)}
    if not a.render_only:
        jdump(rdir / "eval_run.json", meta)
    a.results_md.write_text(render_results_md(meta, det, vis, agent, fail, aws, expert, rdir), encoding="utf-8")
    print(f"wrote {rel(a.results_md)} and {rel(rdir)}/*.json in {meta['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
