"""Count agreement of a YOLOv7 ONNX (OpenCV 5 DNN) with the production count `ai_count`.

`ai_count` is the Roboflow RF-DETR Small count FlyBot stored at capture time: it is a model, **not
human ground truth**. This script measures how close our detector is to production, nothing more.
Human counts come from eval/label_counts.py (data/labels/counts.csv); pass --labels to add them.

    uv run python eval/count_vs_production.py models/yolov7-tiny/fruitfly_deploy.onnx --n 48
    uv run python eval/count_vs_production.py ONNX --resize letterbox stretch --engine new

Images: the first --n rows of data/labels/gt_sample.csv (round-robin over camera x time of day x count
bucket strata, eval/select_gt_sample.py). Outputs eval/results/count_vs_production__<onnx stem>.csv
(per image) and .json (summary: MAE, bias = mean(ours - production), per bucket, per camera; and the
max abs difference of the raw output between the new and classic engines on the first 3 images).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.vision.detect_yolov7 import YoloV7Detector, preprocess


def summarize(pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {"n": 0}
    err = [o - p for o, p in pairs]
    return {
        "n": len(pairs),
        "mae": round(statistics.fmean(abs(e) for e in err), 2),
        "bias_mean_ours_minus_prod": round(statistics.fmean(err), 2),
        "median_abs_err": statistics.median(abs(e) for e in err),
        "max_abs_err": max(abs(e) for e in err),
        "within_2": sum(abs(e) <= 2 for e in err),
        "sum_ours": sum(o for o, _ in pairs),
        "sum_prod": sum(p for _, p in pairs),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--sample", type=Path, default=ROOT / "data" / "labels" / "gt_sample.csv")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--engine", default="new", choices=["new", "classic", "auto"])
    ap.add_argument("--resize", nargs="+", default=["letterbox", "stretch"], choices=["letterbox", "stretch"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--labels", type=Path, default=None, help="data/labels/counts.csv: add human counts")
    ap.add_argument("--exclude-camera", action="append", default=[],
                    help="skip this device_serial (repeatable), e.g. a camera with no trap in view")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "results")
    args = ap.parse_args(argv)

    rows = [r for r in csv.DictReader(args.sample.open(encoding="utf-8"))
            if r["device_serial"] not in args.exclude_camera][: args.n]
    human = {}
    if args.labels and args.labels.exists():
        for r in csv.DictReader(args.labels.open(encoding="utf-8")):
            if r.get("true_count") not in (None, "") and not r.get("labeler", "").startswith("agent-test"):
                human[r["snapshot_id"]] = int(r["true_count"])

    dets = {m: YoloV7Detector(args.onnx, args.engine, resize=m, conf=args.conf, iou=args.iou) for m in args.resize}
    per_image, ms = [], defaultdict(list)
    t0 = time.time()
    for r in rows:
        img = cv2.imread(str(args.data_root / r["image_path"]), cv2.IMREAD_COLOR)
        rec = {k: r[k] for k in ("order", "snapshot_id", "device_serial", "local_time", "time_of_day",
                                 "count_bucket", "ai_count")}
        rec["image_wh"] = f"{img.shape[1]}x{img.shape[0]}"
        for m, d in dets.items():
            res = d.detect(img)
            rec[f"count_{m}"] = res["count"]
            ms[m].append(res["forward_ms"])
        rec["human_count"] = human.get(r["snapshot_id"], "")
        per_image.append(rec)
        print(rec, flush=True)

    # new vs classic raw output on the first 3 images (same blob)
    parity = []
    if args.engine == "new":
        nets = {e: cv2.dnn.readNet(str(args.onnx), "", "", eng)
                for e, eng in (("new", cv2.dnn.ENGINE_NEW), ("classic", cv2.dnn.ENGINE_CLASSIC))}
        for r in rows[:3]:
            img = cv2.imread(str(args.data_root / r["image_path"]), cv2.IMREAD_COLOR)
            blob, _, _ = preprocess(img, dets[args.resize[0]].size, args.resize[0])
            outs = {}
            for e, net in nets.items():
                net.setInput(blob)
                outs[e] = net.forward().copy()
            d = np.abs(outs["new"] - outs["classic"])
            parity.append({"snapshot_id": r["snapshot_id"], "max_abs_diff_box_px": float(d[..., :4].max()),
                           "max_abs_diff_scores": float(d[..., 4:].max())})

    summary = {
        "what": "our detector count vs production ai_count (Roboflow RF-DETR Small); NOT human ground truth",
        "onnx": str(args.onnx), "onnx_meta": dets[args.resize[0]].meta.get("sha256"),
        "opencv": cv2.__version__, "engine_requested": args.engine,
        "engine_used": dets[args.resize[0]].engine_used,
        "conf": args.conf, "iou": args.iou, "sample": str(args.sample), "n": len(per_image),
        "excluded_cameras": args.exclude_camera,
        "seconds": round(time.time() - t0, 1),
        "forward_ms_median": {m: round(statistics.median(v), 1) for m, v in ms.items()},
        "forward_ms_note": "measured on a shared dev laptop while other jobs ran; not a benchmark",
        "new_vs_classic_parity": parity,
        "by_resize": {},
    }
    for m in args.resize:
        pairs = [(p[f"count_{m}"], int(p["ai_count"])) for p in per_image]
        by_bucket, by_cam = defaultdict(list), defaultdict(list)
        for p in per_image:
            by_bucket[p["count_bucket"]].append((p[f"count_{m}"], int(p["ai_count"])))
            by_cam[p["device_serial"]].append((p[f"count_{m}"], int(p["ai_count"])))
        block = {"all": summarize(pairs),
                 "by_count_bucket": {k: summarize(v) for k, v in sorted(by_bucket.items())},
                 "by_camera": {k: summarize(v) for k, v in sorted(by_cam.items())}}
        hp = [(p[f"count_{m}"], p["human_count"]) for p in per_image if p["human_count"] != ""]
        if hp:
            block["vs_human"] = summarize(hp)
            block["production_vs_human"] = summarize([(int(p["ai_count"]), p["human_count"])
                                                      for p in per_image if p["human_count"] != ""])
        summary["by_resize"][m] = block

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"count_vs_production__{args.onnx.stem}{'__' + args.tag if args.tag else ''}"
    with (args.out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0]))
        w.writeheader()
        w.writerows(per_image)
    (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("engine_used", "n", "forward_ms_median", "new_vs_classic_parity")}, indent=2))
    for m, b in summary["by_resize"].items():
        print(m, json.dumps(b["all"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
