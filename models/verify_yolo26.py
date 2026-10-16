"""Day-1 check of a YOLO26n ONNX on OpenCV 5 DNN (new engine) against ONNX Runtime, on sample photos.

    cd lambda/vision
    uv run --with onnxruntime python ../../models/verify_yolo26.py ../../models/yolo26/fruitfly_yolo26n.onnx \
        --images ../../data/samples/images --n 20 [--engine new] [--out ../../eval/results/yolo26_parity.json]

What it checks, per photo (same letterbox preprocessing for both runtimes, `sentinel.vision.detect_yolov7.preprocess`):
  1. `cv2.dnn.readNet(..., ENGINE_NEW)` loads the graph and `forward()` returns finite values of the expected shape;
  2. the raw outputs agree with onnxruntime (max abs diff on the top-300 rows for the end-to-end head, or on all
     anchors for the one-to-many head);
  3. the decoded boxes agree after `sentinel.vision.detect_yolo26.decode` (count, and per-box max abs diff in
     pixels / score, matched greedily by IoU).
Exit 0 when OpenCV loads the model and every photo agrees within --tol-px / --tol-score; exit 2 when OpenCV cannot
load or run the graph (unsupported op); exit 1 on a numeric mismatch.

Fallback when exit 2 (an op of the NMS-free end-to-end head is unsupported by the new DNN engine): re-export with
the one-to-many head (`end2end=False` in models/colab_train_yolo26.ipynb, output [1, 4+nc, 8400]) and keep NMS in
Python (`decode_one_to_many` -> cv2.dnn.NMSBoxes). The detector module handles both shapes: nothing else changes.
Record the outcome in models/MODEL_CARD.md and docs/DECISIONS.md with the JSON this script writes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lambda" / "vision"))

from sentinel.vision.detect_smoke import ENGINES, engine_of  # noqa: E402
from sentinel.vision.detect_yolo26 import decode  # noqa: E402
from sentinel.vision.detect_yolov7 import preprocess  # noqa: E402


def iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match(a: list[dict], b: list[dict]) -> tuple[float, float, int]:
    """Greedy IoU matching -> (max abs box diff px, max abs score diff, unmatched)."""
    used, dpx, dsc, unmatched = set(), 0.0, 0.0, 0
    for da in a:
        best, bi = 0.0, None
        for j, db in enumerate(b):
            if j in used:
                continue
            v = iou(da["xywh"], db["xywh"])
            if v > best:
                best, bi = v, j
        if bi is None or best < 0.5:
            unmatched += 1
            continue
        used.add(bi)
        dpx = max(dpx, max(abs(x - y) for x, y in zip(da["xywh"], b[bi]["xywh"], strict=True)))
        dsc = max(dsc, abs(da["score"] - b[bi]["score"]))
    return dpx, dsc, unmatched + (len(b) - len(used))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("onnx")
    ap.add_argument("--images", default=str(HERE.parent / "data" / "samples" / "images"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="new")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--tol-px", type=float, default=0.5)
    ap.add_argument("--tol-score", type=float, default=1e-3)
    ap.add_argument("--out", default=str(HERE.parent / "eval" / "results" / "yolo26_parity.json"))
    a = ap.parse_args(argv)

    report: dict = {"onnx": str(Path(a.onnx).resolve()), "opencv": cv2.__version__, "engine_requested": a.engine,
                    "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou, "photos": [], "status": None}
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime missing: run with `uv run --with onnxruntime python ...`", file=sys.stderr)
        return 2
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    report["onnxruntime"] = ort.__version__
    report["graph_output_shape"] = [o.shape for o in sess.get_outputs()]
    try:
        t0 = time.perf_counter()
        net = cv2.dnn.readNet(a.onnx, "", "", ENGINES[a.engine])
        report["opencv_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        report["engine_used"] = engine_of(net)
    except cv2.error as e:
        report["status"] = "opencv_cannot_load"
        report["error"] = str(e)[:1000]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"OpenCV cannot load {a.onnx}: {str(e)[:300]}\n-> fallback: export with end2end=False (see docstring)")
        return 2

    images = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: a.n]
    if not images:
        print(f"no images under {a.images}", file=sys.stderr)
        return 2
    worst_px = worst_sc = 0.0
    bad = 0
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        blob, scale, offset = preprocess(img, a.imgsz)
        row = {"image": p.name, "width": img.shape[1], "height": img.shape[0]}
        try:
            net.setInput(blob)
            t0 = time.perf_counter()
            out_cv = net.forward()
            row["opencv_forward_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except cv2.error as e:
            report["status"] = "opencv_cannot_run"
            report["error"] = str(e)[:1000]
            row["error"] = str(e)[:300]
            report["photos"].append(row)
            break
        t0 = time.perf_counter()
        out_ort = sess.run(None, {in_name: blob})[0]
        row["onnxruntime_forward_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row["output_shape"] = {"opencv": list(out_cv.shape), "onnxruntime": list(out_ort.shape)}
        row["output_finite"] = bool(np.isfinite(out_cv).all())
        if out_cv.shape == out_ort.shape:
            row["raw_max_abs_diff"] = float(np.abs(out_cv.astype(np.float32) - out_ort.astype(np.float32)).max())
        dets_cv = decode(out_cv, a.conf, a.iou, scale, offset)
        dets_ort = decode(out_ort, a.conf, a.iou, scale, offset)
        dpx, dsc, unmatched = match(dets_cv, dets_ort)
        row.update(count_opencv=len(dets_cv), count_onnxruntime=len(dets_ort), box_max_abs_diff_px=round(dpx, 4),
                   score_max_abs_diff=round(dsc, 6), unmatched=unmatched)
        ok = row["output_finite"] and unmatched == 0 and dpx <= a.tol_px and dsc <= a.tol_score
        row["ok"] = ok
        bad += not ok
        worst_px, worst_sc = max(worst_px, dpx), max(worst_sc, dsc)
        report["photos"].append(row)
        print(f"{p.name:40} cv={len(dets_cv):3} ort={len(dets_ort):3} dpx={dpx:.3f} dscore={dsc:.5f} "
              f"{'ok' if ok else 'MISMATCH'}  {row['opencv_forward_ms']} / {row['onnxruntime_forward_ms']} ms")
    if report["status"] is None:
        report["status"] = "ok" if bad == 0 else "mismatch"
    report.update(photos_checked=len(report["photos"]), mismatches=bad, worst_box_diff_px=round(worst_px, 4),
                  worst_score_diff=round(worst_sc, 6), head="end2end" if out_cv.shape[-1] == 6 else "one_to_many")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{report['status']}: {len(report['photos'])} photos, {bad} mismatches, worst box diff {worst_px:.4f} px, "
          f"worst score diff {worst_sc:.6f}, engine {report.get('engine_used')}, head {report['head']} -> {a.out}")
    return {"ok": 0, "mismatch": 1}.get(report["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
