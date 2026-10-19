"""Tiled detection (DETECTOR_TILES, sentinel.vision.tiling) measured on the clean photos of SYNTHETIC sets (eval/synth).

    uv run --project lambda/vision python -m eval.synth.tiles [--sets data/synth/demo-30d data/synth/pool] [--limit N]
    uv run --project lambda/vision python -m eval.synth.tiles --handler off=data/synth/demo-30d sticky=DIR
    uv run --project lambda/vision python -m eval.synth.tiles --latency N

Per photo the detector runs once on the whole frame and once on each 2x2 window (5 forwards, the windows' boxes kept
in window pixels); every DETECTOR_TILES variant is then scored from those boxes with the detector's own code
(`tiling.decide`, `tiling.merge`, `box_rules.apply_rules`), so the variants differ only in the rule, not in the run.
This is the handler's detector step (YOLO26n on OpenCV 5 DNN, NMS, box rules) without the trap ROI and the other ops.
Writes eval/results/synth_tiles.json; eval.synth.report puts its table into eval/SYNTHETIC_RESULTS.md.

`--handler` adds whole-handler runs to that file: eval.synth.check on the same set, once per DETECTOR_TILES value
(DIR = the `--out` folder of eval.synth.check, so the runs' check.jsonl stay apart), compared
on the clean photos with the trap ROI, quality ops and agent included.
`--latency N` adds the wall time of `detect()` off vs sticky on N sticky photos (the cost of the 4 forwards).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.synth import ROOT
from eval.synth.check import gt_boxes, match

RESULTS = ROOT / "eval" / "results" / "synth_tiles.json"
DEFAULT_SETS = [ROOT / "data" / "synth" / "demo-30d", ROOT / "data" / "synth" / "pool"]
VARIANTS = {  # name -> Tiling fields (grid 2, overlap 0.12, edge 2 px unless given)
    "off": {"mode": "off"},
    "on": {"mode": "on"},
    "sticky": {"mode": "sticky"},
    "auto20": {"mode": "auto", "auto_px": 20.0},
    "auto24": {"mode": "auto", "auto_px": 24.0},
    "auto28": {"mode": "auto", "auto_px": 28.0},
    "auto24-full": {"mode": "auto", "auto_px": 24.0, "keep_full": False},
}


def xyxy(dets: list[dict]) -> list[list[float]]:
    return [[d["xywh"][0], d["xywh"][1], d["xywh"][0] + d["xywh"][2], d["xywh"][1] + d["xywh"][3]] for d in dets]


def passes(det, img) -> dict:
    """The whole-frame pass and the 2x2 window passes (boxes after NMS, before the box rules), with their times."""
    from sentinel.vision import tiling

    H, W = img.shape[:2]
    full, _, _, scale, ms = det._pass(img)
    wins = tiling.windows(W, H, 2, 0.12)
    per, margins, win_ms = [], [], []
    for x0, y0, x1, y1 in wins:
        d, _, _, s, m = det._pass(img[y0:y1, x0:x1])
        per.append(d)
        margins.append(2.0 / s)
        win_ms.append(m)
    return {"full": full, "scale": scale, "wins": wins, "per": per, "margins": margins, "full_ms": ms, "win_ms": win_ms}


def variant(ps: dict, cfg: dict, trap_kind: str, W: int, H: int, size: int, iou: float, rules) -> tuple[list[dict], int, bool]:
    """Boxes of one DETECTOR_TILES variant from the cached passes: (boxes, forwards, tiled)."""
    from sentinel.vision import tiling
    from sentinel.vision.box_rules import apply_rules

    t = tiling.Tiling(**{"keep_full": cfg.get("mode") == "auto", **cfg})
    first = t.mode in ("off", "auto") or (t.mode == "sticky" and trap_kind != "sticky")
    use, _ = tiling.decide(t, trap_kind, ps["full"] if first else None, ps["scale"], max(W, H), size)
    dets = ps["full"]
    if use:
        dets, _ = tiling.merge(ps["per"], ps["wins"], W, H, iou, ps["margins"], ps["full"] if first and t.keep_full else None)
    return apply_rules(dets, W, H, rules)[0], int(first) + 4 * int(use), use


def score(rows: list[dict]) -> dict:
    if not rows:
        return {"photos": 0}
    true, det, tp = sum(r["true"] for r in rows), sum(r["det"] for r in rows), sum(r["tp30"] for r in rows)
    err = [r["det"] - r["true"] for r in rows]
    p, rc = (tp / det if det else None), (tp / true if true else None)
    return {"photos": len(rows), "true_flies": true, "detected": det, "mae": round(statistics.fmean(abs(e) for e in err), 3),
            "bias": round(statistics.fmean(err), 3), "precision_iou30": None if p is None else round(p, 4),
            "recall_iou30": None if rc is None else round(rc, 4),
            "f1_iou30": None if not (p and rc) else round(2 * p * rc / (p + rc), 4),
            "tiled_photos": sum(r["tiled"] for r in rows), "forwards_per_photo": round(statistics.fmean(r["forwards"] for r in rows), 2)}


def run(sets: list[Path], limit: int | None = None, log=print) -> dict:
    import cv2

    from sentinel.vision.box_rules import BoxRules
    from sentinel.vision.detect_yolo26 import Yolo26Detector
    from sentinel.vision.detector import resolve_onnx

    rules = BoxRules()
    det = Yolo26Detector(resolve_onnx("yolo26"))  # decoder defaults conf 0.25, NMS 0.45, as the handler
    rows: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    full_ms, win_ms, used_sets = [], [], []
    t_run = time.perf_counter()
    for d in sets:
        m = json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))
        if m.get("synthetic") is not True:
            raise SystemExit(f"{d} is not a synthetic set")
        photos = [p for p in m["photos"] if not p.get("defect")]
        photos = photos[:limit] if limit else photos
        used_sets.append({"set": m.get("name"), "seed": m.get("seed"), "clean_photos": len(photos),
                          "generator_git": m["generator"]["git"]})
        for i, p in enumerate(photos, 1):
            img = cv2.imread(str(Path(d) / p["file"]), cv2.IMREAD_COLOR)
            ps = passes(det, img)
            full_ms.append(ps["full_ms"])
            win_ms.append(sum(ps["win_ms"]))
            gt = gt_boxes(p)
            for name, cfg in VARIANTS.items():
                boxes, fw, tiled = variant(ps, cfg, p["trapKind"], p["width"], p["height"], det.size, det.iou, rules)
                rows[name].append({"set": m.get("name"), "kind": p["trapKind"], "true": len(gt), "det": len(boxes),
                                   "tp30": match(gt, xyxy(boxes), 0.3), "forwards": fw, "tiled": tiled})
            if i % 50 == 0 or i == len(photos):
                log(f"{m.get('name')} {i}/{len(photos)} ({time.perf_counter() - t_run:.0f} s)")
    out = {"kind": "measurement on SYNTHETIC composites (eval/synth), not field photos",
           "generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "sets": used_sets,
           "model": f"{det.name} on opencv-{det.cv_version}-dnn-{det.engine_used}", "onnx": det.onnx_path.name,
           "scope": "detector step only (decoder conf 0.25, NMS 0.45, default box rules), whole frame, no trap ROI; "
                    "clean photos (no injected defect); true flies = visible pasted flies; greedy one-to-one at IoU 0.3",
           "tiling": "2x2 windows, overlap 0.12 x frame / 2, a box within 2 detector-input px of an inner window edge "
                     "is dropped, class-agnostic NMS 0.45 in full-frame pixels, then the box rules",
           "variants": {k: {**v, "keep_full": v.get("keep_full", v["mode"] == "auto")} for k, v in VARIANTS.items()},
           "results": {name: {kind: score([r for r in rs if r["kind"] == kind]) for kind in ("lure", "sticky")}
                       for name, rs in rows.items()},
           "results_by_set": {name: {f"{s['set']} / {kind}": score([r for r in rs if r["set"] == s["set"] and r["kind"] == kind])
                                     for s in used_sets for kind in ("lure", "sticky")} for name, rs in rows.items()},
           "forward_ms": {"whole_frame_p50": round(statistics.median(full_ms), 1),
                          "four_windows_p50": round(statistics.median(win_ms), 1),
                          "note": "cv2.dnn forward() only, summed over the 4 windows; dev laptop (CPU) shared with "
                                  "other jobs, not the Lambda"},
           "run_s": round(time.perf_counter() - t_run, 1)}
    RESULTS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


def handler_runs(runs: dict[str, Path]) -> dict:
    """Whole-handler runs of eval.synth.check on one set with different DETECTOR_TILES (name -> folder holding its
    check.jsonl and check.meta.json): count and box metrics of the clean photos per trap kind, handler and DNN time."""
    from eval.synth.report import pct, summarize

    out = {}
    for name, d in runs.items():
        meta = json.loads((d / "check.meta.json").read_text(encoding="utf-8"))
        with open(d / "check.jsonl", encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        clean = [{**r, "true": r["visible"]} for r in recs if not r.get("defect")]
        out[name] = {"set": meta["set"], "photos": meta["ok"], "prev_photo": meta["prev_photo"], "at": meta["at"]}
        for kind in ("lure", "sticky"):
            rows = [r for r in clean if r["trapKind"] == kind]
            s = summarize(rows)
            out[name][kind] = {k: s.get(k) for k in ("photos", "true_flies", "detected", "mae", "bias")} | {
                "precision_iou30": (s.get("iou30") or {}).get("precision"), "recall_iou30": (s.get("iou30") or {}).get("recall"),
                "f1_iou30": (s.get("iou30") or {}).get("f1"),
                "dnn_ms_p50": pct([r["dnnMs"] for r in recs if r["trapKind"] == kind], 50),
                "total_ms_p50": pct([r["totalMs"] for r in recs if r["trapKind"] == kind], 50)}
    return out


def latency(set_dir: Path, n: int, repeat: int = 3) -> dict:
    """Wall time of Yolo26Detector.detect (preprocessing, forwards, decoding, merge, box rules) on `n` clean sticky
    photos of `set_dir`, DETECTOR_TILES off vs sticky, alternating in one process after a warm-up; best of `repeat`."""
    import cv2

    from sentinel.vision.box_rules import BoxRules
    from sentinel.vision.detect_yolo26 import Yolo26Detector
    from sentinel.vision.detector import resolve_onnx
    from sentinel.vision.tiling import Tiling

    m = json.loads((set_dir / "manifest.json").read_text(encoding="utf-8"))
    photos = [p for p in m["photos"] if p["trapKind"] == "sticky" and not p.get("defect")][:n]
    det = Yolo26Detector(resolve_onnx("yolo26"), rules=BoxRules())
    modes = {"off": Tiling(), "sticky": Tiling(mode="sticky")}
    imgs = [cv2.imread(str(set_dir / p["file"]), cv2.IMREAD_COLOR) for p in photos]
    det.detect(imgs[0], trap_kind="sticky")  # warm-up
    wall: dict[str, list[float]] = {k: [] for k in modes}
    fwd: dict[str, list[float]] = {k: [] for k in modes}
    for img in imgs:
        for name, t in modes.items():
            det.tiling = t
            best, best_fwd = None, None
            for _ in range(repeat):
                t0 = time.perf_counter()
                res = det.detect(img, trap_kind="sticky")
                ms = (time.perf_counter() - t0) * 1000
                if best is None or ms < best:
                    best, best_fwd = ms, res["forward_ms"]
            wall[name].append(best)
            fwd[name].append(best_fwd)
    return {"set": m.get("name"), "photos": len(imgs), "size": f"{photos[0]['width']}x{photos[0]['height']}",
            "repeat": repeat, **{name: {"detect_ms_p50": round(statistics.median(wall[name]), 1),
                                        "forward_ms_p50": round(statistics.median(fwd[name]), 1)} for name in modes},
            "note": "per photo, best of the repeats, dev laptop CPU (no other job running), not the Lambda; the handler "
                    "also detects the previous photo, so a sticky request with prevPhoto pays the difference twice"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", type=Path, nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--limit", type=int, default=None, help="clean photos per set (smoke run)")
    ap.add_argument("--latency", type=int, metavar="N", default=None,
                    help="only add the detect() time, off vs sticky, on N sticky photos of the first set")
    ap.add_argument("--handler", nargs="+", metavar="NAME=DIR", default=None,
                    help="only add whole-handler runs (eval.synth.check folders, e.g. off=data/synth/demo-30d "
                         "sticky=data/synth/demo-30d-tiles) to the existing results")
    a = ap.parse_args(argv)
    if a.handler:
        out = json.loads(RESULTS.read_text(encoding="utf-8"))
        out["handler"] = handler_runs({k: Path(v) for k, v in (h.split("=", 1) for h in a.handler)})
        RESULTS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(out["handler"], indent=1))
        return 0
    if a.latency:
        out = json.loads(RESULTS.read_text(encoding="utf-8"))
        out["latency"] = latency(a.sets[0], a.latency)
        RESULTS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(out["latency"], indent=1))
        return 0
    out = run(a.sets, a.limit)
    for name, by_kind in out["results"].items():
        print(name.ljust(12), "  ".join(f"{k}: R {v.get('recall_iou30')} P {v.get('precision_iou30')} MAE {v.get('mae')} "
                                         f"fw {v.get('forwards_per_photo')}" for k, v in by_kind.items()))
    print(f"-> {RESULTS.relative_to(ROOT)} ({out['run_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
