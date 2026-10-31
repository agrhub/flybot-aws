"""Counting fixes for the YOLO26n detector step, A/B on synthetic and real photos from one cached run.

    uv run --project lambda/vision python -m eval.detector_ab cache [--only synth|real|roboflow] [--every N]
    uv run --project lambda/vision python -m eval.detector_ab score [--sweep production tta_lure_hflip]   # -> eval/results/detector_ab.json
    uv run --project lambda/vision python -m eval.detector_ab handler NAME=DIR ...      # whole-handler runs, same file

`cache` runs, once per photo, everything the variants need, with the Lambda's own code: op 1 (sentinel.vision.card:
the lure tray floor or the sticky card), the whole-frame pass, the 2x2 frame windows and, on lure photos with a
floor, 2x2 windows over the floor's bounding box (boxes after NMS at the decoder floor 0.25, before the box rules).
`score` then rebuilds each variant's boxes from that cache with the same functions the detector calls
(`tiling.merge`, `regions.apply_regions`, `box_rules.apply_rules`), so the variants differ only in their settings.
`handler` adds whole-handler runs of eval.synth.check (`--out DIR`, one per setting, on the same set) to the results:
the count with the trap ROI, quality ops and agent, as FlyBot gets it.

Photo sets (the detector step only: no quality ops, no agent):
  synth     clean photos of data/synth/demo-30d and data/synth/pool (SIMULATED, eval/synth): boxes vs the pasted
            flies, greedy one-to-one at IoU 0.3, count vs the visible count. The generator places flies inside the
            floor polygon op 1 finds on the background, so an ROI filter cannot lose a synthetic fly there: the real
            sets below are the check on that
  real      real lure-trap photos of the 2026-10 export (BE4089629, BE4111159; data/export, every
            4th photo): empty traps. v4 trained on 36-42 empty BE4111159 photos of 8-14 Oct as hard negatives, the
            days of this set's BE4111159 photos, so for v4 and later only the BE4089629 count is a clean check. No exported photo has more than 10 production (RF-DETR) flies, and the photos
            where production counts some show an empty floor with specks and stains when looked at, so every box
            here is counted as a false fly. These photos cannot show a real fly lost to a rule
  roboflow  the camos/fruit-fly v3 test split (76 white-disc pictures, human boxes; the split behind the box-rule
            defaults): precision / recall at IoU 0.5, run as a lure and as a sticky photo
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lambda" / "vision"))
CACHE = ROOT / "data" / "cache" / "detector_ab"
RESULTS = ROOT / "eval" / "results" / "detector_ab.json"
SYNTH_SETS = [ROOT / "data" / "synth" / "demo-30d", ROOT / "data" / "synth" / "pool"]
EXPORT = ROOT / "data"
ROBOFLOW = ROOT.parent / "camos-ai-camera" / "models" / "fruit-fly-v3" / "test"
GRID, OVERLAP, EDGE_PX = 2, 0.12, 2.0
CACHE_CONF = 0.10
TILE_VIEWS = ("hflip",)  # test-time views cached for the frame windows (each costs 4 forwards a photo)


def xyxy(d: dict) -> list[float]:
    x, y, w, h = d["xywh"]
    return [x, y, x + w, y + h]


# ---------------------------------------------------------------- cache


def passes(det, img, kind: str) -> dict:
    """Op 1 surface, whole-frame pass (+ each test-time view), frame windows (+ their hflip view), floor windows
    (lure with a floor). View boxes are mapped back to the pixels of the image the view was made from."""
    from sentinel.vision import regions, tiling, tta
    from sentinel.vision.card import find_card, roi_polygon_px

    H, W = img.shape[:2]
    card = find_card(img, trap_kind=kind)
    poly = roi_polygon_px(card)
    rec = {"W": W, "H": H, "roi": poly, "roi_source": card.get("roi_source"), "ms": {}}
    t0 = time.perf_counter()
    rec["full"] = det._pass(img)[0]
    rec["ms"]["full"] = (time.perf_counter() - t0) * 1000
    rec["views"] = {name: tta.unmap(det.view_pass(img, name)[0], name, W, H) for name in tta.VIEWS}

    def tile(rect, views=()):
        wins = tiling.windows(W, H, GRID, OVERLAP, rect)
        per, margins, per_view = [], [], {name: [] for name in views}
        for x0, y0, x1, y1 in wins:
            win = img[y0:y1, x0:x1]
            d, _, _, s, _ = det._pass(win)
            per.append(d)
            margins.append(EDGE_PX / s)
            for name in views:
                per_view[name].append(tta.unmap(det.view_pass(win, name)[0], name, x1 - x0, y1 - y0))
        return {"wins": wins, "per": per, "margins": margins, "rect": rect, "views": per_view}

    t0 = time.perf_counter()
    rec["frame_tiles"] = tile(None, TILE_VIEWS)
    rec["ms"]["frame_tiles"] = (time.perf_counter() - t0) * 1000
    rect = regions.roi_rect(poly, W, H) if kind == "lure" else None
    if rect is not None:
        t0 = time.perf_counter()
        rec["roi_tiles"] = tile(rect)
        rec["ms"]["roi_tiles"] = (time.perf_counter() - t0) * 1000
    return rec


def synth_items(every: int):
    from eval.synth.check import gt_boxes

    for d in SYNTH_SETS:
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        photos = [p for p in m["photos"] if not p.get("defect")][::every]
        for p in photos:
            yield {"set": m["name"], "id": p["photoId"], "trap": p["trap"], "kind": p["trapKind"], "path": d / p["file"],
                   "truth": gt_boxes(p)}


def real_items(every: int):
    rows = {r["snapshot_id"]: r for r in csv.DictReader(open(EXPORT / "manifest.csv", encoding="utf-8"))}
    for cam in ("BE4089629", "BE4111159"):
        for path in sorted((EXPORT / "export" / "images" / cam).glob("*.jpg"))[::every]:
            r = rows.get(path.stem)
            if r is None:
                continue
            yield {"set": "export", "id": path.stem, "trap": cam, "kind": "lure", "path": path,
                   "ref": int(r["ai_count"] or 0), "captured": r["captured_at"]}


def roboflow_items(every: int):
    from eval.run_eval import roboflow_split

    for path, gts in roboflow_split(ROBOFLOW)[::every]:
        for kind in ("lure", "sticky"):
            yield {"set": f"roboflow-test/{kind}", "id": path.stem, "trap": "roboflow", "kind": kind, "path": path,
                   "truth": gts}


def cache(only: list[str], every: int) -> None:
    import cv2

    from sentinel.vision.detect_yolo26 import Yolo26Detector
    from sentinel.vision.detector import resolve_onnx

    # NMS 0.45 as the Lambda; decoder floor CACHE_CONF (Lambda 0.25) so the floor sweep can go below it. NMS keeps the
    # higher score of a pair, so boxes at or above 0.25 are the same as with the Lambda's decoder floor.
    det = Yolo26Detector(resolve_onnx("yolo26"), conf=CACHE_CONF)
    CACHE.mkdir(parents=True, exist_ok=True)
    sources = {"synth": synth_items, "real": real_items, "roboflow": roboflow_items}
    for name in only:
        out, t_run = [], time.perf_counter()
        items = list(sources[name](every))
        for i, it in enumerate(items, 1):
            img = cv2.imread(str(it["path"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out.append({**{k: v for k, v in it.items() if k != "path"}, "file": str(it["path"]),
                        "classes": det.meta.get("classes"), **passes(det, img, it["kind"])})
            if i % 50 == 0 or i == len(items):
                print(f"{name} {i}/{len(items)} ({time.perf_counter() - t_run:.0f} s)", flush=True)
        meta = {"model": f"{det.name} on opencv-{det.cv_version}-dnn-{det.engine_used}", "onnx": det.onnx_path.name,
                "sha256": det.meta.get("sha256"), "classes": det.meta.get("classes"), "conf": CACHE_CONF, "every": every, "at": datetime.now(UTC).isoformat(timespec="seconds"), "photos": len(out)}
        with open(CACHE / f"{name}.pkl", "wb") as fh:
            pickle.dump({"meta": meta, "records": out}, fh)


# ---------------------------------------------------------------- variants


def with_views(dets: list[dict], view_dets: dict, views: list[str], iou: float) -> list[dict]:
    """A pass's boxes joined by its test-time views' boxes, class-agnostic NMS (as Yolo26Detector._pass)."""
    from sentinel.vision.box_rules import nms

    return nms(dets + [d for name in views for d in view_dets[name]], iou) if views else dets


def boxes_for(rec: dict, v: dict, iou: float = 0.45) -> tuple[list[dict], dict]:
    """One variant's counted boxes for a cached photo. v = {tiles: {kind: off|frame|roi|roi+full}, tta: {kind: [view]},
    mask, roi, rim, floor: {kind: min_score}}."""
    from sentinel.vision import regions, tiling
    from sentinel.vision.box_rules import BoxRules, apply_rules
    from sentinel.vision.detect_yolo26 import split_by_class

    kind, W, H = rec["kind"], rec["W"], rec["H"]
    mode = v.get("tiles", {}).get(kind, "off")
    t = None
    if mode == "frame":
        t = rec["frame_tiles"]
    elif mode.startswith("roi"):
        t = rec.get("roi_tiles") or (rec["frame_tiles"] if v.get("no_floor") == "frame" else None)
    views = list(v.get("tta", {}).get(kind, []))
    if t is None:
        dets, info = with_views(rec["full"], rec.get("views", {}), views, iou), {"forwards": 1 + len(views)}
    else:
        full = with_views(rec["full"], rec.get("views", {}), views, iou) if mode == "roi+full" else None
        per = [with_views(d, {k: t["views"][k][i] for k in views}, views, iou) for i, d in enumerate(t["per"])]
        dets, _ = tiling.merge(per, t["wins"], W, H, iou, t["margins"], full, t["rect"])
        info = {"forwards": (4 + (full is not None)) * (1 + len(views))}
    reg = regions.Regions(masks=regions.parse_masks(v.get("mask")), roi=v.get("roi", "off"), rim=v.get("rim", 0.0))
    dets, removed = regions.apply_regions(dets, W, H, reg, rec["roi"], kind)
    rules = replace(BoxRules(), min_score=v.get("floor", {}).get(kind, 0.35))
    kept, _ = apply_rules(dets, W, H, rules)
    kept, others = split_by_class(kept, rec.get("classes"))  # v4: the fruit_fly class only, as the Lambda counts
    return kept, {**info, **removed, "others": len(others)}


def score_synth(recs: list[dict], v: dict) -> dict:
    from eval.synth.check import match

    rows = []
    for r in recs:
        kept, info = boxes_for(r, v)
        rows.append({"kind": r["kind"], "set": r["set"], "trap": r["trap"], "true": len(r["truth"]), "det": len(kept),
                     "tp": match(r["truth"], [xyxy(d) for d in kept], 0.3), "forwards": info["forwards"]})
    out = {}
    for key in ("lure", "sticky", "demo-lure-a", "demo-lure-b", "demo-lure-c", "demo-sticky-d"):
        rs = [r for r in rows if key in (r["kind"], r["trap"])]
        if not rs:
            continue
        true, det, tp = sum(r["true"] for r in rs), sum(r["det"] for r in rs), sum(r["tp"] for r in rs)
        p, rc = (tp / det if det else None), (tp / true if true else None)
        out[key] = {"photos": len(rs), "true": true, "det": det, "ratio": round(det / true, 4) if true else None,
                    "mae": round(statistics.fmean(abs(r["det"] - r["true"]) for r in rs), 3),
                    "under": sum(r["det"] < r["true"] for r in rs), "over": sum(r["det"] > r["true"] for r in rs),
                    "precision": None if p is None else round(p, 4), "recall": None if rc is None else round(rc, 4),
                    "f1": round(2 * p * rc / (p + rc), 4) if p and rc else None,
                    "forwards": round(statistics.fmean(r["forwards"] for r in rs), 2)}
    return out


def score_real(recs: list[dict], v: dict) -> dict:
    n = [len(boxes_for(r, v)[0]) for r in recs]
    by_cam = {}
    for r, k in zip(recs, n, strict=True):
        c = by_cam.setdefault(r["trap"], {"photos": 0, "false_flies": 0})
        c["photos"], c["false_flies"] = c["photos"] + 1, c["false_flies"] + k
    return {"photos": len(recs), "false_flies": sum(n), "photos_with_a_box": sum(k > 0 for k in n),
            "production_count_sum": sum(r["ref"] for r in recs), "by_camera": by_cam}


def score_roboflow(recs: list[dict], v: dict) -> dict:
    from eval.run_eval import match_image

    out = {}
    for kind in ("lure", "sticky"):
        tp = fp = n_gt = 0
        for r in (r for r in recs if r["kind"] == kind):
            kept, _ = boxes_for(r, v)
            n_gt += len(r["truth"])
            for _, t in match_image([[*xyxy(d), d["score"]] for d in kept], r["truth"], 0.5):
                tp, fp = tp + t, fp + (not t)
        p, rc = round(tp / (tp + fp), 4) if tp + fp else 0.0, round(tp / n_gt, 4) if n_gt else 0.0
        out[kind] = {"precision": p, "recall": rc, "tp": tp, "fp": fp, "fn": n_gt - tp,
                     "meets_box_rule_result": p >= 0.9025 and rc >= 0.8987}  # eval/results/yolo26_box_rules.json
    return out


# production since 2026-10-23 (v4 weights since 2026-10-29): DETECTOR_TILES=sticky, DETECTOR_ROI=all, floor 0.35
BASE = {"tiles": {"sticky": "frame"}, "roi": "all"}
VIEW_SETS = (["hflip"], ["hflip", "vflip"], ["hflip", "vflip", "rot90"])


def variants() -> dict[str, dict]:
    v = {"production": BASE,
         "no_roi": {"tiles": {"sticky": "frame"}},  # production 2026-10-19 .. 10-23
         "mask": {**BASE, "mask": "osd"},
         "roi_lure": {"tiles": {"sticky": "frame"}, "roi": "lure"},
         "tiles_lure_frame": {"tiles": {"sticky": "frame", "lure": "frame"}},
         "tiles_lure_frame+roi": {"tiles": {"sticky": "frame", "lure": "frame"}, "roi": "lure"},
         "tiles_lure_roi": {"tiles": {"sticky": "frame", "lure": "roi"}, "roi": "lure"},
         "tiles_lure_roi+full": {"tiles": {"sticky": "frame", "lure": "roi+full"}, "roi": "lure"}}
    for rim in (0.02, 0.05):
        v[f"tiles_lure_roi_rim{rim}"] = {"tiles": {"sticky": "frame", "lure": "roi"}, "roi": "lure", "rim": rim}
    for s in (0.45, 0.55):
        v[f"tiles_lure_frame+roi_floor{s}"] = {**v["tiles_lure_frame+roi"], "floor": {"lure": s}}
    for views in VIEW_SETS:  # test-time views (sentinel.vision.tta) on the whole frame of lure photos
        v["tta_lure_" + "+".join(views)] = {**BASE, "tta": {"lure": views}}
    v["tta_sticky_hflip"] = {**BASE, "tta": {"sticky": ["hflip"]}}  # on each of the 4 windows: 8 forwards
    v["tta_all_hflip"] = {**BASE, "tta": {"lure": ["hflip"], "sticky": ["hflip"]}}
    v["tta_lure_hflip+vflip_sticky_hflip"] = {**BASE, "tta": {"lure": ["hflip", "vflip"], "sticky": ["hflip"]}}
    # recommended 2026-10-31: DETECTOR_TTA_LURE=hflip,vflip DETECTOR_TTA_STICKY=hflip DETECTOR_MIN_SCORE_LURE=0.30
    v["recommended"] = {**v["tta_lure_hflip+vflip_sticky_hflip"], "floor": {"lure": 0.30}}
    return v


def floor_sweep(base: dict) -> dict[str, dict]:
    out = {}
    for kind in ("lure", "sticky"):
        for s in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
            out[f"{kind}={s:.2f}"] = {**base, "floor": {kind: s}}
    return out


def load(name: str) -> dict | None:
    p = CACHE / f"{name}.pkl"
    if not p.is_file():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def score(sweep_bases: list[str]) -> dict:
    data = {k: load(k) for k in ("synth", "real", "roboflow")}
    vs = variants()
    out = {"kind": "detector step A/B (no quality ops, no agent); synthetic rows are SIMULATED data",
           "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "cache": {k: d["meta"] for k, d in data.items() if d},
           "notes": __doc__.split("Photo sets", 1)[1].strip(),
           "variants": vs, "results": {}, "floor_sweep": {}}
    scorers = {"synth": score_synth, "real": score_real, "roboflow": score_roboflow}

    def run(v):
        return {k: scorers[k](d["records"], v) for k, d in data.items() if d}

    for name, v in vs.items():
        out["results"][name] = run(v)
        print(name, flush=True)
    for base in sweep_bases:  # {base variant: {"<kind>=<floor>": results}}
        out["floor_sweep"][base] = {name: run(v) for name, v in floor_sweep(vs[base]).items()}
        print("sweep", base, flush=True)
    times = {}
    for k, d in data.items():
        if d:
            for part in ("full", "frame_tiles", "roi_tiles"):
                ms = [r["ms"][part] for r in d["records"] if part in r["ms"]]
                if ms:
                    times[f"{k}/{part}"] = round(statistics.median(ms), 1)
    out["ms_p50"] = times
    RESULTS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["cache", "score", "handler"])
    ap.add_argument("runs", nargs="*", metavar="NAME=DIR", help="handler: eval.synth.check output folders")
    ap.add_argument("--only", nargs="+", default=["synth", "real", "roboflow"])
    ap.add_argument("--every", type=int, default=1, help="cache every N-th photo of each set")
    ap.add_argument("--sweep", nargs="*", default=[], help="score: sweep the score floor per trap kind on top of these variants")
    a = ap.parse_args(argv)
    if a.step == "cache":
        cache(a.only, a.every)
        return 0
    if a.step == "handler":
        from eval.synth.tiles import handler_runs

        out = json.loads(RESULTS.read_text(encoding="utf-8"))
        runs = handler_runs({k: Path(v) for k, v in (r.rsplit("=", 1) for r in a.runs)}, common=True)
        out["handler"] = {**out.get("handler", {}), **runs}  # one call per set; photos common to the call's runs
        RESULTS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(out["handler"], indent=1))
        return 0
    out = score(a.sweep)
    for name, r in out["results"].items():
        s = r.get("synth", {})
        line = [name.ljust(32)]
        for k in ("lure", "sticky"):
            if k in s:
                line.append(f"{k} P {s[k]['precision']} R {s[k]['recall']} ratio {s[k]['ratio']} MAE {s[k]['mae']}")
        if "real" in r:
            line.append(f"real FF {r['real']['false_flies']}")
        if "roboflow" in r:
            line.append("rf " + " ".join(f"{k} {v['precision']}/{v['recall']}" for k, v in r["roboflow"].items()))
        print(" | ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
