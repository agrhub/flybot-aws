"""Ops 3+4 new-catch check with YOLOv7-tiny boxes instead of the interim blob boxes .

    uv run python -m eval.diff_with_yolo_boxes [--out eval/results/diff_with_yolo_boxes.json]

`sentinel.vision.eval_diff` has no box-source option (its `Frame` always calls `detect_dark_blobs`), so this
script reuses its pieces (`daily_rise`, `summarize`, `farm_day`) and `sentinel.vision.diff` / `register`
unchanged, and only swaps the boxes: YOLOv7-tiny detections (data/cache/yolov7/<snapshot_id>.json, filled by
eval/replay.py or computed here on a cache miss), mapped from source pixels to the canonical frame with the
same matrix `_canonical` returns, then `diff_detections` on consecutive pairs of each trap camera.

Pairs: consecutive photos of a camera in time order after dropping excluded cameras and the repeated 640x640
test pictures (eval/replay.py rules), so a pair never straddles a test picture. The production "daily rise"
comes from data/export/daily_stats.json and is computed by FlyBot from ALL photos, test pictures included
(it is therefore inflated until 2026-10-06); the traps look empty since 2026-10-07, so near-zero new catches
is the expected result, not evidence that op 4 works. Matching itself is covered by synthetic tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import replay
from sentinel.vision import eval_diff
from sentinel.vision.card import find_card
from sentinel.vision.diff import _canonical, diff_detections, transform_boxes
from sentinel.vision.register import extract, register_features

TRAPS_EMPTY_SINCE = "2026-10-07"


class YoloFrame:
    """One photo prepared once: canonical image, SIFT features, YOLOv7 boxes in canonical coordinates."""

    def __init__(self, row: dict, data_dir: Path, source: replay.Yolov7Source, long_side: int):
        path = data_dir / row["image_path"]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(row["image_path"])
        self.row = row
        self.res = f"{img.shape[1]}x{img.shape[0]}"
        card = find_card(img)
        self.canon, M, self.source = _canonical(img, card, long_side)
        self.feat = extract(self.canon, "sift_flann", max_keypoints=2000)
        det = source.detect(row, path)
        xyxy = [[x, y, x + w, y + h] for x, y, w, h in (b["xywh"] for b in det["boxes"])]
        self.boxes = transform_boxes(xyxy, M)
        self.n_boxes = len(xyxy)


def build_pairs(rows: list[dict], data_dir: Path, source: replay.Yolov7Source, long_side: int, limit: int) -> list[dict]:
    by_cam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cam[r["device_serial"]].append(r)
    pairs = []
    t0 = time.perf_counter()
    for cam, cam_rows in sorted(by_cam.items()):
        cam_rows.sort(key=lambda r: (r["captured_at"], r["snapshot_id"]))
        prev: YoloFrame | None = None
        done = 0
        for r in cam_rows:
            cur = YoloFrame(r, data_dir, source, long_side)
            if prev is not None and not (limit and done >= limit):
                reg = register_features(prev.feat, cur.feat)
                d = diff_detections(prev.boxes, cur.boxes, reg["H"], registration_ok=reg["ok"],
                                    prev_img=prev.canon, cur_img=cur.canon)
                pairs.append({
                    "camera": cam, "prev": prev.row["snapshot_id"], "cur": r["snapshot_id"],
                    "prev_at": prev.row["captured_at"], "cur_at": r["captured_at"],
                    "day": eval_diff.farm_day(r["captured_at"]), "res": [prev.res, cur.res],
                    "source": [prev.source, cur.source], "reg_ok": reg["ok"], "inliers": reg["inliers"],
                    "matches": reg["matches"], "new": d["new_count"], "persisted": d["persisted_count"],
                    "disappeared": d["disappeared_count"], "prev_boxes": d["prev_count"], "cur_boxes": d["cur_count"],
                    "card_change": d["card_change_suspected"], "reasons": d["card_change_reasons"],
                    "ai_prev": int(prev.row["ai_count"] or 0), "ai_cur": int(r["ai_count"] or 0),
                })
                done += 1
                if done % 200 == 0:
                    print(f"  {cam}: {done} pairs, {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
            prev = cur
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--excluded", type=Path, default=ROOT / "data" / "excluded_cameras.csv")
    ap.add_argument("--onnx", type=Path, default=ROOT / "models" / "yolov7-tiny" / "fruitfly_deploy.onnx")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "yolov7")
    ap.add_argument("--out", type=Path, default=None,
                    help="default eval/results/diff_with_<yolo|production>_boxes.json")
    ap.add_argument("--box-source", choices=["yolov7", "production"], default="yolov7")
    ap.add_argument("--long-side", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="pairs per camera (0 = all)")
    a = ap.parse_args(argv)
    if a.out is None:
        name = "diff_with_yolo_boxes.json" if a.box_source == "yolov7" else "diff_with_production_boxes.json"
        a.out = ROOT / "eval" / "results" / name

    data_dir = a.manifest.parent
    all_rows = replay.load_manifest(a.manifest)
    info = replay.image_info(all_rows, data_dir, data_dir / "cache" / "imginfo.json")
    tests = replay.find_test_pictures(all_rows, info)
    missing = {sid for sid, i in info.items() if i["bytes"] is None}
    rows, report = replay.select_rows(all_rows, excluded=replay.load_excluded(a.excluded), test_ids=tests,
                                      date_from=None, date_to=None, days=None, cameras=None,
                                      missing_image_ids=missing)
    source = (replay.ProductionBoxSource(data_dir / "export" / "boxes") if a.box_source == "production"
              else replay.Yolov7Source(a.onnx, a.cache_dir))
    t0 = time.perf_counter()
    pairs = build_pairs(rows, data_dir, source, a.long_side, a.limit)
    export = data_dir / "export"
    stats = json.loads((export / "daily_stats.json").read_text(encoding="utf-8"))
    events = json.loads((export / "maintenance_events.json").read_text(encoding="utf-8"))
    rise = eval_diff.daily_rise(stats, events)

    def pick(since: str | None = None, before: str | None = None) -> list[dict]:
        return [p for p in pairs if (since is None or p["day"] >= since)
                and (before is None or p["day"] < before)]

    def summ(ps: list[dict]) -> dict:
        s = eval_diff.summarize(ps, rise, events)
        s["new_catches_total"] = int(sum(p["new"] for p in ps))
        s["pairs_with_yolo_boxes_gt0"] = int(sum(1 for p in ps if p["cur_boxes"] > 0 or p["prev_boxes"] > 0))
        return s

    result = {
        "what": ("op 4 new catches (YOLOv7-tiny boxes, SIFT registration) vs FlyBot daily rise; traps are empty since "
                 f"{TRAPS_EMPTY_SINCE} so near-zero is expected; no human ground truth"),
        "box_source": ("yolov7-tiny fruitfly_deploy.onnx (letterbox, conf 0.25, iou 0.45), data/cache/yolov7"
                       if a.box_source == "yolov7" else "FlyBot production boxes, data/export/boxes (sparse)"),
        "filters": report,
        "all_native_pairs": summ(pairs),
        f"before_{TRAPS_EMPTY_SINCE}": summ(pick(before=TRAPS_EMPTY_SINCE)),
        f"since_{TRAPS_EMPTY_SINCE}": summ(pick(since=TRAPS_EMPTY_SINCE)),
        "run": {"pairs": len(pairs), "seconds": round(time.perf_counter() - t0, 1), "long_side": a.long_side,
                "registration": "sift_flann", "detector_cache_hits": source.hits, "detector_cache_misses": source.misses},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))

    a.out.write_text(json.dumps(result, indent=1, default=default), encoding="utf-8")
    brief = {k: ({cam: {kk: vv for kk, vv in c.items() if kk in ("days_compared", "sum_new", "sum_daily_rise", "mae",
                                                                  "pairs", "card_change_flags")}
                  for cam, c in v["per_camera"].items()} | {"new_catches_total": v["new_catches_total"]})
             for k, v in result.items() if isinstance(v, dict) and "per_camera" in v}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
