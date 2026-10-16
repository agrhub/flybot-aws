"""Evaluate ops 3+4 on the FlyBot export without human ground truth (self-consistency).

    uv run python -m sentinel.vision.eval_diff [--manifest data/manifest.csv] \
        [--out docs/evidence/vision/diff_eval.json] [--aliked-sample 12] [--limit N]

For every consecutive pair of a camera (manifest previous_snapshot_id): rectify (op 1, whole frame
when no card), register (op 3), take boxes, diff (op 4). Per camera and farm day (UTC+7) the sum of
new_count is compared with FlyBot's "daily rise" (max(0, peak(day) - peak(day - 1)) from
data/export/daily_stats.json, unknown on card-change days; ezviz-manager treatmentService.dailyCatches).
Checks whether the card_replaced event of data/export/maintenance_events.json is flagged.

Box source: the export has counts but no boxes (data/README.md), so boxes come from the interim
`blobs.detect_dark_blobs` on the rectified image. The full run uses SIFT+FLANN (ALIKED+LightGlue
costs seconds per image on CPU, see timings.json); `--aliked-sample` pairs are registered with both.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.blobs import detect_dark_blobs
from sentinel.vision.card import find_card
from sentinel.vision.diff import _canonical, diff_detections
from sentinel.vision.register import extract, models_available, register_features, write_evidence

FARM_TZ = timedelta(hours=7)
EVIDENCE_DIR = Path("docs/evidence/vision/diff")
SAMPLE_RES = "640x640"  # uploaded test pictures, not camera frames (see result note)


def farm_day(iso: str) -> str:
    return (datetime.fromisoformat(iso) + FARM_TZ).date().isoformat()


def daily_rise(stats: list[dict], events: list[dict]) -> dict[tuple[str, str], int | None]:
    peaks = {(s["deviceSerial"], s["day"]): s["peak"] for s in stats}
    card_days = {(e["deviceSerial"], farm_day(e["at"])) for e in events if e["type"] == "card_replaced"}
    out = {}
    for (cam, day), peak in peaks.items():
        prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        if (cam, day) in card_days or (cam, prev) not in peaks:
            out[(cam, day)] = None
        else:
            out[(cam, day)] = max(0, peak - peaks[(cam, prev)])
    return out


class Frame:
    """One photo prepared once: canonical image, SIFT features, interim boxes."""

    def __init__(self, row: dict, data_dir: Path, long_side: int):
        img = cv2.imread(str(data_dir / row["image_path"]))
        if img is None:
            raise FileNotFoundError(row["image_path"])
        self.row = row
        self.res = f"{img.shape[1]}x{img.shape[0]}"
        self.card = find_card(img)
        self.canon, _, self.source = _canonical(img, self.card, long_side)
        self.feat = extract(self.canon, "sift_flann", max_keypoints=2000)
        self.boxes = detect_dark_blobs(self.canon)["boxes"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="docs/evidence/vision/diff_eval.json")
    ap.add_argument("--long-side", type=int, default=640)
    ap.add_argument("--aliked-sample", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="pairs per camera (0 = all)")
    ap.add_argument("--evidence-per-camera", type=int, default=2)
    a = ap.parse_args(argv)

    mpath = Path(a.manifest)
    data_dir = mpath.parent
    export = data_dir / "export"
    with open(mpath, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {r["snapshot_id"]: r for r in rows}
    stats = json.loads((export / "daily_stats.json").read_text(encoding="utf-8"))
    events = json.loads((export / "maintenance_events.json").read_text(encoding="utf-8"))
    rise = daily_rise(stats, events)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    by_cam = defaultdict(list)
    for r in rows:
        by_cam[r["device_serial"]].append(r)

    pairs_out = []
    t_start = time.perf_counter()
    for cam, cam_rows in sorted(by_cam.items()):
        cam_rows.sort(key=lambda r: r["captured_at"])
        cache: dict[str, Frame] = {}
        done = 0
        ev_left = a.evidence_per_camera
        for r in cam_rows:
            pid = r["previous_snapshot_id"]
            if not pid or pid not in by_id:
                continue
            if a.limit and done >= a.limit:
                break
            for sid in (pid, r["snapshot_id"]):
                if sid not in cache:
                    cache[sid] = Frame(by_id[sid], data_dir, a.long_side)
            fa, fb = cache[pid], cache[r["snapshot_id"]]
            reg = register_features(fa.feat, fb.feat)
            d = diff_detections(fa.boxes, fb.boxes, reg["H"], registration_ok=reg["ok"], prev_img=fa.canon, cur_img=fb.canon)
            ev = {}
            want_ev = d["card_change_suspected"] and ev_left > 0
            if want_ev or (ev_left > 0 and done % 200 == 100):
                ev_left -= 1
                name = f"{cam}_{r['snapshot_id']}"
                pix = "full_frame" in (fa.source, fb.source)
                d2 = diff_detections(fa.boxes, fb.boxes, reg["H"], registration_ok=reg["ok"], prev_img=fa.canon,
                                     cur_img=fb.canon, out_dir=EVIDENCE_DIR, name=name, pixelate_evidence=pix)
                ev = {**write_evidence(fa.canon, fb.canon, reg["H"], fa.feat, fb.feat, reg["inlier_pairs"], EVIDENCE_DIR, name, pix),
                      **d2["evidence"]}
            pairs_out.append({
                "camera": cam, "prev": pid, "cur": r["snapshot_id"],
                "prev_at": fa.row["captured_at"], "cur_at": r["captured_at"], "day": farm_day(r["captured_at"]),
                "res": [fa.res, fb.res], "source": [fa.source, fb.source],
                "reg_ok": reg["ok"], "inliers": reg["inliers"], "matches": reg["matches"],
                "new": d["new_count"], "persisted": d["persisted_count"], "disappeared": d["disappeared_count"],
                "prev_boxes": d["prev_count"], "cur_boxes": d["cur_count"], "change_ratio": d["change_ratio"],
                "card_change": d["card_change_suspected"], "reasons": d["card_change_reasons"],
                "ai_prev": int(fa.row["ai_count"] or 0), "ai_cur": int(r["ai_count"] or 0), "evidence": ev,
            })
            done += 1
            # keep the cache small: only frames that can still be someone's previous photo
            if len(cache) > 6:
                for k in list(cache)[:-4]:
                    del cache[k]
            if done % 100 == 0:
                print(f"  {cam}: {done} pairs, {time.perf_counter() - t_start:.0f} s", flush=True)

    result = {"all_pairs": summarize(pairs_out, rise, events)}
    native = [p for p in pairs_out if SAMPLE_RES not in p["res"]]
    result["native_pairs"] = summarize(native, rise, events)
    result["note"] = ("640x640 photos are not EZVIZ frames: they are a rotation of a few uploaded fly-disc pictures "
                      "(Roboflow-style 640x640) mixed into BE4089629/BE4111159/AC6527557 until 2026-10-06; "
                      "native_pairs leaves out every pair that involves one. Production daily_rise is computed "
                      "from all photos, so it includes those pictures.")
    result["run"] = {"pairs": len(pairs_out), "seconds": round(time.perf_counter() - t_start, 1),
                     "long_side": a.long_side, "registration": "sift_flann", "box_source": "classic_blobs (interim)"}
    if a.aliked_sample:
        result["aliked_sample"] = aliked_sample(pairs_out, by_id, data_dir, a.long_side, a.aliked_sample)
    Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    Path(a.out).with_name("diff_pairs.json").write_text(json.dumps(pairs_out), encoding="utf-8")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per_day"} if isinstance(v, dict) else v)
                      for k, v in result.items()}, indent=1)[:6000])
    return 0


def summarize(pairs: list[dict], rise: dict, events: list[dict]) -> dict:
    per_day = defaultdict(lambda: {"new": 0, "pairs": 0, "ai_rise_pairs": 0})
    for p in pairs:
        k = (p["camera"], p["day"])
        per_day[k]["new"] += p["new"]
        per_day[k]["pairs"] += 1
        per_day[k]["ai_rise_pairs"] += max(0, p["ai_cur"] - p["ai_prev"])
    rows = []
    for (cam, day), v in sorted(per_day.items()):
        rows.append({"camera": cam, "day": day, "new_sum": v["new"], "pairs": v["pairs"],
                     "daily_rise": rise.get((cam, day)), "ai_pairwise_rise_sum": v["ai_rise_pairs"]})
    per_cam = {}
    for cam in sorted({r["camera"] for r in rows}):
        known = [r for r in rows if r["camera"] == cam and r["daily_rise"] is not None]
        err = [abs(r["new_sum"] - r["daily_rise"]) for r in known]
        pos = [r for r in known if r["daily_rise"] > 0]
        cam_pairs = [p for p in pairs if p["camera"] == cam]
        same = [p for p in cam_pairs if p["res"][0] == p["res"][1]]
        inl = [p["inliers"] for p in same]
        per_cam[cam] = {
            "days_compared": len(known),
            "mae": round(float(np.mean(err)), 2) if err else None,
            "mape_days_with_rise": round(float(np.mean([abs(r["new_sum"] - r["daily_rise"]) / r["daily_rise"] for r in pos])) * 100, 1) if pos else None,
            "days_with_rise": len(pos),
            "days_within_15pct": sum(1 for r in pos if abs(r["new_sum"] - r["daily_rise"]) <= 0.15 * r["daily_rise"]),
            "sum_new": sum(r["new_sum"] for r in known),
            "sum_daily_rise": sum(r["daily_rise"] for r in known),
            "pairs": len(cam_pairs),
            "pairs_same_resolution": len(same),
            "reg_ok_same_resolution": sum(p["reg_ok"] for p in same),
            "inliers_same_resolution_p50": float(np.median(inl)) if inl else None,
            "inliers_same_resolution_p10": float(np.percentile(inl, 10)) if inl else None,
            "pairs_ge_80_inliers": sum(1 for v in inl if v >= 80),
            "card_change_flags": sum(p["card_change"] for p in cam_pairs),
            "card_change_flags_resolution_switch": sum(p["card_change"] for p in cam_pairs if p["res"][0] != p["res"][1]),
        }
    ev_check = []
    for e in events:
        if e["type"] != "card_replaced":
            continue
        cand = [p for p in pairs if p["camera"] == e["deviceSerial"] and p["prev_at"] < e["at"] <= p["cur_at"]]
        ev_check.append({"event": e, "pair": cand[0] if cand else None,
                         "flagged": bool(cand and cand[0]["card_change"])})
    event_pairs = {(c["pair"]["prev"], c["pair"]["cur"]) for c in ev_check if c["pair"]}
    false_flags = [p for p in pairs if p["card_change"] and (p["prev"], p["cur"]) not in event_pairs]
    n_events = len(ev_check)
    return {
        "per_camera": per_cam,
        "card_change": {
            "events": n_events,
            "recall": round(sum(c["flagged"] for c in ev_check) / n_events, 3) if n_events else None,
            "event_checks": ev_check,
            "false_flags": len(false_flags),
            "false_flags_resolution_switch": sum(1 for p in false_flags if p["res"][0] != p["res"][1]),
            "false_flags_by_reason": {r: sum(1 for p in false_flags if r in p["reasons"])
                                      for r in ("disappeared>>persisted", "registration_failed+appearance_change")},
            "pairs": len(pairs),
        },
        "per_day": rows,
    }


def aliked_sample(pairs: list[dict], by_id: dict, data_dir: Path, long_side: int, n: int) -> dict:
    """Register `n` same-resolution pairs (spread over cameras) with ALIKED+LightGlue and SIFT."""
    av = models_available()
    if not (av["aliked"] and av["lightglue"]):
        return {"status": "models missing", **av}
    same = [p for p in pairs if p["res"][0] == p["res"][1] and SAMPLE_RES not in p["res"]]
    cams = sorted({p["camera"] for p in same})
    pick = []
    for cam in cams:
        cp = [p for p in same if p["camera"] == cam]
        k = max(1, n // len(cams))
        pick += [cp[int(i * (len(cp) - 1) / max(1, k - 1))] for i in range(k)] if len(cp) >= k else cp
    out = []
    for i, p in enumerate(pick):
        imgs = []
        for sid in (p["prev"], p["cur"]):
            img = cv2.imread(str(data_dir / by_id[sid]["image_path"]))
            imgs.append(_canonical(img, find_card(img), long_side)[0])
        fa, fb = extract(imgs[0], "aliked_lightglue"), extract(imgs[1], "aliked_lightglue")
        r = register_features(fa, fb)
        sa, sb = extract(imgs[0], "sift_flann", 2000), extract(imgs[1], "sift_flann", 2000)
        s = register_features(sa, sb)
        ev = {}
        if i < 3:
            pix = True  # whole frames
            ev = write_evidence(imgs[0], imgs[1], r["H"], fa, fb, r["inlier_pairs"], EVIDENCE_DIR,
                                f"aliked_{p['camera']}_{p['cur']}", pix)
        out.append({"camera": p["camera"], "prev": p["prev"], "cur": p["cur"], "res": p["res"][0],
                    "aliked": {"ok": r["ok"], "inliers": r["inliers"], "matches": r["matches"],
                               "extract_ms": [round(fa["time_ms"], 1), round(fb["time_ms"], 1)], "match_ms": r["match_ms"]},
                    "sift": {"ok": s["ok"], "inliers": s["inliers"], "matches": s["matches"],
                             "extract_ms": [round(sa["time_ms"], 1), round(sb["time_ms"], 1)], "match_ms": s["match_ms"]},
                    "evidence": ev})
        print(f"  aliked sample {i + 1}/{len(pick)}: {out[-1]['aliked']['inliers']} vs sift {out[-1]['sift']['inliers']}", flush=True)
    ai = [o["aliked"]["inliers"] for o in out]
    si = [o["sift"]["inliers"] for o in out]
    return {
        "pairs": len(out),
        "aliked_inliers_p50": float(np.median(ai)), "sift_inliers_p50": float(np.median(si)),
        "aliked_ok": sum(o["aliked"]["ok"] for o in out), "sift_ok": sum(o["sift"]["ok"] for o in out),
        "aliked_ge_80": sum(1 for v in ai if v >= 80), "sift_ge_80": sum(1 for v in si if v >= 80),
        "aliked_ms_per_pair_p50": float(np.median([sum(o["aliked"]["extract_ms"]) + o["aliked"]["match_ms"] for o in out])),
        "sift_ms_per_pair_p50": float(np.median([sum(o["sift"]["extract_ms"]) + o["sift"]["match_ms"] for o in out])),
        "rows": out,
    }


if __name__ == "__main__":
    raise SystemExit(main())
