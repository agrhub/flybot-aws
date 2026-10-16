"""Calibrate the op-2 quality gate against FlyBot production flags (manifest `quality_issues`).

    uv run python -m sentinel.vision.calibrate_quality [--manifest data/manifest.csv] \
        [--out docs/evidence/vision/quality_calibration.json] [--cache scores.json]

Each camera's photos are replayed in time order with a rolling `Baseline`, like production.
Production only knows dark / overexposed / blurry / low_contrast, so agreement is measured on that
vocabulary; our extra issues (glare, dirty, ir, no_card, partial_card) are counted separately.
Production quality checks started with the FlyBot deploy of 2026-10-10/11 (ezviz-manager commit
0cfb70a): before `--checked-from` an empty `quality_issues` means "not checked", so agreement is
reported on photos captured from that time on.

`--trap-kind` (default `lure`) is the op 1 profile used for every photo, as the agent runs these cameras:
the manifest's `trap_kind` column says `sticky` (FlyBot's default) but both trap cameras look into white lure
trays. `manifest` uses that column. Before 2026-10-15 this script always ran the sticky finder, so the
lure-floor rules (and the false `blurry` of the live AWS run) were never exercised here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.card import find_card
from sentinel.vision.quality import Baseline, assess, load_thresholds, measure

PROD_ISSUES = ["dark", "overexposed", "blurry", "low_contrast"]
FARM_TZ = timedelta(hours=7)


def farm_hour(iso: str) -> int:
    return (datetime.fromisoformat(iso) + FARM_TZ).hour


def compute_scores(rows: list[dict], data_dir: Path, t: dict, trap_kind: str = "lure") -> dict[str, dict]:
    out = {}
    for i, r in enumerate(rows):
        img = cv2.imread(str(data_dir / r["image_path"])) if r["image_path"] else None
        if img is None:
            continue
        kind = (r.get("trap_kind") or "sticky") if trap_kind == "manifest" else trap_kind
        card = find_card(img, trap_kind=kind)
        s, _ = measure(img, card, t)
        s["card_profile_found"] = card["found"]
        out[r["snapshot_id"]] = s
        if i % 250 == 0:
            print(f"  scored {i}/{len(rows)}", flush=True)
    return out


def replay(rows: list[dict], scores: dict[str, dict], t: dict) -> dict[str, list[str]]:
    """Issues per snapshot with a per-camera rolling baseline (time order)."""
    by_cam = defaultdict(list)
    for r in rows:
        if r["snapshot_id"] in scores:
            by_cam[r["device_serial"]].append(r)
    issues = {}
    for cam_rows in by_cam.values():
        cam_rows.sort(key=lambda r: r["captured_at"])
        base = Baseline(int(t["baseline_size"]), int(t["baseline_min"]))
        for r in cam_rows:
            s = scores[r["snapshot_id"]]
            iss = assess(s, t, base.get(), farm_hour(r["captured_at"]))
            issues[r["snapshot_id"]] = iss
            base.update(s, iss)
    return issues


def agreement(rows: list[dict], ours: dict[str, list[str]]) -> dict:
    n = exact = any_agree = 0
    per = {k: Counter() for k in PROD_ISSUES}
    extra = Counter()
    for r in rows:
        sid = r["snapshot_id"]
        if sid not in ours:
            continue
        prod = {x for x in r["quality_issues"].split(";") if x}
        mine = set(ours[sid])
        mine_p = mine & set(PROD_ISSUES)
        n += 1
        exact += prod & set(PROD_ISSUES) == mine_p
        any_agree += bool(prod) == bool(mine_p)
        for k in PROD_ISSUES:
            per[k][("tp" if k in mine else "fn") if k in prod else ("fp" if k in mine else "tn")] += 1
        extra.update(mine - set(PROD_ISSUES))
    per_issue = {}
    for k, c in per.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        per_issue[k] = {
            "prod_flags": tp + fn,
            "ours_flags": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": c["tn"],
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    return {
        "photos": n,
        "exact_agreement": round(exact / n, 4) if n else None,
        "any_issue_agreement": round(any_agree / n, 4) if n else None,
        "per_issue": per_issue,
        "our_extra_issues": dict(extra),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="docs/evidence/vision/quality_calibration.json")
    ap.add_argument("--cache", default=None, help="JSON file for per-photo scores (reused if present)")
    ap.add_argument("--checked-from", default="2026-10-11T00:00:00Z")
    ap.add_argument("--trap-kind", default="lure", choices=["lure", "sticky", "manifest"])
    a = ap.parse_args(argv)

    t = load_thresholds()
    mpath = Path(a.manifest)
    with open(mpath, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cache = Path(a.cache) if a.cache else None
    if cache and cache.exists():
        scores = json.loads(cache.read_text(encoding="utf-8"))
    else:
        scores = compute_scores(rows, mpath.parent, t, a.trap_kind)
        if cache:
            cache.write_text(json.dumps(scores), encoding="utf-8")

    ours = replay(rows, scores, t)
    checked = [r for r in rows if r["captured_at"] >= a.checked_from]
    result = {
        "source": "data/manifest.csv quality_issues (FlyBot imageQuality.js), replayed per camera in time order",
        "thresholds": str(Path("sentinel/vision/quality_thresholds.yaml")),
        "checked_from": a.checked_from,
        "trap_kind": a.trap_kind,
        "checked_window": agreement(checked, ours),
        "all_photos": agreement(rows, ours),
        "card_found": sum(1 for s in scores.values() if s.get("card_profile_found")),
        "blurry_by_camera": dict(Counter(r["device_serial"] for r in rows if "blurry" in ours.get(r["snapshot_id"], []))),
        "sweep_ratio_contrast": {},
    }
    # Transparency: how the main relative threshold moves agreement (YAML value is the one used above)
    for rc in (0.4, 0.5, 0.6, 0.7):
        t2 = json.loads(json.dumps(t))
        t2["low_contrast"]["ratio_contrast"] = rc
        ag = agreement(checked, replay(rows, scores, t2))
        result["sweep_ratio_contrast"][str(rc)] = {
            "exact_agreement": ag["exact_agreement"],
            "low_contrast": ag["per_issue"]["low_contrast"],
        }
    bright = np.array([s["brightness"] for s in scores.values()])
    result["score_ranges"] = {"brightness_p5_p50_p95": [round(float(np.percentile(bright, p)), 1) for p in (5, 50, 95)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("trap_kind", "checked_window", "all_photos", "card_found", "blurry_by_camera")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
