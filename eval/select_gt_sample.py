"""Pick the count ground-truth sample (T0.7b): 150 exported photos, stratified, deterministic.

    uv run python eval/select_gt_sample.py                       # -> data/labels/gt_sample.csv
    uv run python eval/select_gt_sample.py --n 150 --seed 20261015 --manifest data/manifest.csv

Strata = camera x time of day x production count bucket (`ai_count`, the Roboflow RF-DETR count):
- time of day, camera local time (Vietnam, UTC+7): night 18-06, morning 06-10, midday 10-14,
  afternoon 14-18;
- count buckets 0-5, 6-20, 21-50, >50.

Allocation: equal share per non-empty stratum, capped by the stratum size, leftovers handed out
again equally (water filling); photos inside a stratum drawn with `random.Random(seed)`. The output
is ordered round-robin over strata, so labeling only the first K rows still covers every stratum.
Rows whose image file is missing are skipped. Same manifest + seed -> same sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TZ = timezone(timedelta(hours=7))  # Vietnam; all three cameras are there
TOD = [("night", 18, 6), ("morning", 6, 10), ("midday", 10, 14), ("afternoon", 14, 18)]
BUCKETS = [("0-5", 0, 5), ("6-20", 6, 20), ("21-50", 21, 50), (">50", 51, 10**9)]
FIELDS = ["order", "snapshot_id", "device_serial", "captured_at", "local_time", "time_of_day",
          "count_bucket", "ai_count", "quality_issues", "image_path", "stratum"]


def time_of_day(captured_at: str) -> tuple[str, str]:
    t = datetime.fromisoformat(captured_at).astimezone(LOCAL_TZ)
    for name, start, end in TOD:
        if (start <= t.hour < end) if start < end else (t.hour >= start or t.hour < end):
            return name, t.strftime("%Y-%m-%d %H:%M")
    raise AssertionError(t)


def count_bucket(n: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    raise ValueError(n)


def allocate(sizes: dict, n: int) -> dict:
    """Water filling: equal share per stratum, capped by its size, until n is used up."""
    alloc = {k: 0 for k in sizes}
    left = min(n, sum(sizes.values()))
    while left:
        open_ = sorted(k for k in sizes if alloc[k] < sizes[k])
        share = max(1, left // len(open_))
        for k in open_:
            take = min(share, sizes[k] - alloc[k], left)
            alloc[k] += take
            left -= take
            if not left:
                break
    return alloc


def select(manifest: Path, n: int, seed: int, data_root: Path) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    strata: dict[tuple, list[dict]] = defaultdict(list)
    missing = 0
    for r in rows:
        if not r.get("ai_count") or not (data_root / r["image_path"]).exists():
            missing += 1
            continue
        tod, local = time_of_day(r["captured_at"])
        r = dict(r, time_of_day=tod, local_time=local, count_bucket=count_bucket(int(r["ai_count"])))
        strata[(r["device_serial"], tod, r["count_bucket"])].append(r)
    for rows_k in strata.values():
        rows_k.sort(key=lambda r: r["snapshot_id"])  # input order must not matter
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)
    rng = random.Random(seed)
    picked = {k: rng.sample(strata[k], alloc[k]) for k in sorted(strata)}
    out, i = [], 0
    while any(picked.values()):
        for k in sorted(picked):
            if picked[k]:
                r = picked[k].pop(0)
                i += 1
                out.append({**{f: r.get(f, "") for f in FIELDS}, "order": i, "stratum": "|".join(k)})
    report = {
        "manifest": str(manifest), "seed": seed, "n_requested": n, "n_selected": len(out),
        "rows_in_manifest": len(rows), "rows_skipped_missing_image_or_count": missing,
        "strata_non_empty": len(strata), "strata_possible": 3 * len(TOD) * len(BUCKETS),
        "allocation": {"|".join(k): {"available": len(strata[k]), "selected": alloc[k]} for k in sorted(strata)},
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--data-root", type=Path, default=None, help="image_path is relative to this (default: manifest folder)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "labels" / "gt_sample.csv")
    args = ap.parse_args(argv)
    data_root = args.data_root or args.manifest.parent
    out, report = select(args.manifest, args.n, args.seed, data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "allocation"}, indent=2))
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
