"""Replay the bundled sample (data/samples: 24 real lure-trap photos of the two orchard cameras) through the
vision handler exactly as FlyBot would call it (docs/CONTRACTS.md section 1), offline: rule planner, no AWS,
annotated evidence JPEGs on disk.

    make replay-sample
    cd lambda/vision && EVIDENCE_DIR=../../data/replay/sample/evidence uv run python ../../eval/replay_sample.py \
        --manifest ../../data/samples/samples_manifest.csv --out ../../data/replay/sample [--limit N] [--url http://localhost:9001/invoke]

Per photo: `prevPhoto` = the camera's previous photo in the manifest, `history` = the manifest's production counts of
the previous days (FlyBot's `history` field), `capturedAt` from the manifest. Writes <out>/responses.jsonl (one
contract response per photo), <out>/summary.json (counts, flags, proposals, timings: p50 / p95 of dnnMs and totalMs
on this machine, which is a laptop, not the Lambda: the deployed numbers come from `make cold-start`) and a
markdown table on stdout. `--url` sends the same requests to a running `make lambda-local` (or any endpoint that
speaks the contract) instead of calling the handler in-process.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lambda" / "vision"))


def rows_of(manifest: Path) -> list[dict]:
    with open(manifest, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: (r["captured_at"], r["snapshot_id"]))
    return rows


def request_for(row: dict, prev: dict | None, history: list[dict], data_dir: Path) -> dict:
    cam = {"id": row["device_serial"], "trapKind": row.get("trap_kind") or "lure", "platform": row.get("provider") or "ezviz"}
    if row.get("lat") and row.get("lon"):
        cam["location"] = {"lat": float(row["lat"]), "lng": float(row["lon"])}
    req = {"v": 1, "mode": "detect", "photo": {"url": (data_dir / row["image_path"]).resolve().as_uri()},
           "camera": cam, "capturedAt": row["captured_at"], "history": history, "locale": "vi",
           "photoId": row["snapshot_id"]}
    if prev is not None:
        req["prevPhoto"] = {"url": (data_dir / prev["image_path"]).resolve().as_uri()}
    return req


def call_http(url: str, req: dict) -> dict:
    import urllib.request

    r = urllib.request.Request(url, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read())


def pct(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, round(p / 100 * (len(vals) - 1))))
    return round(vals[k], 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(HERE.parent / "data" / "samples" / "samples_manifest.csv"))
    ap.add_argument("--out", default=str(HERE.parent / "data" / "replay" / "sample"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--url", default=None, help="POST the requests here instead of calling the handler in-process")
    ap.add_argument("--no-prev", action="store_true", help="skip prevPhoto (no diff / registration: faster)")
    a = ap.parse_args(argv)
    manifest = Path(a.manifest).resolve()
    data_dir = manifest.parent
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_of(manifest)[: a.limit]
    if not a.url:
        from sentinel import handler as h

    last: dict[str, dict] = {}
    hist: dict[str, list[dict]] = {}
    responses, errors = [], []
    t_run = time.perf_counter()
    with open(out_dir / "responses.jsonl", "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            cam = row["device_serial"]
            prev = None if a.no_prev else last.get(cam)
            req = request_for(row, prev, hist.get(cam, [])[-14 * 48:], data_dir)
            t0 = time.perf_counter()
            try:
                res = call_http(a.url, req) if a.url else h.analyze(req)
                res["_replay"] = {"snapshot_id": row["snapshot_id"], "production_count": row.get("ai_count"),
                                  "wall_ms": round((time.perf_counter() - t0) * 1000, 1), "index": i}
                responses.append(res)
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                print(f"{i:3} {cam} {row['captured_at'][:19]} count={res['counts']['total']:3} prod={row.get('ai_count') or '-':>3} "
                      f"flags={','.join(res['quality']['flags']) or '-'} proposals={','.join(p['action'] for p in res['proposals']) or '-'} "
                      f"dnn={res['timings'].get('dnnMs')} total={res['timings'].get('totalMs')} ms")
            except Exception as e:  # noqa: BLE001 - one bad photo must not stop the replay
                msg = str(e)
                errors.append({"snapshot_id": row["snapshot_id"], "error": msg[:500]})
                print(f"{i:3} {cam} ERROR {msg[:200]}")
            last[cam] = row
            if row.get("ai_count") not in (None, ""):
                hist.setdefault(cam, []).append({"capturedAt": row["captured_at"], "count": int(float(row["ai_count"])),
                                                 "snapshotId": row["snapshot_id"]})
    dnn = [r["timings"].get("dnnMs") for r in responses]
    total = [r["timings"].get("totalMs") for r in responses]
    flags = Counter(f for r in responses for f in r["quality"]["flags"])
    props = Counter(p["action"] for r in responses for p in r["proposals"])
    rules = Counter(p["rule"] for r in responses for p in r["proposals"])
    prod = [(r["counts"]["total"], int(float(r["_replay"]["production_count"]))) for r in responses
            if r["_replay"]["production_count"] not in (None, "")]
    mae = round(statistics.fmean(abs(a - b) for a, b in prod), 2) if prod else None
    summary = {
        "generated_at": datetime.now(UTC).isoformat(), "manifest": str(manifest), "photos": len(rows),
        "ok": len(responses), "errors": errors, "transport": a.url or "in-process",
        "model": responses[0]["model"] if responses else None, "opencv": responses[0].get("opencv") if responses else None,
        "planner": Counter(r["trace"]["planner"] for r in responses),
        "counts": {"total_flies": sum(r["counts"]["total"] for r in responses),
                   "mae_vs_production_ai_count": mae, "n_with_production_count": len(prod),
                   "note": "production ai_count is FlyBot's Roboflow count at capture time, not human ground truth"},
        "quality_flags": dict(flags), "proposals": dict(props), "proposal_rules": dict(rules),
        "diff_available": sum(1 for r in responses if r.get("diff")),
        "timings_ms": {"dnn_p50": pct(dnn, 50), "dnn_p95": pct(dnn, 95), "total_p50": pct(total, 50), "total_p95": pct(total, 95),
                       "run_s": round(time.perf_counter() - t_run, 1),
                       "note": "this machine (dev laptop), not the Lambda; see bench/results/lambda_latency.json"},
        "evidence_dir": str(out_dir / "evidence"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n| photos | ok | flies | MAE vs production | flags | proposals | dnn p50 ms | total p50 ms |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| {len(rows)} | {len(responses)} | {summary['counts']['total_flies']} | {mae} | {dict(flags)} | {dict(props)} | "
          f"{summary['timings_ms']['dnn_p50']} | {summary['timings_ms']['total_p50']} |")
    print(f"\nwritten: {out_dir / 'responses.jsonl'}, {out_dir / 'summary.json'}")
    return 1 if errors and not responses else 0


if __name__ == "__main__":
    raise SystemExit(main())
