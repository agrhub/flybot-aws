"""merge bench/results/<config>.json into the Bench-page summary, add $/1000 photos, dump build info.

    uv run python bench/summarize.py [--region ap-southeast-2] [--bucket B] [--upload] [--cool-fee-usd-hr X]

Reads every bench/results/*.json with schema sentinel.bench.pipeline/1 (smoke files `*_smoke.json` skipped),
writes
    bench/results/summary_t41.json       arms list in the web Bench page shape (p50_ms, p95_ms, fps, cpu_pct,
                                         rss_mb, usd_per_1000, per_op) + micro table + output equivalence;
                                         COOL arms that were not run appear with status "pending subscription"
    bench/results/prices.json            on-demand Linux prices used (AWS Price List API, us-east-1 endpoint)
    docs/evidence/bench/<config>_build_info.txt   cv2.__version__, cv2.__file__, cv2.getBuildInformation()
and with --upload copies the summary to s3://<bucket>/bench/summary_t41.json and each result to
s3://<bucket>/bench/results/ (where the instances already put them).

Cost per 1000 photos = (on-demand $/h + Marketplace software $/h) / (photos per hour) * 1000, with photos per
hour from the parallel throughput (one process per vCPU, the way a worker fleet runs); the sequential variant
(one photo at a time, OpenCV threads) is given as usd_per_1000_sequential. EBS, data transfer and the
instance's boot/install time are not included.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
EVID = ROOT / "docs" / "evidence" / "bench"
# fallback only if the Price List API is not reachable: ap-southeast-2 Linux on-demand, read 2026-10-15
FALLBACK_PRICES = {"c8g.xlarge": 0.2076, "c7g.xlarge": 0.1887, "c7i.xlarge": 0.2331}
COOL_PLACEHOLDERS = {"cool_c8g": ("c8g.xlarge", "Graviton4 COOL listing aajkmdd4qo3r7yhqg61a7aah9"),
                     "cool_c7g": ("c7g.xlarge", "Graviton3 COOL listing 9u1i10i9fbqhc92kgsb6ni7k9")}


def price(itype: str, region: str, profile: str) -> tuple[float | None, str]:
    cmd = ["aws", "pricing", "get-products", "--region", "us-east-1", "--profile", profile, "--service-code", "AmazonEC2",
           "--filters", f"Type=TERM_MATCH,Field=instanceType,Value={itype}",
           f"Type=TERM_MATCH,Field=regionCode,Value={region}", "Type=TERM_MATCH,Field=operatingSystem,Value=Linux",
           "Type=TERM_MATCH,Field=tenancy,Value=Shared", "Type=TERM_MATCH,Field=preInstalledSw,Value=NA",
           "Type=TERM_MATCH,Field=capacitystatus,Value=Used", "--output", "json"]
    try:
        out = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True, shell=False).stdout)
        for pl in out["PriceList"]:
            p = json.loads(pl)
            for term in p["terms"]["OnDemand"].values():
                for dim in term["priceDimensions"].values():
                    if dim["unit"] == "Hrs":
                        return float(dim["pricePerUnit"]["USD"]), f"AWS Price List API ({p.get('publicationDate', '')})"
    except (subprocess.SubprocessError, OSError, KeyError, ValueError, json.JSONDecodeError):
        pass
    if itype in FALLBACK_PRICES:
        return FALLBACK_PRICES[itype], "hardcoded list price read 2026-10-15 (Price List API unreachable)"
    return None, "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="ap-southeast-2")
    ap.add_argument("--profile", default="sentinel")
    ap.add_argument("--bucket", default="sentinel-685527430355-dev")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--cool-fee-usd-hr", type=float, default=None,
                    help="COOL Marketplace software fee per instance-hour (unknown/0 during the free trial)")
    a = ap.parse_args()
    docs = {}
    for p in sorted(RES.glob("*.json")):
        if p.stem.endswith("_smoke") or p.name in ("summary_t41.json", "prices.json"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("schema") == "sentinel.bench.pipeline/1":
            docs[d["config"]] = d
    prices = {}
    EVID.mkdir(parents=True, exist_ok=True)
    arms = []
    for cfg, d in docs.items():
        arm = dict(d["arms"][0])
        itype = arm.get("instance") or d["machine"].get("instance_type")
        if itype not in prices:
            prices[itype] = dict(zip(("usd_hr", "source"), price(itype, a.region, a.profile)))
        hr = prices[itype]["usd_hr"]
        fee = a.cool_fee_usd_hr if cfg.startswith("cool") else 0.0
        total_hr = None if hr is None or (cfg.startswith("cool") and fee is None) else hr + (fee or 0.0)
        fps_par, fps_seq = arm.get("fps_parallel"), arm.get("fps")
        arm["usd_per_hour"] = total_hr
        arm["usd_per_1000"] = round(total_hr / (fps_par * 3600) * 1000, 4) if total_hr and fps_par else None
        arm["usd_per_1000_sequential"] = round(total_hr / (fps_seq * 3600) * 1000, 4) if total_hr and fps_seq else None
        arm["ami"] = d["machine"].get("ami_id")
        arm["instance_id"] = d["machine"].get("instance_id")
        arm["status"] = "measured"
        arm["fps_stdev"] = d["sequential"]["fps_stdev"]
        arm["runtime_s"] = d.get("runtime_s")
        arms.append(arm)
        (EVID / f"{cfg}_build_info.txt").write_text(
            f"config = {cfg}\ninstance = {itype} {arm['instance_id']} ami {arm['ami']}\nopencv source = {arm['build']}\n"
            f"cv2.__version__ = {d['opencv']['version']}\ncv2.__file__ = {d['opencv']['file']}\n"
            f"cv2.getNumThreads() = {d['opencv']['default_threads']}\n\n{d['opencv']['build_info']}", encoding="utf-8")
    for cfg, (itype, listing) in COOL_PLACEHOLDERS.items():
        if cfg not in docs:
            arms.insert(0, {"label": f"{cfg} (pending subscription)", "config": cfg, "build": "COOL AMI", "instance": itype,
                            "arch": "aarch64", "status": "pending subscription", "listing": listing, "p50_ms": None,
                            "p95_ms": None, "fps": None, "cpu_pct": None, "rss_mb": None, "usd_per_1000": None,
                            "per_op": {}})

    # do all configurations compute the same outputs? (run 1, per pair)
    ref_cfg = next((c for c in ("pip_arm64_c8g", "pip_arm64_c7g", "pip_x86_c7i") if c in docs), None)
    equiv = {}
    if ref_cfg:
        ref = {o["idx"]: o for o in docs[ref_cfg]["sequential"]["outputs_run1"]}
        for cfg, d in docs.items():
            outs = d["sequential"]["outputs_run1"]
            same = {k: sum(1 for o in outs if o.get(k) == ref[o["idx"]].get(k)) for k in
                    ("count", "floor_found", "registered", "new", "quality_ok", "camera_moved", "occluded")}
            inl = [abs((o.get("inliers") or 0) - (ref[o["idx"]].get("inliers") or 0)) for o in outs]
            equiv[cfg] = {"reference": ref_cfg, "pairs": len(outs), "identical": same,
                          "inliers_abs_diff_mean": round(sum(inl) / max(1, len(inl)), 2),
                          "count_total": sum(o.get("count") or 0 for o in outs),
                          "flags_run1": d["sequential"]["flags_run1"], "dnn_engine": d["sequential"]["dnn_engine_used"]}
    micro = {cfg: {th: {op: s["p50"] for op, s in ops.items()} for th, ops in d["micro"]["threads"].items()}
             for cfg, d in docs.items() if d.get("micro")}
    summary = {
        "schema": "sentinel.bench.summary/1", "title": "COOL vs pip OpenCV 5: real per-photo pipeline",
        "note": "500 real lure-trap photo pairs (BE4089629/BE4111159), decode + YOLOv7-tiny on both + analyze(lure, "
                "sift), 5 sequential runs after a warm-up; usd_per_1000 = on-demand $/h / parallel throughput "
                "(1 process per vCPU). COOL arms: pending subscription (Free Plan account cannot buy the paid "
                "Marketplace offer) unless measured. Source: bench/results/<config>.json, bench/REPORT_COOL.md",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "region": a.region,
        "arms": arms, "prices": prices, "cool_fee_usd_hr": a.cool_fee_usd_hr, "equivalence": equiv,
        "micro_p50_ms": micro,
    }
    (RES / "summary_t41.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (RES / "prices.json").write_text(json.dumps(prices, indent=1), encoding="utf-8")
    print(json.dumps({a_["label"]: {k: a_.get(k) for k in ("p50_ms", "p95_ms", "fps", "fps_parallel", "cpu_pct", "rss_mb",
                                                            "usd_per_1000")} for a_ in arms}, indent=1))
    if a.upload:
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(RES / "summary_t41.json"),
                        f"s3://{a.bucket}/bench/summary_t41.json", "--profile", a.profile, "--region", a.region,
                        "--content-type", "application/json"], check=True)
        print(f"uploaded s3://{a.bucket}/bench/summary_t41.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
