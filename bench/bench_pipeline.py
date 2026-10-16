"""benchmark: the real per-photo pipeline + an OpenCV op micro-benchmark, on whatever machine runs it.

    python bench/bench_pipeline.py --inputs DIR --config NAME --out RESULT.json
        [--runs 5] [--warmup 20] [--micro-iters 200] [--parallel N] [--limit N] [--skip-micro]

DIR is the unpacked bench_inputs.tar (pairs.csv, images/, model/) from bench/select_inputs.py; the `sentinel`
package must be importable (repo root or the unpacked sentinel_src.tar on PYTHONPATH).

Workload per photo (the exact call the production worker makes, sentinel.vision.pipeline.analyze_photo):
  decode current + previous JPEG from memory (cv2.imdecode; file bytes are read into RAM once, so disk I/O
  is not measured) -> op 5 YOLOv7-tiny ONNX (OpenCV 5 new DNN engine) on both photos -> analyze(): op 1 lure
  floor on both, op 2 quality, op 3 SIFT+FLANN registration, op 4 box diff (Hungarian), op 6 (n/a for lure),
  op 8 motion/occlusion. trap_kind "lure", method "sift", no evidence files written.

Phases
  1. warm-up: the first --warmup pairs once (not recorded; loads the ONNX, JITs nothing, fills caches)
  2. sequential: all pairs, --runs times, OpenCV default threading (cv2.getNumThreads()); per-photo end-to-end
     latency (decode + detect + analyze), per-op timings from the pipeline's own timings_ms, system-wide CPU %
     (psutil), process CPU %, peak RSS (psutil sampling every 0.2 s + getrusage ru_maxrss)
  3. parallel throughput: --parallel worker processes (default: os.cpu_count()), cv2.setNumThreads(1) each,
     every worker warms up on 5 pairs, waits on a barrier, then processes its share of all pairs once;
     throughput = pairs / (last finish - first start). This is how a worker fleet would run; $/1000 uses it.
  4. micro-benchmark: OpenCV ops COOL/KleidiCV claims to accelerate, on a 1920x1080 frame made from a real
     2304x1296 trap photo (INTER_AREA), --micro-iters iterations each after 10 warm-up, at default threads and
     at 1 thread.

Output: one JSON (schema sentinel.bench.pipeline/1) with an `arms` list of one entry in the shape the web
Bench page reads (p50_ms, p95_ms, fps, cpu_pct, rss_mb, usd_per_1000 (filled later by bench/summarize.py),
per_op), plus details (per-run stats, micro, per-pair outputs of run 1 for cross-config equivalence, build info).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform

try:
    import resource
except ImportError:  # Windows dev machine (smoke tests only)
    resource = None
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = "sentinel.bench.pipeline/1"


def pct(v: list[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(v, dtype=float), q)), 2) if v else None


def stats(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    return {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "mean": round(float(np.mean(v)), 2),
            "min": round(float(np.min(v)), 2), "max": round(float(np.max(v)), 2)}


def load_pairs(inputs: Path, limit: int | None) -> list[dict]:
    with open(inputs / "pairs.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = rows[:limit] if limit else rows
    for r in rows:
        r["prev_bytes"] = np.frombuffer((inputs / r["prev_path"]).read_bytes(), np.uint8)
        r["cur_bytes"] = np.frombuffer((inputs / r["cur_path"]).read_bytes(), np.uint8)
    return rows


def one_photo(r: dict) -> tuple[float, float, dict]:
    """-> (end-to-end ms, decode ms, pipeline result)."""
    import cv2

    from sentinel.vision.pipeline import analyze_photo

    t0 = time.perf_counter()
    prev = cv2.imdecode(r["prev_bytes"], cv2.IMREAD_COLOR)
    cur = cv2.imdecode(r["cur_bytes"], cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    res = analyze_photo(cur, prev, trap_kind="lure", out_dir=None, method="sift")
    t2 = time.perf_counter()
    return (t2 - t0) * 1000.0, (t1 - t0) * 1000.0, res


def digest(r: dict, res: dict) -> dict:
    """Compact per-pair output, to check that every configuration computes the same thing."""
    reg, diff, card, mo = res.get("register") or {}, res.get("diff") or {}, res.get("card") or {}, res.get("motion") or {}
    return {"idx": int(r["idx"]), "count": res.get("count"), "floor_found": card.get("found"),
            "registered": reg.get("ok"), "inliers": reg.get("inliers"), "new": diff.get("new_count"),
            "persisted": diff.get("persisted_count"), "quality_ok": (res.get("quality") or {}).get("ok"),
            "issues": (res.get("quality") or {}).get("issues"), "camera_moved": mo.get("camera_moved"),
            "occluded": mo.get("occluded_by_large_object"), "errors": sorted(k for k in res.get("errors", {})
                                                                            if not k.endswith("_trace"))}


class Sampler:
    """Peak RSS of this process, sampled every 0.2 s."""

    def __init__(self):
        import psutil

        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self._stop.wait(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def _cpu_busy(t) -> float:
    """Non-idle CPU seconds of a psutil cpu_times() sample (fields differ per OS)."""
    return (t.user + getattr(t, "nice", 0) + t.system + getattr(t, "irq", 0)
            + getattr(t, "softirq", 0) + getattr(t, "interrupt", 0) + getattr(t, "dpc", 0))


def _cpu_total(t) -> float:
    return _cpu_busy(t) + t.idle + getattr(t, "iowait", 0) + getattr(t, "steal", 0)


def sequential(pairs: list[dict], runs: int, warmup: int) -> dict:
    import psutil

    proc = psutil.Process()
    for r in pairs[:warmup]:
        one_photo(r)
    run_stats, all_e2e, all_dec, per_op = [], [], [], {}
    outputs, flags = [], {"floor_found": 0, "registered": 0, "occluded": 0, "camera_moved": 0, "errors": 0}
    engine = None
    with Sampler() as smp:
        for k in range(runs):
            sys_t0, proc_t0 = psutil.cpu_times(), proc.cpu_times()
            w0 = time.perf_counter()
            e2e = []
            for r in pairs:
                ms, dec, res = one_photo(r)
                e2e.append(ms)
                all_dec.append(dec)
                for op, v in res["timings_ms"].items():
                    per_op.setdefault(op, []).append(v)
                if k == 0:
                    outputs.append(digest(r, res))
                    flags["floor_found"] += bool(res["card"] and res["card"]["found"])
                    flags["registered"] += bool(res["register"] and res["register"]["ok"])
                    flags["occluded"] += bool(res["motion"] and res["motion"]["occluded_by_large_object"])
                    flags["camera_moved"] += bool(res["motion"] and res["motion"]["camera_moved"])
                    flags["errors"] += bool([e for e in res["errors"] if not e.endswith("_trace")])
                    engine = engine or (res.get("detect") or {}).get("engine_used")
            wall = time.perf_counter() - w0
            sys_t1, proc_t1 = psutil.cpu_times(), proc.cpu_times()
            sys_busy = _cpu_busy(sys_t1) - _cpu_busy(sys_t0)
            sys_pct = 100.0 * sys_busy / max(1e-9, _cpu_total(sys_t1) - _cpu_total(sys_t0))
            proc_pct = 100.0 * ((proc_t1.user + proc_t1.system) - (proc_t0.user + proc_t0.system)) / wall
            all_e2e += e2e
            run_stats.append({"run": k + 1, "wall_s": round(wall, 2), "fps": round(len(pairs) / wall, 4),
                              "p50_ms": pct(e2e, 50), "p95_ms": pct(e2e, 95), "mean_ms": round(statistics.fmean(e2e), 2),
                              "cpu_pct_system": round(sys_pct, 1), "cpu_pct_process": round(proc_pct, 1)})
    peak_rss = max(smp.peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 if resource else 0)
    fps_runs = [s["fps"] for s in run_stats]
    return {
        "runs": run_stats,
        "latency_ms": stats(all_e2e), "decode_ms": stats(all_dec),
        "per_op_ms": {op: stats(v) for op, v in per_op.items()},
        "fps_mean": round(statistics.fmean(fps_runs), 4),
        "fps_stdev": round(statistics.stdev(fps_runs), 4) if len(fps_runs) > 1 else 0.0,
        "p50_ms_run_stdev": round(statistics.stdev([s["p50_ms"] for s in run_stats]), 2) if len(run_stats) > 1 else 0.0,
        "cpu_pct_system": round(statistics.fmean(s["cpu_pct_system"] for s in run_stats), 1),
        "cpu_pct_process": round(statistics.fmean(s["cpu_pct_process"] for s in run_stats), 1),
        "peak_rss_mb": round(peak_rss / 2**20, 1),
        "flags_run1": flags, "dnn_engine_used": engine, "outputs_run1": outputs,
    }


def _worker(inputs: str, idxs: list[int], limit, barrier, q):
    import cv2

    cv2.setNumThreads(1)
    pairs = load_pairs(Path(inputs), limit)
    mine = [pairs[i] for i in idxs]
    for r in mine[:5]:
        one_photo(r)
    barrier.wait()
    t0 = time.time()
    lat = [one_photo(r)[0] for r in mine]
    q.put({"t0": t0, "t1": time.time(), "n": len(mine), "lat": lat})


def parallel(inputs: Path, n_pairs: int, workers: int, limit) -> dict:
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(workers), ctx.Queue()
    shards = [list(range(i, n_pairs, workers)) for i in range(workers)]
    procs = [ctx.Process(target=_worker, args=(str(inputs), s, limit, barrier, q)) for s in shards]
    for p in procs:
        p.start()
    got = [q.get(timeout=7200) for _ in procs]
    for p in procs:
        p.join()
    t0, t1 = min(g["t0"] for g in got), max(g["t1"] for g in got)
    lat = [v for g in got for v in g["lat"]]
    return {"workers": workers, "threads_per_worker": 1, "pairs": sum(g["n"] for g in got),
            "wall_s": round(t1 - t0, 2), "fps": round(sum(g["n"] for g in got) / (t1 - t0), 4),
            "latency_ms": stats(lat)}


def micro(inputs: Path, iters: int) -> dict:
    import cv2

    with open(inputs / "pairs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    src = next(r["cur_path"] for r in rows if r["cur_size"] == "2304x1296")
    big = cv2.imread(str(inputs / src))
    img = cv2.resize(big, (1920, 1080), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    H = np.array([[1.02, 0.03, -20.0], [-0.02, 0.99, 15.0], [1.5e-5, -1.0e-5, 1.0]])
    sift = cv2.SIFT_create(nfeatures=2000)
    net = cv2.dnn.readNet(str(inputs / "model" / "fruitfly_deploy.onnx"), "", "", cv2.dnn.ENGINE_NEW)
    from sentinel.vision.detect_smoke import letterbox

    boxed, _, _ = letterbox(img, 640)
    blob = cv2.dnn.blobFromImage(boxed, 1 / 255.0, (640, 640), swapRB=True, crop=False)

    def fwd():
        net.setInput(blob)
        return net.forward()

    ops = {
        "resize_linear_1080p_to_640x360": lambda: cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR),
        "resize_area_1080p_to_960x540": lambda: cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA),
        "cvtColor_bgr2gray": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "cvtColor_bgr2hsv": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
        "GaussianBlur_5x5_bgr": lambda: cv2.GaussianBlur(img, (5, 5), 0),
        "adaptiveThreshold_gauss_31": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                    cv2.THRESH_BINARY_INV, 31, 5),
        "adaptiveThreshold_mean_31": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                                   cv2.THRESH_BINARY_INV, 31, 5),
        "findContours_external": lambda: cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE),
        "warpPerspective_1080p_linear": lambda: cv2.warpPerspective(img, H, (1920, 1080), flags=cv2.INTER_LINEAR),
        "SIFT_detectAndCompute_1080p_2000": lambda: sift.detectAndCompute(gray, None),
        "dnn_blobFromImage_640": lambda: cv2.dnn.blobFromImage(boxed, 1 / 255.0, (640, 640), swapRB=True),
        "dnn_forward_yolov7tiny_640": fwd,
    }
    default_threads = cv2.getNumThreads()
    out = {"frame": "1920x1080 from " + src, "iters": iters, "warmup": 10, "default_threads": default_threads,
           "threads": {}}
    for label, nthreads in (("default", default_threads), ("1", 1)):
        cv2.setNumThreads(nthreads)
        res = {}
        for name, fn in ops.items():
            for _ in range(10):
                fn()
            v = []
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                v.append((time.perf_counter() - t0) * 1000.0)
            res[name] = stats(v)
        out["threads"][label] = res
    cv2.setNumThreads(default_threads)
    return out


def machine_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
            "cpus": os.cpu_count()}
    try:
        txt = Path("/proc/cpuinfo").read_text()
        for key in ("model name", "CPU part", "CPU implementer", "Features", "flags"):
            for line in txt.splitlines():
                if line.lower().startswith(key.lower()):
                    info[key.replace(" ", "_").lower()] = line.split(":", 1)[1].strip()[:400]
                    break
    except OSError:
        pass
    for k in ("INSTANCE_TYPE", "INSTANCE_ID", "AMI_ID", "AZ", "BENCH_CONFIG_LABEL", "OPENCV_SOURCE"):
        if os.environ.get(k):
            info[k.lower()] = os.environ[k]
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True, help="config id, e.g. cool_c8g, pip_arm64_c8g, pip_x86_c7i")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--micro-iters", type=int, default=200)
    ap.add_argument("--parallel", type=int, default=0, help="worker processes (0 = os.cpu_count())")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N pairs (smoke test)")
    ap.add_argument("--skip-micro", action="store_true")
    ap.add_argument("--skip-parallel", action="store_true")
    a = ap.parse_args(argv)
    inputs = Path(a.inputs)
    os.environ.setdefault("SENTINEL_DETECTOR_ONNX", str(inputs / "model" / "fruitfly_deploy.onnx"))

    import cv2

    import sentinel
    from sentinel.vision import pipeline

    version = sentinel.require_opencv5()
    build_info = cv2.getBuildInformation()
    t_start = time.time()
    pairs = load_pairs(inputs, a.limit)
    print(f"[bench] {a.config}: OpenCV {version} from {cv2.__file__}, {len(pairs)} pairs, threads {cv2.getNumThreads()}",
          flush=True)
    seq = sequential(pairs, a.runs, a.warmup)
    print(f"[bench] sequential done: p50 {seq['latency_ms']['p50']} ms, fps {seq['fps_mean']}", flush=True)
    par = None
    if not a.skip_parallel:
        par = parallel(inputs, len(pairs), a.parallel or os.cpu_count(), a.limit)
        print(f"[bench] parallel done: fps {par['fps']}", flush=True)
    mic = None if a.skip_micro else micro(inputs, a.micro_iters)
    print("[bench] micro done", flush=True)

    per_op = {op: {"p50_ms": s["p50"], "p95_ms": s["p95"], "mean_ms": s["mean"]} for op, s in seq["per_op_ms"].items()}
    per_op = {"decode": {"p50_ms": seq["decode_ms"]["p50"], "p95_ms": seq["decode_ms"]["p95"],
                         "mean_ms": seq["decode_ms"]["mean"]}} | per_op
    m = machine_info()
    arm = {
        "label": a.config, "config": a.config, "build": os.environ.get("OPENCV_SOURCE", "pip"),
        "instance": m.get("instance_type"), "arch": m["machine"], "opencv": version,
        "n_images": len(pairs), "runs": a.runs,
        "p50_ms": seq["latency_ms"]["p50"], "p95_ms": seq["latency_ms"]["p95"], "mean_ms": seq["latency_ms"]["mean"],
        "fps": seq["fps_mean"], "fps_parallel": par["fps"] if par else None,
        "cpu_pct": seq["cpu_pct_system"], "cpu_pct_process": seq["cpu_pct_process"], "rss_mb": seq["peak_rss_mb"],
        "usd_per_1000": None, "per_op": per_op,
    }
    doc = {
        "schema": SCHEMA, "title": f"pipeline benchmark, {a.config}", "config": a.config,
        "note": "per-photo latency = decode prev+cur + YOLOv7-tiny on both + analyze(lure, sift); fps = sequential "
                "(OpenCV default threads); fps_parallel = one process per vCPU, 1 OpenCV thread each; "
                "usd_per_1000 is filled by bench/summarize.py from the on-demand price",
        "arms": [arm],
        "opencv": {"version": version, "file": cv2.__file__, "default_threads": cv2.getNumThreads(),
                   "build_info_sha256": hashlib.sha256(build_info.encode()).hexdigest(),
                   "build_info": build_info},
        "machine": m, "sequential": seq, "parallel": par, "micro": mic,
        "inputs": {"pairs": len(pairs), "pairs_csv_sha256": hashlib.sha256((inputs / "pairs.csv").read_bytes()).hexdigest(),
                   "onnx_sha256": hashlib.sha256((inputs / "model" / "fruitfly_deploy.onnx").read_bytes()).hexdigest()},
        "pipeline_schema": pipeline.SCHEMA, "runtime_s": round(time.time() - t_start, 1),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[bench] wrote {a.out} in {doc['runtime_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
