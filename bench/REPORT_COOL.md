# T4.1: COOL vs vanilla OpenCV 5 on AWS, with the real FlyBot Sentinel pipeline

Status 2026-10-15 (evening): **all three configurations are measured.** Configuration A ran on the COOL Graviton4 AMI after the account was upgraded to the Paid Plan and the listing subscribed (agreement 11:35 UTC; software fee USD 0.01/h on c8g.xlarge, 7-day free trial). A first COOL run (i-0a67b755f1198c60c) was discarded because the blind cv2 search picked a reference wheel left inside a containerd snapshot on the AMI, not the COOL build; the user-data now prefers `/opt/cool/venvs/python_3.x` + `LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib` as the listing documents (bench/results/logs/wrong-build/ keeps that run). The valid COOL run is i-0d127c2d251f3e25e (31 min). **Result: on this real pipeline COOL is slightly faster on the OpenCV image ops but slower on the DNN forward, and net 4.6 % slower per photo than the pip wheel on the same c8g instance; it does not beat x86 c7i on cost.**

Every number below comes from `bench/results/<config>.json` (raw, written on the instance) or `bench/results/summary_t41.json` (merged, with prices). Build info dumps are in `docs/evidence/bench/`.

## 1. What is measured

**Workload.** The per-photo call the production vision worker makes (`sentinel.vision.pipeline.analyze_photo`, the same code as the Lambda worker), on 500 fixed real photo pairs (current + previous photo of the same camera):
- decode both JPEGs from memory (`cv2.imdecode`; file bytes are read into RAM once, so disk I/O is not timed);
- op 5: YOLOv7-tiny fruit-fly detector, ONNX `models/yolov7-tiny/fruitfly_deploy.onnx` (sha256 `0cf45768…a16f`), on the OpenCV 5 **new DNN engine** (`engine_used = new` recorded on every configuration), on both photos;
- `analyze(cur, prev, trap_kind="lure", boxes, prev_boxes, method="sift")`: op 1 lure tray floor on both photos; op 2 quality gate; op 3 SIFT + FLANN registration; op 4 box diff with Hungarian matching; op 6 n/a for lure traps; op 8 motion / occlusion. No evidence files are written.

**Inputs** (`bench/select_inputs.py`, deterministic): consecutive photo pairs of the lure-trap EZVIZ cameras BE4089629 / BE4111159 from the FlyBot export (real camera frames only). Out of 1,291 eligible pairs every 2.58th pair is taken, covering 2026-10-03 to 2026-10-14: 500 pairs, 1,000 distinct images (313 pairs from BE4089629, 187 from BE4111159; 440 current photos at 768x432, 60 at 2304x1296). The list with the sha256 of every image is `bench/inputs/pairs.csv`. Package on S3: `s3://sentinel-685527430355-dev/bench/inputs/bench_inputs.tar` (62.2 MB, sha256 `f4c39b7e…10c4a1`), beside `sentinel_src.tar` and `bench_pipeline.py`.

**Phases** (`bench/bench_pipeline.py`, the same script on every instance):
1. Warm-up on the first 20 pairs (not recorded).
2. **Sequential:** all 500 pairs, **5 runs** (2,500 photos), OpenCV default threading (4 threads on 4 vCPUs). Per-photo end-to-end latency p50/p95, throughput, system CPU % (psutil), process CPU %, peak RSS (psutil sampling every 0.2 s plus `ru_maxrss`), and the pipeline's own per-op timings.
3. **Parallel throughput:** 4 worker processes (one per vCPU), `cv2.setNumThreads(1)` each; warm-up 5 pairs, barrier, then all 500 pairs once. Throughput = 500 / (last finish − first start). This is how a worker fleet runs, so **$/1000 photos uses this number**.
4. **Micro-benchmark:** ops COOL / KleidiCV claim to speed up, on a 1920x1080 frame made from a real 2304x1296 trap photo; 200 iterations after 10 warm-ups, at default threads and at 1 thread.

**Instances.** One per configuration, all xlarge (4 vCPU), on-demand, ap-southeast-2c, account 685527430355, launched together. Instance profile `sentinel-bench-ec2` (S3 `GetObject` on `bench/inputs/*`, `PutObject` on `bench/results/*` and `bench/logs/*` only). Security group `sentinel-bench-nosg` (no ingress), IMDSv2 required. `InstanceInitiatedShutdownBehavior=terminate`; user data ends with `shutdown -h now` on success and failure (EXIT trap) plus a hard cap `shutdown -h +300`.

## 2. Configurations

| Config | Instance (CPU) | AMI | OpenCV | Instance id |
|---|---|---|---|---|
| A `cool_c8g` | c8g.xlarge (Graviton4), same host type as B | COOL Graviton4 `ami-0a393ffc6f5d928e2` (product `aajkmdd4qo3r7yhqg61a7aah9`, listing v3.1) | the AMI's build: `/opt/cool/python_3.12/site-packages/cv2`, **OpenCV 5.1.0-dev**, GCC 13.3, baseline `SVE NEON FP16 NEON_DOTPROD NEON_FP16 NEON_BF16` (no runtime dispatch), KleidiCV **0.7.0**, pthreads, MLAS SGEMM; numpy from the AMI | i-0d127c2d251f3e25e |
| A `cool_c7g` | c7g.xlarge | COOL Graviton3 `ami-097ba4b5cac2b90d2` | not subscribed (separate listing) | not run |
| B `pip_arm64_c8g` | c8g.xlarge, Neoverse-V2 (Graviton4 2.8 GHz), 4 cores | Canonical Ubuntu 24.04 arm64 `ami-0ad88ad11d3502d29` | pip `opencv-python-headless==5.0.0.93`, numpy 2.5.4, scipy 1.18.1, Python 3.12.3 venv | i-0994a17ce967f927b |
| B `pip_arm64_c7g` | c7g.xlarge, Neoverse-V1 (Graviton3 2.6 GHz), 4 cores | same | same | i-04b5a42a618e8a82d |
| C `pip_x86_c7i` | c7i.xlarge, Xeon Platinum 8488C (Sapphire Rapids), 2 cores x 2 threads | Canonical Ubuntu 24.04 amd64 `ami-0f5e1e0151e5b95ed` | same pip versions (x86_64 wheel) | i-033dbe92ee78992c3 |

B runs on both Graviton generations, so the matching pip baseline exists whichever COOL listing gets subscribed.

**Build info**: `cv2.__version__` = 5.0.0 everywhere; full `cv2.getBuildInformation()` in `docs/evidence/bench/<config>_build_info.txt`, CPU in `<config>.lscpu.txt`.

Important for COOL: **the vanilla pip arm64 wheel already contains KleidiCV.** Its build info says `Custom HAL: YES (carotene (ver 0.0.1) KleidiCV (ver 26.03))`; baseline `NEON FP16`, dispatched `NEON_DOTPROD NEON_FP16 NEON_BF16`, **no SVE/SVE2 code paths** although Graviton4 has `sve sve2 i8mm bf16`; `DNN MLAS: YES (SGEMM-only, vendored)` with NEON SGEMM kernels; pthreads; GCC 14.2.1 `-O3`. The x86 wheel has the IPP HAL, AVX2/AVX512_SKX dispatch and AVX512F MLAS kernels. The c8g and c7g build info are byte-identical (same wheel). COOL's advantage over this baseline has to come from something other than "KleidiCV on/off": compiler flags (`-mcpu=neoverse-v2`, SVE), DNN kernels, or the threading backend. This is why we measure instead of quoting the claimed ~20%.

## 3. Results: real pipeline (500 pairs x 5 runs)

| Config | p50 ms | p95 ms | mean ms | fps (sequential) | fps (parallel, 4 proc) | CPU % (system, sequential) | Peak RSS MB | $/h on-demand | **$ / 1000 photos** (parallel) | $ / 1000 (sequential) |
|---|---|---|---|---|---|---|---|---|---|---|
| A COOL c8g (Graviton4) | 437.3 | 450.1 | 437.1 | 2.287 | 3.066 | 76.3 | 424.4 | 0.2076 + 0.01 fee = 0.2176 | **0.0197** | 0.0264 |
| B pip arm64 c8g (Graviton4) | 418.0 | 436.2 | 418.7 | 2.387 | 3.320 | 74.4 | 419.7 | 0.2076 | **0.0174** | 0.0242 |
| B pip arm64 c7g (Graviton3) | 495.7 | 506.5 | 494.8 | 2.020 | 2.788 | 74.9 | 440.5 | 0.1887 | **0.0188** | 0.0260 |
| C pip x86 c7i | 345.4 | 363.7 | 344.8 | 2.899 | 4.189 | 71.9 | 473.2 | 0.2331 | **0.0155** | 0.0223 |

**Run-to-run stability** (5 runs): fps standard deviation 0.001 (c8g), 0.002 (c7g), 0.003 (c7i); per-run p50 standard deviation 0.28 / 0.44 / 0.49 ms. Differences between configurations are about 100x the run-to-run noise.

Other numbers: process CPU during the sequential phase 297 % (c8g), 299 % (c7g), 287 % (c7i) of one core; latency per photo inside the parallel phase (4 in flight, 1 thread each) p50 1204 ms (c8g), 1432 ms (c7g), 934 ms (c7i). Prices from the AWS Price List API (publication 2026-10-12, Linux on-demand ap-southeast-2, `bench/results/prices.json`). $/1000 counts the instance-hour only (no EBS, data transfer, boot or install time).

### Per-op p50 (ms, from the pipeline's `timings_ms`, 2,500 photos each)

| Op | B c8g | B c7g | C c7i | COOL |
|---|---|---|---|---|
| decode prev + cur | 1.59 | 1.92 | 1.69 | 1.71 |
| op 5 detect (YOLOv7-tiny x 2, new DNN engine) | **218.5** | 261.8 | **145.1** | **243.0** |
| op 1 lure floor (x 2) | 28.4 | 29.7 | 24.5 | 28.6 |
| op 2 quality | 4.8 | 5.6 | 5.6 | 4.8 |
| op 3 SIFT + FLANN registration | 69.2 | 82.7 | 79.4 | **65.4** |
| op 4 box diff | 2.6 | 3.6 | 2.8 | 2.8 |
| op 6 crowd (n/a for lure) | 0.01 | 0.01 | 0.01 | 0.01 |
| op 8 motion / occlusion | 91.5 | 108.0 | 84.5 | **89.2** |
| `analyze()` total (ops 1-8 without op 5) | **197.9** | 230.7 | **198.2** | **191.9** |

Reading: COOL is 3 % faster than the pip wheel on the OpenCV image ops (`analyze()` 191.9 vs 197.9 ms: SIFT registration −5.5 %, motion −2.5 %) but **11 % slower on the DNN forward** (243.0 vs 218.5 ms), so the whole photo is 4.6 % slower (437.3 vs 418.0 ms). Without the detector, Graviton4 with the pip wheel is as fast as Sapphire Rapids (197.9 vs 198.2 ms), faster on SIFT registration and slower on op 8 and op 1. **The whole gap between c8g and c7i is the DNN** (218.5 vs 145.1 ms), which is where an optimized Arm build would have to win. Graviton3 is 16 to 20 % slower than Graviton4 on every op.

### Micro-benchmark: 1920x1080 frame, 200 iterations, p50 ms

| Op | B c8g default (4 thr) | B c8g 1 thr | B c7g default | B c7g 1 thr | C c7i default | C c7i 1 thr | A COOL c8g default / 1 thr |
|---|---|---|---|---|---|---|---|
| resize linear → 640x360 | 0.12 | 0.31 | 0.14 | 0.36 | 0.23 | 0.46 | 0.29 / 1.05 |
| resize area → 960x540 | 0.25 | 0.33 | 0.31 | 0.30 | 0.23 | 0.49 | 0.22 / 0.23 |
| cvtColor BGR→GRAY | 0.16 | 0.63 | 0.19 | 0.75 | 0.28 | 0.62 | 0.17 / 0.62 |
| cvtColor BGR→HSV | 0.80 | 3.15 | 0.98 | 3.83 | 0.72 | 1.68 | 0.79 / 3.13 |
| GaussianBlur 5x5 (BGR) | 0.57 | 1.85 | 0.50 | 1.91 | 1.02 | 2.46 | 0.49 / 1.71 |
| adaptiveThreshold Gaussian 31 | 4.46 | 10.34 | 4.73 | 13.02 | 5.23 | 7.51 | 4.88 / 15.60 |
| adaptiveThreshold mean 31 | 3.53 | 3.52 | 4.02 | 4.02 | 2.24 | 2.25 | **1.86** / 6.55 |
| findContours (external) | 14.12 | 14.22 | 16.40 | 16.44 | 11.27 | 11.13 | 13.75 / 13.82 |
| warpPerspective linear 1080p | 2.69 | 10.67 | 3.14 | 12.41 | 4.47 | 11.69 | 2.79 / 11.11 |
| SIFT detectAndCompute (2000 kp) | 114.8 | 326.4 | 141.9 | 424.2 | 166.7 | 330.5 | 113.8 / 322.5 |
| dnn blobFromImage 640 | 0.50 | 0.50 | 0.49 | 0.49 | 0.92 | 0.90 | 0.49 / 0.49 |
| **dnn forward YOLOv7-tiny 640** | **107.9** | **339.4** | 129.6 | 407.9 | **68.9** | **137.7** | **120.4 / 371.9** |

Per-op findings (COOL vs pip on the same c8g): COOL wins clearly only on mean adaptiveThreshold (1.86 vs 3.53 ms, −47 %) and marginally on GaussianBlur, resize-area and SIFT; it is slower on linear resize (0.29 vs 0.12 ms; 1.05 vs 0.31 single-threaded), Gaussian adaptiveThreshold (+9 %, +51 % single-threaded) and the DNN forward (+12 %, +10 % single-threaded). KleidiCV is in both builds (26.03 in the pip wheel, 0.7.0 in COOL), so this is a compiler/flags comparison (GCC 13.3 with SVE baseline vs GCC 14.2 NEON dispatch), not a KleidiCV on/off comparison. On the imgproc ops KleidiCV covers (resize, cvtColor gray, GaussianBlur, warpPerspective), Graviton4 with the pip wheel is already **1.6 to 1.9x faster** than c7i at default threads, and faster on SIFT. Graviton4 is slower on HSV conversion, mean adaptiveThreshold and findContours, and **much slower on the DNN forward**: 1.6x at 4 threads, **2.5x single-threaded**.

### Do all configurations compute the same thing?

`summary_t41.json` → `equivalence`, run 1, per pair, against `pip_arm64_c8g`: **c7g and c8g are bit-identical on all 500 pairs**. **c7i (x86) differs slightly**: detection count identical on 469/500 pairs (total 244 vs 247), registration ok/not ok identical on 492/500 (mean absolute inlier difference 1.7), floor/quality/motion flags identical on all 500; cause: floating-point differences between AVX-512 and NEON kernels (detector scores near the 0.25 threshold, SIFT descriptors). COOL vs pip arm64 on c8g: the run-1 flags are identical (floor found 499/500, registered 469, camera moved 1, occluded 0, errors 0); a box-level diff like the x86 one was not produced for this pair. Flags in run 1 (same on all configurations except registration): floor found 499/500, registered 469 (471 on x86), camera moved 1, occluded 0, errors 0.

## 4. Cost

- Per 1000 photos (parallel throughput, on-demand, Sydney): c7i USD 0.0155, c8g 0.0174, c7g 0.0188.
- COOL on c8g: USD 0.0197 per 1000 photos (3.066 photos/s in parallel at USD 0.2176/h incl. the 0.01 fee), i.e. **13 % more expensive than the pip wheel on the same instance and 27 % more than x86 c7i**.
- With the pip wheels, **x86 c7i is the cheapest and the fastest** for this workload. Graviton4 costs 11 % less per hour but is 21 % slower per photo, because of the DNN.
- **Break-even for COOL** was 3.73 photos/s in parallel on c8g to match c7i (3.91 with the fee); measured: **3.07**, below the pip wheel (3.32). COOL would have to speed up the DNN forward, which dominates this pipeline, to pay off here.
- Spend of this task (upper bounds, `bench/results/logs/instances.txt`): smoke runs c8g 4.8 min, c7g 4.5 min, c7i 3.8 + 2.6 min; full runs c8g 25.0 min, c7g 31.2 min, c7i 21.8 min; total **about USD 0.33** plus under USD 0.01 of EBS.

## 5. COOL subscription check (2026-10-15, before the Paid Plan upgrade)

`aws ec2 run-instances --dry-run` in ap-southeast-2, profile `sentinel`:

| AMI | Listing | Result |
|---|---|---|
| ami-0a393ffc6f5d928e2 (Graviton4 v3, 2026-07-15) | aajkmdd4qo3r7yhqg61a7aah9 | OptInRequired |
| ami-06a19ba5979af120e (Graviton4 v2) | aajkmdd4qo3r7yhqg61a7aah9 | OptInRequired |
| ami-097ba4b5cac2b90d2 (Graviton3 v3) | 9u1i10i9fbqhc92kgsb6ni7k9 | OptInRequired |
| ami-0de67a99debacb610 (Graviton2 V3) | 4eb70u7a6gb5bj1ijix5jlokv | OptInRequired |
| ami-04bd68f2b7445010d (Graviton5) | 216fpdsdup19yw3vwuiwx254l | OptInRequired |

The account's Marketplace agreements (`aws marketplace-agreement search-agreements`, us-east-1) contained only 3 SaaS agreements (proposer 276720623826, accepted 2026-10-15); none is a COOL AMI. `run_matrix.sh` repeats the dry-run each time and launches only the COOL configs that pass it. After the upgrade and subscription, `ami-0a393ffc6f5d928e2` passed the dry-run (`DryRunOperation`); the Graviton3 listing is still not subscribed, so `cool_c7g` was not run.

## 6. How configuration A was run (and how to rerun it)

1. Subscribe to a COOL listing (done for Graviton4 on the Paid Plan). Delete stale `bench/logs/cool_*.status.txt` objects in S3 first, otherwise `run_matrix.sh` treats the config as already done.
2. Run `bash bench/run_matrix.sh --only cool` (for another account: `--profile <P> --region <R> --bucket <bucket there>`). The script picks the newest AMI per product code in that region, creates the role, instance profile and security group there, and uploads the same `bench/build/*.tar`, `bench_pipeline.py` and `pairs.csv` to `s3://<bucket>/bench/inputs/`.
3. With `MODE=cool`, the user data does not install OpenCV. It prefers `/opt/cool/venvs/python_3.12` (fallback `/opt/cool/python_3.12/site-packages` on PYTHONPATH) with `LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib`, as the listing documents; only if those are absent does it look for the AMI's own cv2 and records the search in `bench/logs/cool_*.discovery.txt` (motd, `/opt`, cv2*.so, libopencv_core, KleidiCV files, venvs, system site-packages), uses the first interpreter that imports cv2, adds only psutil / boto3 / scipy / pyyaml with **numpy pinned to the AMI's version**, and records `OPENCV_SOURCE` and `cv2.__file__`. `sentinel.require_opencv5()` refuses a 4.x build.
4. Run `uv run python bench/summarize.py --upload [--cool-fee-usd-hr X]` to merge the COOL columns, the prices and `docs/evidence/bench/cool_*_build_info.txt` (done for this report with `--cool-fee-usd-hr 0.01`).

Expected: about 25 min (c8g) or 30 min (c7g) plus about 1 min of setup; under USD 0.15 per config plus the software fee.

## 7. Reproduce

```bash
uv run python bench/select_inputs.py          # once: bench/inputs/pairs.csv + bench/build/*.tar (needs data/export)
bash bench/run_matrix.sh --dry-run            # AMIs, COOL subscription check, plan
bash bench/run_matrix.sh --only b,c --smoke   # 6 pairs, harness check (~5 min)
bash bench/run_matrix.sh --only b,c           # this report's B + C (~35 min wall, in parallel)
bash bench/run_matrix.sh --only cool          # configuration A once subscribed
uv run python bench/summarize.py --upload     # merge, prices, build-info dumps, s3://.../bench/summary_t41.json
```

The run of this report: RUN_ID `t41-20261015T102851Z`; instances launched 10:30:50–10:31:23Z, all finished by 11:00:28Z; script runtimes 1415.8 s (c8g), 1685.7 s (c7g), 1157.5 s (c7i). The web Bench page (https://d146vb7dg01iax.cloudfront.net) lists `bench/summary_t41.json` (5 arms; `cool_c8g` measured, `cool_c7g` not subscribed) and `bench/results/<config>.json`.

## 8. Not measured / limits

- COOL on Graviton3 (c7g) and the Docker variant of COOL: not measured (not subscribed).
- COOL build info is `docs/evidence/bench/cool_c8g_build_info.txt`; the AMI reports OpenCV 5.1.0-dev, not 5.0.0, so the comparison is also across minor versions.
- CPU % and RSS during the parallel phase.
- ALIKED + LightGlue registration (production default is SIFT).
- Evidence writing, S3 upload, SQS/Lambda overhead, cold starts.
- One instance per configuration, one AZ, one time window; variation between hosts of the same type is not measured (run-to-run variation on one host is under 0.2 %).
- c7i.xlarge has 2 physical cores (4 hyperthreads); the Graviton xlarge types have 4 physical cores. The comparison is per instance and per dollar, as AWS sells them.
- Energy / carbon.
