#!/bin/bash
# T4.1 EC2 user data: install, benchmark, upload, self-terminate. bench/run_matrix.sh prepends the variables:
#   CONFIG (e.g. pip_arm64_c8g)  MODE (cool | pip)  BUCKET  REGION  RUNS  WARMUP  MICRO_ITERS  LIMIT  MAX_MIN
# The instance is launched with InstanceInitiatedShutdownBehavior=terminate, so every `shutdown -h` here
# terminates it: at the end (success or failure, EXIT trap) and by a hard cap of MAX_MIN minutes set first.
#
# MODE=pip : Ubuntu 24.04 python3 venv + opencv-python-headless==5.0.0.93 (the wheel the Lambda worker uses).
# MODE=cool: the COOL AMI's own OpenCV. Nothing from PyPI may shadow it: the script finds where the AMI exposes
#            cv2 (system site-packages, a venv/conda env, or a cv2 package dir under /opt, /usr/local, /home),
#            records the discovery, and only adds psutil/boto3/scipy/pyyaml with numpy pinned to what is there.
set -uo pipefail
: "${CONFIG:?}" "${MODE:?}" "${BUCKET:?}" "${REGION:?}"
RUNS="${RUNS:-5}"; WARMUP="${WARMUP:-20}"; MICRO_ITERS="${MICRO_ITERS:-200}"; LIMIT="${LIMIT:-}"; MAX_MIN="${MAX_MIN:-240}"
W=/opt/bench; mkdir -p "$W"; LOG="$W/bench.log"
exec > >(tee -a "$LOG" > /dev/console) 2>&1
shutdown -h "+$MAX_MIN" || true                      # hard cap, whatever happens below
export AWS_DEFAULT_REGION="$REGION" AWS_REGION="$REGION" DEBIAN_FRONTEND=noninteractive HOME=/root
echo "[ud] $(date -u +%FT%TZ) start CONFIG=$CONFIG MODE=$MODE"

TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')
md() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
export INSTANCE_ID=$(md instance-id) INSTANCE_TYPE=$(md instance-type) AMI_ID=$(md ami-id) AZ=$(md placement/availability-zone)
echo "[ud] instance $INSTANCE_ID $INSTANCE_TYPE $AMI_ID $AZ"

PY=""            # python that runs the benchmark
s3() {           # s3 get|put LOCAL KEY  (boto3 from the bench python)
  "$PY" - "$@" <<'EOF'
import sys, boto3, os
op, local, key = sys.argv[1:4]
c = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
if op == "get":
    c.download_file(os.environ["BUCKET"], key, local)
else:
    c.upload_file(local, os.environ["BUCKET"], key)
EOF
}
export BUCKET
STATUS=failed
finish() {
  rc=$?
  echo "[ud] $(date -u +%FT%TZ) finish status=$STATUS rc=$rc"
  if [ -n "$PY" ]; then
    echo "$STATUS rc=$rc $(date -u +%FT%TZ) $INSTANCE_ID" > "$W/status.txt"
    s3 put "$W/status.txt" "bench/logs/$CONFIG.status.txt" || true
    s3 put "$LOG" "bench/logs/$CONFIG.log" || true
  fi
  sync; shutdown -h now
}
trap finish EXIT

# ---------------------------------------------------------------- environment
apt-get update -qq || true
if [ "$MODE" = pip ]; then
  apt-get install -y -qq python3-venv python3-pip > /dev/null
  python3 -m venv "$W/venv"; PY="$W/venv/bin/python"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q opencv-python-headless==5.0.0.93 numpy==2.5.4 scipy==1.18.1 pyyaml==6.0.3 psutil boto3
  export OPENCV_SOURCE="pip opencv-python-headless==5.0.0.93 (PyPI manylinux wheel) in a python3 venv"
else
  D="$W/cool_discovery.txt"
  {
    echo "== uname / os"; uname -a; cat /etc/os-release
    echo "== motd"; cat /etc/motd 2>/dev/null; ls /etc/update-motd.d 2>/dev/null
    for f in /etc/update-motd.d/*; do echo "--- $f"; head -50 "$f"; done 2>/dev/null
    echo "== /opt /usr/local /home"; ls -la /opt /usr/local /usr/local/lib /home /home/* 2>/dev/null
    echo "== README-like files"; find / -xdev \( -iname '*readme*' -o -iname '*cool*' -o -iname '*kleidi*' \) \
         -not -path '/proc/*' -not -path '/sys/*' -not -path '/usr/share/doc/*' -not -path '/snap/*' 2>/dev/null | head -100
    echo "== cv2 packages"; find / -xdev \( -name 'cv2*.so' -o -name 'cv2' -type d \) -not -path '/proc/*' 2>/dev/null
    echo "== libopencv"; find / -xdev -name 'libopencv_core*' -not -path '/proc/*' 2>/dev/null
    echo "== libkleidicv"; find / -xdev -iname '*kleidicv*' -not -path '/proc/*' 2>/dev/null | head -50
    echo "== pythons"; ls -la /usr/bin/python3* /usr/local/bin/python* 2>/dev/null; find / -xdev -path '*/bin/activate' -not -path '/proc/*' 2>/dev/null
    echo "== system python3 import cv2"; python3 -c 'import cv2,sys; print(cv2.__version__, cv2.__file__, sys.version)' 2>&1
    echo "== pip list (system)"; python3 -m pip list 2>&1 | head -80
    echo "== dpkg opencv"; dpkg -l | grep -i -E 'opencv|kleidi' 2>/dev/null
    echo "== env files"; cat /etc/environment; ls /etc/profile.d; cat /etc/profile.d/*.sh 2>/dev/null | head -80
    echo "== ld.so.conf.d"; cat /etc/ld.so.conf.d/*.conf 2>/dev/null
  } > "$D" 2>&1
  cat "$D"
  # pick the interpreter + path that import the AMI's cv2
  CAND_PY=""; CAND_PP=""
  # The COOL listing documents its build under /opt/cool (venvs per Python version, C++ SDK libs).
  # Prefer it explicitly: the AMI also carries an unrelated reference wheel inside a containerd
  # snapshot (…/venvs/pip_ref) that a blind search picks up first (run t41, 2026-10-15).
  export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  for v in /opt/cool/venvs/python_3.12 /opt/cool/venvs/python_3.11 /opt/cool/venvs/python_3.10; do
    if [ -x "$v/bin/python" ] && "$v/bin/python" -c 'import cv2' 2>/dev/null; then CAND_PY="$v/bin/python"; OPENCV_SOURCE="COOL venv $v (LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib)"; break; fi
  done
  if [ -z "$CAND_PY" ]; then
    for pp in /opt/cool/python_3.12/site-packages /opt/cool/python_3.11/site-packages /opt/cool/python_3.10/site-packages; do
      if [ -d "$pp" ] && PYTHONPATH="$pp" python3 -c 'import cv2' 2>/dev/null; then CAND_PY=python3; CAND_PP="$pp"; OPENCV_SOURCE="COOL PYTHONPATH=$pp (LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib)"; break; fi
    done
  fi
  if [ -n "$CAND_PY" ]; then :
  elif python3 -c 'import cv2' 2>/dev/null; then CAND_PY=python3; OPENCV_SOURCE="system python3 site-packages"
  else
    for act in $(find / -xdev -path '*/bin/activate' -not -path '/proc/*' 2>/dev/null); do
      b="$(dirname "$act")/python"; if [ -x "$b" ] && "$b" -c 'import cv2' 2>/dev/null; then CAND_PY="$b"; OPENCV_SOURCE="env $(dirname "$(dirname "$act")")"; break; fi
    done
  fi
  if [ -z "$CAND_PY" ]; then
    for so in $(find / -xdev -name 'cv2*.so' -not -path '/proc/*' 2>/dev/null); do
      d="$(dirname "$so")"; [ "$(basename "$d")" = cv2 ] || [ -d "$d/cv2" ] || true
      for pp in "$(dirname "$d")" "$d"; do
        if PYTHONPATH="$pp" python3 -c 'import cv2' 2>/dev/null; then CAND_PY=python3; CAND_PP="$pp"; OPENCV_SOURCE="PYTHONPATH=$pp ($so)"; break 2; fi
      done
    done
  fi
  if [ -z "$CAND_PY" ]; then echo "[ud] no importable cv2 on the COOL AMI"; PY=$(command -v python3); apt-get install -y -qq python3-pip >/dev/null; \
     python3 -m pip install -q --break-system-packages boto3 || true; s3 put "$D" "bench/logs/$CONFIG.discovery.txt"; exit 3; fi
  export OPENCV_SOURCE="COOL AMI $AMI_ID: $OPENCV_SOURCE"
  echo "[ud] cv2 via $OPENCV_SOURCE"
  # venv that sees the AMI's packages; numpy pinned to the AMI's numpy so cv2's ABI is untouched
  apt-get install -y -qq python3-venv python3-pip > /dev/null || true
  if [ "$CAND_PY" = python3 ]; then
    python3 -m venv --system-site-packages "$W/venv"; PY="$W/venv/bin/python"
  else PY="$CAND_PY"; fi
  [ -n "$CAND_PP" ] && export PYTHONPATH="$CAND_PP"
  NPV=$(PYTHONPATH="${CAND_PP}" "$PY" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || true)
  [ -n "$NPV" ] && echo "numpy==$NPV" > "$W/constraints.txt" || : > "$W/constraints.txt"
  "$PY" -m pip install -q -c "$W/constraints.txt" scipy pyyaml psutil boto3 || "$PY" -m pip install -q --break-system-packages -c "$W/constraints.txt" scipy pyyaml psutil boto3
  "$PY" -c 'import cv2; print("[ud] cv2", cv2.__version__, cv2.__file__)'
  s3 put "$D" "bench/logs/$CONFIG.discovery.txt" || true
fi
"$PY" -m pip freeze > "$W/pip_freeze.txt" 2>&1 || true
s3 put "$W/pip_freeze.txt" "bench/logs/$CONFIG.pip_freeze.txt" || true

# ---------------------------------------------------------------- inputs + code
mkdir -p "$W/in" "$W/src"
s3 get "$W/bench_inputs.tar" bench/inputs/bench_inputs.tar
s3 get "$W/sentinel_src.tar" bench/inputs/sentinel_src.tar
s3 get "$W/bench_pipeline.py" bench/inputs/bench_pipeline.py
sha256sum "$W"/*.tar "$W/bench_pipeline.py"
tar -xf "$W/bench_inputs.tar" -C "$W/in"; tar -xf "$W/sentinel_src.tar" -C "$W/src"
export PYTHONPATH="$W/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import cv2; open("/opt/bench/build_info.txt","w").write(f"cv2.__version__ = {cv2.__version__}\ncv2.__file__ = {cv2.__file__}\n\n" + cv2.getBuildInformation())'
s3 put "$W/build_info.txt" "bench/logs/$CONFIG.build_info.txt" || true
lscpu > "$W/lscpu.txt" 2>&1; s3 put "$W/lscpu.txt" "bench/logs/$CONFIG.lscpu.txt" || true

# progress: push the log every 5 min
( while sleep 300; do s3 put "$LOG" "bench/logs/$CONFIG.log" >/dev/null 2>&1; done ) &

# ---------------------------------------------------------------- benchmark
EXTRA=(); [ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")
cd "$W"
"$PY" "$W/bench_pipeline.py" --inputs "$W/in" --config "$CONFIG" --out "$W/result.json" \
  --runs "$RUNS" --warmup "$WARMUP" --micro-iters "$MICRO_ITERS" "${EXTRA[@]}" || exit 4
s3 put "$W/result.json" "bench/results/$CONFIG.json" || exit 5
STATUS=done
