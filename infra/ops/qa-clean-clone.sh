#!/usr/bin/env bash
# Clean-clone reproducibility run (A9, T4.6 / T5.6): clones this repo (local clone, no network for git), runs
# what a judge without hardware, AWS or the git-ignored models can run, and prints a markdown table
# (command, wall time, exit code). Never deploys and never calls AWS.
#
#   bash infra/ops/qa-clean-clone.sh [DEST]          DEST default: $TMPDIR/qa-clone (deleted first)
#   KEEP=1 bash infra/ops/qa-clean-clone.sh          keep DEST and the isolated caches afterwards
#
# Package downloads (PyPI, npm) do need the network; the uv and npm caches are isolated under DEST.cache so the
# run measures a cold machine. Make targets are run as their underlying commands (Windows has no make).
set -uo pipefail
SRC="$(git rev-parse --show-toplevel)" || exit 2
DEST="${1:-${TMPDIR:-/tmp}/qa-clone}"
CACHE="$DEST.cache"
export UV_CACHE_DIR="$CACHE/uv" npm_config_cache="$CACHE/npm"
rm -rf "$DEST" "$CACHE"
LOG="$CACHE/logs"; mkdir -p "$LOG"
ROWS=()

MIN_FREE_MB="${MIN_FREE_MB:-1500}"  # the run needs ~1 GB (venv, node_modules, caches); stop before the disk fills
free_mb() { df -Pm "$(dirname "$DEST")" | awk 'NR==2 {print $4}'; }

step() {  # step "<label>" <dir> <command...>
  local label="$1" dir="$2"; shift 2
  if [ "$(free_mb)" -lt "$MIN_FREE_MB" ]; then
    ROWS+=("| \`$label\` | — | not run | less than ${MIN_FREE_MB} MB free on the clone's disk |")
    echo "[skip] $label: less than ${MIN_FREE_MB} MB free" >&2; return
  fi
  local f; f="$LOG/$(printf '%02d' ${#ROWS[@]}).log"
  local t0; t0=$(date +%s)
  (cd "$dir" && "$@") > "$f" 2>&1
  local rc=$? t=$(( $(date +%s) - t0 ))
  local tail; tail="$(grep -E 'passed|failed|# (pass|fail) |Tests +[0-9]|error|Error|vulnerabilit|built in' "$f" | tail -1 | tr '|' '/' | cut -c1-90)"
  ROWS+=("| \`$label\` | ${t}s | $([ $rc = 0 ] && echo pass || echo "FAIL ($rc)") | ${tail} |")
  echo "[$([ $rc = 0 ] && echo ok || echo FAIL)] $label (${t}s) -> $f" >&2
}

step "git clone (local)" / git clone -q "$SRC" "$DEST"
step "make setup = uv sync --group dev" "$DEST" uv sync --group dev
step "make test = uv run pytest -q" "$DEST" uv run pytest -q -rs
step "make lint = uv run ruff check ." "$DEST" uv run ruff check .
step "uv run pytest -q docker/vision-worker" "$DEST" uv run pytest -q docker/vision-worker
step "python cv2 is 5.x" "$DEST" uv run python -c "import cv2; assert cv2.__version__.startswith('5.'); print(cv2.__version__)"
step "node --test camos_cloud/test/*.test.mjs" "$DEST" bash -c 'node --test camos_cloud/test/*.test.mjs'
step "node --test camos_cloud/ui-api/test/*.test.mjs" "$DEST" bash -c 'node --test camos_cloud/ui-api/test/*.test.mjs'
step "node --test camos_cloud/flybot-poller/test/*.test.mjs" "$DEST" bash -c 'node --test camos_cloud/flybot-poller/test/*.test.mjs'
step "web: npm ci" "$DEST/web" npm ci --no-audit --no-fund
step "web: npm test" "$DEST/web" npm test
step "web: npm run build" "$DEST/web" npm run build
step "web: npm audit --omit=dev" "$DEST/web" npm audit --omit=dev
step "python -m sentinel.agent.demo --out <tmp>" "$DEST" uv run python -m sentinel.agent.demo --out "$CACHE/demo"
if [ -f "$DEST/data/samples/samples_manifest.csv" ]; then
  step "make replay-sample (bundled sample, cached boxes)" "$DEST" env SENTINEL_PLANNER=rule uv run python -m eval.replay \
    --manifest data/samples/samples_manifest.csv --box-source cached --cache-dir data/samples/yolov7 \
    --out-dir "$CACHE/replay" --run-id sample --approve-after-hours 2
  img="$(ls "$DEST"/data/samples/images/*/*.jpg 2>/dev/null | head -2 | tr '\n' ' ')"
  set -- $img
  [ $# = 2 ] && step "vision pipeline on 2 sample photos (no detector)" "$DEST" uv run python -m sentinel.vision.pipeline \
    --image "$2" --prev "$1" --trap-kind lure --out "$CACHE/pipeline"
fi
step "make replay DAYS=30 (needs git-ignored data/manifest.csv)" "$DEST" uv run python -m eval.replay --days 30 --approve-after-hours 2
step "make eval (needs git-ignored export + ONNX)" "$DEST" uv run python -m eval.run_eval --quick

echo
echo "| Command | Wall time | Result | Last summary line |"
echo "|---|---|---|---|"
printf '%s\n' "${ROWS[@]}"
echo
echo "Logs: $LOG"
if [ "${KEEP:-0}" != 1 ]; then rm -rf "$DEST"; echo "removed $DEST (logs kept in $LOG; rm -rf $CACHE to free the caches)"; fi
