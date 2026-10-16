#!/usr/bin/env bash
# Cold start + warm latency of the deployed vision Lambda, measured, written to bench/results/lambda_latency.json.
#
#   make cold-start [STAGE=dev] [N=5]
#
# Uploads one sample photo to the photos bucket (samples/), forces a cold start by touching an environment variable
# (BENCH_NONCE), invokes once (cold), then N times (warm), and reads Init Duration / Duration from the REPORT line
# of each invocation's log tail. The numbers are what CloudWatch reports; the summary keeps p50 / max.
set -euo pipefail
STAGE="${STAGE:-dev}"; PROFILE="${PROFILE:-sentinel}"; REGION="${REGION:-ap-southeast-2}"; N="${N:-5}"
STACK="flybot-$STAGE"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# shellcheck source=infra/lib.sh
. infra/lib.sh
find_tools
aws_or_die
out() { "$AWS" cloudformation describe-stacks --stack-name "$STACK" --profile "$PROFILE" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
FN="$(out VisionFunctionName)"; BUCKET="$(out PhotosBucket)"
[ -n "$FN" ] && [[ "$FN" == flybot-vision-* ]] || { echo "no Lambda in $STACK yet" >&2; exit 1; }
PHOTO="data/samples/images/BE4089629/6ac327d44c3227d11cfbd084.jpg"; PREV="data/samples/images/BE4089629/6ac32eda4c3227d11cfbd0b4.jpg"
"$AWS" s3 cp "$PHOTO" "s3://$BUCKET/samples/cur.jpg" --only-show-errors --profile "$PROFILE" --region "$REGION"
"$AWS" s3 cp "$PREV" "s3://$BUCKET/samples/prev.jpg" --only-show-errors --profile "$PROFILE" --region "$REGION"
REQ="{\"v\":1,\"mode\":\"detect\",\"photo\":{\"s3Bucket\":\"$BUCKET\",\"s3Key\":\"samples/cur.jpg\"},\"prevPhoto\":{\"s3Bucket\":\"$BUCKET\",\"s3Key\":\"samples/prev.jpg\"},\"camera\":{\"id\":\"BENCH\",\"trapKind\":\"lure\",\"platform\":\"ezviz\"},\"capturedAt\":\"2026-10-16T08:00:00Z\",\"locale\":\"en\"}"
mkdir -p bench/results infra/.build
# force a cold start: a new environment variable value redeploys the function's execution environment
ENVJSON="$("$AWS" lambda get-function-configuration --function-name "$FN" --query 'Environment.Variables' --output json --profile "$PROFILE" --region "$REGION")"
NEWENV="$(printf '%s' "$ENVJSON" | py -c 'import json,sys,time;d=json.load(sys.stdin);d["BENCH_NONCE"]=str(int(time.time()));print(json.dumps({"Variables":d}))')"
"$AWS" lambda update-function-configuration --function-name "$FN" --environment "$NEWENV" --profile "$PROFILE" --region "$REGION" >/dev/null
"$AWS" lambda wait function-updated --function-name "$FN" --profile "$PROFILE" --region "$REGION"
invoke() {  # -> "init_ms duration_ms billed_ms memory_mb count"
  local log; log="$("$AWS" lambda invoke --function-name "$FN" --payload "$REQ" --cli-binary-format raw-in-base64-out --log-type Tail \
    --query 'LogResult' --output text --profile "$PROFILE" --region "$REGION" infra/.build/invoke-out.json | base64 -d)"
  local init dur billed mem count
  init="$(printf '%s' "$log" | grep -o 'Init Duration: [0-9.]*' | grep -o '[0-9.]*' || echo 0)"
  # the first "Duration:" after RequestId (a greedy match would pick "Init Duration:" at the end of the REPORT line)
  dur="$(printf '%s' "$log" | grep -o 'REPORT RequestId: [0-9a-f-]*[[:space:]]*Duration: [0-9.]*' | head -1 | grep -o '[0-9.]*$')"
  billed="$(printf '%s' "$log" | grep -o 'Billed Duration: [0-9]*' | grep -o '[0-9]*')"
  mem="$(printf '%s' "$log" | grep -o 'Max Memory Used: [0-9]*' | grep -o '[0-9]*')"
  count="$(py -c 'import json;print(json.load(open("infra/.build/invoke-out.json")).get("counts",{}).get("total","err"))')"
  echo "${init:-0} ${dur:-0} ${billed:-0} ${mem:-0} $count"
}
echo "cold invoke"; COLD="$(invoke)"; echo "  init/duration/billed/mem/count: $COLD"
WARM=()
for i in $(seq 1 "$N"); do W="$(invoke)"; WARM+=("$W"); echo "  warm $i: $W"; done
py - "$FN" "$COLD" "${WARM[@]}" <<'EOF'
import json, statistics, sys, time
fn, cold, *warm = sys.argv[1:]
c = cold.split(); w = [x.split() for x in warm]
durs = [float(x[1]) for x in w]
doc = {"function": fn, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "memory_mb_configured": 2048,
       "cold": {"init_ms": float(c[0]), "duration_ms": float(c[1]), "billed_ms": int(c[2]), "max_memory_mb": int(c[3]), "count": c[4]},
       "warm": {"n": len(w), "duration_ms_p50": statistics.median(durs), "duration_ms_max": max(durs), "max_memory_mb": max(int(x[3]) for x in w)},
       "request": "data/samples BE4089629 cur+prev (lure), detect mode, S3 photo", "source": "CloudWatch REPORT lines via lambda invoke --log-type Tail"}
json.dump(doc, open("bench/results/lambda_latency.json", "w"), indent=2)
print(json.dumps(doc, indent=2))
EOF
