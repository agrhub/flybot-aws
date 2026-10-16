#!/usr/bin/env bash
# T4.1 benchmark matrix on EC2: COOL AMI vs pip arm64 vs pip x86, same inputs, same script.
#
#   bash bench/run_matrix.sh [--only LIST] [--profile P] [--region R] [--bucket B] [--smoke] [--dry-run]
#
#   --only     comma list of configs or groups (default: all = every subscribed COOL config + the pip ones)
#              groups: cool (cool_c8g, cool_c7g: only those whose AMI passes the dry-run), pip_arm64
#              (pip_arm64_c8g, pip_arm64_c7g), pip_x86 (pip_x86_c7i), b (= pip_arm64), c (= pip_x86)
#   --profile  AWS CLI profile (default sentinel)        --region  region (default ap-southeast-2)
#   --bucket   bucket for bench/inputs, bench/results, bench/logs (default sentinel-<account>-dev; for a
#              fallback account give a bucket there, the inputs are copied from bench/build/ to it)
#   --smoke    6 pairs, 1 run, 5 micro iterations (harness check, results go to <config>_smoke.json)
#   --dry-run  resolve AMIs and print the launch plan, launch nothing
#
# Steps: resolve AMIs (COOL newest per product code + EC2 dry-run for the Marketplace subscription; Canonical
# Ubuntu 24.04 via SSM) -> ensure IAM role/instance profile sentinel-bench-ec2 (S3 get bench/inputs/*, put
# bench/results/* and bench/logs/* of that bucket only) and a no-ingress security group -> upload inputs
# (bench/build/*.tar from bench/select_inputs.py + bench/bench_pipeline.py) -> launch one on-demand instance per
# config with bench/instance_user_data.sh (InstanceInitiatedShutdownBehavior=terminate, MAX_MIN hard cap) ->
# poll S3 -> download results + logs -> terminate anything still alive (EXIT trap, also on Ctrl-C/failure)
# -> bench/summarize.py. Region-agnostic: AMIs are looked up, not hardcoded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
command -v aws >/dev/null || export PATH="$LOCALAPPDATA/Programs/Amazon/AWSCLIV2:/c/Users/$USERNAME/AppData/Local/Programs/Amazon/AWSCLIV2:$PATH"
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
PROFILE=sentinel REGION=ap-southeast-2 BUCKET="" ONLY=all SMOKE=0 DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2;; --profile) PROFILE="$2"; shift 2;; --region) REGION="$2"; shift 2;;
    --bucket) BUCKET="$2"; shift 2;; --smoke) SMOKE=1; shift;; --dry-run) DRY=1; shift;;
    -h|--help) sed -n 2,22p "$0"; exit 0;; *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
export AWS_PROFILE="$PROFILE" AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" MSYS_NO_PATHCONV=1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="${BUCKET:-sentinel-$ACCOUNT-dev}"
RUN_ID="t41-$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/bench/results"; LOGS="$ROOT/bench/results/logs"; mkdir -p "$OUT" "$LOGS"
ROLE=sentinel-bench-ec2 SG_NAME=sentinel-bench-nosg
G4_PRODUCT=aajkmdd4qo3r7yhqg61a7aah9 G3_PRODUCT=9u1i10i9fbqhc92kgsb6ni7k9
if [ "$SMOKE" = 1 ]; then RUNS=1 WARMUP=2 MICRO=5 LIMIT=6 MAX_MIN=60 SUFFIX=_smoke; else RUNS=5 WARMUP=20 MICRO=200 LIMIT="" MAX_MIN=300 SUFFIX=""; fi
wp() { if command -v cygpath >/dev/null; then cygpath -w "$1"; else echo "$1"; fi; }  # local path for aws.exe
log() { echo "[matrix $(date -u +%H:%M:%S)] $*" >&2; }

# ---------------------------------------------------------------- AMIs
newest_by_product() {
  aws ec2 describe-images --owners aws-marketplace --filters "Name=product-code,Values=$1" "Name=architecture,Values=arm64" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text
}
dry_ok() {  # AMI TYPE -> 0 if the account may launch it (DryRunOperation), 1 otherwise (OptInRequired...)
  local out; out=$(aws ec2 run-instances --dry-run --image-id "$1" --instance-type "$2" --count 1 2>&1 || true)
  echo "$out" | grep -q DryRunOperation && return 0
  log "dry-run $1 on $2: $(echo "$out" | grep -o 'An error occurred ([A-Za-z]*)' | head -1)"; return 1
}
UBU_ARM=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
UBU_X86=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
COOL_G4=$(newest_by_product $G4_PRODUCT); COOL_G3=$(newest_by_product $G3_PRODUCT)
log "AMIs: ubuntu arm64 $UBU_ARM, ubuntu x86 $UBU_X86, COOL G4 $COOL_G4, COOL G3 $COOL_G3"

# config -> "mode ami type"
declare -A CFG
CFG[pip_arm64_c8g]="pip $UBU_ARM c8g.xlarge"
CFG[pip_arm64_c7g]="pip $UBU_ARM c7g.xlarge"
CFG[pip_x86_c7i]="pip $UBU_X86 c7i.xlarge"
COOL_OK=()
[ "$COOL_G4" != None ] && dry_ok "$COOL_G4" c8g.xlarge && CFG[cool_c8g]="cool $COOL_G4 c8g.xlarge" && COOL_OK+=(cool_c8g)
[ "$COOL_G3" != None ] && dry_ok "$COOL_G3" c7g.xlarge && CFG[cool_c7g]="cool $COOL_G3 c7g.xlarge" && COOL_OK+=(cool_c7g)
log "COOL configs launchable: ${COOL_OK[*]:-none (not subscribed)}"

SEL=()
IFS=, read -ra WANT <<< "$ONLY"
for w in "${WANT[@]}"; do
  case "$w" in
    all) SEL+=("${COOL_OK[@]}" pip_arm64_c8g pip_arm64_c7g pip_x86_c7i);;
    cool) SEL+=("${COOL_OK[@]}"); [ ${#COOL_OK[@]} -gt 0 ] || { log "no COOL AMI is subscribed in $ACCOUNT/$REGION"; exit 3; };;
    pip_arm64|b) SEL+=(pip_arm64_c8g pip_arm64_c7g);; pip_x86|c) SEL+=(pip_x86_c7i);;
    *) [ -n "${CFG[$w]:-}" ] || { log "config $w unknown or not launchable"; exit 2; }; SEL+=("$w");;
  esac
done
for c in "${SEL[@]}"; do log "plan: $c -> ${CFG[$c]}"; done
[ "$DRY" = 1 ] && exit 0
[ ${#SEL[@]} -gt 0 ] || { log "nothing to run"; exit 2; }
for c in "${SEL[@]}"; do set -- ${CFG[$c]}; dry_ok "$2" "$3" || { log "cannot launch $c"; exit 3; }; done

# ---------------------------------------------------------------- IAM + SG
if ! aws iam get-role --role-name $ROLE >/dev/null 2>&1; then
  log "creating role $ROLE"
  aws iam create-role --role-name $ROLE --description "T4.1 bench instances: S3 bench/ prefix only" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --tags Key=Project,Value=sentinel Key=Purpose,Value=bench-t41 >/dev/null
fi
aws iam put-role-policy --role-name $ROLE --policy-name bench-s3 --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
 {\"Effect\":\"Allow\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::$BUCKET/bench/inputs/*\"},
 {\"Effect\":\"Allow\",\"Action\":\"s3:PutObject\",\"Resource\":[\"arn:aws:s3:::$BUCKET/bench/results/*\",\"arn:aws:s3:::$BUCKET/bench/logs/*\"]}]}"
if ! aws iam get-instance-profile --instance-profile-name $ROLE >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name $ROLE >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
  sleep 15   # instance profile propagation
fi
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$SG_NAME Name=vpc-id,Values=$VPC --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = None ]; then
  SG=$(aws ec2 create-security-group --group-name $SG_NAME --description "T4.1 bench: no ingress" --vpc-id "$VPC" --query GroupId --output text)
fi
log "role $ROLE, sg $SG (no ingress), vpc $VPC"

# ---------------------------------------------------------------- inputs
[ -f "$ROOT/bench/build/bench_inputs.tar" ] || (cd "$ROOT" && uv run python bench/select_inputs.py)
for f in bench_inputs.tar sentinel_src.tar; do
  aws s3 cp --only-show-errors "$(wp "$ROOT/bench/build/$f")" "s3://$BUCKET/bench/inputs/$f"
done
aws s3 cp --only-show-errors "$(wp "$ROOT/bench/bench_pipeline.py")" "s3://$BUCKET/bench/inputs/bench_pipeline.py"
aws s3 cp --only-show-errors "$(wp "$ROOT/bench/inputs/pairs.csv")" "s3://$BUCKET/bench/inputs/pairs.csv"
log "inputs in s3://$BUCKET/bench/inputs/ ($(sha256sum "$ROOT/bench/build/bench_inputs.tar" | cut -c1-16))"

# ---------------------------------------------------------------- launch
declare -A IID
cleanup() {
  local alive=()
  for c in "${!IID[@]}"; do
    st=$(aws ec2 describe-instances --instance-ids "${IID[$c]}" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo unknown)
    [ "$st" = terminated ] || alive+=("${IID[$c]}")
  done
  if [ ${#alive[@]} -gt 0 ]; then log "terminating ${alive[*]}"; aws ec2 terminate-instances --instance-ids "${alive[@]}" >/dev/null || true; fi
}
trap cleanup EXIT INT TERM
SUBNETS=($(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text))
for c in "${SEL[@]}"; do
  set -- ${CFG[$c]}; MODE=$1 AMI=$2 TYPE=$3; NAME="$c$SUFFIX"
  UD=$(mktemp)
  { echo '#!/bin/bash'
    echo "CONFIG=$NAME MODE=$MODE BUCKET=$BUCKET REGION=$REGION RUNS=$RUNS WARMUP=$WARMUP MICRO_ITERS=$MICRO LIMIT=$LIMIT MAX_MIN=$MAX_MIN"
    echo "export CONFIG MODE BUCKET REGION RUNS WARMUP MICRO_ITERS LIMIT MAX_MIN"
    tail -n +2 "$ROOT/bench/instance_user_data.sh"; } > "$UD"
  ROOTDEV=$(aws ec2 describe-images --image-ids "$AMI" --query 'Images[0].RootDeviceName' --output text)
  ROOTGB=$(aws ec2 describe-images --image-ids "$AMI" --query 'Images[0].BlockDeviceMappings[0].Ebs.VolumeSize' --output text)
  [ "$ROOTGB" -lt 16 ] && ROOTGB=16
  for try in 1 2 3 4 5; do
    for sn in "${SUBNETS[@]}"; do
      id=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --count 1 --subnet-id "$sn" \
        --security-group-ids "$SG" --iam-instance-profile Name=$ROLE --user-data "file://$(wp "$UD")" \
        --instance-initiated-shutdown-behavior terminate --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --block-device-mappings "[{\"DeviceName\":\"$ROOTDEV\",\"Ebs\":{\"VolumeSize\":$ROOTGB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=sentinel-bench-$NAME},{Key=Project,Value=sentinel},{Key=Purpose,Value=bench-t41},{Key=RunId,Value=$RUN_ID}]" \
        --query 'Instances[0].InstanceId' --output text 2>"$UD.err") && break 2
      log "launch $NAME in $sn failed: $(tail -1 "$UD.err")"
      grep -q -E 'Invalid IamInstanceProfile|InsufficientInstanceCapacity|Unsupported' "$UD.err" || break 2
    done
    sleep 10
  done
  rm -f "$UD"
  [ -n "${id:-}" ] && [[ "$id" == i-* ]] || { log "could not launch $NAME"; exit 4; }
  IID[$NAME]=$id
  log "launched $NAME = $id ($TYPE, $AMI)"
  echo "$RUN_ID $NAME $id $TYPE $AMI $(date -u +%FT%TZ)" >> "$LOGS/instances.txt"
  unset id
done

# ---------------------------------------------------------------- poll
declare -A DONE START END
for c in "${!IID[@]}"; do START[$c]=$(date +%s); done
DEADLINE=$(( $(date +%s) + (MAX_MIN + 10) * 60 ))
while :; do
  pending=0
  for c in "${!IID[@]}"; do
    [ -n "${DONE[$c]:-}" ] && continue
    st=$(aws ec2 describe-instances --instance-ids "${IID[$c]}" --query 'Reservations[0].Instances[0].State.Name' --output text)
    s=$(aws s3 cp --only-show-errors "s3://$BUCKET/bench/logs/$c.status.txt" - 2>/dev/null || true)
    if [ -n "$s" ] || [ "$st" = terminated ] || [ "$st" = shutting-down ]; then
      DONE[$c]="${s:-no status (state $st)}"; END[$c]=$(date +%s)
      log "$c (${IID[$c]}): ${DONE[$c]} after $(( (END[$c] - START[$c]) / 60 )) min"
    else
      pending=$((pending + 1))
      tailline=$(aws s3 cp --only-show-errors "s3://$BUCKET/bench/logs/$c.log" - 2>/dev/null | grep '\[bench\]\|\[ud\]' | tail -1 || true)
      log "$c: $st ${tailline:0:140}"
    fi
  done
  [ $pending -eq 0 ] && break
  [ "$(date +%s)" -gt $DEADLINE ] && { log "deadline reached"; break; }
  sleep 120
done

# ---------------------------------------------------------------- collect
for c in "${!IID[@]}"; do
  aws s3 cp --only-show-errors "s3://$BUCKET/bench/results/$c.json" "$(wp "$OUT/$c.json")" 2>/dev/null || log "no result for $c"
  for k in log status.txt build_info.txt lscpu.txt pip_freeze.txt discovery.txt; do
    aws s3 cp --only-show-errors "s3://$BUCKET/bench/logs/$c.$k" "$(wp "$LOGS/$c.${k%.txt}.txt")" 2>/dev/null || true
  done
  echo "$RUN_ID $c ${IID[$c]} start=$(date -u -d @${START[$c]} +%FT%TZ 2>/dev/null || echo ${START[$c]}) end=$(date -u -d @${END[$c]:-$(date +%s)} +%FT%TZ 2>/dev/null) status=${DONE[$c]:-none}" >> "$LOGS/instances.txt"
done
cleanup
trap - EXIT
# instance-hours actually billed: launch -> terminated (from CloudTrail-free EC2 data: describe-instances keeps
# terminated instances ~1 h; the summary uses the local timestamps in instances.txt as the upper bound)
if [ "$SMOKE" = 0 ]; then
  (cd "$ROOT" && uv run python bench/summarize.py --region "$REGION" --bucket "$BUCKET" --upload)
fi
log "done: results in $OUT, logs in $LOGS"
