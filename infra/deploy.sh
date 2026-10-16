#!/usr/bin/env bash
# FlyBot on AWS in one command, driven by deploy.env (copy deploy.env.example, fill it once).
#
#   ./deploy.sh                      everything below, idempotent (re-run it after any change)
#   ./deploy.sh --dry-run            preflight + discovery for real (read-only AWS calls), then the plan; changes nothing
#   ./deploy.sh --only <step>        one step (preflight and discovery always run first)
#   ./deploy.sh --show-ip            the Elastic IP and the DNS record of the existing stack
#   ./deploy.sh --yes                allow destructive recoveries (delete a stack stuck in ROLLBACK_COMPLETE)
#   ./deploy.sh --skip-wait          do not wait for the GitHub deploy key (the EC2 retries for 30 min itself)
#   ./deploy.sh --env-file F         another settings file (default: deploy.env)
#
# Steps: preflight, discover, models, secrets, repoaccess, stack, image, lambda, bootstrap, health, smoke, coldstart, summary.
# Log: infra/.build/deploy-<stage>.log (no secret value is ever printed). Exit 2 = AWS session expired (aws login).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=infra/lib.sh
. infra/lib.sh
find_tools

STEPS=(models secrets repoaccess stack image lambda bootstrap health smoke coldstart summary)
ENV_FILE=deploy.env DRY=0 ONLY="" YES=0 SHOW_IP=0 SKIP_WAIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="${2:?--only needs a step: ${STEPS[*]}}"; shift ;;
    --only=*) ONLY="${1#--only=}" ;;
    --yes|-y) YES=1 ;;
    --skip-wait) SKIP_WAIT=1 ;;
    --show-ip) SHOW_IP=1 ;;
    --env-file) ENV_FILE="${2:?}"; shift ;;
    --env-file=*) ENV_FILE="${1#--env-file=}" ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
  shift
done
if [ -n "$ONLY" ] && [[ " ${STEPS[*]} " != *" $ONLY "* ]]; then die "--only $ONLY: not a step (${STEPS[*]})"; fi

# ---------------------------------------------------------------- settings (deploy.env < environment)
CFG_LINES=""
if [ -f "$ENV_FILE" ]; then
  CFG_LINES="$(py infra/deploy_helpers.py settings "$ENV_FILE")" || die "cannot read $ENV_FILE"
  eval "$CFG_LINES"   # only CFG_<KEY>=<shell-quoted literal> and HAS_<KEY>=1 lines (deploy_helpers.py settings)
elif [ "$SHOW_IP" = 0 ]; then
  die "$ENV_FILE not found: cp deploy.env.example deploy.env, fill it, then run ./deploy.sh again"
fi
cfg() {  # cfg KEY DEFAULT: environment variable, else deploy.env, else DEFAULT
  local k="$1" d="${2:-}" e c
  e="${!k:-}"; c="CFG_$k"; c="${!c:-}"
  if [ -n "$e" ]; then printf '%s' "$e"; elif [ -n "$c" ]; then printf '%s' "$c"; else printf '%s' "$d"; fi
}
has() { local h="HAS_$1"; [ "${!h:-0}" = 1 ]; }

STAGE="$(cfg STAGE dev)"
PROFILE="$(cfg AWS_PROFILE sentinel)"
REGION="$(cfg AWS_REGION ap-southeast-2)"
DOMAIN_NAME="$(cfg DOMAIN_NAME flybot.agrhub.com)"
ALARM_EMAIL="$(cfg ALARM_EMAIL)"
ACME_EMAIL="$(cfg ACME_EMAIL "$ALARM_EMAIL")"
OWNER_CIDR="$(cfg OWNER_CIDR)"
DETECTOR="$(cfg DETECTOR yolo26)"
PLANNER_MODE="$(cfg PLANNER_MODE rule)"
INSTANCE_TYPE="$(cfg INSTANCE_TYPE t4g.small)"
KEY_NAME="$(cfg KEY_NAME)"
REPO_URL="$(cfg REPO_URL https://github.com/agrhub/flybot-aws.git)"
REPO_REF="$(cfg REPO_REF main)"
AMI_ID="$(cfg AMI_ID)"
ANTHROPIC_SECRET_NAME="$(cfg ANTHROPIC_SECRET_NAME sentinel-dev-anthropic)"
ANTHROPIC_MODEL="$(cfg ANTHROPIC_MODEL claude-haiku-5-5)"
GCP_FILE="$(cfg GCP_SERVICE_ACCOUNT_FILE)"
DNS_WAIT_MIN="$(cfg DNS_WAIT_MIN 30)"
MODELS_BUCKET="$(cfg MODELS_BUCKET)"
DEMO_CAMS="$(cfg DEMO_MAX_CAMERAS_PER_USER 3)"
DEMO_SNAPS="$(cfg DEMO_MAX_SNAPSHOTS_PER_DAY 48)"
DEMO_UNCLAIMED="$(cfg DEMO_MAX_UNCLAIMED 200)"
DEVICE_EVENTS_URL="$(cfg DEVICE_EVENTS_URL)"
SKIP_COLDSTART="$(cfg SKIP_COLDSTART 0)"
FLYBOT_REPO="$(cfg FLYBOT_REPO agrhub/flybot)"
REPO_WAIT_MIN="$(cfg REPO_WAIT_MIN 30)"
STACK="flybot-$STAGE"
APP_SECRET="flybot-$STAGE-app"
GCP_SECRET="flybot-$STAGE-gcp"
export AWS_PROFILE="$PROFILE" AWS_REGION="$REGION" STAGE PROFILE REGION

mkdir -p infra/.build
LOG="infra/.build/deploy-$STAGE.log"
exec > >(tee -a "$LOG") 2>&1
TMP="$(mktemp -d)"; chmod 700 "$TMP"
trap 'rm -rf "$TMP"' EXIT
printf '\n######## %s  ./deploy.sh %s  (stage %s, %s)\n' "$(date -u +%FT%TZ)" \
  "$([ "$DRY" = 1 ] && echo '--dry-run ')$([ -n "$ONLY" ] && echo "--only $ONLY")" "$STAGE" "$ENV_FILE"

SUMMARY=()
row() { SUMMARY+=("$(printf '%-22s %s' "$1" "$2")"); }
step() { printf '\n== %-10s %s\n' "$1" "$2"; }
plan() { printf '%s  plan  %s\n' "$(date +%H:%M:%S)" "$*"; }
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
out() { awsx cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null | grep -v '^None$' || true; }
param() { awsx cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Parameters[?ParameterKey=='$1'].ParameterValue" --output text 2>/dev/null | grep -v '^None$' || true; }

dns_parts() {  # HOST and PARENT of DOMAIN_NAME (flybot / agrhub.com)
  local n; n="$(printf '%s' "$DOMAIN_NAME" | tr -cd '.' | wc -c)"
  if [ "$n" -ge 2 ]; then DNS_HOST="${DOMAIN_NAME%%.*}"; DNS_PARENT="${DOMAIN_NAME#*.}"; else DNS_HOST="@"; DNS_PARENT="$DOMAIN_NAME"; fi
}
box() {  # box TITLE LINE...: a highlighted block the owner must act on
  local l w=76 bar
  for l in "$@"; do [ ${#l} -gt "$w" ] && w=${#l}; done
  bar="$(printf '%*s' $((w + 2)) '' | tr ' ' '=')"
  printf '\n  +%s+\n' "$bar"
  for l in "$@"; do printf '  | %-*s |\n' "$w" "$l"; done
  printf '  +%s+\n\n' "$bar"
}
dns_box() {  # $1 = EIP
  dns_parts
  local where="your DNS provider for $DNS_PARENT"
  [ "$DNS_PARENT" = agrhub.com ] && where="Namecheap: Domain List > agrhub.com > Manage > Advanced DNS > Add New Record"
  printf '\n'
  printf '  +==========================================================================+\n'
  printf '  |  DNS RECORD TO ADD NOW (the build keeps running; the health step waits)  |\n'
  printf '  |  %-72s|\n' "$where"
  printf '  |  %-72s|\n' "Type: A Record    Host: $DNS_HOST    Value: $1    TTL: Automatic (300)"
  printf '  |  %-72s|\n' "= $DOMAIN_NAME.  A  $1"
  printf '  +==========================================================================+\n\n'
}
doh_a() {  # public resolver ($1 = 1.1.1.1 | 8.8.8.8) -> the A records of DOMAIN_NAME, space separated
  local body=""
  case "$1" in
    1.1.1.1) body="$(curl -s --max-time 8 -H 'accept: application/dns-json' "https://1.1.1.1/dns-query?name=$DOMAIN_NAME&type=A" || true)" ;;
    8.8.8.8) body="$(curl -s --max-time 8 "https://8.8.8.8/resolve?name=$DOMAIN_NAME&type=A" || true)" ;;
  esac
  printf '%s' "$body" | grep -o '"data":"[0-9][0-9.]*"' | grep -o '[0-9][0-9.]*' | tr '\n' ' ' | sed 's/ $//'
}

# ================================================================ preflight
preflight() {
  step preflight "tools, session, account, region, service limits"
  [ -n "$AWS_BIN" ] || die "AWS CLI v2 not found (install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)"
  [ -n "$SAM" ] || [ "$SHOW_IP" = 1 ] || die "SAM CLI not found (install: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)"
  ok "aws  $("$AWS" --version 2>&1 | head -1 | cut -d' ' -f1)  ($AWS_BIN)"
  [ -n "$SAM" ] && ok "sam  $("$SAM" --version 2>&1 | tr -d '\r' | head -1)"
  py -c 1 || die "no usable Python 3.9+"
  ok "python $(py -c 'import sys;print(sys.version.split()[0])')  (${PY_CMD[*]})"
  command -v curl >/dev/null || die "curl not found"
  command -v openssl >/dev/null || warn "openssl not found: VAPID keys cannot be generated (web push off)"
  if ! has FLYBOT_REPO_TOKEN; then
    { command -v ssh-keygen && command -v ssh; } >/dev/null || die "ssh / ssh-keygen not found (needed for the GitHub deploy key; or set FLYBOT_REPO_TOKEN)"
  fi
  [ "$REGION" = ap-southeast-2 ] || die "AWS_REGION=$REGION refused: the account's SCP allows ap-southeast-2 only"
  aws_or_die
  ACCOUNT="$(awsx sts get-caller-identity --query Account --output text)"
  ok "session ok: account $ACCOUNT ($(awsx sts get-caller-identity --query Arn --output text | sed 's#.*:assumed-role/##; s#/.*##'), profile $PROFILE)"
  [ "$SHOW_IP" = 1 ] && return 0

  # deploy.env must never reach git
  if git ls-files --error-unmatch "$ENV_FILE" >/dev/null 2>&1; then die "$ENV_FILE is tracked by git: git rm --cached $ENV_FILE (and rotate its secrets)"; fi
  if [ "$ENV_FILE" = deploy.env ] && ! git check-ignore -q deploy.env; then die "deploy.env is not git-ignored (.gitignore)"; fi
  if [ -n "$GCP_FILE" ]; then
    [ -f "$GCP_FILE" ] || die "GCP_SERVICE_ACCOUNT_FILE=$GCP_FILE does not exist"
    case "$(cd "$(dirname "$GCP_FILE")" && pwd)/" in
      "$ROOT"/*) git check-ignore -q "$GCP_FILE" || die "$GCP_FILE is inside the repo and not git-ignored: move it out of the repo" ;;
    esac
  fi
  if [ -n "$DEVICE_EVENTS_URL" ]; then
    py infra/deploy_helpers.py get "$ENV_FILE" DEVICE_EVENTS_SECRET --out "$TMP/des" || die "DEVICE_EVENTS_URL needs DEVICE_EVENTS_SECRET"
    DEVICE_EVENTS_SECRET="$(cat "$TMP/des")"
    [[ "$DEVICE_EVENTS_SECRET" =~ ^[A-Za-z0-9]{32,}$ ]] || die "DEVICE_EVENTS_SECRET: letters and digits only, 32+ characters"
  fi

  local azs; azs="$(awsx ec2 describe-availability-zones --query 'AvailabilityZones[].ZoneName' --output text 2>&1)" \
    || die "EC2 refused in $REGION (SCP?): $(printf '%s' "$azs" | head -1)"
  ok "region $REGION allowed (AZs: $azs)"
  local name cmd fails=0 msg
  while IFS='|' read -r name cmd; do
    # shellcheck disable=SC2086  # word splitting of the canned argument list is intended
    if msg="$(awsx $cmd 2>&1 >/dev/null)"; then ok "api $name"; else warn "api $name: $(printf '%s' "$msg" | tail -1 | cut -c1-160)"; fails=$((fails + 1)); fi
  done <<'EOF'
CloudFormation|cloudformation list-stacks --max-items 1
IoT Core|iot describe-endpoint --endpoint-type iot:Data-ATS
Kinesis Video|kinesisvideo list-signaling-channels --max-results 1
Lambda|lambda list-functions --max-items 1
ECR|ecr describe-repositories --max-items 1
SSM|ssm describe-instance-information --max-results 5
Secrets Manager|secretsmanager list-secrets --max-results 1
SNS|sns list-topics
S3|s3api list-buckets --max-items 1
IAM|iam get-account-summary
EOF
  [ "$fails" = 0 ] || die "$fails AWS API(s) not answering (permissions or SCP): see above"

  local vpc offered
  vpc="$(awsx ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  [ -n "$vpc" ] && [ "$vpc" != None ] || die "no default VPC in $REGION (the template launches into it)"
  offered="$(awsx ec2 describe-instance-type-offerings --location-type availability-zone --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
             --query 'InstanceTypeOfferings[].Location' --output text)"
  [ -n "$offered" ] || die "$INSTANCE_TYPE is not offered in $REGION"
  ok "default VPC $vpc; $INSTANCE_TYPE offered in $offered"
  if [ -z "$(out InstanceId)" ]; then
    local eq eu vq vu
    eq="$(awsx service-quotas get-service-quota --service-code ec2 --quota-code L-0263D0A3 --query 'Quota.Value' --output text 2>/dev/null || echo 5)"
    eu="$(awsx ec2 describe-addresses --query 'length(Addresses)' --output text)"
    [ "${eu%.*}" -lt "${eq%.*}" ] || die "Elastic IP quota reached ($eu of ${eq%.*}): release one or raise quota L-0263D0A3"
    vq="$(awsx service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query 'Quota.Value' --output text 2>/dev/null || echo 0)"
    vu="$(awsx ec2 describe-instances --filters Name=instance-state-name,Values=pending,running \
          --query 'Reservations[].Instances[].CpuOptions.[CoreCount,ThreadsPerCore]' --output text | awk '{s+=$1*$2} END {print s+0}')"
    if [ "${vq%.*}" -gt 0 ] && [ $((vu + 2)) -gt "${vq%.*}" ]; then die "On-Demand vCPU quota: $vu of ${vq%.*} used, $INSTANCE_TYPE needs 2"; fi
    ok "quotas: Elastic IPs $eu of ${eq%.*} used; On-Demand vCPUs $vu of ${vq%.*} used (needs 2)"
  fi
  # what the EC2 will run: it clones REPO_REF from GitHub, not this folder
  if [ -z "${SKIP_GIT_CHECK:-}" ]; then
    local remote head; remote="$(git ls-remote "$REPO_URL" "refs/heads/$REPO_REF" 2>/dev/null | cut -c1-12 || true)"; head="$(git rev-parse HEAD | cut -c1-12)"
    if [ -z "$remote" ]; then warn "cannot read $REPO_URL $REPO_REF (the EC2 clones it: is it public?)"
    elif [ "$remote" != "$head" ]; then warn "the EC2 deploys $REPO_REF@$remote but local HEAD is $head: push first if you changed FlyBot or the compose file"
    else ok "GitHub $REPO_REF = local HEAD $head"; fi
  fi
  local free; free="$(df -Pk . | awk 'NR==2 {print int($4/1024)}')"
  [ "$free" -ge 1024 ] || warn "only $free MB free on this drive"
}

# ================================================================ discover
discover() {
  step discover "account facts, existing stack, topic, secrets, models"
  IOT_ENDPOINT="$(awsx iot describe-endpoint --endpoint-type iot:Data-ATS --query endpointAddress --output text)"
  ok "IoT Data-ATS endpoint $IOT_ENDPOINT"
  STACK_STATUS="$(awsx cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo NONE)"
  CUR_IMAGE="$(param ImageUri)"; CUR_INSTANCE="$(out InstanceId)"; CUR_EIP="$(out ElasticIp)"
  ok "stack $STACK: $STACK_STATUS${CUR_INSTANCE:+, instance $CUR_INSTANCE}${CUR_EIP:+, EIP $CUR_EIP}${CUR_IMAGE:+, Lambda image ${CUR_IMAGE##*@}}"

  if [ -n "$AMI_ID" ]; then ok "AMI $AMI_ID (deploy.env)"
  elif [ -n "$(param AmiId)" ]; then AMI_ID="$(param AmiId)"; ok "AMI $AMI_ID (pinned by the stack: a new AMI would replace the instance and its MongoDB)"
  else
    AMI_ID="$(awsx ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023*-kernel-6.12-arm64" \
              "Name=state,Values=available" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)"
    [[ "$AMI_ID" == ami-* ]] || die "no Amazon Linux 2023 arm64 AMI found"
    ok "AMI $AMI_ID (newest AL2023 kernel-6.12 arm64)"
  fi
  if [ -z "$OWNER_CIDR" ]; then
    local ip; ip="$(curl -s --max-time 8 https://checkip.amazonaws.com | tr -d '\r\n ' || true)"
    if [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then OWNER_CIDR="$ip/32"; ok "SSH source $OWNER_CIDR (this laptop's public IP)"
    else OWNER_CIDR="$(param OwnerCidr)"; OWNER_CIDR="${OWNER_CIDR:-0.0.0.0/32}"; warn "public IP not detected: SSH source stays $OWNER_CIDR"; fi
  fi

  # alarm topic: an existing one whose email subscription is confirmed (never this stack's own topic)
  ALARM_TOPIC_ARN="" ALARM_NOTE=""
  local t subs pick=""
  for t in $(awsx sns list-topics --query 'Topics[].TopicArn' --output text | tr '\t' '\n' | grep -v ":flybot-$STAGE-alarms\$" || true); do
    subs="$(awsx sns list-subscriptions-by-topic --topic-arn "$t" \
            --query "Subscriptions[?Protocol=='email' && SubscriptionArn!='PendingConfirmation'].Endpoint" --output text 2>/dev/null || true)"
    [ -n "$subs" ] && [ "$subs" != None ] || continue
    if [ -n "$ALARM_EMAIL" ]; then
      local e want="${ALARM_EMAIL,,}"
      for e in $subs; do [ "${e,,}" = "$want" ] && pick="$t"; done   # (grep -iF aborts in some Git Bash builds)
      [ -n "$pick" ] && break
    else
      case "$t" in *":sentinel-$STAGE-alarms"|*":sentinel-dev-alarms") pick="$t"; break ;; esac
      [ -z "$pick" ] && pick="$t"
    fi
  done
  if [ -n "$pick" ]; then ALARM_TOPIC_ARN="$pick"; ALARM_NOTE="reuse ${pick##*:} (confirmed email)"
  elif [ -n "$ALARM_EMAIL" ]; then ALARM_NOTE="new topic flybot-$STAGE-alarms; CONFIRM the email AWS sends to the alarm address"
  else ALARM_NOTE="new topic flybot-$STAGE-alarms without subscriber (set ALARM_EMAIL to get alarms)"; fi
  ok "alarms: $ALARM_NOTE"

  PHOTOS="flybot-photos-$ACCOUNT-$STAGE"; PHOTOS_EXISTS=false
  if awsx cloudformation describe-stack-resource --stack-name "$STACK" --logical-resource-id PhotosBucket >/dev/null 2>&1; then
    ok "photos bucket $PHOTOS (owned by the stack)"
  elif awsx s3api head-bucket --bucket "$PHOTOS" >/dev/null 2>&1; then
    PHOTOS_EXISTS=true; ok "photos bucket $PHOTOS exists outside the stack (kept by destroy --keep-photos): reused"
  else ok "photos bucket $PHOTOS: will be created"; fi

  APP_STATE="$(secret_state "$APP_SECRET")"; GCP_STATE="$(secret_state "$GCP_SECRET")"
  if [ "$(secret_state "$ANTHROPIC_SECRET_NAME")" = active ]; then ANTHROPIC_NOTE="reuse $ANTHROPIC_SECRET_NAME"
  elif has ANTHROPIC_API_KEY; then ANTHROPIC_SECRET_NAME="flybot-$STAGE-anthropic"; ANTHROPIC_NOTE="$ANTHROPIC_SECRET_NAME from deploy.env"
  else ANTHROPIC_NOTE="none: deep mode answers with the rule planner's template"; fi
  ok "secrets: $APP_SECRET $APP_STATE, $GCP_SECRET $GCP_STATE, Anthropic $ANTHROPIC_NOTE"

  MODELS_BUCKET="${MODELS_BUCKET:-sentinel-$ACCOUNT-models-public}"
  YOLO26_URL="" YOLO26_SHA="" ONNX=models/yolo26/fruitfly_yolo26n.onnx
  if [ "$DETECTOR" = yolo26 ]; then
    if [ -f "$ONNX" ] && [ -f "$ONNX.json" ]; then
      YOLO26_SHA="$(sha256sum "$ONNX" | cut -d' ' -f1)"
      local side; side="$(py -c 'import json,sys;print(json.load(open(sys.argv[1])).get("sha256",""))' "$ONNX.json")"
      [ "$side" = "$YOLO26_SHA" ] || warn "sidecar sha256 ${side:0:12} != file ${YOLO26_SHA:0:12}: the file wins"
      YOLO26_URL="https://$MODELS_BUCKET.s3.$REGION.amazonaws.com/models/yolo26/fruitfly_yolo26n.onnx"
      ok "detector yolo26 ${YOLO26_SHA:0:12} (baked into the image; fallback URL on $MODELS_BUCKET)"
    else
      warn "DETECTOR=yolo26 but $ONNX(.json) missing (models/COLAB_RUNBOOK.md): falling back to yolov7"
      DETECTOR=yolov7
    fi
  fi
  [ "$DETECTOR" = yolov7 ] && ok "detector yolov7-tiny (baseline, baked into the image)"
  CTX_TAG="ctx-$(py infra/deploy_helpers.py ctx-hash)"
  ok "Lambda build inputs $CTX_TAG"
}

secret_state() {  # missing | active | deleted
  local r; r="$(awsx secretsmanager describe-secret --secret-id "$1" --query DeletedDate --output text 2>&1)" || { echo missing; return 0; }
  if [ "$r" = None ] || [ -z "$r" ]; then echo active; else echo deleted; fi
}
secret_write() {  # name file description state
  local f; f="file://$(winpath "$2")"
  case "$4" in
    missing) awsx secretsmanager create-secret --name "$1" --description "$3" --secret-string "$f" \
               --tags Key=project,Value=flybot-aws "Key=stage,Value=$STAGE" --query Name --output text >/dev/null ;;
    deleted) awsx secretsmanager restore-secret --secret-id "$1" --query Name --output text >/dev/null
             awsx secretsmanager put-secret-value --secret-id "$1" --secret-string "$f" --query Name --output text >/dev/null ;;
    *)       awsx secretsmanager put-secret-value --secret-id "$1" --secret-string "$f" --query Name --output text >/dev/null ;;
  esac
}
secret_read() {  # name file (empty file when missing/deleted)
  : > "$2"
  [ "$(secret_state "$1")" = active ] || return 0
  awsx secretsmanager get-secret-value --secret-id "$1" --query SecretString --output text > "$2" 2>/dev/null || : > "$2"
}

# ================================================================ models
step_models() {
  step models "YOLO26 weights on the public model bucket (fallback for the baked-in copy)"
  if [ "$DETECTOR" != yolo26 ]; then ok "detector $DETECTOR: nothing to upload"; return 0; fi
  if ! awsx s3api head-bucket --bucket "$MODELS_BUCKET" >/dev/null 2>&1; then
    warn "model bucket $MODELS_BUCKET not found: the Lambda uses the copy baked into the image only"; YOLO26_URL=""; return 0
  fi
  local f key remote
  for f in "$ONNX" "$ONNX.json" models/yolo26/fruitfly_yolo26n-o2m.onnx; do
    [ -f "$f" ] || continue
    key="models/yolo26/$(basename "$f")"
    remote="$(awsx s3api head-object --bucket "$MODELS_BUCKET" --key "$key" --query 'Metadata.sha256' --output text 2>/dev/null || true)"
    if [ "$remote" = "$YOLO26_SHA" ]; then ok "s3://$MODELS_BUCKET/$key up to date"; continue; fi
    if [ "$DRY" = 1 ]; then plan "upload $f -> s3://$MODELS_BUCKET/$key (sha256 metadata ${YOLO26_SHA:0:12})"; continue; fi
    awsx s3 cp "$f" "s3://$MODELS_BUCKET/$key" --metadata "sha256=$YOLO26_SHA" --only-show-errors
    ok "uploaded $key"
  done
  [ "$DRY" = 1 ] && return 0
  local code; code="$(curl -s -o /dev/null -w '%{http_code}' -I --max-time 15 "$YOLO26_URL" || true)"
  if [ "$code" = 200 ]; then ok "anonymous GET $YOLO26_URL -> 200"; row "Model URL" "$YOLO26_URL"
  else warn "anonymous GET of the model -> $code (bucket policy allows models/* only): the Lambda keeps the baked-in copy"; YOLO26_URL=""; fi
}

# ================================================================ secrets
step_secrets() {
  step secrets "Secrets Manager $APP_SECRET, $GCP_SECRET, Anthropic (values never printed)"
  SECRETS_CHANGED=0
  local rep changed
  secret_read "$APP_SECRET" "$TMP/app.cur"
  rep="$(py infra/deploy_helpers.py merge-app --env "$ENV_FILE" --current "$TMP/app.cur" --out "$TMP/app.new" \
         --contact "${ACME_EMAIL:-$ALARM_EMAIL}" ${GCP_FILE:+--gcp "$GCP_FILE"})"
  changed="$(printf '%s' "$rep" | py -c 'import json,sys;print(int(json.load(sys.stdin)["changed"]))')"
  printf '%s' "$rep" | py -c '
import json,sys
r=json.load(sys.stdin)
for k in ("set","generated","kept","default","empty"):
    if r[k]: print("          %-9s %s" % (k, ", ".join(r[k])))'
  if [ "$changed" = 0 ]; then ok "$APP_SECRET unchanged"
  elif [ "$DRY" = 1 ]; then plan "$([ "$APP_STATE" = active ] && echo update || echo create) $APP_SECRET"
  else secret_write "$APP_SECRET" "$TMP/app.new" "FlyBot application secrets (FLYBOT_SECRETS), managed by infra/deploy.sh from deploy.env" "$APP_STATE"
       ok "$APP_SECRET written"; SECRETS_CHANGED=1; fi

  secret_read "$GCP_SECRET" "$TMP/gcp.cur"
  rep="$(py infra/deploy_helpers.py gcp ${GCP_FILE:+--file "$GCP_FILE"} --current "$TMP/gcp.cur" --out "$TMP/gcp.new")" \
    || die "GCP_SERVICE_ACCOUNT_FILE is not a service-account key"
  changed="$(printf '%s' "$rep" | py -c 'import json,sys;d=json.load(sys.stdin);print(int(d["changed"]), d["status"], d.get("project") or "-")')"
  case "$changed" in
    0\ *) ok "$GCP_SECRET unchanged (${changed#0 })" ;;
    *) if [ "$DRY" = 1 ]; then plan "write $GCP_SECRET (${changed#1 })"
       else secret_write "$GCP_SECRET" "$TMP/gcp.new" "Google service-account JSON for the FlyBot chatbot, managed by infra/deploy.sh" "$GCP_STATE"
            ok "$GCP_SECRET written (${changed#1 })"; SECRETS_CHANGED=1; fi ;;
  esac
  case "$changed" in *placeholder*) CHATBOT="off (no GCP_SERVICE_ACCOUNT_FILE)" ;; *) CHATBOT="Vertex AI project ${changed##* }" ;; esac

  if [ "$ANTHROPIC_SECRET_NAME" = "flybot-$STAGE-anthropic" ]; then
    py infra/deploy_helpers.py get "$ENV_FILE" ANTHROPIC_API_KEY --out "$TMP/ak.new"
    secret_read "$ANTHROPIC_SECRET_NAME" "$TMP/ak.cur"
    if cmp -s "$TMP/ak.new" "$TMP/ak.cur"; then ok "$ANTHROPIC_SECRET_NAME unchanged"
    elif [ "$DRY" = 1 ]; then plan "write $ANTHROPIC_SECRET_NAME"
    else secret_write "$ANTHROPIC_SECRET_NAME" "$TMP/ak.new" "Anthropic API key for the FlyBot vision Lambda" "$(secret_state "$ANTHROPIC_SECRET_NAME")"
         ok "$ANTHROPIC_SECRET_NAME written"; fi
  fi
  rm -f "$TMP"/app.* "$TMP"/gcp.* "$TMP"/ak.*
}

# ================================================================ repoaccess (private submodule agrhub/flybot)
GH_KNOWN_HOSTS='github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl'
git_anon() { GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never git -c credential.helper= ls-remote "https://github.com/$FLYBOT_REPO.git" HEAD >/dev/null 2>&1; }
git_key() {  # $1 = private key file
  GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -i $1 -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=$TMP/gh_known_hosts -o StrictHostKeyChecking=yes" \
    git ls-remote "git@github.com:$FLYBOT_REPO.git" HEAD >/dev/null 2>&1
}
git_token() {  # $1 = token file; the helper reads it at call time, the token is never on a command line
  # shellcheck disable=SC2016
  GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never git -c credential.helper= \
    -c "credential.helper=!f() { echo username=x-access-token; echo password=\$(cat '$1'); }; f" \
    ls-remote "https://github.com/$FLYBOT_REPO.git" HEAD >/dev/null 2>&1
}
step_repoaccess() {
  step repoaccess "read access for the EC2 to the private submodule $FLYBOT_REPO"
  local ks="flybot-$STAGE-deploykey" ts="flybot-$STAGE-repotoken" t=0 pub
  printf '%s\n' "$GH_KNOWN_HOSTS" > "$TMP/gh_known_hosts"
  if has FLYBOT_REPO_TOKEN; then
    py infra/deploy_helpers.py get "$ENV_FILE" FLYBOT_REPO_TOKEN --out "$TMP/tok"
    git_token "$TMP/tok" || die "FLYBOT_REPO_TOKEN cannot read $FLYBOT_REPO (fine-grained PAT: repository $FLYBOT_REPO, Contents read-only)"
    ok "FLYBOT_REPO_TOKEN reads $FLYBOT_REPO"
    secret_read "$ts" "$TMP/tok.cur"
    if cmp -s "$TMP/tok" "$TMP/tok.cur"; then ok "$ts unchanged"
    elif [ "$DRY" = 1 ]; then plan "write $ts"
    else secret_write "$ts" "$TMP/tok" "GitHub fine-grained token (contents read-only) for the private FlyBot submodule, managed by infra/deploy.sh" "$(secret_state "$ts")"
         ok "$ts written (read by the host at bootstrap, never kept in its .env)"; fi
    rm -f "$TMP"/tok*; REPO_ACCESS="token ($ts)"; return 0
  fi
  if [ "$(secret_state "$ks")" = active ]; then
    secret_read "$ks" "$TMP/dk"; chmod 600 "$TMP/dk"; ok "deploy key in $ks (reused)"
  elif [ "$DRY" = 1 ]; then
    plan "generate an ed25519 deploy key, store the private half in $ks, print the public half to add on GitHub"
    REPO_ACCESS="deploy key $ks (to create, NOT ADDED YET)"; return 0
  else
    ssh-keygen -q -t ed25519 -N "" -C "flybot-$STAGE-ec2" -f "$TMP/dk" || die "ssh-keygen failed"
    secret_write "$ks" "$TMP/dk" "Read-only GitHub deploy key of the FlyBot EC2 for $FLYBOT_REPO, managed by infra/deploy.sh" "$(secret_state "$ks")"
    ok "new deploy key stored in $ks"
  fi
  pub="$(ssh-keygen -y -f "$TMP/dk" 2>/dev/null | cut -d' ' -f1,2) flybot-$STAGE-ec2"
  if git_key "$TMP/dk"; then ok "deploy key reads $FLYBOT_REPO"; REPO_ACCESS="deploy key $ks (ok)"; rm -f "$TMP/dk"; return 0; fi
  box "ADD THIS READ-ONLY DEPLOY KEY ON GITHUB (the EC2 clones the private submodule with it)" \
      "github.com/$FLYBOT_REPO > Settings > Deploy keys > Add deploy key" \
      "Title: flybot-$STAGE-ec2      Allow write access: leave UNCHECKED" \
      "Key:" "$pub"
  if git_anon; then
    ok "$FLYBOT_REPO is still public: the EC2 clones it anonymously now; add the key before making the repo private"
    REPO_ACCESS="public repo; deploy key $ks NOT ADDED YET"; rm -f "$TMP/dk"; return 0
  fi
  if [ "$SKIP_WAIT" = 1 ]; then
    warn "deploy key not accepted yet (--skip-wait): the EC2 retries for 30 min at boot"
    REPO_ACCESS="deploy key $ks (NOT ADDED YET)"; rm -f "$TMP/dk"; return 0
  fi
  while ! git_key "$TMP/dk"; do
    if [ "$t" -ge $((REPO_WAIT_MIN * 60)) ]; then rm -f "$TMP/dk"; die "deploy key not accepted after $REPO_WAIT_MIN min: add it, then ./deploy.sh again (or --skip-wait)"; fi
    log "waiting for the deploy key on github.com/$FLYBOT_REPO ($((t / 60))/$REPO_WAIT_MIN min; --skip-wait to go on without)"
    sleep 20; t=$((t + 20))
  done
  ok "deploy key reads $FLYBOT_REPO"; REPO_ACCESS="deploy key $ks (ok)"; rm -f "$TMP/dk"
}

# ================================================================ stack
stack_params() {  # $1 = ImageUri
  local alarm_email=""
  if [ -z "$ALARM_TOPIC_ARN" ]; then alarm_email="$ALARM_EMAIL"; fi
  PARAMS=("Stage=$STAGE" "DomainName=$DOMAIN_NAME" "OwnerCidr=$OWNER_CIDR" "ImageUri=$1" "Detector=$DETECTOR"
          "PlannerMode=$PLANNER_MODE" "RepoUrl=$REPO_URL" "RepoRef=$REPO_REF" "AnthropicSecretName=$ANTHROPIC_SECRET_NAME"
          "AnthropicModel=$ANTHROPIC_MODEL" "IotEndpoint=$IOT_ENDPOINT" "AmiId=$AMI_ID" "AcmeEmail=$ACME_EMAIL"
          "InstanceType=$INSTANCE_TYPE" "KeyName=$KEY_NAME" "PhotosBucketExists=$PHOTOS_EXISTS"
          "AlarmEmail=$alarm_email" "ExistingAlarmTopicArn=$ALARM_TOPIC_ARN"
          "Yolo26Url=$YOLO26_URL" "Yolo26Sha256=$YOLO26_SHA"
          "DemoMaxCamerasPerUser=$DEMO_CAMS" "DemoMaxSnapshotsPerDay=$DEMO_SNAPS" "DemoMaxUnclaimed=$DEMO_UNCLAIMED"
          "DeviceEventsWebhookUrl=$DEVICE_EVENTS_URL" "DeviceEventsSecret=${DEVICE_EVENTS_SECRET:-}")
}
sam_deploy() {  # $1 = ImageUri
  stack_params "$1"
  local mask="${DEVICE_EVENTS_SECRET:-__none__}"
  "$SAM" deploy --template-file infra/template.yaml --stack-name "$STACK" --resolve-s3 --s3-prefix "$STACK" \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND --parameter-overrides "${PARAMS[@]}" \
    --tags project=flybot-aws "stage=$STAGE" --no-confirm-changeset --no-fail-on-empty-changeset --no-progressbar \
    --profile "$PROFILE" --region "$REGION" 2>&1 | tr -d '\r' | sed "s/$mask/********/g"
  return "${PIPESTATUS[0]}"
}
stack_wait_idle() {
  local s
  while :; do
    s="$(awsx cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo NONE)"
    case "$s" in *_IN_PROGRESS) log "stack $STACK $s, waiting"; sleep 20 ;; *) STACK_STATUS="$s"; return 0 ;; esac
  done
}
step_stack() {
  step stack "CloudFormation $STACK: bucket, ECR, IoT, roles, EC2 + EIP (Lambda kept as is)"
  stack_wait_idle
  case "$STACK_STATUS" in
    NONE|CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE|IMPORT_COMPLETE) ;;
    ROLLBACK_COMPLETE|ROLLBACK_FAILED|CREATE_FAILED|DELETE_FAILED)
      [ "$YES" = 1 ] || die "stack $STACK is $STACK_STATUS (a failed first create): it must be deleted. Re-run with --yes to delete and recreate it (the photos bucket is retained)"
      if [ "$DRY" = 1 ]; then plan "delete $STACK ($STACK_STATUS) and create it again"
      else log "deleting $STACK ($STACK_STATUS)"; awsx cloudformation delete-stack --stack-name "$STACK"
           awsx cloudformation wait stack-delete-complete --stack-name "$STACK"; STACK_STATUS=NONE; CUR_IMAGE=""; CUR_INSTANCE=""; fi ;;
    *) die "stack $STACK is $STACK_STATUS: fix it in the CloudFormation console (events tab), then re-run" ;;
  esac
  if [ "$DRY" = 1 ]; then
    stack_params "$CUR_IMAGE"
    plan "sam deploy $STACK ($([ "$STACK_STATUS" = NONE ] && echo create || echo update)) with:"
    printf '          %s\n' "${PARAMS[@]}" | sed 's/^\(          DeviceEventsSecret=\).\+/\1********/'
    return 0
  fi
  sam_deploy "$CUR_IMAGE" || die "sam deploy failed (CloudFormation console > $STACK > Events)"
  EIP="$(out ElasticIp)"; INSTANCE_ID="$(out InstanceId)"
  [ -n "$CUR_INSTANCE" ] && [ "$CUR_INSTANCE" = "$INSTANCE_ID" ] && INSTANCE_NEW=0 || INSTANCE_NEW=1
  ok "stack ready: instance $INSTANCE_ID ($([ "$INSTANCE_NEW" = 1 ] && echo new || echo existing)), EIP $EIP"
  if [ "$(doh_a 1.1.1.1)" = "$EIP" ]; then ok "$DOMAIN_NAME already resolves to $EIP"; else dns_box "$EIP"; fi
}

# ================================================================ image
image_lookup() {  # sets REPO and IMAGE_URI when the image of CTX_TAG is in ECR
  REPO="$(out VisionRepositoryUri)"; IMAGE_URI=""
  [ -n "$REPO" ] || return 0
  local digest; digest="$(awsx ecr describe-images --repository-name "${REPO##*/}" --image-ids "imageTag=$CTX_TAG" \
                          --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)"
  [[ "$digest" == sha256:* ]] && IMAGE_URI="$REPO@$digest"
  return 0
}
step_image() {
  step image "vision Lambda image $CTX_TAG (built on the FlyBot EC2 through SSM when not in ECR)"
  image_lookup
  if [ -n "$IMAGE_URI" ]; then ok "in ECR already: $IMAGE_URI"; return 0; fi
  if [ "$DRY" = 1 ]; then plan "build + push $CTX_TAG on the instance (10-15 min the first time; waits for the host bootstrap)"; return 0; fi
  [ -n "$REPO" ] || die "no ECR repository output: run the stack step first"
  TAG="$CTX_TAG" STAGE="$STAGE" bash infra/build-image.sh || die "image build failed: infra/.build/build-log-$STAGE.txt, ./deploy.sh --only image to retry"
  image_lookup
  [ -n "$IMAGE_URI" ] || die "image $CTX_TAG not found in ECR after the build"
  ok "image $IMAGE_URI"
}

# ================================================================ lambda
step_lambda() {
  step lambda "Lambda flybot-vision-$STAGE + alarms on the image"
  [ -n "${IMAGE_URI:-}" ] || image_lookup
  if [ -z "$IMAGE_URI" ]; then
    if [ "$DRY" = 1 ]; then plan "sam deploy $STACK with ImageUri=<$CTX_TAG digest> (after the image step)"; return 0; fi
    die "no image for $CTX_TAG in ECR: ./deploy.sh --only image"
  fi
  local cur; cur="$(param ImageUri)"
  if [ "$cur" = "$IMAGE_URI" ]; then ok "Lambda already on $CTX_TAG"; return 0; fi
  if [ "$DRY" = 1 ]; then plan "sam deploy $STACK with ImageUri=$IMAGE_URI"; return 0; fi
  sam_deploy "$IMAGE_URI" || die "sam deploy (Lambda) failed"
  ok "Lambda $(out VisionFunctionName) on $CTX_TAG"
}

# ================================================================ bootstrap
step_bootstrap() {
  step bootstrap "FlyBot on the EC2: docker compose up with the current .env and secrets"
  INSTANCE_ID="${INSTANCE_ID:-$(out InstanceId)}"
  if [ -z "$INSTANCE_ID" ]; then [ "$DRY" = 1 ] && { plan "wait for the first-boot bootstrap of the new instance"; return 0; }; die "no instance"; fi
  if [ "$DRY" = 1 ]; then plan "re-run the current user-data on $INSTANCE_ID through SSM (git pull, .env, secrets, compose up -d --build)"; return 0; fi
  ssm_wait_online 900 || die "SSM agent on $INSTANCE_ID not online after 15 min"
  if [ "${INSTANCE_NEW:-0}" = 1 ]; then
    log "first boot: waiting for 'bootstrap done' (docker install + FlyBot image build, ~10 min)"
    ssm_run "$BOOTSTRAP_WAIT" 1800 || die "first-boot bootstrap did not finish: ./deploy.sh --only bootstrap, or make host CMD='tail -50 /var/log/flybot-bootstrap.log'"
  else
    log "re-running the bootstrap (picks up secrets${SECRETS_CHANGED:+ (changed)}, template and git changes)"
    ssm_run "$BOOTSTRAP_RERUN" 1800 || die "bootstrap failed: make host CMD='tail -50 /var/log/flybot-bootstrap.log'"
  fi
  ok "FlyBot containers up: $(ssm_run 'cd /opt/flybot/flybot-aws && docker compose ps --format "{{.Service}}={{.State}}" | tr "\n" " "' 120 || echo '?')"
  if ssm_run 'test -f /opt/flybot/flybot-aws/flybot/package.json && git -C /opt/flybot/flybot-aws/flybot log -1 --format=%h' 120 > "$TMP/sub" 2>/dev/null; then
    ok "private submodule $FLYBOT_REPO checked out on the host at $(tr -d '\n' < "$TMP/sub")"
  else warn "no $FLYBOT_REPO checkout on the host: make host CMD='grep -i submodule /var/log/flybot-bootstrap.log | tail'"; fi
}

# ================================================================ health
step_health() {
  step health "DNS -> $DOMAIN_NAME, HTTPS certificate, FlyBot answering"
  EIP="${EIP:-$(out ElasticIp)}"
  if [ "$DRY" = 1 ]; then plan "poll 1.1.1.1 and 8.8.8.8 for $DOMAIN_NAME = ${EIP:-<EIP>} (up to $DNS_WAIT_MIN min), then https://$DOMAIN_NAME/health = 200"; return 0; fi
  [ -n "$EIP" ] || die "no Elastic IP output"
  local t=0 a b
  DNS_OK=0
  while [ "$t" -le $((DNS_WAIT_MIN * 60)) ]; do
    a="$(doh_a 1.1.1.1)"; b="$(doh_a 8.8.8.8)"
    if [ "$a" = "$EIP" ] && [ "$b" = "$EIP" ]; then DNS_OK=1; break; fi
    [ "$t" = 0 ] && dns_box "$EIP"
    log "DNS $DOMAIN_NAME: 1.1.1.1=${a:-none} 8.8.8.8=${b:-none}, want $EIP ($((t / 60))/$DNS_WAIT_MIN min)"
    sleep 30; t=$((t + 30))
  done
  if [ "$DNS_OK" = 0 ]; then warn "DNS not resolving to $EIP after $DNS_WAIT_MIN min: add the record, then ./deploy.sh --only health"
    row "DNS" "PENDING: $DOMAIN_NAME. A $EIP"; row "HTTPS" "not checked (DNS)"; return 0; fi
  ok "DNS $DOMAIN_NAME = $EIP on 1.1.1.1 and 8.8.8.8"; row "DNS" "$DOMAIN_NAME. A $EIP (ok)"
  local code="" i
  for i in $(seq 1 60); do   # Caddy asks Let's Encrypt on the first request once the name resolves
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 --resolve "$DOMAIN_NAME:443:$EIP" "https://$DOMAIN_NAME/health" || true)"
    [ "$code" = 200 ] && break
    [ $((i % 4)) = 1 ] && log "https://$DOMAIN_NAME/health -> ${code:-no answer} (certificate / FlyBot starting), retrying"
    sleep 15
  done
  if [ "$code" != 200 ]; then warn "https://$DOMAIN_NAME/health -> $code after 15 min: make host CMD='cd /opt/flybot/flybot-aws && docker compose logs --tail 50 caddy flybot'"
    row "HTTPS" "FAILED ($code)"; return 0; fi
  local home dev; home="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$DOMAIN_NAME/" || true)"
  dev="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$DOMAIN_NAME/api/device/time" || true)"
  ok "https://$DOMAIN_NAME valid certificate, /health 200, / $home, /api/device/time $dev (401 = device API on)"
  row "HTTPS" "https://$DOMAIN_NAME (certificate ok, /health 200, / $home)"
}

# ================================================================ smoke
step_smoke() {
  step smoke "one contract call to the Lambda with a sample trap photo"
  local fn bucket photo=data/samples/images/BE4089629/6ac327d44c3227d11cfbd084.jpg fe
  fn="$(out VisionFunctionName)"; bucket="$(out PhotosBucket)"
  if [ "$DRY" = 1 ]; then plan "upload $photo to s3://${bucket:-$PHOTOS}/samples/, invoke ${fn:-flybot-vision-$STAGE}, check the response contract"; return 0; fi
  [[ "$fn" == flybot-vision-* ]] || die "no Lambda yet: ./deploy.sh --only lambda"
  awsx s3 cp "$photo" "s3://$bucket/samples/smoke.jpg" --only-show-errors
  printf '{"v":1,"mode":"detect","photo":{"s3Bucket":"%s","s3Key":"samples/smoke.jpg"},"camera":{"id":"SMOKE","trapKind":"lure","platform":"ezviz"},"capturedAt":"2026-10-16T08:00:00Z","locale":"en"}' \
    "$bucket" > "infra/.build/smoke-req-$STAGE.json"
  fe="$(awsx lambda invoke --function-name "$fn" --payload "file://$(winpath "infra/.build/smoke-req-$STAGE.json")" \
        --cli-binary-format raw-in-base64-out --cli-read-timeout 180 --query FunctionError --output text "infra/.build/smoke-out-$STAGE.json")"
  local res; res="$(py infra/deploy_helpers.py contract "infra/.build/smoke-out-$STAGE.json" --detector "$DETECTOR")" || {
    printf '%s\n' "$res"; warn "Lambda smoke test failed (FunctionError ${fe:-none}): CloudWatch /aws/lambda/$fn"; row "Lambda smoke" "FAILED"; return 0; }
  ok "$res"; row "Lambda smoke" "${res#contract ok: }"
}

step_coldstart() {
  step coldstart "cold + warm latency (bench/results/lambda_latency.json)"
  if [ "$SKIP_COLDSTART" = 1 ]; then ok "skipped (SKIP_COLDSTART=1)"; return 0; fi
  if [ "$DRY" = 1 ]; then plan "infra/measure-lambda.sh: 1 forced cold start + 5 warm calls"; return 0; fi
  STAGE="$STAGE" bash infra/measure-lambda.sh >/dev/null || { warn "cold-start measurement failed"; return 0; }
  local l; l="$(py infra/deploy_helpers.py latency bench/results/lambda_latency.json)"; ok "$l"; row "Lambda latency" "$l"
}

step_summary() {
  step summary ""
  local eip fn; eip="$(out ElasticIp)"; fn="$(out VisionFunctionName)"
  local sha=""; [ "$DETECTOR" = yolo26 ] && sha=" ${YOLO26_SHA:0:12}"
  {
    printf '%-22s %s\n' "URL" "https://$DOMAIN_NAME"
    printf '%-22s %s\n' "Elastic IP" "${eip:-<not created>}"
    printf '%-22s %s\n' "Instance" "$(out InstanceId) ($INSTANCE_TYPE, AMI $AMI_ID)"
    printf '%-22s %s\n' "Lambda" "${fn:-<not deployed>} (image $CTX_TAG)"
    printf '%-22s %s\n' "Detector" "$DETECTOR$sha"
    printf '%-22s %s\n' "IoT endpoint" "$IOT_ENDPOINT"
    printf '%-22s %s\n' "KVS viewer role" "$(out KvsViewerRoleArn)"
    printf '%-22s %s\n' "Photos bucket" "$PHOTOS"
    printf '%-22s %s\n' "Secrets" "$APP_SECRET, $GCP_SECRET (chatbot ${CHATBOT:-?}), Anthropic: $ANTHROPIC_NOTE"
    printf '%-22s %s\n' "Alarms" "$ALARM_NOTE"
    printf '%-22s %s\n' "FlyBot repo access" "${REPO_ACCESS:-?}"
    [ ${#SUMMARY[@]} -gt 0 ] && printf '%s\n' "${SUMMARY[@]}"
    printf '%-22s %s\n' "Cost" "~24 USD/month at demo scale, EC2-dominated (infra/COST.md)"
    printf '%-22s %s\n' "Log" "$LOG"
    printf '\nLeft for a human:\n'
    printf '  - Register the first account at https://%s/auth/login: it becomes the admin (do it before sharing the URL).\n' "$DOMAIN_NAME"
    case " ${SUMMARY[*]:-} " in *PENDING*) printf '  - DNS: add  %s. A %s  (Namecheap > Advanced DNS), then ./deploy.sh --only health\n' "$DOMAIN_NAME" "$eip" ;; esac
    case "${REPO_ACCESS:-}" in *"NOT ADDED"*) printf '  - Add the deploy key printed above: github.com/%s > Settings > Deploy keys (read-only).\n' "$FLYBOT_REPO" ;; esac
    case "$ALARM_NOTE" in *CONFIRM*) printf '  - Click the confirmation link AWS SNS emailed to the alarm address.\n' ;; esac
  } | tee "infra/.build/deploy-$STAGE-summary.txt"
}

# ================================================================ main
preflight
if [ "$SHOW_IP" = 1 ]; then
  EIP="$(out ElasticIp)"; [ -n "$EIP" ] || die "stack $STACK has no Elastic IP (not deployed yet)"
  dns_box "$EIP"; a="$(doh_a 1.1.1.1)"; b="$(doh_a 8.8.8.8)"
  log "now: 1.1.1.1=${a:-none} 8.8.8.8=${b:-none}"; exit 0
fi
discover
INSTANCE_ID="$CUR_INSTANCE"; EIP="$CUR_EIP"; INSTANCE_NEW=0; SECRETS_CHANGED=0; CHATBOT=""
for s in "${STEPS[@]}"; do
  want "$s" || continue
  "step_$s"
done
[ "$DRY" = 1 ] && printf '\ndry run: nothing was changed. Run ./deploy.sh without --dry-run to apply.\n'
exit 0
