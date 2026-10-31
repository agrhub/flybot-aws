#!/usr/bin/env bash
# FlyBot on AWS in one command, driven by deploy.env (copy deploy.env.example, fill it once).
#
#   ./deploy.sh                      everything below, idempotent (re-run it after any change)
#   ./deploy.sh --dry-run            preflight + discovery for real (read-only AWS calls), then the plan; changes nothing
#   ./deploy.sh --only <step>        one step (preflight and discovery always run first)
#   ./deploy.sh --show-ip            the Elastic IP and the DNS record of the existing stack
#   ./deploy.sh --rollback           put the host back on the previous FlyBot release
#   ./deploy.sh --app                a new FlyBot release only, on an existing stack: build, publish, bootstrap, /health
#                                    (skips the infrastructure checks, the stack, the image and the Lambda)
#   ./deploy.sh --allow-dirty        build FlyBot from a checkout with uncommitted changes (the release is marked dirty)
#   ./deploy.sh --npm-ci             force `npm ci` in the FlyBot checkout before the build
#   ./deploy.sh --prune-releases     delete FlyBot deploy builds on GitHub beyond the newest FLYBOT_KEEP_RELEASES (5)
#   ./deploy.sh --yes                allow destructive recoveries (delete a stack stuck in ROLLBACK_COMPLETE)
#   ./deploy.sh --env-file F         another settings file (default: deploy.env)
#
# Steps: preflight, discover, models, secrets, stack, flybot, image, lambda, bootstrap, health, smoke, coldstart, summary.
# Log: infra/.build/deploy-<stage>.log (no secret value is ever printed). Exit 2 = AWS session expired (aws login).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=infra/lib.sh
. infra/lib.sh
find_tools

STEPS=(models secrets stack flybot image lambda bootstrap health smoke coldstart summary)
ENV_FILE=deploy.env DRY=0 ONLY="" YES=0 SHOW_IP=0 ROLLBACK=0 APP=0 ALLOW_DIRTY=0 NPM_CI=0 PRUNE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="${2:?--only needs a step: ${STEPS[*]}}"; shift ;;
    --only=*) ONLY="${1#--only=}" ;;
    --yes|-y) YES=1 ;;
    --show-ip) SHOW_IP=1 ;;
    --rollback) ROLLBACK=1 ;;
    --app) APP=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --npm-ci) NPM_CI=1 ;;
    --prune-releases) PRUNE=1 ;;
    --env-file) ENV_FILE="${2:?}"; shift ;;
    --env-file=*) ENV_FILE="${1#--env-file=}" ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
  shift
done
if [ -n "$ONLY" ] && [[ " ${STEPS[*]} " != *" $ONLY "* ]]; then die "--only $ONLY: not a step (${STEPS[*]})"; fi
if [ "$APP" = 1 ] && { [ -n "$ONLY" ] || [ "$ROLLBACK" = 1 ] || [ "$SHOW_IP" = 1 ]; }; then die "--app goes alone (not with --only, --rollback or --show-ip)"; fi

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
DETECTOR_TILES="$(cfg DETECTOR_TILES off)"
DETECTOR_ROI="$(cfg DETECTOR_ROI off)"
DETECTOR_TTA_LURE="$(cfg DETECTOR_TTA_LURE off)"
DETECTOR_TTA_STICKY="$(cfg DETECTOR_TTA_STICKY off)"
DETECTOR_MIN_SCORE_LURE="$(cfg DETECTOR_MIN_SCORE_LURE)"
PLANNER_MODE="$(cfg PLANNER_MODE rule)"
INSTANCE_TYPE="$(cfg INSTANCE_TYPE t4g.micro)"
KEY_NAME="$(cfg KEY_NAME)"
AMI_ID="$(cfg AMI_ID)"
ANTHROPIC_SECRET_NAME="$(cfg ANTHROPIC_SECRET_NAME sentinel-dev-anthropic)"
ANTHROPIC_MODEL="$(cfg ANTHROPIC_MODEL claude-haiku-5-5)"
GCP_FILE="$(cfg GCP_SERVICE_ACCOUNT_FILE)"
MODELS_BUCKET="$(cfg MODELS_BUCKET)"
DEMO_CAMS="$(cfg DEMO_MAX_CAMERAS_PER_USER 3)"
DEMO_SNAPS="$(cfg DEMO_MAX_SNAPSHOTS_PER_DAY 48)"
DEMO_UNCLAIMED="$(cfg DEMO_MAX_UNCLAIMED 200)"
APP_MODE="$(cfg APP_MODE hybrid)"
case "$APP_MODE" in b2c|b2b|hybrid) ;; *) die "deploy.env: APP_MODE must be b2c, b2b or hybrid (got $APP_MODE)" ;; esac
PAYMENTS_ENABLED="$(cfg PAYMENTS_ENABLED false)"
case "$PAYMENTS_ENABLED" in true|false) ;; *) die "deploy.env: PAYMENTS_ENABLED must be true or false (got $PAYMENTS_ENABLED)" ;; esac
if [ "$PAYMENTS_ENABLED" = true ] && [ "$APP_MODE" = b2b ]; then
  die "deploy.env: PAYMENTS_ENABLED=true is not allowed with APP_MODE=b2b (FlyBot refuses online payment for organizations)"
fi
DEVICE_EVENTS_URL="$(cfg DEVICE_EVENTS_URL)"
SKIP_COLDSTART="$(cfg SKIP_COLDSTART 0)"
FLYBOT_SRC="$(cfg FLYBOT_SRC ../ezviz-manager)"
VITE_GOOGLE_MAP_KEY="$(cfg VITE_GOOGLE_MAP_KEY)"
S3_BUCKET_NAME="$(cfg S3_BUCKET_NAME)"
S3_REGION="$(cfg S3_REGION)"
S3_ENDPOINT="$(cfg S3_ENDPOINT)"
NODE_VERSION="$(cfg NODE_VERSION)"
CADDY_VERSION="$(cfg CADDY_VERSION 2.10.2)"
RELEASE_STORE="$(cfg FLYBOT_RELEASE_STORE github)"
case "$RELEASE_STORE" in github|s3) ;; *) die "deploy.env: FLYBOT_RELEASE_STORE must be github or s3 (got $RELEASE_STORE)" ;; esac
GITHUB_REPO="$(cfg GITHUB_REPO agrhub/flybot-aws)"
FLYBOT_MEDIA="$(cfg FLYBOT_MEDIA separate)"
case "$FLYBOT_MEDIA" in separate|bundled) ;; *) die "deploy.env: FLYBOT_MEDIA must be separate or bundled (got $FLYBOT_MEDIA)" ;; esac
KEEP_RELEASES="$(cfg FLYBOT_KEEP_RELEASES 5)"
# photos: stack bucket (empty S3_BUCKET_NAME), an AWS bucket by name, or S3-compatible storage at S3_ENDPOINT
if [ -z "$S3_BUCKET_NAME" ]; then STORAGE_MODE=stack
elif [ -n "$S3_ENDPOINT" ] && [[ "$S3_ENDPOINT" != *amazonaws.com* ]]; then STORAGE_MODE=external
else STORAGE_MODE=aws; S3_ENDPOINT=""; fi
# MongoDB: MONGO_URI (secret, deploy.env) = an external database; empty = MongoDB 7 on the instance
if has MONGO_URI; then LOCAL_MONGO=false; else LOCAL_MONGO=true; fi
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
  "$([ "$DRY" = 1 ] && echo '--dry-run ')$([ -n "$ONLY" ] && echo "--only $ONLY")$([ "$ROLLBACK" = 1 ] && echo '--rollback')$([ "$APP" = 1 ] && echo '--app')" "$STAGE" "$ENV_FILE"

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
box() {  # box TITLE LINE...: a highlighted block that asks for a manual action
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
  box "DNS RECORD TO ADD (the deploy does not wait for it)" "$where" \
      "Type: A Record    Host: $DNS_HOST    Value: $1    TTL: Automatic (300)" "= $DOMAIN_NAME.  A  $1"
}
doh_a() {  # public resolver ($1 = 1.1.1.1 | 8.8.8.8) -> the A records of DOMAIN_NAME, space separated
  local body=""
  case "$1" in
    1.1.1.1) body="$(curl -s --max-time 8 -H 'accept: application/dns-json' "https://1.1.1.1/dns-query?name=$DOMAIN_NAME&type=A" || true)" ;;
    8.8.8.8) body="$(curl -s --max-time 8 "https://8.8.8.8/resolve?name=$DOMAIN_NAME&type=A" || true)" ;;
  esac
  printf '%s' "$body" | grep -o '"data":"[0-9][0-9.]*"' | grep -o '[0-9][0-9.]*' | tr '\n' ' ' | sed 's/ $//'
}

# ---------------------------------------------------------------- GitHub REST API (FlyBot releases)
# The token stays on this computer: it is read from deploy.env into a header file in the private temp folder (never on a
# command line, never printed, never sent to the server). CURL can point at a stub (infra/tests).
gh_api() {  # METHOD URL [data-file [content-type]] -> prints the HTTP status; the body is in $TMP/gh.out
  local m="$1" u="$2" d="${3:-}" ct="${4:-application/json}" code
  local args=(-sS -o "$TMP/gh.out" -w '%{http_code}' -X "$m" -H "@$(winpath "$TMP/gh.hdr")"
              -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2022-11-28')
  [ -n "$d" ] && args+=(-H "Content-Type: $ct" --data-binary "@$(winpath "$d")")
  code="$("${CURL:-curl}" "${args[@]}" "$u" 2>/dev/null)" || code=000
  printf '%s' "${code:-000}"
}
gh_anon_status() {  # URL -> HTTP status of an anonymous download (redirects followed)
  "${CURL:-curl}" -sIL -o /dev/null -w '%{http_code}' --max-time 30 "$1" 2>/dev/null || printf '000'
}

# ================================================================ preflight
preflight() {
  step preflight "tools, session, account, region, service limits, FlyBot checkout"
  [ -n "$AWS_BIN" ] || die "AWS CLI v2 not found (install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)"
  if [ -z "$SAM" ] && [ "$SHOW_IP" != 1 ] && [ "$ROLLBACK" != 1 ]; then
    printf 'SAM CLI candidates that did not run:\n%s' "$SAM_TRIED" >&2
    die "no working SAM CLI; install it from your own terminal (not from an app sandbox): uv tool install --force aws-sam-cli, or the MSI from https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"
  fi
  ok "aws  $("$AWS" --version 2>&1 | head -1 | cut -d' ' -f1)  ($AWS_BIN)"
  [ -n "$SAM" ] && ok "sam  $("$SAM" --version 2>&1 | tr -d '\r' | head -1)  ($SAM)"
  py -c 1 || die "no usable Python 3.9+"
  ok "python $(py -c 'import sys;print(sys.version.split()[0])')  (${PY_CMD[*]})"
  command -v curl >/dev/null || die "curl not found"
  command -v openssl >/dev/null || warn "openssl not found: VAPID keys cannot be generated (web push off)"
  [ "$REGION" = ap-southeast-2 ] || die "AWS_REGION=$REGION refused: the account's SCP allows ap-southeast-2 only"
  aws_or_die
  ACCOUNT="$(awsx sts get-caller-identity --query Account --output text)"
  ok "session ok: account $ACCOUNT ($(awsx sts get-caller-identity --query Arn --output text | sed 's#.*:assumed-role/##; s#/.*##'), profile $PROFILE)"
  [ "$SHOW_IP" = 1 ] || [ "$ROLLBACK" = 1 ] && return 0

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

  # the FlyBot checkout this laptop builds (want flybot / --only flybot)
  if want flybot; then
    find_node
    [ -n "$NODE_BIN" ] || { printf 'node candidates rejected:\n%s' "$NODE_TRIED" >&2; die "Node.js 22.12+ with npm not found (nvm install 22, or https://nodejs.org)"; }
    ok "node $("$NODE_BIN" --version | tr -d '\r'), npm $("$NPM_BIN" --version 2>/dev/null | tr -d '\r')  ($NODE_BIN)"
    case "$FLYBOT_SRC" in /*|[A-Za-z]:*) ;; *) FLYBOT_SRC="$ROOT/$FLYBOT_SRC" ;; esac
    [ -d "$FLYBOT_SRC" ] && FLYBOT_SRC="$(cd "$FLYBOT_SRC" && pwd)"
    [ -f "$FLYBOT_SRC/package.json" ] && grep -q '"name": *"flybot"' "$FLYBOT_SRC/package.json" \
      || die "FLYBOT_SRC=$FLYBOT_SRC is not a FlyBot checkout (package.json name flybot)"
    FB_COMMIT="$(git -C "$FLYBOT_SRC" rev-parse --short=12 HEAD)"; FB_BRANCH="$(git -C "$FLYBOT_SRC" rev-parse --abbrev-ref HEAD)"
    FB_DIRTY="$(git -C "$FLYBOT_SRC" status --porcelain --untracked-files=no | head -20)"
    if [ -n "$FB_DIRTY" ] && [ "$ALLOW_DIRTY" != 1 ]; then
      printf '%s\n' "$FB_DIRTY" | sed 's/^/      /' >&2
      die "FlyBot checkout $FLYBOT_SRC has uncommitted changes (above): commit them, or --allow-dirty for a test release"
    fi
    FB_ID="$FB_COMMIT"; [ -n "$FB_DIRTY" ] && FB_ID="$FB_COMMIT-dirty-$(git -C "$FLYBOT_SRC" diff HEAD | sha256sum | cut -c1-8)"
    ok "FlyBot checkout $FLYBOT_SRC: $FB_BRANCH@$FB_COMMIT$([ -n "$FB_DIRTY" ] && echo ' (DIRTY, --allow-dirty)')"
    if [ "$RELEASE_STORE" = github ] && ! has GITHUB_TOKEN; then
      warn "GITHUB_TOKEN is empty in $ENV_FILE: the FlyBot release goes to the artifacts bucket instead (FLYBOT_RELEASE_STORE=s3)"
      RELEASE_STORE=s3
    fi
    if [ "$RELEASE_STORE" = github ]; then
      py infra/deploy_helpers.py get "$ENV_FILE" GITHUB_TOKEN --out "$TMP/ghtok"
      py -c 'import sys;t=open(sys.argv[1],encoding="utf-8").read().strip();open(sys.argv[2],"w").write("Authorization: Bearer "+t+"\n")' "$TMP/ghtok" "$TMP/gh.hdr"
      rm -f "$TMP/ghtok"
      local gc; gc="$(gh_api GET "https://api.github.com/repos/$GITHUB_REPO")"
      [ "$gc" = 200 ] || die "GITHUB_TOKEN cannot read $GITHUB_REPO (HTTP $gc): a fine-grained token for repository $GITHUB_REPO with Contents: read and write"
      [ "$(py -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8")).get("private"))' "$TMP/gh.out")" = False ] \
        || die "$GITHUB_REPO is private: the server downloads release assets anonymously; make it public or set FLYBOT_RELEASE_STORE=s3"
      ok "FlyBot releases on GitHub $GITHUB_REPO (token ok, public repository)"
    else
      ok "FlyBot releases in the artifacts bucket (FLYBOT_RELEASE_STORE=s3)"
    fi
  fi
  # --app: the stack exists already; the checks below are for creating or changing it
  [ "$APP" = 1 ] && return 0

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
CodeBuild|codebuild list-projects
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
  # photo storage given in deploy.env
  case "$STORAGE_MODE" in
    aws)
      awsx s3api head-bucket --bucket "$S3_BUCKET_NAME" >/dev/null 2>&1 || die "S3_BUCKET_NAME=$S3_BUCKET_NAME: not found or not accessible with profile $PROFILE"
      local loc; loc="$(awsx s3api get-bucket-location --bucket "$S3_BUCKET_NAME" --query LocationConstraint --output text 2>/dev/null || true)"
      [ "$loc" = None ] && loc=us-east-1
      [ -z "$S3_REGION" ] && S3_REGION="$loc"
      [ "$loc" = "$REGION" ] || warn "bucket $S3_BUCKET_NAME is in $loc, the Lambda in $REGION: cross-region reads add latency and transfer cost"
      ok "photos: AWS bucket $S3_BUCKET_NAME ($loc), read/write granted to the instance role and the Lambda" ;;
    external)
      { has S3_ACCESS_KEY && has S3_SECRET_KEY; } || die "S3_ENDPOINT=$S3_ENDPOINT needs S3_ACCESS_KEY and S3_SECRET_KEY in deploy.env"
      [ -n "$S3_REGION" ] || S3_REGION=auto
      case "$S3_ENDPOINT" in http://*|https://*) ;; *) die "S3_ENDPOINT=$S3_ENDPOINT: needs http:// or https://" ;; esac
      local s3host="${S3_ENDPOINT#*://}"; s3host="${s3host%%/*}"; s3host="${s3host%:*}"
      case "$s3host" in localhost|127.*|10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*|*.local|*.lan)
        die "S3_ENDPOINT host $s3host is a private address: the Lambda and the server reach it over the internet" ;; esac
      local s3code; s3code="$(${CURL:-curl} -s -o /dev/null -w '%{http_code}' --max-time 10 "$S3_ENDPOINT/" 2>/dev/null || true)"
      if [ -z "$s3code" ] || [ "$s3code" = 000 ]; then warn "S3_ENDPOINT $S3_ENDPOINT does not answer from this computer: the Lambda and the server must reach it from the internet"
      else ok "S3_ENDPOINT answers (HTTP $s3code without credentials)"; fi
      [[ "$S3_ENDPOINT" == http://* ]] && warn "S3_ENDPOINT is plain http: photos, and the presigned URLs FlyBot gives the Lambda, travel unencrypted between AWS and $s3host"
      ok "photos: S3-compatible $S3_ENDPOINT bucket $S3_BUCKET_NAME (the Lambda gets presigned URLs and stores no picture)" ;;
    *) ok "photos: the stack bucket flybot-photos-$ACCOUNT-$STAGE" ;;
  esac
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
  local cur_type; cur_type="$(param InstanceType)"
  if [ -n "$cur_type" ] && [ "$cur_type" != "$INSTANCE_TYPE" ]; then
    warn "instance type $cur_type -> $INSTANCE_TYPE: CloudFormation stops and starts the instance (Elastic IP and disk stay); the bootstrap step re-runs afterwards"
  fi
  if [ "$(param LocalMongo)" = true ] && [ "$LOCAL_MONGO" = false ]; then
    warn "database: the server's MongoDB -> MONGO_URI. The local MongoDB is stopped; its data (/var/lib/mongo on the instance) is NOT copied to the new database (copy it first with mongodump/mongorestore if you need it)"
  fi

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
  # this stack's own topic (made by an earlier deploy): its subscription may be confirmed already
  local own own_state=""
  own="$(awsx sns list-topics --query 'Topics[].TopicArn' --output text | tr '\t' '\n' | grep ":flybot-$STAGE-alarms\$" || true)"
  if [ -z "$pick" ] && [ -n "$own" ] && [ -n "$ALARM_EMAIL" ]; then
    own_state="$(awsx sns list-subscriptions-by-topic --topic-arn "$own" \
                 --query "Subscriptions[?Protocol=='email'].[Endpoint,SubscriptionArn]" --output text 2>/dev/null || true)"
    own_state="$(printf '%s\n' "$own_state" | awk -v w="${ALARM_EMAIL,,}" 'tolower($1) == w { print ($2 == "PendingConfirmation") ? "pending" : "confirmed"; exit }')"
  fi
  if [ -n "$pick" ]; then ALARM_TOPIC_ARN="$pick"; ALARM_NOTE="reuse ${pick##*:} (confirmed email)"
  elif [ "$own_state" = confirmed ]; then ALARM_NOTE="topic flybot-$STAGE-alarms, $ALARM_EMAIL confirmed"
  elif [ "$own_state" = pending ]; then ALARM_NOTE="topic flybot-$STAGE-alarms; CONFIRM the email AWS sent to $ALARM_EMAIL (subscription pending)"
  elif [ -n "$ALARM_EMAIL" ]; then ALARM_NOTE="new topic flybot-$STAGE-alarms; CONFIRM the email AWS sends to the alarm address"
  else ALARM_NOTE="new topic flybot-$STAGE-alarms without subscriber (set ALARM_EMAIL to get alarms)"; fi
  ok "alarms: $ALARM_NOTE"

  PHOTOS="flybot-photos-$ACCOUNT-$STAGE"; PHOTOS_EXISTS=false
  ARTIFACTS="flybot-artifacts-$ACCOUNT-$STAGE"
  if [ "$STORAGE_MODE" = aws ]; then
    ok "photos bucket $S3_BUCKET_NAME (yours; the stack creates no photo bucket)"
  elif awsx cloudformation describe-stack-resource --stack-name "$STACK" --logical-resource-id PhotosBucket >/dev/null 2>&1; then
    ok "stack bucket $PHOTOS (owned by the stack$([ "$STORAGE_MODE" = external ] && echo ', for requested evidence only'))"
  elif awsx s3api head-bucket --bucket "$PHOTOS" >/dev/null 2>&1; then
    PHOTOS_EXISTS=true; ok "stack bucket $PHOTOS exists outside the stack (kept by destroy --keep-photos): reused"
  else ok "stack bucket $PHOTOS: will be created$([ "$STORAGE_MODE" = external ] && echo ' (for requested evidence only)')"; fi
  if [ "$LOCAL_MONGO" = true ]; then MONGO_NOTE="MongoDB 7 on the instance (cache 0.25 GB, data on its EBS volume)"
  else MONGO_NOTE="external (MONGO_URI in $APP_SECRET); no MongoDB on the instance"; fi
  ok "database: $MONGO_NOTE"

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

# ================================================================ stack
stack_params() {  # $1 = ImageUri
  local alarm_email=""
  if [ -z "$ALARM_TOPIC_ARN" ]; then alarm_email="$ALARM_EMAIL"; fi
  PARAMS=("Stage=$STAGE" "DomainName=$DOMAIN_NAME" "OwnerCidr=$OWNER_CIDR" "ImageUri=$1" "Detector=$DETECTOR"
          "DetectorTiles=$DETECTOR_TILES" "DetectorRoi=$DETECTOR_ROI" "DetectorTtaLure=$DETECTOR_TTA_LURE"
          "DetectorTtaSticky=$DETECTOR_TTA_STICKY" "DetectorMinScoreLure=$DETECTOR_MIN_SCORE_LURE"
          "PlannerMode=$PLANNER_MODE" "AnthropicSecretName=$ANTHROPIC_SECRET_NAME"
          "AnthropicModel=$ANTHROPIC_MODEL" "IotEndpoint=$IOT_ENDPOINT" "AmiId=$AMI_ID" "AcmeEmail=$ACME_EMAIL"
          "InstanceType=$INSTANCE_TYPE" "KeyName=$KEY_NAME" "PhotosBucketExists=$PHOTOS_EXISTS"
          "StorageMode=$STORAGE_MODE" "S3BucketName=$S3_BUCKET_NAME" "S3Region=$S3_REGION" "S3Endpoint=$S3_ENDPOINT"
          "LocalMongo=$LOCAL_MONGO" "NodeVersion=$NODE_VERSION" "CaddyVersion=$CADDY_VERSION"
          "AlarmEmail=$alarm_email" "ExistingAlarmTopicArn=$ALARM_TOPIC_ARN"
          "Yolo26Url=$YOLO26_URL" "Yolo26Sha256=$YOLO26_SHA"
          "DemoMaxCamerasPerUser=$DEMO_CAMS" "DemoMaxSnapshotsPerDay=$DEMO_SNAPS" "DemoMaxUnclaimed=$DEMO_UNCLAIMED" "AppMode=$APP_MODE"
          "DeviceEventsWebhookUrl=$DEVICE_EVENTS_URL" "DeviceEventsSecret=${DEVICE_EVENTS_SECRET:-}")
  # SAM CLI rejects "Key=" with an empty value; an omitted parameter takes the template default ("").
  local p kept=()
  for p in "${PARAMS[@]}"; do [ -n "${p#*=}" ] && kept+=("$p"); done
  PARAMS=("${kept[@]}")
}
sam_deploy() {  # $1 = ImageUri
  stack_params "$1"
  local mask="${DEVICE_EVENTS_SECRET:-__none__}"
  "$SAM" deploy --template-file infra/template.yaml --stack-name "$STACK" --resolve-s3 --s3-prefix "$STACK" \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND --parameter-overrides "${PARAMS[@]}" \
    --image-repository "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/flybot-vision-$STAGE" \
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
  step stack "CloudFormation $STACK: buckets, ECR, CodeBuild, IoT, roles, EC2 + EIP (Lambda kept as is)"
  stack_wait_idle
  case "$STACK_STATUS" in
    NONE|CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE|IMPORT_COMPLETE) ;;
    ROLLBACK_COMPLETE|ROLLBACK_FAILED|CREATE_FAILED|DELETE_FAILED)
      [ "$YES" = 1 ] || die "stack $STACK is $STACK_STATUS (a failed first create): it must be deleted. Re-run with --yes to delete and recreate it (the photos bucket is retained)"
      if [ "$DRY" = 1 ]; then plan "delete $STACK ($STACK_STATUS) and create it again"
      else log "deleting $STACK ($STACK_STATUS)"
           awsx s3 rm "s3://$ARTIFACTS" --recursive --only-show-errors 2>/dev/null || true
           awsx cloudformation delete-stack --stack-name "$STACK"
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
  ok "stack ready: instance $INSTANCE_ID ($([ "$INSTANCE_NEW" = 1 ] && echo new || echo existing), $INSTANCE_TYPE), EIP $EIP"
  if [ "$(doh_a 1.1.1.1)" = "$EIP" ]; then ok "$DOMAIN_NAME already resolves to $EIP"; else dns_box "$EIP"; fi
}

# ================================================================ flybot (prebuilt release, built on this laptop)
flybot_current() {  # the release the host is told to run: "<url or key> <sha256>" or empty
  awsx s3 cp "s3://$ARTIFACTS/flybot/current.json" - 2>/dev/null \
    | py -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d.get("url") or d.get("key"), d["sha256"])
except Exception:
    pass' 2>/dev/null || true
}
flybot_build() {  # npm ci (when needed) + npm run build in FLYBOT_SRC, then the release tarball; prints its sha256
  local tgz="$1" free
  free="$(df -Pk "$FLYBOT_SRC" | awk 'NR==2 {print int($4/1024)}')"
  [ "$free" -ge 1500 ] || warn "only $free MB free on the FlyBot checkout's drive (npm ci needs ~1.5 GB)"
  if [ "$NPM_CI" = 1 ] || [ ! -f "$FLYBOT_SRC/node_modules/.package-lock.json" ] || [ "$FLYBOT_SRC/package-lock.json" -nt "$FLYBOT_SRC/node_modules/.package-lock.json" ]; then
    log "npm ci in $FLYBOT_SRC (package-lock.json changed or node_modules missing)" >&2
    (cd "$FLYBOT_SRC" && "$NPM_BIN" ci --no-audit --no-fund --loglevel=error) >&2 || die "npm ci failed in $FLYBOT_SRC"
  fi
  log "npm run build (Vue client + server bundle)" >&2
  (cd "$FLYBOT_SRC" && VITE_GOOGLE_MAP_KEY="$VITE_GOOGLE_MAP_KEY" NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=4096}" "$NPM_BIN" run build) \
    > "infra/.build/flybot-build-$STAGE.log" 2>&1 || { tail -30 "infra/.build/flybot-build-$STAGE.log" >&2; die "FlyBot build failed (infra/.build/flybot-build-$STAGE.log)"; }
  printf '{"commit":"%s","branch":"%s","dirty":%s,"built_at":"%s","node":"%s","media":"%s"}\n' "$FB_COMMIT" "$FB_BRANCH" \
    "$([ -n "$FB_DIRTY" ] && echo true || echo false)" "$(date -u +%FT%TZ)" "$("$NODE_BIN" --version | tr -d '\r')" "$FLYBOT_MEDIA" > "$TMP/release.json"
  local extra=(); [ "$FLYBOT_MEDIA" = separate ] && extra=(--exclude-media)
  py infra/deploy_helpers.py pack-flybot --src "$FLYBOT_SRC" --out "$tgz" --meta "$TMP/release.json" "${extra[@]}" \
    || die "packing the release failed"
}
flybot_media() {  # FLYBOT_MEDIA=separate: the videos once to s3://<artifacts>/media/ (sha256 metadata), synced by the host
  [ "$FLYBOT_MEDIA" = separate ] || return 0
  local sha size rel key remote n=0 up=0 todo=0
  [ -d "$FLYBOT_SRC/client/dist/media" ] || { warn "no $FLYBOT_SRC/client/dist/media: videos not published (build FlyBot once)"; return 0; }
  while read -r sha size rel; do
    [ -n "$rel" ] || continue
    n=$((n + 1)); key="$rel"
    remote="$(awsx s3api head-object --bucket "$ARTIFACTS" --key "$key" --query 'Metadata.sha256' --output text 2>/dev/null || true)"
    [ "$remote" = "$sha" ] && continue
    if [ "$DRY" = 1 ]; then plan "upload $rel ($((size / 1048576)) MB) -> s3://$ARTIFACTS/$key"; todo=$((todo + 1)); continue; fi
    awsx s3 cp "$FLYBOT_SRC/client/dist/$rel" "s3://$ARTIFACTS/$key" --metadata "sha256=$sha" --content-type video/mp4 --only-show-errors
    up=$((up + 1))
  done < <(py infra/deploy_helpers.py media-list --src "$FLYBOT_SRC" | tr -d '\r')
  if [ "$DRY" = 1 ]; then ok "videos: $n, $((n - todo)) already in s3://$ARTIFACTS/media/, $todo to upload; served by Caddy from the server's local copy"
  else ok "videos: $n in s3://$ARTIFACTS/media/ ($up uploaded now), served by Caddy from the server's local copy"; fi
}
flybot_github() {  # TGZ-or-empty: find or create the pre-release + its assets; sets SHA (and RELEASE_URL)
  local tgz="$1" tag="flybot-$FB_ID" asset="flybot-$FB_ID.tgz" code rel_id="" line
  local api="https://api.github.com/repos/$GITHUB_REPO"
  RELEASE_URL="https://github.com/$GITHUB_REPO/releases/download/$tag/$asset"
  code="$(gh_api GET "$api/releases/tags/$tag")"
  if [ "$code" = 200 ]; then rel_id="$(py infra/deploy_helpers.py gh-release "$TMP/gh.out" | awk '$1=="id" {print $2}')"
  elif [ "$code" != 404 ]; then die "GitHub API $code reading release $tag of $GITHUB_REPO"; fi
  if [ -z "$tgz" ]; then   # look-up only: reuse a complete release
    SHA=""
    [ -n "$rel_id" ] || return 0
    py infra/deploy_helpers.py gh-release "$TMP/gh.out" | grep -q "^asset $asset [0-9]* uploaded " || return 0
    py infra/deploy_helpers.py gh-release "$TMP/gh.out" | grep -q "^asset $asset.sha256 [0-9]* uploaded " || return 0
    SHA="$("${CURL:-curl}" -fsSL --max-time 30 "$RELEASE_URL.sha256" 2>/dev/null | awk '{print $1}' || true)"
    [[ "$SHA" =~ ^[0-9a-f]{64}$ ]] || SHA=""
    return 0
  fi
  local size_mb=$(( $(wc -c < "$tgz") / 1048576 ))
  if [ -z "$rel_id" ]; then
    py infra/deploy_helpers.py gh-new-release --tag "$tag" --commit "$FB_ID" --branch "$FB_BRANCH" --built "$(date -u +%FT%TZ)" \
      --asset "$asset" --sha "$SHA" --size-mb "$size_mb" --out "$TMP/gh-new.json"
    code="$(gh_api POST "$api/releases" "$TMP/gh-new.json")"
    [ "$code" = 201 ] || die "GitHub API $code creating release $tag (token needs Contents: read and write on $GITHUB_REPO): $(head -c 300 "$TMP/gh.out")"
    rel_id="$(py infra/deploy_helpers.py gh-release "$TMP/gh.out" | awk '$1=="id" {print $2}')"
    ok "GitHub pre-release $tag created in $GITHUB_REPO"
  fi
  # a half-uploaded asset of an earlier run blocks a new upload with the same name
  while read -r line; do
    set -- $line
    case "$2" in "$asset"|"$asset.sha256") gh_api DELETE "$api/releases/assets/$3" >/dev/null ;; esac
  done < <(py infra/deploy_helpers.py gh-release "$TMP/gh.out" | grep '^asset ' || true)
  code="$(gh_api POST "https://uploads.github.com/repos/$GITHUB_REPO/releases/$rel_id/assets?name=$asset" "$tgz" application/gzip)"
  [ "$code" = 201 ] || die "GitHub API $code uploading $asset: $(head -c 300 "$TMP/gh.out")"
  printf '%s  %s\n' "$SHA" "$asset" > "$TMP/$asset.sha256"
  code="$(gh_api POST "https://uploads.github.com/repos/$GITHUB_REPO/releases/$rel_id/assets?name=$asset.sha256" "$TMP/$asset.sha256" text/plain)"
  [ "$code" = 201 ] || die "GitHub API $code uploading $asset.sha256: $(head -c 300 "$TMP/gh.out")"
  ok "uploaded $asset ($size_mb MB) + $asset.sha256 to the GitHub release $tag"
}
flybot_prune() {  # --prune-releases: keep the newest KEEP_RELEASES deploy builds (and the current one)
  [ "$PRUNE" = 1 ] || return 0
  [ "$RELEASE_STORE" = github ] || { warn "--prune-releases applies to GitHub releases only"; return 0; }
  local code id tag n=0
  code="$(gh_api GET "https://api.github.com/repos/$GITHUB_REPO/releases?per_page=100")"
  [ "$code" = 200 ] || { warn "GitHub API $code listing releases: nothing pruned"; return 0; }
  while read -r id tag; do
    [ -n "$id" ] || continue
    n=$((n + 1))
    if [ "$DRY" = 1 ]; then plan "delete the GitHub deploy build $tag (release $id and its tag)"; continue; fi
    code="$(gh_api DELETE "https://api.github.com/repos/$GITHUB_REPO/releases/$id")"
    [ "$code" = 204 ] || { warn "GitHub API $code deleting release $tag"; continue; }
    gh_api DELETE "https://api.github.com/repos/$GITHUB_REPO/git/refs/tags/$tag" >/dev/null
    ok "deleted the deploy build $tag"
  done < <(py infra/deploy_helpers.py gh-prune "$TMP/gh.out" --keep "$KEEP_RELEASES" --current "flybot-$FB_ID")
  [ "$n" = 0 ] && ok "GitHub deploy builds: nothing to prune (keeping $KEEP_RELEASES)"
  return 0
}
step_flybot() {
  local where="the GitHub release flybot-$FB_ID of $GITHUB_REPO"; [ "$RELEASE_STORE" = s3 ] && where="s3://$ARTIFACTS/flybot/"
  step flybot "FlyBot $FB_BRANCH@$FB_ID: npm build here, release to $where (no build on the server)"
  local key="flybot/flybot-$FB_ID.tgz" tgz="infra/.build/flybot-$FB_ID.tgz" ref
  SHA="" RELEASE_URL=""
  if ! awsx s3api head-bucket --bucket "$ARTIFACTS" >/dev/null 2>&1 && [ "$DRY" != 1 ]; then die "artifacts bucket $ARTIFACTS missing: ./deploy.sh --only stack first"; fi
  if [ "$RELEASE_STORE" = github ]; then flybot_github ""
  else
    local meta; meta="$(awsx s3api head-object --bucket "$ARTIFACTS" --key "$key" --query '[Metadata.sha256, Metadata.media]' --output text 2>/dev/null || true)"
    read -r SHA ref <<< "$meta"
    [ "$SHA" = None ] && SHA=""
    if [ -n "$SHA" ] && [ "$ref" != "$FLYBOT_MEDIA" ]; then
      ok "s3://$ARTIFACTS/$key was packed with videos ${ref/None/bundled}, deploy.env says $FLYBOT_MEDIA: rebuilding"; SHA=""
    fi
  fi
  if [ -n "$SHA" ]; then
    ok "release flybot-$FB_ID already published (sha256 ${SHA:0:12}): no build"
  elif [ "$DRY" = 1 ]; then
    plan "npm $( { [ "$NPM_CI" = 1 ] || [ ! -f "$FLYBOT_SRC/node_modules/.package-lock.json" ] || [ "$FLYBOT_SRC/package-lock.json" -nt "$FLYBOT_SRC/node_modules/.package-lock.json" ]; } && echo 'ci + ')run build in $FLYBOT_SRC, pack flybot-$FB_ID.tgz (videos $FLYBOT_MEDIA)"
    if [ "$RELEASE_STORE" = github ]; then
      plan "create the GitHub pre-release flybot-$FB_ID \"FlyBot $FB_ID (deploy build)\" in $GITHUB_REPO, upload flybot-$FB_ID.tgz + .sha256"
    else plan "upload $key to s3://$ARTIFACTS"; fi
  else
    SHA="$(flybot_build "$tgz" | tail -1)"
    [[ "$SHA" =~ ^[0-9a-f]{64}$ ]] || die "no sha256 from the release packer"
    if [ "$RELEASE_STORE" = github ]; then flybot_github "$tgz"
    else awsx s3 cp "$tgz" "s3://$ARTIFACTS/$key" --metadata "sha256=$SHA,commit=$FB_COMMIT,media=$FLYBOT_MEDIA" --only-show-errors
         ok "uploaded $key ($(( $(wc -c < "$tgz") / 1048576 )) MB, sha256 ${SHA:0:12})"; fi
    rm -f "$tgz"
  fi
  if [ "$RELEASE_STORE" = github ] && [ "$DRY" != 1 ]; then
    local code; code="$(gh_anon_status "$RELEASE_URL")"
    [ "$code" = 200 ] || die "the release asset is not downloadable anonymously (HTTP $code): $RELEASE_URL"
    ok "anonymous download ok: $RELEASE_URL"
  fi
  flybot_media
  ref="$key"; [ "$RELEASE_STORE" = github ] && ref="$RELEASE_URL"
  local cur; cur="$(flybot_current)"
  if [ -n "$SHA" ] && [ "$cur" = "$ref $SHA" ]; then ok "host release pointer already on $FB_ID"; RELEASE_CHANGED=0
  elif [ "$DRY" = 1 ]; then plan "point s3://$ARTIFACTS/flybot/current.json at flybot-$FB_ID ($RELEASE_STORE; the server switches at the bootstrap step, no restart of the instance)"; RELEASE_CHANGED=1
  else
    py -c 'import json,sys
store, ref, sha, commit, branch, at = sys.argv[1:7]
d = {"store": store, "sha256": sha, "commit": commit, "branch": branch, "set_at": at}
d["url" if store == "github" else "key"] = ref
open(sys.argv[7], "w").write(json.dumps(d) + "\n")' "$RELEASE_STORE" "$ref" "$SHA" "$FB_ID" "$FB_BRANCH" "$(date -u +%FT%TZ)" "$TMP/current.json"
    awsx s3 cp "$TMP/current.json" "s3://$ARTIFACTS/flybot/current.json" --content-type application/json --only-show-errors
    ok "release pointer -> $FB_ID (was ${cur:-none})"; RELEASE_CHANGED=1
  fi
  flybot_prune
  row "FlyBot release" "$FB_BRANCH@$FB_ID ($RELEASE_STORE${SHA:+, sha256 ${SHA:0:12}})"
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
  step image "vision Lambda image $CTX_TAG (AWS CodeBuild, only when not in ECR)"
  image_lookup
  if [ -n "$IMAGE_URI" ]; then ok "in ECR already: $IMAGE_URI"; return 0; fi
  if [ "$DRY" = 1 ]; then plan "CodeBuild flybot-vision-build-$STAGE (ARM small): build + push $CTX_TAG, 6-12 min"; return 0; fi
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
publish_host_script() {  # the host logic lives in S3, not in the user-data: changing it never restarts the instance
  awsx s3 cp infra/host/bootstrap.sh "s3://$ARTIFACTS/host/bootstrap.sh" --content-type text/x-shellscript --only-show-errors \
    || die "cannot publish infra/host/bootstrap.sh to s3://$ARTIFACTS/host/"
}
host_status() {  # one line: services + release on the host
  ssm_run 'printf "flybot=%s caddy=%s mongod=%s release=%s" "$(systemctl is-active flybot)" "$(systemctl is-active caddy)" "$(systemctl is-active mongod 2>/dev/null || true)" "$(python3 -c "import json;print(json.load(open(\"/opt/flybot/current/.pointer.json\"))[\"commit\"])" 2>/dev/null || echo none)"' 120 2>/dev/null | tr -d '\r' | tail -1 || true
}
step_bootstrap() {
  step bootstrap "the host: Node, Caddy, MongoDB, the FlyBot release under systemd (current user-data through SSM)"
  INSTANCE_ID="${INSTANCE_ID:-$(out InstanceId)}"
  if [ -z "$INSTANCE_ID" ]; then [ "$DRY" = 1 ] && { plan "publish infra/host/bootstrap.sh to s3://$ARTIFACTS/host/; wait for the first boot of the new instance, then run it"; return 0; }; die "no instance"; fi
  if [ "$DRY" = 1 ]; then plan "publish infra/host/bootstrap.sh to s3://$ARTIFACTS/host/, run it on $INSTANCE_ID through SSM (installs the release if it changed)"; return 0; fi
  publish_host_script
  ssm_wait_online 900 || die "SSM agent on $INSTANCE_ID not online after 15 min"
  if [ "${INSTANCE_NEW:-0}" = 1 ]; then
    log "first boot: waiting for 'bootstrap done' (Node, Caddy, MongoDB install, ~5 min)"
    ssm_run "$BOOTSTRAP_WAIT" 1800 || warn "first-boot bootstrap not finished yet; running it again now"
  fi
  local force=0; [ "${SECRETS_CHANGED:-0}" = 1 ] && force=1
  log "running the bootstrap (release ${RELEASE_CHANGED:-?}, secrets changed $force)"
  ssm_run "export FORCE_RESTART=$force; $BOOTSTRAP_RERUN" 1800 || die "bootstrap failed: make host CMD='tail -60 /var/log/flybot-bootstrap.log'"
  local st; st="$(host_status)"
  ok "host: $st"
  case "$st" in *flybot=active*) ;; *) warn "flybot is not active: make host CMD='journalctl -u flybot -n 60 --no-pager'" ;; esac
}

# ================================================================ health
step_health() {
  step health "FlyBot answering on the host; DNS and HTTPS checked once, never waited for"
  EIP="${EIP:-$(out ElasticIp)}"
  INSTANCE_ID="${INSTANCE_ID:-$(out InstanceId)}"
  if [ "$DRY" = 1 ]; then plan "GET http://127.0.0.1:8080/health on the host (SSM); one DNS lookup of $DOMAIN_NAME; https only when it resolves to ${EIP:-<EIP>}"; return 0; fi
  [ -n "$EIP" ] || die "no Elastic IP output"
  # 1. The app itself, from inside the instance (no DNS or certificate involved)
  local code="" i
  for i in $(seq 1 20); do
    code="$(ssm_run 'curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://127.0.0.1:8080/health' 60 2>/dev/null | tr -d '\r' | tail -1 || true)"
    [ "$code" = 200 ] && break
    [ $((i % 4)) = 1 ] && log "FlyBot /health on the host -> ${code:-no answer} (starting), retrying"
    sleep 15
  done
  if [ "$code" = 200 ]; then ok "FlyBot answers on the host: /health 200"; row "FlyBot" "running on $INSTANCE_ID (/health 200)"
  else warn "FlyBot /health on the host -> ${code:-no answer}: make host CMD='journalctl -u flybot -n 60 --no-pager'"
       row "FlyBot" "NOT answering (${code:-no answer})"; fi
  # 2. DNS + HTTPS: one look, no waiting (the record is added by hand; Caddy gets the certificate on its own afterwards)
  local a b; a="$(doh_a 1.1.1.1 || true)"; b="$(doh_a 8.8.8.8 || true)"
  if [ "$a" != "$EIP" ] || [ "$b" != "$EIP" ]; then
    dns_box "$EIP"
    row "DNS" "to add: $DOMAIN_NAME. A $EIP (HTTPS follows by itself; check later with ./deploy.sh --only health)"
    return 0
  fi
  ok "DNS $DOMAIN_NAME = $EIP"; row "DNS" "$DOMAIN_NAME. A $EIP (ok)"
  for i in 1 2 3 4; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 --resolve "$DOMAIN_NAME:443:$EIP" "https://$DOMAIN_NAME/health" || true)"
    [ "$code" = 200 ] && break
    sleep 15
  done
  if [ "$code" = 200 ]; then ok "https://$DOMAIN_NAME/health 200 (valid certificate)"; row "HTTPS" "https://$DOMAIN_NAME (ok)"
  else log "https://$DOMAIN_NAME -> ${code:-no answer}: Caddy is still getting the certificate; ./deploy.sh --only health later"
       row "HTTPS" "pending (certificate)"; fi
}

# ================================================================ smoke
step_smoke() {
  step smoke "one contract call to the Lambda with a sample trap photo"
  local fn bucket photo=data/samples/images/BE4089629/6ac327d44c3227d11cfbd084.jpg fe ref
  fn="$(out VisionFunctionName)"; bucket="$(out PhotosBucket)"
  if [ "$DRY" = 1 ]; then plan "sample photo by $([ "$STORAGE_MODE" = stack ] && echo "s3://${bucket:-$PHOTOS}/samples/" || echo "presigned URL from s3://$ARTIFACTS/lambda/"), invoke ${fn:-flybot-vision-$STAGE}, check the response contract"; return 0; fi
  [[ "$fn" == flybot-vision-* ]] || die "no Lambda yet: ./deploy.sh --only lambda"
  if [ "$STORAGE_MODE" = stack ]; then
    awsx s3 cp "$photo" "s3://$bucket/samples/smoke.jpg" --only-show-errors
    ref="{\"s3Bucket\":\"$bucket\",\"s3Key\":\"samples/smoke.jpg\"}"
  else   # never write test files into a bucket you bring: a presigned URL of the artifacts bucket (CONTRACTS 1 photo.url)
    awsx s3 cp "$photo" "s3://$ARTIFACTS/lambda/smoke.jpg" --only-show-errors
    ref="{\"url\":\"$(awsx s3 presign "s3://$ARTIFACTS/lambda/smoke.jpg" --expires-in 900)\"}"
  fi
  printf '{"v":1,"mode":"detect","photo":%s,"camera":{"id":"SMOKE","trapKind":"lure","platform":"ezviz"},"capturedAt":"2026-10-16T08:00:00Z","locale":"en"}' \
    "$ref" > "infra/.build/smoke-req-$STAGE.json"
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
  local storage="stack bucket $PHOTOS"
  [ "$STORAGE_MODE" = aws ] && storage="your AWS bucket $S3_BUCKET_NAME"
  [ "$STORAGE_MODE" = external ] && storage="$S3_ENDPOINT bucket $S3_BUCKET_NAME (evidence in $PHOTOS)"
  {
    printf '%-22s %s\n' "URL" "https://$DOMAIN_NAME"
    printf '%-22s %s\n' "Elastic IP" "${eip:-<not created>}"
    printf '%-22s %s\n' "Instance" "$(out InstanceId) ($INSTANCE_TYPE, AMI $AMI_ID; no Docker)"
    printf '%-22s %s\n' "Lambda" "${fn:-<not deployed>} (image $CTX_TAG, built by CodeBuild)"
    printf '%-22s %s\n' "Detector" "$DETECTOR$sha"
    printf '%-22s %s\n' "IoT endpoint" "$IOT_ENDPOINT"
    printf '%-22s %s\n' "KVS viewer role" "$(out KvsViewerRoleArn)"
    printf '%-22s %s\n' "Photos" "$storage"
    printf '%-22s %s\n' "Database" "$MONGO_NOTE"
    printf '%-22s %s\n' "Artifacts" "s3://$ARTIFACTS (release pointer, host script, videos, Lambda build context)"
    printf '%-22s %s\n' "Secrets" "$APP_SECRET, $GCP_SECRET (chatbot ${CHATBOT:-?}), Anthropic: $ANTHROPIC_NOTE"
    printf '%-22s %s\n' "Alarms" "$ALARM_NOTE"
    if [ "$PAYMENTS_ENABLED" = true ]; then
      printf '%-22s %s\n' "Payments" "on; webhooks: https://$DOMAIN_NAME/api/payments/webhooks/payos (PayOS), https://$DOMAIN_NAME/api/payments/webhooks/stripe (Stripe)"
    else printf '%-22s %s\n' "Payments" "off (PAYMENTS_ENABLED=false)"; fi
    [ ${#SUMMARY[@]} -gt 0 ] && printf '%s\n' "${SUMMARY[@]}"
    printf '%-22s %s\n' "Cost" "~17 USD/month at demo scale with t4g.micro (infra/COST.md)"
    printf '%-22s %s\n' "Log" "$LOG"
    printf '\nLeft for a human:\n'
    printf '  - Register the first account at https://%s/auth/login: it becomes the admin (do it before sharing the URL).\n' "$DOMAIN_NAME"
    case " ${SUMMARY[*]:-} " in *"to add:"*) printf '  - DNS: add  %s. A %s  (Namecheap > Advanced DNS), then ./deploy.sh --only health\n' "$DOMAIN_NAME" "$eip" ;; esac
    [ "$LOCAL_MONGO" = false ] && printf '  - MongoDB Atlas (or any firewalled database): allow %s/32 in its IP access list.\n' "${eip:-<EIP>}"
    case "$ALARM_NOTE" in *CONFIRM*) printf '  - Click the confirmation link AWS SNS emailed to the alarm address.\n' ;; esac
    if [ "$PAYMENTS_ENABLED" = true ]; then
      printf '  - Payments: enter the webhook URLs above in the PayOS payment channel and the Stripe event destination (FlyBot docs/payments.md).\n'
    fi
  } | tee "infra/.build/deploy-$STAGE-summary.txt"
}

# ================================================================ rollback
rollback() {
  step rollback "the host back on its previous FlyBot release"
  INSTANCE_ID="$(out InstanceId)"; ARTIFACTS="flybot-artifacts-$ACCOUNT-$STAGE"
  [ -n "$INSTANCE_ID" ] || die "stack $STACK has no instance"
  ssm_wait_online 300 || die "SSM agent on $INSTANCE_ID not online"
  ssm_run 'cat /opt/flybot/previous/.pointer.json' 60 > "$TMP/prev.json" 2>/dev/null || die "no previous release on the host (only one was ever installed)"
  py -c 'import json,sys;d=json.load(open(sys.argv[1]));print("previous:", d["commit"], d["key"])' "$TMP/prev.json" || die "unreadable previous release pointer"
  if [ "$DRY" = 1 ]; then plan "write it to s3://$ARTIFACTS/flybot/current.json and re-run the bootstrap"; return 0; fi
  awsx s3 cp "$TMP/prev.json" "s3://$ARTIFACTS/flybot/current.json" --content-type application/json --only-show-errors
  publish_host_script
  ssm_run "export FORCE_RESTART=0; $BOOTSTRAP_RERUN" 1800 || die "bootstrap failed: make host CMD='tail -60 /var/log/flybot-bootstrap.log'"
  ok "host: $(host_status)"
}

# ================================================================ app release
app_release() {  # --app: a new FlyBot release on the existing stack, without discovery, stack, image or Lambda
  INSTANCE_ID="$(out InstanceId)"; ARTIFACTS="flybot-artifacts-$ACCOUNT-$STAGE"
  [ -n "$INSTANCE_ID" ] || die "stack $STACK has no instance: run the whole ./deploy.sh once first"
  INSTANCE_NEW=0; SECRETS_CHANGED=0; RELEASE_CHANGED=0
  step_flybot
  step_bootstrap
  [ "$DRY" = 1 ] && return 0
  local code="" i
  for i in 1 2 3 4 5 6; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$DOMAIN_NAME/health" || true)"
    [ "$code" = 200 ] && break
    sleep 10
  done
  if [ "$code" = 200 ]; then ok "https://$DOMAIN_NAME/health 200"
  else warn "https://$DOMAIN_NAME/health -> ${code:-no answer}: ./deploy.sh --only health"; fi
}

# ================================================================ main
preflight
if [ "$SHOW_IP" = 1 ]; then
  EIP="$(out ElasticIp)"; [ -n "$EIP" ] || die "stack $STACK has no Elastic IP (not deployed yet)"
  dns_box "$EIP"; a="$(doh_a 1.1.1.1 || true)"; b="$(doh_a 8.8.8.8 || true)"
  log "now: 1.1.1.1=${a:-none} 8.8.8.8=${b:-none}"; exit 0
fi
if [ "$ROLLBACK" = 1 ]; then rollback; exit 0; fi
if [ "$APP" = 1 ]; then
  app_release
  [ "$DRY" = 1 ] && printf '\ndry run: nothing was changed. Run ./deploy.sh --app without --dry-run to apply.\n'
  exit 0
fi
discover
INSTANCE_ID="$CUR_INSTANCE"; EIP="$CUR_EIP"; INSTANCE_NEW=0; SECRETS_CHANGED=0; RELEASE_CHANGED=0; CHATBOT=""
for s in "${STEPS[@]}"; do
  want "$s" || continue
  "step_$s"
done
[ "$DRY" = 1 ] && printf '\ndry run: nothing was changed. Run ./deploy.sh without --dry-run to apply.\n'
exit 0
