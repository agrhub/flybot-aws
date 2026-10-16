#!/usr/bin/env bash
# Build the arm64 vision Lambda image and push it to ECR, on the FlyBot EC2 host through SSM Run Command.
#
#   make build-lambda [STAGE=dev]        (= bash infra/build-image.sh)
#
# Why there: the laptop has no Docker and 2 GB of free disk; the FlyBot host is an arm64 Amazon Linux 2023 box with
# Docker (user-data) and an instance role that may push to the stack's ECR repository. Steps:
#   1. stage the build context (lambda/vision + models/fetch_models.py + any fetched models/yolov7-tiny, models/yolo26)
#      as a tar.gz and upload it to s3://<photos bucket>/build/ (7-day lifecycle);
#   2. ssm send-command on the instance: download, docker build (linux/arm64, native), push to ECR, write the digest;
#   3. wait for the command, print its output, record the image URI (with digest) in infra/.build/image-uri-<stage>
#      and the measured image size in infra/.build/image-size-<stage>.
# Env: INSTANCE_ID (default: the stack's), TAG (default: git short sha), NO_CACHE=1, LOCAL=1 (build with the local
# docker instead, when one exists: docker buildx --platform linux/arm64, then push).
set -euo pipefail

STAGE="${STAGE:-dev}"
PROFILE="${PROFILE:-sentinel}"
REGION="${REGION:-ap-southeast-2}"
STACK="flybot-$STAGE"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=infra/lib.sh
. infra/lib.sh
find_tools
aws_or_die
mkdir -p infra/.build

out() { "$AWS" cloudformation describe-stacks --stack-name "$STACK" --profile "$PROFILE" --region "$REGION" \
          --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
ACCOUNT="$("$AWS" sts get-caller-identity --query Account --output text --profile "$PROFILE" --region "$REGION")"
REPO="$(out VisionRepositoryUri)"
BUCKET="$(out PhotosBucket)"
INSTANCE_ID="${INSTANCE_ID:-$(out InstanceId)}"
TAG="${TAG:-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}"
[ -n "$REPO" ] && [ "$REPO" != "None" ] || { echo "stack $STACK has no ECR repository output: deploy phase 1 first" >&2; exit 1; }

# 1. build context
CTX="infra/.build/context-$STAGE.tar.gz"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
mkdir -p "$STAGE_DIR/ctx/models"
cp -r lambda/vision/sentinel lambda/vision/Dockerfile lambda/vision/requirements-lambda.txt lambda/vision/.dockerignore "$STAGE_DIR/ctx/"
cp models/fetch_models.py "$STAGE_DIR/ctx/models/"
for d in yolov7-tiny yolo26; do
  if [ -d "models/$d" ]; then mkdir -p "$STAGE_DIR/ctx/models/$d"; cp models/$d/*.onnx* "$STAGE_DIR/ctx/models/$d/" 2>/dev/null || true; fi
done
find "$STAGE_DIR/ctx" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
tar -C "$STAGE_DIR/ctx" -czf "$CTX" .
echo "build context: $(du -h "$CTX" | cut -f1) ($CTX)"

if [ "${LOCAL:-0}" = 1 ]; then
  "$AWS" ecr get-login-password --profile "$PROFILE" --region "$REGION" | docker login --username AWS --password-stdin "${REPO%%/*}"
  docker buildx build --platform linux/arm64 -t "$REPO:$TAG" --push "$STAGE_DIR/ctx"
  DIGEST="$("$AWS" ecr describe-images --repository-name "${REPO##*/}" --image-ids imageTag="$TAG" --query 'imageDetails[0].imageDigest' --output text --profile "$PROFILE" --region "$REGION")"
  echo "$REPO@$DIGEST" > "infra/.build/image-uri-$STAGE"
  echo "pushed $REPO@$DIGEST"; exit 0
fi

# 2. upload + SSM
KEY="build/vision-$TAG.tar.gz"
"$AWS" s3 cp "$CTX" "s3://$BUCKET/$KEY" --only-show-errors --profile "$PROFILE" --region "$REGION"
NO_CACHE_FLAG=""; [ "${NO_CACHE:-0}" = 1 ] && NO_CACHE_FLAG="--no-cache"
# The instance registers with SSM a minute or two after CloudFormation reports it running
for i in $(seq 1 30); do
  PING="$("$AWS" ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID" --query 'InstanceInformationList[0].PingStatus' --output text --profile "$PROFILE" --region "$REGION" 2>/dev/null || true)"
  [ "$PING" = Online ] && break
  [ "$i" = 1 ] && printf 'waiting for the SSM agent on %s ' "$INSTANCE_ID"; printf '.'; sleep 10
done
[ "$PING" = Online ] || { echo " not online after 5 min (instance role / network?)" >&2; exit 1; }
SCRIPT=$(cat <<EOF
set -euo pipefail
export AWS_DEFAULT_REGION=$REGION
# user-data installs docker and builds FlyBot first (2 GB box: one build at a time); wait for it, up to 25 min
for i in \$(seq 1 150); do grep -qs 'bootstrap done' /var/log/flybot-bootstrap.log && docker info >/dev/null 2>&1 && break; sleep 10; done
docker info >/dev/null || { echo "docker not ready on the host (see /var/log/flybot-bootstrap.log)"; exit 1; }
W=/opt/flybot/build-$TAG; rm -rf "\$W"; mkdir -p "\$W"; cd "\$W"
aws s3 cp "s3://$BUCKET/$KEY" ctx.tar.gz --only-show-errors && tar -xzf ctx.tar.gz && rm ctx.tar.gz
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws || true
t0=\$(date +%s)
docker build $NO_CACHE_FLAG --platform linux/arm64 -t "$REPO:$TAG" -t "$REPO:latest" .
echo "BUILD_SECONDS=\$((\$(date +%s)-t0))"
docker image inspect "$REPO:$TAG" --format 'IMAGE_SIZE_BYTES={{.Size}}'
docker push "$REPO:$TAG" >/dev/null && docker push "$REPO:latest" >/dev/null
DIGEST=\$(aws ecr describe-images --repository-name "${REPO##*/}" --image-ids imageTag="$TAG" --query 'imageDetails[0].imageDigest' --output text)
echo "IMAGE_URI=$REPO@\$DIGEST"
docker image prune -f >/dev/null; docker builder prune -f --keep-storage 2GB >/dev/null || true
cd / && rm -rf "\$W"
EOF
)
PARAMS="$(py -c 'import json,sys;print(json.dumps({"commands":[sys.stdin.read()],"executionTimeout":["3600"]}))' <<<"$SCRIPT")"
CMD_ID="$("$AWS" ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript --comment "flybot vision image $TAG" \
  --parameters "$PARAMS" --timeout-seconds 600 --query 'Command.CommandId' --output text --profile "$PROFILE" --region "$REGION")"
echo "ssm command $CMD_ID on $INSTANCE_ID (docker build on arm64; 5-15 min the first time)"
STATUS=Pending
while [ "$STATUS" = Pending ] || [ "$STATUS" = InProgress ] || [ "$STATUS" = Delayed ]; do
  sleep 15
  STATUS="$("$AWS" ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --query Status --output text --profile "$PROFILE" --region "$REGION" 2>/dev/null || echo Pending)"
  printf '.'
done
echo " $STATUS"
"$AWS" ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --query 'StandardOutputContent' --output text --profile "$PROFILE" --region "$REGION" | tail -40 | tee "infra/.build/build-log-$STAGE.txt"
if [ "$STATUS" != Success ]; then
  "$AWS" ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --query 'StandardErrorContent' --output text --profile "$PROFILE" --region "$REGION" | tail -40 >&2
  exit 1
fi
IMAGE_URI="$(grep -o 'IMAGE_URI=.*' "infra/.build/build-log-$STAGE.txt" | tail -1 | cut -d= -f2)"
SIZE="$(grep -o 'IMAGE_SIZE_BYTES=.*' "infra/.build/build-log-$STAGE.txt" | tail -1 | cut -d= -f2)"
[ -n "$IMAGE_URI" ] || { echo "no IMAGE_URI in the build output" >&2; exit 1; }
echo "$IMAGE_URI" > "infra/.build/image-uri-$STAGE"
echo "{\"stage\":\"$STAGE\",\"tag\":\"$TAG\",\"image\":\"$IMAGE_URI\",\"size_bytes\":${SIZE:-null},\"built_at\":\"$(date -u +%FT%TZ)\"}" > "infra/.build/image-size-$STAGE.json"
echo "image: $IMAGE_URI  size: $(( ${SIZE:-0} / 1048576 )) MB (uncompressed, docker image inspect)"
