#!/usr/bin/env bash
# Build the arm64 vision Lambda image with AWS CodeBuild and push it to ECR (no Docker on the laptop or the EC2).
#
#   ./deploy.sh --only image             (deploy.sh calls this when the image of the current inputs is not in ECR)
#   TAG=ctx-<hash> STAGE=dev bash infra/build-image.sh
#
#   1. zip the build context (lambda/vision/{Dockerfile,sentinel,requirements-lambda.txt} + models/), upload it to
#      s3://flybot-artifacts-<account>-<stage>/lambda/context-<tag>.zip (14-day lifecycle);
#   2. start the CodeBuild project flybot-vision-build-<stage> (ARM_CONTAINER, small) on that zip with IMAGE_TAG=<tag>;
#   3. wait, print the phases, keep the build log in infra/.build/build-log-<stage>.txt, record the image URI (with
#      digest) in infra/.build/image-uri-<stage> and its size in infra/.build/image-size-<stage>.json.
set -euo pipefail

STAGE="${STAGE:-dev}"
STACK="flybot-$STAGE"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=infra/lib.sh
. infra/lib.sh
find_tools
aws_or_die
mkdir -p infra/.build

out() { awsx cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null | grep -v '^None$' || true; }
REPO="$(out VisionRepositoryUri)"; BUCKET="$(out ArtifactsBucket)"; PROJECT="$(out VisionBuildProject)"
[ -n "$REPO" ] && [ -n "$BUCKET" ] && [ -n "$PROJECT" ] || die "stack $STACK has no ECR repository / artifacts bucket / CodeBuild project: ./deploy.sh --only stack first"
TAG="${TAG:-ctx-$(py infra/deploy_helpers.py ctx-hash)}"

ZIP="infra/.build/context-$STAGE.zip"
log "build context: $(py infra/deploy_helpers.py ctx-zip "$ZIP")"
KEY="lambda/context-$TAG.zip"
awsx s3 cp "$ZIP" "s3://$BUCKET/$KEY" --only-show-errors
ID="$(awsx codebuild start-build --project-name "$PROJECT" --source-location-override "$BUCKET/$KEY" \
      --environment-variables-override "name=IMAGE_TAG,value=$TAG,type=PLAINTEXT" --query 'build.id' --output text)"
log "CodeBuild $ID ($PROJECT, ARM small): docker build + push $TAG, 6-12 min the first time"

last="" status=IN_PROGRESS
while [ "$status" = IN_PROGRESS ]; do
  sleep 15
  read -r status phase < <(awsx codebuild batch-get-builds --ids "$ID" --query 'builds[0].[buildStatus,currentPhase]' --output text 2>/dev/null || echo "IN_PROGRESS ?")
  if [ "$phase" != "$last" ]; then log "  phase $phase"; last="$phase"; fi
done

read -r group stream < <(awsx codebuild batch-get-builds --ids "$ID" --query 'builds[0].logs.[groupName,streamName]' --output text)
: > "infra/.build/build-log-$STAGE.txt"
if [ -n "$group" ] && [ "$group" != None ]; then
  # UTF-8 safe (PYTHONUTF8 in lib.sh); control characters from docker's progress output are dropped
  awsx logs get-log-events --log-group-name "$group" --log-stream-name "$stream" --start-from-head --limit 10000 \
    --query 'events[].message' --output text 2>/dev/null | tr '\t' '\n' | tr -d '\000-\010\013\014\016-\037' > "infra/.build/build-log-$STAGE.txt" || true
fi
if [ "$status" != SUCCEEDED ]; then
  tail -40 "infra/.build/build-log-$STAGE.txt" >&2
  die "CodeBuild $ID: $status (full log: infra/.build/build-log-$STAGE.txt, CloudWatch $group)"
fi
DIGEST="$(awsx ecr describe-images --repository-name "${REPO##*/}" --image-ids "imageTag=$TAG" --query 'imageDetails[0].imageDigest' --output text)"
[[ "$DIGEST" == sha256:* ]] || die "image $TAG not in ECR after a successful build"
SIZE="$(grep -o 'IMAGE_SIZE_BYTES=[0-9]*' "infra/.build/build-log-$STAGE.txt" | tail -1 | cut -d= -f2 || true)"
echo "$REPO@$DIGEST" > "infra/.build/image-uri-$STAGE"
printf '{"stage":"%s","tag":"%s","image":"%s","size_bytes":%s,"build":"%s","built_at":"%s"}\n' \
  "$STAGE" "$TAG" "$REPO@$DIGEST" "${SIZE:-null}" "$ID" "$(date -u +%FT%TZ)" > "infra/.build/image-size-$STAGE.json"
ok "image $REPO@$DIGEST  size $(( ${SIZE:-0} / 1048576 )) MB (uncompressed)"
