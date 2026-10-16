#!/usr/bin/env bash
# Upload the YOLO26 weights + sidecar to the public model bucket (anonymous GET; the Lambda fetches them at cold start).
#
#   bash models/upload_yolo26.sh [models/yolo26/fruitfly_yolo26n.onnx] [profile] [region]
#
# Prints the YOLO26_URL / YOLO26_SHA256 values for `make deploy` (infra/deploy.sh reads the sidecar itself).
set -euo pipefail
ONNX="${1:-models/yolo26/fruitfly_yolo26n.onnx}"
PROFILE="${2:-sentinel}"
REGION="${3:-ap-southeast-2}"
BUCKET="${MODELS_BUCKET:-sentinel-685527430355-models-public}"
PREFIX="${MODELS_PREFIX:-yolo26}"
AWS="${AWS:-aws}"
[ -f "$ONNX" ] || { echo "missing $ONNX (download it from the Colab run, see models/COLAB_RUNBOOK.md)" >&2; exit 1; }
SHA="$(sha256sum "$ONNX" | cut -d' ' -f1)"
NAME="$(basename "$ONNX")"
for f in "$ONNX" "$ONNX.json"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
  "$AWS" s3 cp "$f" "s3://$BUCKET/$PREFIX/$(basename "$f")" --metadata "sha256=$SHA" --only-show-errors --profile "$PROFILE" --region "$REGION"
done
URL="https://$BUCKET.s3.$REGION.amazonaws.com/$PREFIX/$NAME"
echo "uploaded $NAME ($SHA)"
echo "anonymous check: curl -sI $URL | head -1   (expect HTTP 200; else add $PREFIX/* to the bucket policy's public GetObject statement)"
echo "YOLO26_URL=$URL"
echo "YOLO26_SHA256=$SHA"
