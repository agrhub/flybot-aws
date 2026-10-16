#!/usr/bin/env bash
# Remove one CamOS camera from AWS IoT Core and sentinel-devices (the reverse of provision-camera.sh).
#
#   bash infra/iot/deprovision-camera.sh <SERIAL> [profile] [region]
#
# For each certificate attached to the thing: detach the policy camos-camera and the thing, set it
# INACTIVE, delete it. Then delete the thing and its sentinel-devices row (STAGE=dev picks the stack).
# The policy camos-camera and the thing type stay. The real camera AIC54F29F5167FA is refused unless FORCE=1.
set -euo pipefail

SERIAL="${1:?serial}"
PROFILE="${2:-sentinel}"
REGION="${3:-ap-southeast-2}"
POLICY_NAME="camos-camera"
AWS="aws --profile $PROFILE --region $REGION --output text"

if [ "$SERIAL" = "AIC54F29F5167FA" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "refusing to deprovision the real camera $SERIAL (FORCE=1 to override)" >&2
  exit 1
fi

for CERT_ARN in $($AWS iot list-thing-principals --thing-name "$SERIAL" --query 'principals[]' 2>/dev/null || true); do
  CERT_ID="${CERT_ARN##*/}"
  $AWS iot detach-policy --policy-name "$POLICY_NAME" --target "$CERT_ARN" || true
  $AWS iot detach-thing-principal --thing-name "$SERIAL" --principal "$CERT_ARN"
  # Detaching is eventually consistent: delete-certificate fails while the thing is still attached
  for i in 1 2 3 4 5 6; do
    [ -z "$($AWS iot list-principal-things --principal "$CERT_ARN" --query 'things[]')" ] && break
    sleep 5
  done
  $AWS iot update-certificate --certificate-id "$CERT_ID" --new-status INACTIVE
  $AWS iot delete-certificate --certificate-id "$CERT_ID"
  echo "certificate $CERT_ID deleted"
done

if $AWS iot describe-thing --thing-name "$SERIAL" >/dev/null 2>&1; then
  $AWS iot delete-thing --thing-name "$SERIAL"
  echo "thing $SERIAL deleted"
fi

STACK="sentinel-${STAGE:-dev}"
DEVICES_TABLE="$($AWS cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='DevicesTableName'].OutputValue" 2>/dev/null || true)"
if [ -n "$DEVICES_TABLE" ] && [ "$DEVICES_TABLE" != "None" ]; then
  $AWS dynamodb delete-item --table-name "$DEVICES_TABLE" --key "{\"serial\":{\"S\":\"$SERIAL\"}}"
  echo "row $SERIAL deleted from $DEVICES_TABLE"
fi
