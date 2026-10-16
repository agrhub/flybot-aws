#!/usr/bin/env bash
# Register one CamOS AI camera in AWS IoT Core .
#
#   bash infra/iot/provision-camera.sh <SERIAL> <device-cert.pem> [profile] [region]
#   STAGE=dev (default) picks the stack sentinel-<stage> for the Device API token; ROTATE_TOKEN=1 issues a new one
#
# The camera keeps its private key; only its certificate (PEM, CN = serial) is registered, without a CA
# (aws iot register-certificate-without-ca). A thing named <SERIAL>, the policy "camos-camera" (scoped to
# camos/<thing>/… topics through ${iot:Connection.Thing.ThingName}) and the certificate are tied together.
# Then a Device API token: random, printed once, only its SHA-256 is stored in sentinel-devices
# (token_sha256). Idempotent: re-running updates nothing destructive and keeps an issued token.
# Then the label verify code for the Open API activation: set-verify-code.sh (VERIFY_CODE=… to
# reuse a printed one); the code goes to data/secrets/<SERIAL>-verify-code.txt, only its hash to the table.
set -euo pipefail

SERIAL="${1:?serial}"
CERT="${2:?device certificate pem}"
PROFILE="${3:-sentinel}"
REGION="${4:-ap-southeast-2}"
POLICY_NAME="camos-camera"
HERE="$(cd "$(dirname "$0")" && pwd)"
# AWS CLI on Windows needs native paths in file:// parameters
winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }

AWS="aws --profile $PROFILE --region $REGION --output text"
ACCOUNT="$($AWS sts get-caller-identity --query Account)"

echo "account $ACCOUNT region $REGION thing $SERIAL"

# 1. Policy (templated on account/region)
if ! $AWS iot get-policy --policy-name "$POLICY_NAME" >/dev/null 2>&1; then
  sed -e "s/\${REGION}/$REGION/g" -e "s/\${ACCOUNT}/$ACCOUNT/g" "$HERE/camera-policy.json" > "$HERE/.policy.$$.json"
  $AWS iot create-policy --policy-name "$POLICY_NAME" --policy-document "file://$(winpath "$HERE/.policy.$$.json")" --query policyArn
  rm -f "$HERE/.policy.$$.json"
else
  echo "policy $POLICY_NAME exists"
fi

# 2. Thing type + thing
$AWS iot create-thing-type --thing-type-name camos-ai-camera >/dev/null 2>&1 || true
$AWS iot create-thing --thing-name "$SERIAL" --thing-type-name camos-ai-camera \
  --attribute-payload '{"attributes":{"model":"AIC-AMB82","protocol":"1"}}' --query thingArn 2>/dev/null \
  || echo "thing $SERIAL exists"

# 3. Certificate (registered without CA, activated)
FINGERPRINT="$(openssl x509 -in "$CERT" -noout -fingerprint -sha256 | sed 's/.*=//; s/://g' | tr 'A-F' 'a-f')"
CERT_ARN="$($AWS iot describe-certificate --certificate-id "$FINGERPRINT" --query certificateDescription.certificateArn 2>/dev/null || true)"
if [ -z "$CERT_ARN" ] || [ "$CERT_ARN" = "None" ]; then
  CERT_ARN="$($AWS iot register-certificate-without-ca --certificate-pem "file://$(winpath "$CERT")" --status ACTIVE --query certificateArn)"
  echo "registered certificate $CERT_ARN"
else
  echo "certificate already registered $CERT_ARN"
fi

# 4. Attach
$AWS iot attach-policy --policy-name "$POLICY_NAME" --target "$CERT_ARN"
$AWS iot attach-thing-principal --thing-name "$SERIAL" --principal "$CERT_ARN"

# 5. Device API token (needs the stack from `make deploy`)
STACK="sentinel-${STAGE:-dev}"
output() { $AWS cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" 2>/dev/null || true; }
API_BASE="$(output ApiBaseUrl)"
DEVICES_TABLE="$(output DevicesTableName)"
TOKEN=""
if [ -z "$DEVICES_TABLE" ] || [ "$DEVICES_TABLE" = "None" ]; then
  echo "stack $STACK not found: no Device API token (run make deploy, then this script again)"
else
  KEY="{\"serial\":{\"S\":\"$SERIAL\"}}"
  HAS="$($AWS dynamodb get-item --table-name "$DEVICES_TABLE" --key "$KEY" --projection-expression token_sha256 --query Item.token_sha256.S)"
  if [ -n "$HAS" ] && [ "$HAS" != "None" ] && [ "${ROTATE_TOKEN:-0}" != "1" ]; then
    echo "device token already issued for $SERIAL (ROTATE_TOKEN=1 issues a new one; the old one then stops working)"
  else
    TOKEN="$(openssl rand -hex 16)"
    HASH="$(printf '%s' "$TOKEN" | openssl dgst -sha256 | sed 's/.*= *//')"
    NOW="$(date +%s)000"
    $AWS dynamodb update-item --table-name "$DEVICES_TABLE" --key "$KEY" \
      --update-expression "SET token_sha256 = :h, token_issued_at = :t, thing_name = :s" \
      --expression-attribute-values "{\":h\":{\"S\":\"$HASH\"},\":t\":{\"N\":\"$NOW\"},\":s\":{\"S\":\"$SERIAL\"}}"
    echo "device token issued for $SERIAL in $DEVICES_TABLE (only its SHA-256 is stored)"
  fi
fi

# 6. Label verify code for the Open API activation; keeps an existing one
if [ -n "$DEVICES_TABLE" ] && [ "$DEVICES_TABLE" != "None" ]; then
  bash "$HERE/set-verify-code.sh" "$SERIAL" "$PROFILE" "$REGION"
fi

ENDPOINT="$($AWS iot describe-endpoint --endpoint-type iot:Data-ATS --query endpointAddress)"
echo
echo "MQTT endpoint: $ENDPOINT:8883"
echo "Camera console:  CAMOS=MQTT $ENDPOINT:8883   (after loading Amazon Root CA 1 with CAMOS=CA, see docs/SPIKE_T0.5_IOT_CORE.md)"
if [ -n "$TOKEN" ]; then
  echo
  echo "Device API token (shown once, keep it with the camera): $TOKEN"
  echo "Camera console:  CAMOS=API $API_BASE/t/$TOKEN   (firmware 0.24 sends only X-Device-Serial: the token rides in the path)"
  echo "                 or CAMOS=API $API_BASE with header X-Device-Token: <token> once the firmware sends it"
  [ ${#API_BASE} -gt 90 ] && echo "warning: $API_BASE/t/<token> is longer than the firmware's 127-char api_base"
  echo "Test upload:     curl -H 'X-Device-Serial: $SERIAL' -H 'X-Device-Token: <token>' -H 'Content-Type: image/jpeg' --data-binary @x.jpg '$API_BASE/device/v1/images?kind=snapshot&w=640&h=480&eventId=test1'"
fi
echo "Test RPC:        aws iot-data publish --profile $PROFILE --region $REGION --topic camos/$SERIAL/rpc/req --cli-binary-format raw-in-base64-out --payload '{\"v\":1,\"id\":\"spike1\",\"method\":\"ping\",\"params\":{},\"timeoutS\":30}'"
