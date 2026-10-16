#!/usr/bin/env bash
# Register one CamOS trap camera (ESP32-S3 or AMB82) in AWS IoT Core for the flybot-<stage> stack.
#
#   make provision-camera SERIAL=ESP32S3-0001 CERT=path/to/device-cert.pem      (= bash infra/provision-camera.sh SERIAL CERT)
#   bash infra/provision-camera.sh <SERIAL> [device-cert.pem] [profile] [region]
#
# Without a certificate file, IoT Core generates a key pair + certificate (aws iot create-keys-and-certificate) and
# the private key is written ONCE to data/secrets/<SERIAL>.key.pem (git-ignored) for the firmware portal upload;
# with one, only the certificate (PEM, CN = serial) is registered without a CA, the camera keeps its key.
# Then: thing <SERIAL> of type camos-camera (stack), policy camos-camera-<stage> (stack) attached, Amazon Root CA 1
# saved next to the key. Idempotent. The FlyBot pairing (device token, KVS channel) is done by FlyBot itself
# (POST /api/device/pair, CONTRACTS.md section 2): this script only does the IoT Core side.
set -euo pipefail

SERIAL="${1:?serial}"
CERT="${2:-}"
PROFILE="${3:-${PROFILE:-sentinel}}"
REGION="${4:-${REGION:-ap-southeast-2}}"
STAGE="${STAGE:-dev}"
STACK="flybot-$STAGE"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=infra/lib.sh
. "$ROOT/infra/lib.sh"
find_tools
aws_or_die
winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }
A="$AWS --profile $PROFILE --region $REGION --output text"
ACCOUNT="$($A sts get-caller-identity --query Account)"
out() { $A cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" 2>/dev/null || true; }
POLICY="$(out CameraPolicyName)"; THING_TYPE="$(out ThingTypeName)"
[ -n "$POLICY" ] && [ "$POLICY" != None ] || { echo "stack $STACK not found: make deploy first" >&2; exit 1; }
echo "account $ACCOUNT region $REGION stack $STACK thing $SERIAL type $THING_TYPE policy $POLICY"

# 1. thing
$A iot create-thing --thing-name "$SERIAL" --thing-type-name "$THING_TYPE" \
  --attribute-payload '{"attributes":{"model":"ESP32S3-TRAP","protocol":"1"}}' --query thingArn 2>/dev/null \
  || echo "thing $SERIAL exists"

# 2. certificate
SECRETS="$ROOT/data/secrets"; mkdir -p "$SECRETS"
if [ -n "$CERT" ]; then
  FP="$(openssl x509 -in "$CERT" -noout -fingerprint -sha256 | sed 's/.*=//; s/://g' | tr 'A-F' 'a-f')"
  CERT_ARN="$($A iot describe-certificate --certificate-id "$FP" --query certificateDescription.certificateArn 2>/dev/null || true)"
  if [ -z "$CERT_ARN" ] || [ "$CERT_ARN" = None ]; then
    CERT_ARN="$($A iot register-certificate-without-ca --certificate-pem "file://$(winpath "$CERT")" --status ACTIVE --query certificateArn)"
    echo "registered certificate $CERT_ARN"
  else
    echo "certificate already registered $CERT_ARN"
  fi
else
  EXISTING="$($A iot list-thing-principals --thing-name "$SERIAL" --query 'principals[0]' 2>/dev/null || true)"
  if [ -n "$EXISTING" ] && [ "$EXISTING" != None ]; then
    CERT_ARN="$EXISTING"; echo "thing already has a certificate $CERT_ARN (no new key generated)"
  else
    JSON="$($AWS iot create-keys-and-certificate --set-as-active --profile "$PROFILE" --region "$REGION" --output json)"
    CERT_ARN="$(printf '%s' "$JSON" | py -c 'import json,sys;print(json.load(sys.stdin)["certificateArn"])')"
    printf '%s' "$JSON" | py -c 'import json,sys;d=json.load(sys.stdin);open(sys.argv[1],"w").write(d["certificatePem"]);open(sys.argv[2],"w").write(d["keyPair"]["PrivateKey"])' \
      "$SECRETS/$SERIAL.cert.pem" "$SECRETS/$SERIAL.key.pem"
    chmod 600 "$SECRETS/$SERIAL.key.pem" 2>/dev/null || true
    echo "created certificate $CERT_ARN; key + cert written ONCE to data/secrets/$SERIAL.{key,cert}.pem (git-ignored)"
  fi
fi

# 3. attach
$A iot attach-policy --policy-name "$POLICY" --target "$CERT_ARN"
$A iot attach-thing-principal --thing-name "$SERIAL" --principal "$CERT_ARN"
cp -n "$ROOT/infra/iot/AmazonRootCA1.pem" "$SECRETS/AmazonRootCA1.pem" 2>/dev/null || true

ENDPOINT="$($A iot describe-endpoint --endpoint-type iot:Data-ATS --query endpointAddress)"
echo
echo "IoT endpoint: $ENDPOINT:8883   thing: $SERIAL   policy: $POLICY"
echo "Firmware portal (camos-esp32-trap docs): IoT host = $ENDPOINT, upload data/secrets/$SERIAL.cert.pem + .key.pem, Amazon Root CA 1 is built in"
echo "Then pair the camera in FlyBot (QR / provision token): FlyBot issues the device token and the KVS channel cam-$SERIAL"
echo "Test RPC: aws iot-data publish --profile $PROFILE --region $REGION --topic camos/$SERIAL/rpc/req --cli-binary-format raw-in-base64-out --payload '{\"v\":1,\"id\":\"t1\",\"method\":\"ping\",\"params\":{},\"timeoutS\":30}'"
