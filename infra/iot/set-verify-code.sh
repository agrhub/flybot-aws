#!/usr/bin/env bash
# Give a camera its label verify code for the Open API (POST /devices/activate, task T2.4).
#
#   bash infra/iot/set-verify-code.sh <SERIAL> [profile] [region]
#   VERIFY_CODE=K7P4QX   use this code (e.g. the one already printed on the label) instead of a random one
#   ROTATE_VERIFY_CODE=1 replace an existing code; STAGE=dev picks the stack; QUIET=1 does not echo the code
#
# Only SHA-256("<SERIAL>:<CODE>") is stored, in sentinel-devices.verify_code_sha256. The code and the label QR
# (protocol 12.1: CAMOS:1;SN:<serial>;VC:<code>;M:<model>;;) are written to data/secrets/<SERIAL>-verify-code.txt
# (git-ignored). Called by provision-camera.sh; safe to re-run (keeps an existing code).
set -euo pipefail

SERIAL="${1:?serial}"
PROFILE="${2:-sentinel}"
REGION="${3:-ap-southeast-2}"
MODEL="${MODEL:-AIC-AMB82}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SECRETS="$HERE/../../data/secrets"
AWS="aws --profile $PROFILE --region $REGION --output text"
STACK="sentinel-${STAGE:-dev}"

TABLE="$($AWS cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='DevicesTableName'].OutputValue" 2>/dev/null || true)"
if [ -z "$TABLE" ] || [ "$TABLE" = "None" ]; then
  echo "stack $STACK not found: no verify code (run make deploy first)" >&2
  exit 1
fi
KEY="{\"serial\":{\"S\":\"$SERIAL\"}}"
HAS="$($AWS dynamodb get-item --table-name "$TABLE" --key "$KEY" --projection-expression verify_code_sha256 --query Item.verify_code_sha256.S)"
if [ -n "$HAS" ] && [ "$HAS" != "None" ] && [ "${ROTATE_VERIFY_CODE:-0}" != "1" ]; then
  echo "verify code already set for $SERIAL (ROTATE_VERIFY_CODE=1 replaces it)"
  exit 0
fi

# 6 characters without 0/O/1/I/L, like the dev stand-in (tools/camos-dev/openapi.js)
gen() {
  local a=ABCDEFGHJKMNPQRSTUVWXYZ23456789 out="" b
  for b in $(openssl rand 6 | od -An -tu1); do out+="${a:$((b % 31)):1}"; done
  echo "$out"
}
CODE="$(printf '%s' "${VERIFY_CODE:-$(gen)}" | tr 'a-z' 'A-Z' | tr -d ' \r\n')"
case "$CODE" in (*[!A-Z0-9]*|"") echo "verify code must be letters/digits" >&2; exit 1 ;; esac
HASH="$(printf '%s' "$SERIAL:$CODE" | openssl dgst -sha256 | sed 's/.*= *//')"
NOW="$(date +%s)000"
$AWS dynamodb update-item --table-name "$TABLE" --key "$KEY" \
  --update-expression "SET verify_code_sha256 = :h, verify_code_set_at = :t, verify_failures = :z REMOVE verify_locked_until" \
  --expression-attribute-values "{\":h\":{\"S\":\"$HASH\"},\":t\":{\"N\":\"$NOW\"},\":z\":{\"N\":\"0\"}}"

mkdir -p "$SECRETS"
OUT="$SECRETS/$SERIAL-verify-code.txt"
umask 077
{
  echo "serial=$SERIAL"
  echo "verifyCode=$CODE"
  echo "labelQr=CAMOS:1;SN:$SERIAL;VC:$CODE;M:$MODEL;;"
  echo "setAt=$(date -u +%Y-%m-%dT%H:%M:%SZ) table=$TABLE"
} > "$OUT"
echo "verify code set for $SERIAL in $TABLE (only its SHA-256 is stored); code and label QR in data/secrets/$SERIAL-verify-code.txt"
[ "${QUIET:-0}" = "1" ] || echo "Verify code (for the label): $CODE"
