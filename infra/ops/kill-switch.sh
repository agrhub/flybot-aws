#!/usr/bin/env bash
# Agent kill switch (A2's guardrails.DynamoKillSwitch, policy.yaml kill_switch): sentinel-devices item
# serial=kill_switch {kill_switch: true} stops every action on every camera; attribute kill_switch=true on a
# camera's own item stops that camera. The agent caches the flag 30 s and fails closed if it cannot read it.
#   bash infra/ops/kill-switch.sh on|off|status [SERIAL]
set -euo pipefail
STAGE="${STAGE:-dev}" PROFILE="${PROFILE:-sentinel}" REGION="${REGION:-ap-southeast-2}"
TABLE="sentinel-devices"; [ "$STAGE" = dev ] || TABLE="sentinel-devices-$STAGE"
MODE="${1:-status}" SERIAL="${2:-kill_switch}"
A=(--profile "$PROFILE" --region "$REGION")
KEY="{\"serial\":{\"S\":\"$SERIAL\"}}"
case "$MODE" in
  on|off)
    V=$([ "$MODE" = on ] && echo true || echo false)
    aws dynamodb update-item --table-name "$TABLE" --key "$KEY" "${A[@]}" \
      --update-expression "SET kill_switch = :v, kill_switch_at = :t" \
      --expression-attribute-values "{\":v\":{\"BOOL\":$V},\":t\":{\"S\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}}" ;;
  status) ;;
  *) echo "usage: $0 on|off|status [SERIAL]" >&2; exit 2 ;;
esac
aws dynamodb get-item --table-name "$TABLE" --key "$KEY" --consistent-read "${A[@]}" \
  --projection-expression "serial, kill_switch, kill_switch_at" --output json
