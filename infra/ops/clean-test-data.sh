#!/usr/bin/env bash
# Remove e2e / simulation test data from stack sentinel-<stage> (A4, before submission and after `make e2e-phase3`).
#
#   bash infra/ops/clean-test-data.sh           dry run: lists what would be deleted
#   YES=1 bash infra/ops/clean-test-data.sh     deletes it
#
# Test serials = prefixes AICSIM, FBSIM, AICTEST (infra/e2e/phase3-e2e.sh, the worker skips AICTEST by design).
# Never touches the real cameras (AIC54F29F5167FA, BE*, AC*): the script refuses a serial outside those prefixes.
# Deletes: IoT things (no certificate attached) and retained messages camos/<serial>/*; S3 frames/, evidence/, clips/,
# e2e/ under each serial; sentinel-traces rows of the serials (per_photo, followup, daily:camera:<serial>:*);
# sentinel-devices rows; sentinel-approvals rows whose cameras are all test serials; webhook dedupe ids snap_e2e*;
# one-shot follow-up schedules (fu-*) whose description names a test serial.
set -euo pipefail
STAGE="${STAGE:-dev}" PROFILE="${PROFILE:-sentinel}" REGION="${REGION:-ap-southeast-2}"
export AWS_PROFILE="$PROFILE" AWS_REGION="$REGION" MSYS_NO_PATHCONV=1
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="sentinel-$ACCOUNT-$STAGE"
if [ "$STAGE" = dev ]; then SFX=""; else SFX="-$STAGE"; fi
TRACES="sentinel-traces$SFX" DEVICES="sentinel-devices$SFX" APPROVALS="sentinel-approvals$SFX" DEDUPE="sentinel-flybot-webhook-dedupe$SFX"
GROUP="sentinel-$STAGE"
TEST_RE='^(AICSIM|FBSIM|AICTEST)[A-Z0-9]*$'
run() { if [ "${YES:-}" = 1 ]; then "$@"; else echo "DRY: $*"; fi; }
n=0

serials="$( { aws iot list-things --query 'things[].thingName' --output text | tr '\t' '\n'
             aws dynamodb scan --table-name "$DEVICES" --projection-expression serial --query 'Items[].serial.S' --output text | tr '\t' '\n'
             for p in frames evidence clips e2e; do aws s3 ls "s3://$BUCKET/$p/" | awk '/PRE/{sub("/$","",$2); print $2}'; done
           } | grep -E "$TEST_RE" | sort -u || true)"
echo "test serials: ${serials:-none}"

for s in $serials; do
  [[ "$s" =~ $TEST_RE ]] || { echo "refusing $s" >&2; exit 1; }
  if aws iot describe-thing --thing-name "$s" >/dev/null 2>&1; then
    if [ -n "$(aws iot list-thing-principals --thing-name "$s" --query 'principals[]' --output text)" ]; then
      echo "skip thing $s: a certificate is attached (detach + revoke it by hand)"
    else
      echo "thing $s"; run aws iot delete-thing --thing-name "$s"; n=$((n + 1))
    fi
  fi
  for p in frames evidence clips e2e; do
    c="$( (aws s3 ls "s3://$BUCKET/$p/$s/" --recursive || true) | wc -l)"
    if [ "$c" -gt 0 ]; then echo "s3 $p/$s/ ($c objects)"; run aws s3 rm "s3://$BUCKET/$p/$s/" --recursive --only-show-errors; n=$((n + c)); fi
  done
  for id in $(aws dynamodb scan --table-name "$TRACES" --projection-expression event_id \
                --filter-expression 'camera = :c OR begins_with(event_id, :d)' \
                --expression-attribute-values "{\":c\":{\"S\":\"$s\"},\":d\":{\"S\":\"daily:camera:$s:\"}}" \
                --query 'Items[].event_id.S' --output text); do
    echo "trace $id"; run aws dynamodb delete-item --table-name "$TRACES" --key "{\"event_id\":{\"S\":\"$id\"}}"; n=$((n + 1))
  done
  if [ -n "$(aws dynamodb get-item --table-name "$DEVICES" --key "{\"serial\":{\"S\":\"$s\"}}" --projection-expression serial --output text)" ]; then
    echo "device row $s"; run aws dynamodb delete-item --table-name "$DEVICES" --key "{\"serial\":{\"S\":\"$s\"}}"; n=$((n + 1))
  fi
done

# retained messages of test serials (an empty retained publish clears the topic)
for t in $(aws iot-data list-retained-messages --query 'retainedTopics[].topic' --output text); do
  s="$(echo "$t" | cut -d/ -f2)"
  if [[ "$s" =~ $TEST_RE ]]; then echo "retained $t"; run aws iot-data publish --topic "$t" --retain --qos 1 --payload ""; n=$((n + 1)); fi
done

# approvals whose cameras are all test serials
for id in $(aws dynamodb scan --table-name "$APPROVALS" --projection-expression approval_id --query 'Items[].approval_id.S' --output text); do
  cams="$(aws dynamodb get-item --table-name "$APPROVALS" --key "{\"approval_id\":{\"S\":\"$id\"}}" --query 'Item.doc.S' --output text \
          | grep -o '"cameras":\[[^]]*\]' | grep -o '"[A-Z0-9]*"' | tr -d '"' || true)"
  [ -n "$cams" ] || continue
  if ! echo "$cams" | grep -qvE "$TEST_RE"; then
    echo "approval $id ($(echo $cams))"; run aws dynamodb delete-item --table-name "$APPROVALS" --key "{\"approval_id\":{\"S\":\"$id\"}}"; n=$((n + 1))
  fi
done

# webhook dedupe ids of the e2e FlyBot posts
for id in $(aws dynamodb scan --table-name "$DEDUPE" --projection-expression id --filter-expression 'begins_with(id, :p)' \
              --expression-attribute-values '{":p":{"S":"snap_e2e"}}' --query 'Items[].id.S' --output text); do
  echo "dedupe $id"; run aws dynamodb delete-item --table-name "$DEDUPE" --key "{\"id\":{\"S\":\"$id\"}}"; n=$((n + 1))
done

# one-shot follow-up schedules of test serials
for name in $(aws scheduler list-schedules --group-name "$GROUP" --name-prefix fu- --query 'Schedules[].Name' --output text); do
  d="$(aws scheduler get-schedule --group-name "$GROUP" --name "$name" --query Description --output text)"
  if echo "$d" | grep -qE '(AICSIM|FBSIM|AICTEST)[A-Z0-9]*'; then
    echo "schedule $name ($d)"; run aws scheduler delete-schedule --group-name "$GROUP" --name "$name"; n=$((n + 1))
  fi
done
echo "$([ "${YES:-}" = 1 ] && echo deleted || echo 'would delete') $n items"
