#!/usr/bin/env bash
# Smoke test of ./deploy.sh --dry-run and ./destroy.sh --dry-run against a fake AWS CLI (no network, no account):
# preflight + discovery run, every step prints its plan, and no mutating call reaches the CLI.
#   bash infra/tests/test_deploy_dry_run.sh        (exit 0 = pass)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT" || exit 2
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
export FAKE_LOG="$T/calls.log" AWS_BIN="$ROOT/infra/tests/fake-aws.sh" SAM="$ROOT/infra/tests/fake-sam.sh" \
       OWNER_CIDR=203.0.113.7/32 SKIP_GIT_CHECK=1
chmod +x "$AWS_BIN" "$SAM"
sed -e 's/^STAGE=dev/STAGE=citest/' -e 's/^APP_KEY=.*/APP_KEY=test-app-key-not-real-0001/' \
    -e 's/^ALARM_EMAIL=.*/ALARM_EMAIL=owner@example.com/' deploy.env.example > "$T/test.env"
fail=0
check() { if grep -qE "$2" "$T/out.txt"; then echo "ok    $1"; else echo "FAIL  $1 (no /$2/)"; fail=1; fi; }

bash infra/deploy.sh --dry-run --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && echo "ok    deploy.sh --dry-run exit 0" || { echo "FAIL  deploy.sh --dry-run exit $rc"; fail=1; tail -30 "$T/out.txt"; }
check "session + account" "session ok: account 123456789012"
check "all API probes" "ok    api IAM"
check "quotas checked" "quotas: Elastic IPs 1 of 5 used"
check "IoT endpoint discovered" "fake123-ats.iot.ap-southeast-2.amazonaws.com"
check "new stack" "stack flybot-citest: NONE"
check "AMI discovered" "ami-0fake0000000000000"
check "confirmed topic reused" "alarms: reuse sentinel-dev-alarms"
check "photos bucket planned" "photos bucket flybot-photos-123456789012-citest: will be created"
check "anthropic secret reused" "Anthropic reuse sentinel-dev-anthropic"
check "app secret planned" "plan  create flybot-citest-app"
check "JWT generated" "generated +JWT_SECRET"
check "app key from deploy.env" "set +APP_KEY"
check "stack params" "ExistingAlarmTopicArn=arn:aws:sns:ap-southeast-2:123456789012:sentinel-dev-alarms"
check "demo limits passed" "DemoMaxUnclaimed=200"
check "image planned" "plan  build \+ push ctx-"
check "health planned" "poll 1.1.1.1 and 8.8.8.8"
check "deploy key planned" "plan  generate an ed25519 deploy key, store the private half in flybot-citest-deploykey"
check "summary lists the deploy key" "Add the deploy key printed above"
check "nothing changed" "dry run: nothing was changed"
if grep -q "test-app-key-not-real-0001" "$T/out.txt" "$ROOT/infra/.build/deploy-citest.log"; then echo "FAIL  a secret value was printed"; fail=1; else echo "ok    no secret value in the output or the log"; fi
mut="$(grep -E '(^| )(create-|put-|delete-|update-|restore-|send-command|invoke|deploy|s3 cp|s3 rm|attach-)' "$FAKE_LOG" | grep -v -- '--version' || true)"
if [ -n "$mut" ]; then echo "FAIL  mutating calls in dry run:"; printf '      %s\n' "$mut"; fail=1; else echo "ok    no mutating AWS/SAM call ($(wc -l < "$FAKE_LOG") read-only calls)"; fi

: > "$FAKE_LOG"
printf 'ALARM_EMAIL=nobody@example.com\nSTAGE=citest\n' > "$T/t2.env"
bash infra/deploy.sh --dry-run --only secrets --env-file "$T/t2.env" > "$T/out.txt" 2>&1
check "--only runs one step" "== secrets"
grep -q "== stack" "$T/out.txt" && { echo "FAIL  --only secrets also ran the stack step"; fail=1; } || echo "ok    --only skips the other steps"
check "unconfirmed email -> new topic" "new topic flybot-citest-alarms; CONFIRM"

EXP="$T/expired-aws"; printf '#!/bin/sh\n[ "$1" = --version ] && echo aws-cli/2 && exit 0\necho "Token has expired" >&2; exit 255\n' > "$EXP"; chmod +x "$EXP"
AWS_BIN="$EXP" bash infra/deploy.sh --dry-run --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" = 2 ] && echo "ok    expired session -> exit 2" || { echo "FAIL  expired session exit $rc"; fail=1; }
check "login hint" "aws login --profile sentinel"

bash infra/destroy.sh --dry-run --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && echo "ok    destroy.sh --dry-run exit 0" || { echo "FAIL  destroy.sh --dry-run exit $rc"; fail=1; tail -20 "$T/out.txt"; }
check "destroy keeps the model bucket" "kept +model bucket sentinel-123456789012-models-public"
bash infra/destroy.sh --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q -- "--yes" "$T/out.txt" && echo "ok    destroy.sh without --yes refuses" || { echo "FAIL  destroy.sh without --yes (exit $rc)"; fail=1; }
rm -f "$ROOT/infra/.build/deploy-citest.log" "$ROOT/infra/.build/destroy-citest.log"
[ "$fail" = 0 ] && echo "PASS" || echo "FAILED"
exit "$fail"
