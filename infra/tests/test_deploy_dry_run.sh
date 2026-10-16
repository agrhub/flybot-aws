#!/usr/bin/env bash
# Smoke test of ./deploy.sh --dry-run and ./destroy.sh --dry-run against a fake AWS CLI (no network, no account):
# preflight + discovery run, every step prints its plan, and no mutating call reaches the CLI. Needs Node 22.12+
# (the FlyBot build step checks it) and git; the FlyBot checkout is a throw-away git repo made here.
#   bash infra/tests/test_deploy_dry_run.sh        (exit 0 = pass)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT" || exit 2
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
export FAKE_LOG="$T/calls.log" AWS_BIN="$ROOT/infra/tests/fake-aws.sh" SAM="$ROOT/infra/tests/fake-sam.sh" \
       OWNER_CIDR=203.0.113.7/32 FLYBOT_SRC="$T/flybot"
chmod +x "$AWS_BIN" "$SAM"
mkdir -p "$T/flybot" && (cd "$T/flybot" && git init -q && printf '{\n  "name": "flybot",\n  "private": true\n}\n' > package.json \
  && git add package.json && git -c user.email=t@example.com -c user.name=t commit -qm init)
sed -e 's/^STAGE=dev/STAGE=citest/' -e 's/^APP_KEY=.*/APP_KEY=test-app-key-not-real-0001/' \
    -e 's/^ALARM_EMAIL=.*/ALARM_EMAIL=owner@example.com/' -e '/^FLYBOT_SRC=/d' deploy.env.example > "$T/test.env"
fail=0
check() { if grep -qE "$2" "$T/out.txt"; then echo "ok    $1"; else echo "FAIL  $1 (no /$2/)"; fail=1; fi; }
nocheck() { if grep -qE "$2" "$T/out.txt"; then echo "FAIL  $1 (/$2/ present)"; fail=1; else echo "ok    $1"; fi; }
mutations() {  # every AWS/SAM call that would change something (an s3 cp FROM s3:// to stdout is a read)
  grep -E '(^| )(create-|put-|delete-|update-|restore-|send-command|invoke|start-build|deploy|s3 cp|s3 rm|attach-)' "$FAKE_LOG" \
    | grep -v -- '--version' | grep -vE ' s3 cp s3://[^ ]+ -( |$)' || true
}

bash infra/deploy.sh --dry-run --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && echo "ok    deploy.sh --dry-run exit 0" || { echo "FAIL  deploy.sh --dry-run exit $rc"; fail=1; tail -30 "$T/out.txt"; }
check "session + account" "session ok: account 123456789012"
check "all API probes incl. CodeBuild" "ok    api CodeBuild"
check "quotas checked" "quotas: Elastic IPs 1 of 5 used"
check "node found for the FlyBot build" "ok    node v2[2-9]"
check "FlyBot checkout recorded" "FlyBot checkout .*: (master|main)@[0-9a-f]{12}"
check "IoT endpoint discovered" "fake123-ats.iot.ap-southeast-2.amazonaws.com"
check "new stack" "stack flybot-citest: NONE"
check "AMI discovered" "ami-0fake0000000000000"
check "confirmed topic reused" "alarms: reuse sentinel-dev-alarms"
check "stack photo bucket planned" "stack bucket flybot-photos-123456789012-citest: will be created"
check "local MongoDB by default" "database: MongoDB 7 on the instance"
check "anthropic secret reused" "Anthropic reuse sentinel-dev-anthropic"
check "app secret planned" "plan  create flybot-citest-app"
check "JWT generated" "generated +JWT_SECRET"
check "app key from deploy.env" "set +APP_KEY"
check "stack params: alarm topic" "ExistingAlarmTopicArn=arn:aws:sns:ap-southeast-2:123456789012:sentinel-dev-alarms"
check "stack params: t4g.micro" "InstanceType=t4g.micro"
check "stack params: stack storage" "StorageMode=stack"
check "stack params: local mongo" "LocalMongo=true"
check "demo limits passed" "DemoMaxUnclaimed=200"
check "FlyBot build planned" "plan  npm .*run build in .*flybot, upload flybot/flybot-[0-9a-f]{12}\.tgz"
check "release pointer planned" "point s3://flybot-artifacts-123456789012-citest/flybot/current.json"
check "CodeBuild image planned" "plan  CodeBuild flybot-vision-build-citest"
check "bootstrap planned" "plan  wait for the first-boot bootstrap"
check "health planned (no DNS wait)" "one DNS lookup of"
check "nothing changed" "dry run: nothing was changed"
nocheck "no GitHub deploy key any more" "deploy key"
if grep -q "test-app-key-not-real-0001" "$T/out.txt" "$ROOT/infra/.build/deploy-citest.log"; then echo "FAIL  a secret value was printed"; fail=1; else echo "ok    no secret value in the output or the log"; fi
mut="$(mutations)"
if [ -n "$mut" ]; then echo "FAIL  mutating calls in dry run:"; printf '      %s\n' "$mut"; fail=1; else echo "ok    no mutating AWS/SAM call ($(wc -l < "$FAKE_LOG") read-only calls)"; fi

# external MongoDB + S3-compatible storage: secrets stay hidden, the stack gets the right parameters
: > "$FAKE_LOG"
sed -e 's#^MONGO_URI=.*#MONGO_URI="mongodb+srv://u:sup3rsecretpw@c0.example.net/flybot"#' \
    -e 's#^S3_BUCKET_NAME=.*#S3_BUCKET_NAME=minio-photos#' -e 's#^S3_ENDPOINT=.*#S3_ENDPOINT=https://minio.example.com#' \
    -e 's#^S3_ACCESS_KEY=.*#S3_ACCESS_KEY=MINIOACCESSKEY0001#' -e 's#^S3_SECRET_KEY=.*#S3_SECRET_KEY=minio-secret-key-not-real-0002#' \
    "$T/test.env" > "$T/ext.env"
bash infra/deploy.sh --dry-run --env-file "$T/ext.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && echo "ok    external db + storage: exit 0" || { echo "FAIL  external db + storage exit $rc"; fail=1; tail -20 "$T/out.txt"; }
check "external storage detected" "photos: S3-compatible https://minio.example.com bucket minio-photos"
check "evidence bucket of the stack" "will be created \(evidence only\)"
check "external database" "database: external \(MONGO_URI in flybot-citest-app\)"
check "stack params external" "StorageMode=external"
check "stack params no local mongo" "LocalMongo=false"
check "mongo + S3 keys go to the secret" "set +.*MONGO_URI.*S3_ACCESS_KEY.*S3_SECRET_KEY"
check "Atlas hint in the summary" "allow <EIP>/32 in its IP access list"
if grep -qE "sup3rsecretpw|minio-secret-key-not-real-0002|MINIOACCESSKEY0001" "$T/out.txt" "$ROOT/infra/.build/deploy-citest.log"; then
  echo "FAIL  MONGO_URI or S3 keys printed"; fail=1; else echo "ok    MONGO_URI and S3 keys never printed"; fi
[ -z "$(mutations)" ] && echo "ok    no mutating call (external case)" || { echo "FAIL  mutating calls (external case)"; fail=1; }

# an existing AWS bucket given by name
sed -e 's#^S3_BUCKET_NAME=.*#S3_BUCKET_NAME=my-own-photos#' "$T/test.env" > "$T/aws.env"
bash infra/deploy.sh --dry-run --only stack --env-file "$T/aws.env" > "$T/out.txt" 2>&1
check "own AWS bucket used" "photos: AWS bucket my-own-photos \(ap-southeast-2\)"
check "stack params own bucket" "S3BucketName=my-own-photos"
check "stack params aws mode" "StorageMode=aws"

# a dirty FlyBot checkout is refused unless --allow-dirty
echo '// local change' >> "$T/flybot/package.json"
bash infra/deploy.sh --dry-run --only flybot --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q "uncommitted changes" "$T/out.txt" && echo "ok    dirty FlyBot checkout refused" || { echo "FAIL  dirty checkout (exit $rc)"; fail=1; }
bash infra/deploy.sh --dry-run --only flybot --allow-dirty --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && echo "ok    --allow-dirty builds a dirty release" || { echo "FAIL  --allow-dirty exit $rc"; fail=1; tail -10 "$T/out.txt"; }
check "dirty release id" "flybot-[0-9a-f]{12}-dirty-[0-9a-f]{8}\.tgz"
(cd "$T/flybot" && git checkout -q -- package.json)

: > "$FAKE_LOG"
printf 'ALARM_EMAIL=nobody@example.com\nSTAGE=citest\n' > "$T/t2.env"
bash infra/deploy.sh --dry-run --only secrets --env-file "$T/t2.env" > "$T/out.txt" 2>&1
check "--only runs one step" "== secrets"
nocheck "--only skips the other steps" "== stack"
check "unconfirmed email -> new topic" "new topic flybot-citest-alarms; CONFIRM"

EXP="$T/expired-aws"; printf '#!/bin/sh\n[ "$1" = --version ] && echo aws-cli/2 && exit 0\necho "Token has expired" >&2; exit 255\n' > "$EXP"; chmod +x "$EXP"
AWS_BIN="$EXP" bash infra/deploy.sh --dry-run --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" = 2 ] && echo "ok    expired session -> exit 2" || { echo "FAIL  expired session exit $rc"; fail=1; }
check "login hint" "aws login --profile sentinel"

bash infra/destroy.sh --dry-run --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && echo "ok    destroy.sh --dry-run exit 0" || { echo "FAIL  destroy.sh --dry-run exit $rc"; fail=1; tail -20 "$T/out.txt"; }
check "destroy keeps the model bucket" "kept +model bucket sentinel-123456789012-models-public"
check "destroy empties the artifacts bucket" "artifacts +s3://flybot-artifacts-123456789012-citest emptied"
bash infra/destroy.sh --env-file "$T/test.env" > "$T/out.txt" 2>&1; rc=$?
[ "$rc" != 0 ] && grep -q -- "--yes" "$T/out.txt" && echo "ok    destroy.sh without --yes refuses" || { echo "FAIL  destroy.sh without --yes (exit $rc)"; fail=1; }
rm -f "$ROOT/infra/.build/deploy-citest.log" "$ROOT/infra/.build/destroy-citest.log" "$ROOT/infra/.build/deploy-citest-summary.txt"
[ "$fail" = 0 ] && echo "PASS" || echo "FAILED"
exit "$fail"
