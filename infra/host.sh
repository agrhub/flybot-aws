#!/usr/bin/env bash
# Run one shell command on the FlyBot EC2 host through SSM Run Command and print its output (no SSH, no key pair).
#
#   make host CMD='tail -20 /var/log/flybot-bootstrap.log' [STAGE=dev]
#   bash infra/host.sh 'systemctl status flybot caddy mongod --no-pager'
#   bash infra/host.sh bootstrap        # re-run the CURRENT user-data (env file, gcp file, FlyBot release, services)
#
# Interactive shell instead: aws ssm start-session --target <InstanceId> --profile sentinel (Session Manager plugin).
set -euo pipefail
CMD="${1:?command, or: bootstrap}"
STAGE="${STAGE:-dev}"
STACK="flybot-$STAGE"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# shellcheck source=infra/lib.sh
. infra/lib.sh
find_tools
aws_or_die
INSTANCE_ID="${INSTANCE_ID:-$(awsx cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)}"
[ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != None ] || die "stack $STACK has no instance: run ./deploy.sh first"
ssm_wait_online 600 || die "SSM agent on $INSTANCE_ID not online"
[ "$CMD" = bootstrap ] && CMD="$BOOTSTRAP_RERUN"
ssm_run "$CMD" "${TIMEOUT_S:-1800}"
