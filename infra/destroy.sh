#!/usr/bin/env bash
# Remove what ./deploy.sh created for one stage, the mirror of infra/deploy.sh.
#
#   ./destroy.sh --yes                       stack flybot-<stage> (EC2 + EIP, Lambda, ECR images, CodeBuild project,
#                                            artifacts bucket, IoT policy and rules, roles, alarms), the stack photos
#                                            bucket and the flybot-<stage>-* secrets (a bucket of your own is never touched)
#   ./destroy.sh --yes --keep-photos         keep s3://flybot-photos-<account>-<stage> (a later deploy reuses it)
#   ./destroy.sh --yes --keep-secrets        keep flybot-<stage>-app/-gcp/-anthropic/-deploykey/-repotoken (JWT stays valid, no re-entry)
#   ./destroy.sh --dry-run                   what would be removed
#   --env-file F                             settings file (default deploy.env; only STAGE / AWS_PROFILE are read)
#
# Never touched: the public model bucket (sentinel-<account>-models-public), the sentinel-dev stack and its secrets,
# IoT things and certificates of registered cameras (they lose access with the policy; delete them in FlyBot or the
# IoT console), the SAM artifacts bucket. Secrets are scheduled for deletion with a 7-day recovery window
# (deploy.sh restores them if you deploy again within that week).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=infra/lib.sh
. infra/lib.sh
find_tools

YES=0 DRY=0 KEEP_PHOTOS=0 KEEP_SECRETS=0 ENV_FILE=deploy.env
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --dry-run) DRY=1 ;;
    --keep-photos) KEEP_PHOTOS=1 ;;
    --keep-secrets) KEEP_SECRETS=1 ;;
    --env-file) ENV_FILE="${2:?}"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
  shift
done
if [ -f "$ENV_FILE" ]; then eval "$(py infra/deploy_helpers.py settings "$ENV_FILE" | grep -E '^CFG_(STAGE|AWS_PROFILE|AWS_REGION)=')"; fi
STAGE="${STAGE:-${CFG_STAGE:-dev}}"
PROFILE="${CFG_AWS_PROFILE:-$PROFILE}"
REGION="${CFG_AWS_REGION:-$REGION}"
STACK="flybot-$STAGE"
[[ "$STAGE" =~ ^[a-z0-9-]{1,16}$ ]] || die "bad stage $STAGE"
[ "$REGION" = ap-southeast-2 ] || die "region $REGION refused (Sydney only)"
aws_or_die
ACCOUNT="$(awsx sts get-caller-identity --query Account --output text)"
PHOTOS="flybot-photos-$ACCOUNT-$STAGE"          # the stack bucket only: a bucket of your own (S3_BUCKET_NAME) is never touched
ARTIFACTS="flybot-artifacts-$ACCOUNT-$STAGE"    # FlyBot releases + Lambda build contexts, emptied so the stack can delete it
mkdir -p infra/.build
exec > >(tee -a "infra/.build/destroy-$STAGE.log") 2>&1

STATUS="$(awsx cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo NONE)"
BUCKET_EXISTS=0; awsx s3api head-bucket --bucket "$PHOTOS" >/dev/null 2>&1 && BUCKET_EXISTS=1
SECRETS=()
for s in "flybot-$STAGE-app" "flybot-$STAGE-gcp" "flybot-$STAGE-anthropic" "flybot-$STAGE-deploykey" "flybot-$STAGE-repotoken"; do
  awsx secretsmanager describe-secret --secret-id "$s" --query DeletedDate --output text 2>/dev/null | grep -qx None && SECRETS+=("$s")
done

printf '\nAccount %s, stage %s:\n' "$ACCOUNT" "$STAGE"
printf '  stack       %s (%s)%s\n' "$STACK" "$STATUS" "$([ "$STATUS" = NONE ] && echo ': nothing to delete')"
printf '  photos      s3://%s %s\n' "$PHOTOS" "$([ $BUCKET_EXISTS = 0 ] && echo '(absent)' || { [ $KEEP_PHOTOS = 1 ] && echo KEPT || echo 'emptied and DELETED (every photo)'; })"
printf '  secrets     %s %s\n' "${SECRETS[*]:-(none)}" "$([ ${#SECRETS[@]} = 0 ] && echo '' || { [ $KEEP_SECRETS = 1 ] && echo KEPT || echo 'scheduled for deletion (7-day recovery)'; })"
printf '  artifacts   s3://%s emptied and deleted with the stack
' "$ARTIFACTS"
printf '  kept        model bucket sentinel-%s-models-public, stack sentinel-dev, IoT things/certificates\n\n' "$ACCOUNT"
if [ "$DRY" = 1 ]; then echo "dry run: nothing deleted"; exit 0; fi
[ "$YES" = 1 ] || die "add --yes to delete the above (no prompt: this script also runs unattended)"

if [ "$STATUS" != NONE ]; then
  log "emptying s3://$ARTIFACTS (FlyBot releases, build contexts)"
  awsx s3 rm "s3://$ARTIFACTS" --recursive --only-show-errors 2>/dev/null || true
  log "deleting stack $STACK"
  awsx cloudformation delete-stack --stack-name "$STACK"
  awsx cloudformation wait stack-delete-complete --stack-name "$STACK" || die "stack deletion failed: CloudFormation console > $STACK > Events"
  ok "stack $STACK deleted"
fi
if [ $BUCKET_EXISTS = 1 ] && [ $KEEP_PHOTOS = 0 ]; then
  log "emptying and deleting s3://$PHOTOS"
  awsx s3 rm "s3://$PHOTOS" --recursive --only-show-errors || true
  awsx s3api delete-bucket --bucket "$PHOTOS" && ok "bucket $PHOTOS deleted"
elif [ $BUCKET_EXISTS = 1 ]; then ok "bucket $PHOTOS kept (the next ./deploy.sh reuses it)"; fi
if [ $KEEP_SECRETS = 0 ]; then
  for s in "${SECRETS[@]}"; do
    awsx secretsmanager delete-secret --secret-id "$s" --recovery-window-in-days 7 --query Name --output text >/dev/null && ok "secret $s scheduled for deletion (7 days)"
  done
fi
rm -f "infra/.build/image-uri-$STAGE"
case " ${SECRETS[*]:-} " in *"flybot-$STAGE-deploykey"*)
  [ $KEEP_SECRETS = 1 ] || log "a GitHub deploy key flybot-$STAGE-ec2 from an earlier layout may still exist: remove it at github.com/agrhub/flybot > Settings > Deploy keys" ;;
esac
ok "done"
