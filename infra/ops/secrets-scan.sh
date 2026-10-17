#!/usr/bin/env bash
# Secrets scan of the whole git history, for machines without gitleaks .
#
#   bash infra/ops/secrets-scan.sh            scan; exit 1 on any finding (and on any scan error)
#   bash infra/ops/secrets-scan.sh --staged   pre-commit check of the staged changes only (make hooks installs it):
#                                             refuses deploy.env / *.env / key files and secret-looking added lines
#
# Three checks, read-only (no git writes):
#  1. pattern scan of every added line in `git log -p --all`: Anthropic keys, FlyBot keys (fbk_), webhook
#     secrets (whsec_), AWS access keys (AKIA/ASIA) and long secret-key values, private keys, long Bearer tokens,
#     GitHub (classic and fine-grained) / Slack / Google / Telegram tokens, 32-64 hex values assigned to token/secret/key/password names.
#     Known public test values are allowed (AWS documentation example key, whsec_0123..., whsec_wrong...,
#     sk_flybot_secret, fbk_test, test values containing notreal0);
#  2. value scan: every value of length >= 12 in .env and data/secrets/* (the real local secrets) is searched for
#     in the history; only the variable or file NAME is printed, never the value. Non-secret config values (model
#     ids, regions, URLs) are reported as "config" hits and do not fail the scan;
#  3. the secret / large-artifact paths are git-ignored and were never tracked; no blob over 2 MB.
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)" || exit 2
cd "$ROOT" || exit 2
TMP="$(mktemp -d)" || exit 2
trap 'rm -rf "$TMP"' EXIT
fail=0

cat > "$TMP/patterns.txt" <<'EOF'
sk-ant-[A-Za-z0-9_-]{20,}
fbk_[A-Za-z0-9]{8,}
whsec_[A-Za-z0-9]{8,}
(AKIA|ASIA)[0-9A-Z]{16}
aws_secret_access_key *= *[A-Za-z0-9/+=]{30,}
secretAccessKey["']? *[:=] *["'][A-Za-z0-9/+=]{30,}
BEGIN [A-Z ]*PRIVATE KEY
Bearer [A-Za-z0-9._~+/-]{20,}
gh[pousr]_[A-Za-z0-9]{30,}
github_pat_[A-Za-z0-9_]{40,}
xox[baprs]-[A-Za-z0-9-]{10,}
AIza[0-9A-Za-z_-]{35}
[0-9]{9,10}:AA[A-Za-z0-9_-]{33}
(token|secret|key|passw(or)?d)[A-Za-z_]*["']?[[:space:]]*[:=][[:space:]]*["']?[0-9a-f]{32,64}([^0-9a-f]|$)
EOF
cat > "$TMP/allow.txt" <<'EOF'
AKIAIOSFODNN7EXAMPLE
wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
whsec_0123456789abcdef
whsec_wrong
sk_flybot_secret
fbk_test
0123456789abcdef0123456789abcdef
sha256
notreal0
EOF

if [ "${1:-}" = --staged ]; then
  bad="$(git diff --cached --name-only --diff-filter=ACMR | grep -E '(^|/)(deploy(\.[^/]+)?\.env|deploy\.env\.[^/]+|\.env(\.[^/]+)?|[^/]*service-account[^/]*\.json|gcp-[^/]*\.json|[^/]*\.(pem|key|p12|pfx))$|^data/secrets/|^secrets/' \
         | grep -vE '(^|/)(deploy\.env\.example|\.env\.example)$|AmazonRootCA1\.pem$' || true)"
  [ -n "$bad" ] && { echo "refusing to commit secret files:"; printf '  %s\n' $bad; fail=1; }
  git diff --cached --no-color -U0 | grep -E '^\+[^+]' | grep -iE -f "$TMP/patterns.txt" | grep -vF -f "$TMP/allow.txt" | cut -c1-120 > "$TMP/staged.txt" || true
  [ -s "$TMP/staged.txt" ] && { echo "secret-looking added lines (git diff --cached):"; cat "$TMP/staged.txt"; fail=1; }
  # values of the local deploy.env / .env (12+ chars) must not appear in the staged diff (names printed, never values)
  for f in deploy.env .env; do
    [ -f "$f" ] || continue
    while IFS= read -r line; do
      k="${line%%=*}"; v="$(printf '%s' "${line#*=}" | sed 's/[[:space:]]#.*$//' | tr -d "\"' \r")"
      [ ${#v} -ge 12 ] || continue
      case "$k" in STAGE|AWS_*|DOMAIN_NAME|*_EMAIL|OWNER_CIDR|DETECTOR|PLANNER_MODE|REPO_*|AMI_ID|*_MODEL|*_LOCATION|ANTHROPIC_SECRET_NAME|GCP_SERVICE_ACCOUNT_FILE|MODELS_BUCKET|EZVIZ_DOMAIN|SENTINEL_*|DEVICE_EVENTS_URL|FLYBOT_SRC|GITHUB_REPO) continue ;; esac
      git diff --cached --no-color -U0 | grep -E '^\+' | grep -F -- "$v" >/dev/null && { echo "value of $f $k is in the staged diff"; fail=1; }
    done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$f")
  done
  [ "$fail" = 0 ] && exit 0
  echo "commit refused by infra/ops/secrets-scan.sh --staged (unstage with: git restore --staged <file>)"; exit 1
fi

H="$TMP/history.patch"
git log -p --all --no-color -U0 > "$H" || { echo "git log failed"; exit 2; }
[ -s "$H" ] || { echo "empty history dump"; exit 2; }

echo "== 1. pattern scan ($(git rev-list --all | wc -l) commits, $(wc -l < "$H") patch lines)"
grep -nE '^\+' "$H" | grep -iE -f "$TMP/patterns.txt" | grep -vF -f "$TMP/allow.txt" > "$TMP/hits.txt"
rc=$?
[ "$rc" -le 1 ] || { echo "grep error $rc"; exit 2; }
if [ -s "$TMP/hits.txt" ]; then
  echo "findings (N = line of git log -p --all; commit: git log -p --all --no-color -U0 | head -N | grep '^commit' | tail -1):"
  cut -c1-200 "$TMP/hits.txt"; fail=1
else
  echo "no pattern hits"
fi

echo "== 2. value scan of local secrets against the history (names only)"
CONFIG_KEYS='^(STAGE|AWS_PROFILE|AWS_REGION|DOMAIN_NAME|ACME_EMAIL|ALARM_EMAIL|OWNER_CIDR|DETECTOR|PLANNER_MODE|ANTHROPIC_SECRET_NAME|ANTHROPIC_MODEL|GCP_SERVICE_ACCOUNT_FILE|VERTEX_LOCATION|VERTEX_MODEL|EZVIZ_DOMAIN|REPO_URL|MODELS_BUCKET|SENTINEL_ANTHROPIC_MODEL|SENTINEL_BEDROCK_MODEL|SENTINEL_BEDROCK_REGION|SENTINEL_AWS_REGION|SENTINEL_AWS_PROFILE|SENTINEL_DETECTOR_ONNX|SENTINEL_PLANNER|SENTINEL_LLM_BACKEND|SENTINEL_STORE|CAMOS_API_URL|FLYBOT_API_URL|FLYBOT_SRC|GITHUB_REPO)$'
checked=0
for f in .env deploy.env data/secrets/*; do
  [ -f "$f" ] || continue
  case "$f" in
    .env|*.env)
      while IFS= read -r line; do
        k="${line%%=*}"; v="$(printf '%s' "${line#*=}" | tr -d "\"' \r")"
        [ ${#v} -ge 12 ] || continue
        checked=$((checked + 1))
        if grep -qF -- "$v" "$H"; then
          if [[ "$k" =~ $CONFIG_KEYS ]]; then echo "config value in history (not a secret): $f $k"
          else echo "SECRET VALUE IN HISTORY: $f $k"; fail=1; fi
        fi
      done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$f")
      ;;
    *)
      v="$(tr -d '\r\n ' < "$f" | head -c 64)"
      [ ${#v} -ge 12 ] || continue
      checked=$((checked + 1))
      if grep -qF -- "$v" "$H"; then echo "SECRET VALUE IN HISTORY: $f"; fail=1; fi
      ;;
  esac
done
echo "checked $checked local secret values"

echo "== 3. ignored and never tracked"
for p in .env deploy.env secrets data/secrets data/export data/cache data/manifest.csv models/yolov7-tiny models/features; do
  ign=no; { git check-ignore -q "$p" || git check-ignore -q "$p/"; } && ign=yes
  ever=no; [ -n "$(git log --all --oneline -- "$p" | head -1)" ] && ever=YES
  printf '%-22s ignored=%-3s ever_tracked=%s\n' "$p" "$ign" "$ever"
  { [ "$ign" = yes ] && [ "$ever" = no ]; } || fail=1
done
echo "== blobs over 2 MB in history"
big="$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' | awk '$1=="blob" && $2>2000000')"
if [ -n "$big" ]; then echo "$big"; fail=1; else echo "none"; fi

[ "$fail" = 0 ] && echo "RESULT: clean" || echo "RESULT: FINDINGS (see above)"
exit "$fail"
