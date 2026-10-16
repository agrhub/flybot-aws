# shellcheck shell=bash
# Shared by the infra/*.sh scripts (sourced, not run). Needs ROOT (repo root) set by the caller.
#
#   find_tools        aws / sam / uv on PATH even when the installers did not add them (Windows laptop)
#   py ...            a Python 3 for the stdlib helpers (infra/deploy_helpers.py); never the Windows Store stub
#   awsx ...          aws with --profile/--region, CRs stripped (aws.exe on Windows prints CRLF)
#   aws_or_die        exit 2 with the exact login command when the session is gone
#   log/ok/warn/die   step log lines
#   winpath P         a path aws.exe understands in file:// arguments
#   ssm_wait_online / ssm_run CMD     SSM Run Command on the FlyBot instance (INSTANCE_ID)

PROFILE="${PROFILE:-${AWS_PROFILE:-sentinel}}"
export SAM_CLI_TELEMETRY="${SAM_CLI_TELEMETRY:-0}"
# aws.exe on Windows prints with the console code page (cp1252) and crashes on docker build output; force UTF-8
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
REGION="${REGION:-${AWS_REGION:-ap-southeast-2}}"

log()  { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '%s  ok    %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '%s  WARN  %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '%s  FAIL  %s\n' "$(date +%H:%M:%S)" "$1" >&2; exit "${2:-1}"; }

# Prepend the usual install folders that exist (AWS CLI v2 MSI per-user/all-users, SAM CLI, uv/pipx ~/.local/bin).
find_tools() {
  local d u="${USERNAME:-${USER:-}}"
  for d in "$HOME/.local/bin" "/c/Users/$u/.local/bin" \
           "/c/Users/$u/AppData/Local/Programs/Amazon/AWSCLIV2" "/c/Program Files/Amazon/AWSCLIV2" \
           "/c/Program Files/Amazon/AWSSAMCLI/bin" "/usr/local/bin" "/opt/homebrew/bin"; do
    if [ -d "$d" ]; then case ":$PATH:" in *":$d:"*) ;; *) PATH="$d:$PATH";; esac; fi
  done
  export PATH
  # AWS_BIN = the real CLI (exported for child scripts); AWS = aws_cr, the same with CRs stripped
  if [ -z "${AWS_BIN:-}" ]; then
    if [ -n "${AWS:-}" ] && [ "${AWS:-}" != aws_cr ]; then AWS_BIN="$AWS"; else AWS_BIN="$(command -v aws || command -v aws.exe || true)"; fi
  fi
  export AWS_BIN
  AWS=aws_cr
  find_sam; SAM="$SAM_FOUND"
  UV="${UV:-$(command -v uv || command -v uv.exe || true)}"
}

# Node 22.12+ and npm for the FlyBot build on this laptop: PATH first, then nvm-windows (NVM_SYMLINK, NVM_HOME,
# D:/Tools/nvm-noinstall), Program Files, nvm on Linux/macOS. Sets NODE_BIN / NPM_BIN and puts the folder on PATH
# (npm runs `node`). NODE_TRIED lists what was rejected, for the preflight message.
# shellcheck disable=SC2034  # NODE_BIN / NPM_BIN are read by deploy.sh
NODE_BIN="" NPM_BIN="" NODE_TRIED=""
find_node() {
  local d c v nvm_home="" sym=""
  [ -n "${NVM_HOME:-}" ] && nvm_home="$(cygpath -u "$NVM_HOME" 2>/dev/null || printf '%s' "$NVM_HOME")"
  [ -n "${NVM_SYMLINK:-}" ] && sym="$(cygpath -u "$NVM_SYMLINK" 2>/dev/null || printf '%s' "$NVM_SYMLINK")"
  local cands=("$(dirname "$(command -v node 2>/dev/null || echo /nonexistent/node)")" "$sym" \
               "/d/Tools/nvm-noinstall/nodejs" "/c/Program Files/nodejs")
  for d in "$nvm_home"/v22* /d/Tools/nvm-noinstall/v22* "$HOME"/.nvm/versions/node/v22*/bin; do [ -d "$d" ] && cands+=("$d"); done
  NODE_BIN="" NPM_BIN="" NODE_TRIED=""
  for d in "${cands[@]}"; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    for c in "$d/node" "$d/node.exe"; do
      [ -x "$c" ] || continue
      v="$("$c" --version 2>/dev/null | tr -d '\r')" || v=""
      if [[ "$v" =~ ^v([0-9]+)\.([0-9]+) ]] && { [ "${BASH_REMATCH[1]}" -gt 22 ] || { [ "${BASH_REMATCH[1]}" = 22 ] && [ "${BASH_REMATCH[2]}" -ge 12 ]; }; } \
         && { [ -f "$d/npm" ] || [ -f "$d/npm.cmd" ]; }; then
        # shellcheck disable=SC2034  # read by deploy.sh
        NODE_BIN="$c"; NPM_BIN="$d/npm"; [ -f "$NPM_BIN" ] || NPM_BIN="$d/npm.cmd"
        case ":$PATH:" in *":$d:"*) ;; *) PATH="$d:$PATH"; export PATH ;; esac
        return 0
      fi
      NODE_TRIED="${NODE_TRIED}    $c -> ${v:-no output}"$'\n'
    done
  done
  return 0
}

# The first SAM CLI that really runs (prints "SAM CLI ..."). On Windows every uv-made sam.exe is a "trampoline" that
# can fail with "uv trampoline failed to canonicalize script path"; the last resort runs `python -m samcli` with the
# uv tool venv's own python.exe (not a trampoline) through a small shim. SAM_TRIED keeps each failure for preflight.
SAM_TRIED="" SAM_FOUND=""
find_sam() {
  local appdata="" c out venv shim
  if [ -n "${APPDATA:-}" ]; then appdata="$(cygpath -u "$APPDATA" 2>/dev/null || printf '%s' "$APPDATA")"; fi
  venv="${appdata:+$appdata/uv/tools/aws-sam-cli}"
  shim=""
  if [ -n "$venv" ] && [ -x "$venv/Scripts/python.exe" ]; then
    shim="${ROOT:-.}/infra/.build/sam-python.sh"
    mkdir -p "$(dirname "$shim")"
    printf '#!/usr/bin/env bash\nexec "%s" -m samcli "$@"\n' "$venv/Scripts/python.exe" > "$shim" && chmod +x "$shim"
  fi
  SAM_TRIED="" SAM_FOUND=""
  for c in "${SAM:-}" \
           "$(command -v sam 2>/dev/null || true)" "$(command -v sam.exe 2>/dev/null || true)" \
           "$(command -v sam.cmd 2>/dev/null || true)" \
           "${venv:+$venv/Scripts/sam.exe}" \
           "${HOME:+$HOME/.local/share/uv/tools/aws-sam-cli/bin/sam}" \
           "/c/Program Files/Amazon/AWSSAMCLI/bin/sam.cmd" "/c/Program Files/Amazon/AWSSAMCLI/bin/sam.exe" \
           "$shim"; do
    [ -n "$c" ] || continue
    out="$("$c" --version 2>&1 | tr -d '\r' | head -1)" || true   # deploy.sh runs with set -e -o pipefail
    case "$out" in "SAM CLI"*) SAM_FOUND="$c"; return 0 ;; esac
    SAM_TRIED="${SAM_TRIED}    $c -> ${out:-no output}"$'\n'
  done
  return 0
}

# A Python that really runs (on Windows `python` may be the Store stub, exit 9009; a uv trampoline may be stale).
PY_CMD=()
py() {
  if [ ${#PY_CMD[@]} -eq 0 ]; then
    local c
    for c in python3 python; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
        PY_CMD=("$c"); break
      fi
    done
    if [ ${#PY_CMD[@]} -eq 0 ] && [ -n "${UV:-}" ]; then PY_CMD=("$UV" run --quiet --no-project python); fi
    [ ${#PY_CMD[@]} -gt 0 ] || die "no Python 3.9+ found (install uv: https://docs.astral.sh/uv/)"
  fi
  "${PY_CMD[@]}" "$@"
}

aws_cr() {
  "$AWS_BIN" "$@" | tr -d '\015'
  return "${PIPESTATUS[0]}"
}

awsx() {
  "$AWS" --profile "$PROFILE" --region "$REGION" "$@" | tr -d '\r'
  return "${PIPESTATUS[0]}"
}

aws_or_die() {
  [ -n "${AWS_BIN:-}" ] || die "AWS CLI v2 not found (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)"
  if ! "$AWS" sts get-caller-identity --profile "$PROFILE" --region "$REGION" >/dev/null 2>&1; then
    printf '\nAWS session expired or missing for profile %s. Run:\n\n    aws login --profile %s\n\nthen run this command again.\n\n' "$PROFILE" "$PROFILE" >&2
    exit 2
  fi
}

winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

ssm_wait_online() {  # $1 = max seconds (default 600)
  local max="${1:-600}" t=0 ping=""
  while [ "$t" -lt "$max" ]; do
    ping="$(awsx ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
            --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)"
    if [ "$ping" = Online ]; then [ "$t" = 0 ] || echo; return 0; fi
    [ "$t" = 0 ] && printf '%s  waiting for the SSM agent on %s ' "$(date +%H:%M:%S)" "$INSTANCE_ID"
    printf '.'; sleep 10; t=$((t + 10))
  done
  echo; return 1
}

# ssm_run CMD [execution timeout s]: runs CMD as root on INSTANCE_ID, prints its stdout, returns 0 on Success
ssm_run() {
  local cmd="$1" tmo="${2:-1800}" params id status err
  params="$(printf '%s' "$cmd" | py -c 'import json,sys;print(json.dumps({"commands":[sys.stdin.read()],"executionTimeout":[sys.argv[1]]}))' "$tmo")"
  id="$(awsx ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript --comment "flybot-aws infra" \
        --parameters "$params" --timeout-seconds 600 --query 'Command.CommandId' --output text)" || return 1
  status=Pending
  while [ "$status" = Pending ] || [ "$status" = InProgress ] || [ "$status" = Delayed ]; do
    sleep 5
    status="$(awsx ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE_ID" --query Status --output text 2>/dev/null || echo Pending)"
  done
  awsx ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE_ID" --query StandardOutputContent --output text
  if [ "$status" != Success ]; then
    err="$(awsx ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE_ID" --query StandardErrorContent --output text 2>/dev/null)"
    [ -n "$err" ] && [ "$err" != None ] && printf '%s\n' "$err" | tail -20 >&2
    warn "ssm command $id: $status"; return 1
  fi
}

# The CURRENT user-data from IMDS, re-run: cloud-init runs it once per instance, and a stack update that changes it
# stops/starts the instance without re-running it. Output goes to /var/log/flybot-bootstrap.log only.
# shellcheck disable=SC2016,SC2034  # expanded on the host, not here; used by the scripts that source this file
BOOTSTRAP_RERUN='set -euo pipefail
T=$(curl -sfX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
curl -sf -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/user-data -o /opt/flybot/bootstrap.sh
chmod 700 /opt/flybot/bootstrap.sh
rc=0; bash /opt/flybot/bootstrap.sh >/dev/null 2>&1 || rc=$?
tail -4 /var/log/flybot-bootstrap.log | cut -c1-200
exit $rc'

# Wait (up to 25 min) until the first boot printed "bootstrap done" (Node, Caddy, MongoDB installed; FlyBot started).
# shellcheck disable=SC2016,SC2034
BOOTSTRAP_WAIT='for i in $(seq 1 150); do grep -qs "bootstrap done" /var/log/flybot-bootstrap.log && break; sleep 10; done
tail -3 /var/log/flybot-bootstrap.log | cut -c1-200; grep -qs "bootstrap done" /var/log/flybot-bootstrap.log'
