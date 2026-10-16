#!/usr/bin/env bash
# FlyBot on AWS in one command: fill deploy.env (cp deploy.env.example deploy.env), `aws login --profile sentinel`, then
#   ./deploy.sh            (options: --dry-run, --only <step>, --show-ip, --yes; see infra/deploy.sh --help)
exec bash "$(dirname "$0")/infra/deploy.sh" "$@"
