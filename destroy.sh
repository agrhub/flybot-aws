#!/usr/bin/env bash
# Remove the FlyBot stack: ./destroy.sh --yes [--keep-photos] [--keep-secrets]   (see infra/destroy.sh --help)
exec bash "$(dirname "$0")/infra/destroy.sh" "$@"
