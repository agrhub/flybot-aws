#!/usr/bin/env bash
printf 'sam %s\n' "$*" >> "${FAKE_LOG:-/dev/null}"
[ "${1:-}" = --version ] && { echo "SAM CLI, version 1.99.0"; exit 0; }
echo "fake sam: refusing $*" >&2; exit 1
