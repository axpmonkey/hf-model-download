#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

UPGRADE_DEPS=false
ARGS=()

for arg in "$@"; do
  case "$arg" in
    --upgrade-deps) UPGRADE_DEPS=true ;;
    *) ARGS+=("$arg") ;;
  esac
done

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  UPGRADE_DEPS=true
fi

if [ "$UPGRADE_DEPS" = true ]; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install --upgrade -q "huggingface_hub>=1.5" "typer>=0.24" "python-dotenv>=1.2"
fi

.venv/bin/python download_models.py "${ARGS[@]}"
