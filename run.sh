#!/usr/bin/env bash
set -e

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q "huggingface_hub>=1.5" "typer>=0.24" "python-dotenv>=1.2"
python download_models.py "$@"
deactivate
