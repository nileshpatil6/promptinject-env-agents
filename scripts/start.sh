#!/usr/bin/env bash
# One-command launcher — starts the environment server on port 7860
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uv &>/dev/null; then
    uv run python -m uvicorn server.main:app --host 0.0.0.0 --port 7860 --workers 1
elif command -v python &>/dev/null; then
    python -m uvicorn server.main:app --host 0.0.0.0 --port 7860 --workers 1
else
    echo "ERROR: neither uv nor python found in PATH" >&2
    exit 1
fi
