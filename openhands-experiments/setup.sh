#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OPENHANDS_DIR="${OPENHANDS_DIR:-$WORKSPACE_ROOT/openhands}"

git -C "$WORKSPACE_ROOT" submodule update --init --recursive openhands

cd "$OPENHANDS_DIR"
test "$(git rev-parse HEAD)" = "d39f7ae0e66174c50bdc714304fed4078b5e3b72"

if ! command -v docker >/dev/null || ! docker info >/dev/null 2>&1; then
  echo "Docker with Linux containers is required." >&2
  exit 2
fi

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry 2.1.2 is required." >&2
  exit 2
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is required." >&2
  exit 2
fi

# The full OpenHands `make build` also installs the web frontend, Playwright,
# Chromium system dependencies, and Git hooks. SWE-bench inference needs only
# the Python runtime and evaluation dependency groups. Skipping those optional
# builds avoids Playwright's sudo/apt prompt on shared servers.
export INSTALL_PLAYWRIGHT=false
export SKIP_VSCODE_BUILD=true
poetry env use python3.12
poetry install --with dev,test,runtime,evaluation
cp "$SCRIPT_DIR/config.toml.example" config.toml

echo "Setup complete. Create $SCRIPT_DIR/.env, then run $SCRIPT_DIR/run.sh"
