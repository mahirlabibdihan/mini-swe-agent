#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OPENHANDS_DIR="${OPENHANDS_DIR:-$WORKSPACE_ROOT/openhands}"

git -C "$WORKSPACE_ROOT" submodule update --init --recursive openhands

cd "$OPENHANDS_DIR"
test "$(git rev-parse HEAD)" = "d39f7ae0e66174c50bdc714304fed4078b5e3b72"

if ! command -v docker >/dev/null || ! docker info >/dev/null 2>&1; then
  echo "Docker with Linux containers is required." >&2
  exit 2
fi

make build
poetry install --with dev,test,runtime,evaluation
make setup-config
cp "$SCRIPT_DIR/config.toml.example" config.toml

echo "Setup complete. Create $SCRIPT_DIR/.env, then run $SCRIPT_DIR/run.sh"
