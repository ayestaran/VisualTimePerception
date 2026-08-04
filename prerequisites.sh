#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
  echo "[prerequisites] $*"
}

fail() {
  echo "[prerequisites] ERROR: $*" >&2
  exit 1
}

log "Project root: $ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "Python not found. Install Python 3 and retry."
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
log "Using $PYTHON_BIN ($PYTHON_VERSION)"

if ! "$PYTHON_BIN" -c 'import venv' >/dev/null 2>&1; then
  fail "The venv module is unavailable. Reinstall Python with venv support and retry."
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Virtual environment already exists at $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python3"
if [[ ! -x "$VENV_PYTHON" ]]; then
  VENV_PYTHON="$VENV_DIR/bin/python"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  fail "No Python executable found in virtual environment at $VENV_DIR/bin"
fi

log "Upgrading pip, setuptools, and wheel"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  log "Installing Python dependencies from requirements.txt"
  "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
else
  fail "requirements.txt not found in $ROOT_DIR"
fi

log "Setup complete"
log "Activate env: source $VENV_DIR/bin/activate"
log "Start app: ./launch.sh start"
