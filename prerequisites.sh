#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"

log() {
  echo "[prerequisites] $*"
}

fail() {
  echo "[prerequisites] ERROR: $*" >&2
  exit 1
}

resolve_python_bin() {
  if [[ -n "$PYTHON_BIN" ]]; then
    if [[ -x "$PYTHON_BIN" ]]; then
      echo "$PYTHON_BIN"
      return 0
    fi
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      command -v "$PYTHON_BIN"
      return 0
    fi
    return 1
  fi

  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

log "Project root: $ROOT_DIR"

if ! PYTHON_CMD="$(resolve_python_bin)"; then
  fail "Python not found. Install Python 3 and retry."
fi

PYTHON_VERSION="$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
log "Using $PYTHON_CMD ($PYTHON_VERSION)"

if ! "$PYTHON_CMD" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
  fail "Python 3.9 or newer is required."
fi

if ! "$PYTHON_CMD" -c 'import venv' >/dev/null 2>&1; then
  fail "The venv module is unavailable. Reinstall Python with venv support and retry."
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_CMD" -m venv "$VENV_DIR"
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

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

log "Upgrading pip, setuptools, and wheel"
"$VENV_PYTHON" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  log "Installing Python dependencies from requirements.txt"
  "$VENV_PYTHON" -m pip install --disable-pip-version-check -r "$ROOT_DIR/requirements.txt"
else
  fail "requirements.txt not found in $ROOT_DIR"
fi

pip3 install matplotlib

log "Setup complete"
log "Activate env: source $VENV_DIR/bin/activate"
log "Start app: ./launch.sh start"

