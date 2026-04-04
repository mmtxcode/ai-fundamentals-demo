#!/usr/bin/env bash
# AI Fundamentals Demo — launcher
# Creates a virtual environment, installs dependencies, and runs the chat.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="${PYTHON:-python3}"

# ── Python version check (requires 3.10+) ────────────────────────────────────
PY_VERSION=$("$PYTHON" -c "import sys; print(sys.version_info >= (3,10))" 2>/dev/null || echo "False")
if [ "$PY_VERSION" != "True" ]; then
    echo "Error: Python 3.10 or higher is required."
    echo "Current: $("$PYTHON" --version 2>&1)"
    echo "Install it from https://python.org or via your package manager."
    exit 1
fi

# ── Ensure python3-venv is available (Debian/Ubuntu systems) ─────────────────
if ! "$PYTHON" -m venv --help > /dev/null 2>&1; then
    echo "Python venv module not found. Attempting to install..."
    if command -v apt > /dev/null 2>&1; then
        PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        sudo apt install -y "python${PY_VER}-venv"
    else
        echo "Error: python venv module is missing and could not be installed automatically."
        echo "Please install it manually and re-run this script."
        exit 1
    fi
fi

# ── Create virtual environment if needed ─────────────────────────────────────
# Check for the activate script specifically — a missing activate means the
# venv directory exists but was created in a previous failed attempt.
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    # Clean up any broken partial venv before retrying
    if [ -d "$VENV_DIR" ]; then
        echo "Removing incomplete virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ── Activate ──────────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── Install / upgrade dependencies ───────────────────────────────────────────
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# ── Launch ────────────────────────────────────────────────────────────────────
python "$SCRIPT_DIR/chat.py" "$@"
