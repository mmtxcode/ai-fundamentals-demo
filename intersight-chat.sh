#!/usr/bin/env bash
# Intersight AI Demo — launcher
# Sets up the virtual environment, installs dependencies, and starts the chat.
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

# ── Ensure python3-venv is available (Debian/Ubuntu) ─────────────────────────
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
if [ ! -f "$VENV_DIR/bin/activate" ]; then
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

# ── Load .env if present ─────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ── Check credentials are configured ─────────────────────────────────────────
if [ -z "$INTERSIGHT_CLIENT_ID" ] && [ -z "$INTERSIGHT_OAUTH_TOKEN" ] && [ -z "$INTERSIGHT_API_KEY_ID" ]; then
    echo ""
    echo "  No Intersight credentials found."
    echo ""
    echo "  Copy .env.example to .env and add your credentials:"
    echo "    cp .env.example .env"
    echo ""
    echo "  Then set one of:"
    echo "    INTERSIGHT_CLIENT_ID + INTERSIGHT_CLIENT_SECRET  (OAuth2 — recommended)"
    echo "    INTERSIGHT_OAUTH_TOKEN                           (pre-fetched token)"
    echo "    INTERSIGHT_API_KEY_ID + INTERSIGHT_API_SECRET_KEY_FILE  (HTTP Signature)"
    echo ""
    exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────
python "$SCRIPT_DIR/intersight-chat.py" "$@"
