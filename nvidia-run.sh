#!/usr/bin/env bash
# nvidia-run.sh — set up venv and launch the NVIDIA inference demo
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv-nvidia"

# ── Create or reuse venv ───────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV ..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# ── Install / upgrade dependencies ────────────────────────────────────────────
echo "Checking dependencies..."
pip install --quiet --upgrade pip
pip install --quiet openai "rich>=13.7.0" pynvml

# ── Optional: warn if no NVIDIA driver ────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "  NOTE: nvidia-smi not found. GPU metrics will be skipped."
    echo "        That's fine for a Mac/dev machine — run this on the L40S server"
    echo "        and pynvml will light up automatically."
    echo ""
fi

# ── Environment defaults (override before running if needed) ──────────────────
export INFERENCE_BASE_URL="${INFERENCE_BASE_URL:-http://localhost:8000/v1}"
export INFERENCE_API_KEY="${INFERENCE_API_KEY:-token-abc123}"

echo ""
echo "  Inference server : $INFERENCE_BASE_URL"
echo "  Model override   : ${INFERENCE_MODEL:-(auto-detect)}"
echo ""

# ── Launch ─────────────────────────────────────────────────────────────────────
python "$SCRIPT_DIR/nvidia-inference-demo.py" "$@"
