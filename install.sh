#!/bin/bash
# =============================================================
# install.sh — Local environment setup for Lead-Lag Pipeline
# Works on: Mac, Linux, Windows Git Bash, Windows WSL
# Usage: bash install.sh
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================="
echo "Lead-Lag Pipeline — Local Environment Setup"
echo "================================================="

# ── Python version check ───────────────────────────────────────────────────
echo ""
echo "[1/4] Checking Python version..."
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  ERROR: Python not found. Install from https://python.org"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "  Python $PYTHON_VERSION found ($PYTHON_CMD)"

# ── Create virtual environment ─────────────────────────────────────────────
echo ""
echo "[2/4] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists"
fi

# Activate — handle both Unix (bin/) and Windows Git Bash (Scripts/)
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "  Virtual environment activated (Windows path)"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "  Virtual environment activated (Unix path)"
else
    echo "  ERROR: Could not find venv activate script."
    exit 1
fi

# ── Install dependencies ───────────────────────────────────────────────────
echo ""
echo "[3/4] Installing Python dependencies..."
python -m pip install --quiet --upgrade pip 2>/dev/null || true
python -m pip install -r requirements.txt
echo "  Dependencies installed"

# ── GCP Authentication check ───────────────────────────────────────────────
echo ""
echo "[4/4] Checking GCP authentication..."

if command -v gcloud &>/dev/null; then
    ADC_UNIX="$HOME/.config/gcloud/application_default_credentials.json"
    ADC_WIN="${APPDATA:-}/gcloud/application_default_credentials.json"
    if [ -f "$ADC_UNIX" ] || [ -f "$ADC_WIN" ]; then
        echo "  Application Default Credentials found"
    else
        echo "  No ADC found. Running: gcloud auth application-default login"
        gcloud auth application-default login
    fi

    PROJECT_ID=$(python -c "
import yaml
with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg['gcp']['project_id'])
")
    gcloud config set project "$PROJECT_ID" --quiet
    echo "  GCP project set to: $PROJECT_ID"
else
    echo "  WARNING: gcloud CLI not found."
    echo "  Install from: https://cloud.google.com/sdk/docs/install"
    echo "  Then run: gcloud auth application-default login"
fi

echo ""
echo "================================================="
echo "Setup complete!"
echo ""
echo "NEXT STEPS:"
echo "  1. Activate the virtual environment:"
if [ -f ".venv/Scripts/activate" ]; then
    echo "     source .venv/Scripts/activate   (Git Bash)"
    echo "     .venv\\Scripts\\activate          (CMD / PowerShell)"
else
    echo "     source .venv/bin/activate"
fi
echo ""
echo "  2. Run one-time BigQuery setup:"
echo "     python run_local.py --mode setup"
echo ""
echo "  3. Run a single window:"
echo "     python run_local.py --window-start 2022-01-01 --window-end 2022-12-31"
echo ""
echo "  See README_LOCAL.md for full documentation."
echo "================================================="
