#!/usr/bin/env bash
# MultiView-AgriHealth — macOS/Linux Environment Setup
# Run from the project root: bash scripts/setup_venv.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

echo ""
echo "======================================================"
echo "  MultiView-AgriHealth — Python Virtual Environment"
echo "======================================================"
echo ""

# 1. Detect Python
PYTHON_EXE=""
for py in python3.11 python3.10 python3.12 python3 python; do
    if command -v "$py" &>/dev/null; then
        VERSION=$("$py" --version 2>&1)
        if echo "$VERSION" | grep -q "Python 3"; then
            PYTHON_EXE="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON_EXE" ]; then
    echo "[ERROR] No compatible Python 3 found. Install Python 3.10+ first."
    exit 1
fi

echo "[OK] Detected: $($PYTHON_EXE --version)  ($PYTHON_EXE)"

# 2. Create .venv
if [ -d "$VENV_PATH" ]; then
    echo "[INFO] .venv already exists at: $VENV_PATH"
    echo "       Delete it manually to recreate."
else
    echo "[...] Creating virtual environment at: $VENV_PATH"
    "$PYTHON_EXE" -m venv "$VENV_PATH" --prompt "agrihealth"
    echo "[OK] Virtual environment created."
fi

VENV_PYTHON="$VENV_PATH/bin/python"
VENV_PIP="$VENV_PATH/bin/pip"

# 3. Upgrade pip
echo "[...] Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet
echo "[OK] pip upgraded."

# 4. Install requirements
REQ_FILE="$PROJECT_ROOT/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "[...] Installing requirements..."
    "$VENV_PYTHON" -m pip install -r "$REQ_FILE" --quiet
    echo "[OK] Dependencies installed."
fi

DEV_REQ="$PROJECT_ROOT/requirements-dev.txt"
if [ -f "$DEV_REQ" ]; then
    echo "[...] Installing dev requirements..."
    "$VENV_PYTHON" -m pip install -r "$DEV_REQ" --quiet
    echo "[OK] Dev dependencies installed."
fi

echo ""
echo "======================================================"
echo "  Setup Complete"
echo "======================================================"
echo "  Python: $($VENV_PYTHON --version)"
echo "  Python exe: $VENV_PYTHON"
echo ""
echo "  To activate:  source .venv/bin/activate"
echo "  To run backend:"
echo "    python -m uvicorn backend.app.main:app --reload --port 8000"
echo "======================================================"
echo ""
