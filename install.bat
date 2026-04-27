@echo off
REM =============================================================
REM install.bat — Local environment setup for Lead-Lag Pipeline
REM =============================================================
REM Run once to install Python dependencies and verify GCP auth.
REM Usage: install.bat
REM =============================================================

echo =================================================
echo Lead-Lag Pipeline — Local Environment Setup
echo =================================================

REM ── Python version check ──────────────────────────────────────────────────
echo.
echo [1/4] Checking Python version...
python --version
if %ERRORLEVEL% neq 0 (
    echo   ERROR: Python not found. Install from https://python.org
    exit /b 1
)
echo   OK

REM ── Create virtual environment ────────────────────────────────────────────
echo.
echo [2/4] Setting up virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo   Created .venv
) else (
    echo   .venv already exists
)

REM ── Install dependencies ──────────────────────────────────────────────────
echo.
echo [3/4] Installing Python dependencies...
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo   ERROR: pip install failed
    exit /b 1
)
echo   Dependencies installed

REM ── GCP Authentication check ──────────────────────────────────────────────
echo.
echo [4/4] Checking GCP authentication...
where gcloud >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set ADC_PATH=%APPDATA%\gcloud\application_default_credentials.json
    if exist "%ADC_PATH%" (
        echo   Application Default Credentials found
    ) else (
        echo   No ADC found. Running: gcloud auth application-default login
        gcloud auth application-default login
    )

    REM Set project from config
    for /f "delims=" %%i in ('python -c "import yaml; cfg=yaml.safe_load(open('config/config.yaml')); print(cfg['gcp']['project_id'])"') do set PROJECT_ID=%%i
    gcloud config set project %PROJECT_ID% --quiet
    echo   GCP project set to: %PROJECT_ID%
) else (
    echo   WARNING: gcloud CLI not found.
    echo   Install from: https://cloud.google.com/sdk/docs/install
    echo   Then run: gcloud auth application-default login
)

echo.
echo =================================================
echo Setup complete!
echo.
echo NEXT STEPS:
echo   1. Activate the virtual environment:
echo      .venv\Scripts\activate
echo.
echo   2. Run one-time BigQuery setup:
echo      python run_local.py --mode setup
echo.
echo   3. Run a single window:
echo      python run_local.py --window-start 2022-01-01 --window-end 2022-12-31
echo.
echo   4. Run a historical backfill:
echo      python run_local.py --mode backfill --step residuals
echo      python run_local.py --mode backfill --step pairs --workers 4
echo      python run_local.py --mode backfill --step aggregation
echo.
echo   See README_LOCAL.md for full documentation.
echo =================================================
