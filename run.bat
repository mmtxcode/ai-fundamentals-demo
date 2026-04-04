@echo off
:: AI Fundamentals Demo — Windows launcher
:: Creates a virtual environment, installs dependencies, and runs the chat.

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv

:: ── Python version check ─────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Install it from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo Error: Python 3.10+ required. Found %PY_VER%
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo Error: Python 3.10+ required. Found %PY_VER%
    pause
    exit /b 1
)

:: ── Create virtual environment if needed ─────────────────────────────────────
if not exist "%VENV_DIR%\" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

:: ── Activate ──────────────────────────────────────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"

:: ── Install / upgrade dependencies ───────────────────────────────────────────
pip install -q --upgrade pip
pip install -q -r "%SCRIPT_DIR%requirements.txt"

:: ── Launch ────────────────────────────────────────────────────────────────────
python "%SCRIPT_DIR%chat.py" %*
