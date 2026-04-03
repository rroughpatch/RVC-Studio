@echo off
setlocal

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo uv is not installed. Run uv-install.bat first or install it from:
    echo https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

uv python install 3.10
if %errorlevel% neq 0 (
    echo Failed to install Python 3.10 with uv.
    pause
    exit /b %errorlevel%
)

uv sync
if %errorlevel% neq 0 (
    echo Failed to create the project environment.
    pause
    exit /b %errorlevel%
)

uv run streamlit run Home.py
endlocal
