@echo off
setlocal

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing uv...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo Failed to install uv.
        pause
        exit /b %errorlevel%
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

call "%~dp0uv-start.bat"
endlocal
