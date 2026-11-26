@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment .venv not found in %~dp0
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python webcam_app.py %*
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
