@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Error: Virtual environment .venv not found in %~dp0
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python -c "import cv2, mediapipe, torch, PIL, sounddevice, scipy, tkinter" 2>nul
if errorlevel 1 (
    echo Error: Required dependencies missing in .venv. Run: pip install -r requirements.txt
    pause
    exit /b 1
)
python app.py
if errorlevel 1 (
    echo.
    echo SignBridge AI exited with an error.
    pause
)
