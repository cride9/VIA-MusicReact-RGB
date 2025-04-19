@echo off
if not DEFINED IS_MINIMIZED set IS_MINIMIZED=1 && start "" /min "%~dpnx0" %* && exit
echo Enhanced Music-Reactive Keyboard LED Controller GUI
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or newer from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Check for required packages and install if missing
echo Checking for required packages...

python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing numpy...
    pip install numpy
)

python -c "import scipy" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing scipy...
    pip install scipy
)

python -c "import pyaudio" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing pyaudio...
    pip install pyaudio
)

python -c "import hid" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing hidapi...
    pip install hidapi
)

python -c "import PyQt6" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyQt6...
    pip install PyQt6
)

echo.
echo All required packages are installed.
echo.
echo Starting Enhanced Music-Reactive Keyboard LED Controller GUI...
echo.
echo New Features:
echo - System tray icon (minimize to taskbar)
echo - RGB color control (quiet to loud color transitions)
echo - Support for any QMK RGB keyboard
echo - Enhanced configuration saving
echo.
echo Note: If this is your first time running the program, you may need to:
echo 1. Enable "Stereo Mix" in your Windows sound settings
echo 2. Connect your QMK keyboard
echo.

python fixed_music_reactive_keyboard_gui.py

if %errorlevel% neq 0 (
    echo.
    echo Error starting the application.
    echo Please check that all dependencies are installed correctly.
    echo.
    pause
)
