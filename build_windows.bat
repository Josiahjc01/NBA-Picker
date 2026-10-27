@echo off
REM Build NBA-Picker.exe on Windows (PyInstaller). Run from this folder in cmd.exe.
REM Requires: Python 3.10+ with Tcl/Tk, and a venv with requirements installed.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating venv...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv. Is Python on PATH?
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
if errorlevel 1 (
  echo pip install failed.
  exit /b 1
)

echo.
echo Building windowed NBA-Picker.exe (this can take a few minutes)...
python -m PyInstaller --noconfirm nba_picker.spec
if errorlevel 1 (
  echo PyInstaller failed.
  exit /b 1
)

echo.
echo Done.
echo   Exe:  %cd%\dist\NBA-Picker.exe
echo Run the .exe on Windows. Cache/ and output/ are created next to the exe when possible.
echo.
endlocal
