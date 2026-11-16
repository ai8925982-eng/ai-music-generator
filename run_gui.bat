@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Virtual environment not found - running setup first...
  call setup_windows.bat
)
call .venv\Scripts\activate.bat
python run_app.py
if errorlevel 1 pause
