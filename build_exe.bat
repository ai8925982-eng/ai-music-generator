@echo off
REM Build standalone Windows .exe files (needs setup_windows.bat run once).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
pip install pyinstaller
pyinstaller --noconfirm --clean --onefile --windowed --name AIMusicGenerator --paths src run_app.py
pyinstaller --noconfirm --onefile --console --name AIMusicGenerator-cli --paths src run_app.py
echo.
echo Done. Find the programs in the "dist" folder.
pause
