@echo off
REM Build the standalone Windows programs on your own PC (run setup_windows.bat once first).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
pip install pyinstaller soundfile numpy
set OPTS=--noconfirm --paths src --collect-all soundfile --collect-submodules aimusicgen
pyinstaller %OPTS% --clean --onefile --windowed --name AIMusicGenerator run_app.py
pyinstaller %OPTS% --onedir --windowed --name AIMusicGenerator-portable run_app.py
pyinstaller %OPTS% --onefile --console --name AIMusicGenerator-cli run_app.py
echo.
echo Done. In the "dist" folder:
echo   AIMusicGenerator.exe              single-file program (double-click)
echo   AIMusicGenerator-portable\        folder version, starts faster (copy the whole folder anywhere / USB)
echo   AIMusicGenerator-cli.exe          command-line version
pause
