@echo off
REM Example: run_cli.bat -g lofi --bars 32 -o output
REM          run_cli.bat --vocal "C:\path\my_vocal.wav" -g pop
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run_app.py %*
pause
