@echo off
REM Upload the whole project to GitHub (needs Git for Windows: https://git-scm.com)
REM Usage: push_to_github.bat https://github.com/YOUR_NAME/YOUR_REPO.git
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
  echo Git not found. Install Git for Windows from https://git-scm.com/download/win and try again.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Usage: push_to_github.bat https://github.com/YOUR_NAME/YOUR_REPO.git
  pause
  exit /b 1
)
if not exist .git git init
git add -A
git commit -m "Add full AI Music Generator project"
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin %1
REM --force replaces what is on GitHub (e.g. the lone workflow file created in the browser)
git push -u origin main --force
echo.
echo Done. Open the Actions tab on GitHub to watch the build.
pause
