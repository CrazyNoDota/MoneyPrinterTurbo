@echo off
cd /d "%~dp0"

echo Starting MoneyPrinterTurbo...

:: Check if uv is available locally or in PATH
if exist ".uv\uv.exe" (
    set UV=".uv\uv.exe"
    goto run
)

where uv >nul 2>&1
if %errorlevel%==0 (
    set UV=uv
    goto run
)

where python >nul 2>&1
if %errorlevel%==0 (
    python -m pip install uv --quiet
    set UV=python -m uv
    goto run
)

echo ERROR: Python is not installed. Please install Python 3.11 from https://www.python.org/downloads/
pause
exit /b 1

:run
%UV% sync --frozen
%UV% run streamlit run ./webui/Main.py --browser.gatherUsageStats=False
pause
