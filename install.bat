@echo off
cd /d "%~dp0"

echo ============================================================
echo  MoneyPrinterTurbo one-click install (Windows 11)
echo  Checks prerequisites, builds the .venv, installs the app,
echo  sets up the motion-graphics toolchain, writes config.toml,
echo  and (optionally) registers the bot to autostart at logon.
echo  Safe to re-run: completed steps are skipped.
echo ============================================================
echo.

:: --- Prerequisite: Python 3.11+ on PATH -----------------------------------
echo ===== Checking prerequisites
where python >nul 2>&1
if not %errorlevel%==0 (
    echo ERROR: Python not found on PATH. Install Python 3.11+ from
    echo        https://www.python.org/downloads/windows/ and re-run install.bat.
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>&1
if not %errorlevel%==0 (
    echo ERROR: Python 3.11 or newer is required. Install it from
    echo        https://www.python.org/downloads/windows/ and re-run install.bat.
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo  - %%v detected

:: --- Prerequisite: Node.js 22+ on PATH ------------------------------------
where node >nul 2>&1
if not %errorlevel%==0 (
    echo ERROR: Node.js not found on PATH. Install Node.js 22+ from
    echo        https://nodejs.org and re-run install.bat.
    exit /b 1
)
node -e "process.exit(parseInt(process.versions.node.split('.')[0],10) >= 22 ? 0 : 1)" >nul 2>&1
if not %errorlevel%==0 (
    echo ERROR: Node.js 22 or newer is required. Install it from
    echo        https://nodejs.org and re-run install.bat.
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version') do echo  - Node.js %%v detected

:: --- Python virtual environment -------------------------------------------
echo.
echo ===== Setting up the Python environment (.venv)
if exist ".venv\Scripts\python.exe" (
    echo  - .venv already exists; keeping it. Delete the folder to rebuild.
    goto venvready
)
echo  - creating .venv
python -m venv .venv
if not %errorlevel%==0 (
    echo ERROR: failed to create .venv.
    exit /b 1
)

:venvready
echo  - bootstrapping pip
".venv\Scripts\python.exe" -m ensurepip --upgrade
if not %errorlevel%==0 (
    echo ERROR: failed to bootstrap pip in .venv.
    exit /b 1
)

echo  - installing requirements.txt (this can take a while)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if not %errorlevel%==0 (
    echo ERROR: failed to install requirements.txt.
    exit /b 1
)

:: --- Hyperframes motion-graphics toolchain (required) ---------------------
echo.
echo ===== Setting up the motion-graphics toolchain (hyperframes)
call setup-hyperframes.bat
if not %errorlevel%==0 (
    echo ERROR: setup-hyperframes.bat failed.
    exit /b 1
)

:: --- Optional: local Russian TTS (Qwen3-TTS) ------------------------------
echo.
echo ===== Optional components
set "QWEN="
set /p QWEN="Install local Russian TTS (Qwen3-TTS, several GB)? [y/N]: "
if /i "%QWEN%"=="y" (
    call setup-qwen.bat
) else (
    echo  - skipping Qwen3-TTS.
)

:: --- Optional: Wav2Lip avatar fallback ------------------------------------
set "AVATAR="
set /p AVATAR="Install Wav2Lip avatar fallback (~400MB model + deps)? [y/N]: "
if /i "%AVATAR%"=="y" (
    call setup-avatar.bat
) else (
    echo  - skipping Wav2Lip avatar fallback.
)

:: --- config.toml ----------------------------------------------------------
echo.
echo ===== Configuration
if exist "config.toml" (
    echo  - config.toml already exists; leaving it untouched.
    goto autostart
)
echo  - creating config.toml from config.example.toml
copy /y "config.example.toml" "config.toml" >nul
if not exist "config.toml" (
    echo ERROR: failed to create config.toml.
    exit /b 1
)
set "TGTOKEN="
set /p TGTOKEN="Paste your Telegram bot token from @BotFather (Enter to skip): "
if not defined TGTOKEN (
    echo  - no token entered; set [telegram] bot_token in config.toml later.
    goto autostart
)
:: Token is passed to PowerShell via the live environment (no delayed expansion
:: needed). The literal `bot_token = ""` exists only in the [telegram] block.
:: Single-arg -Command (multi-arg form broke dynamic param binding in PS 5.1);
:: no $ anchor in the regex: .NET multiline $ does not match before CRLF.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='config.toml'; $c=[IO.File]::ReadAllText($p); $n=$c -replace '(?m)^bot_token = \"\"', ('bot_token = \"'+$env:TGTOKEN+'\"'); if ($n -eq $c) { exit 1 }; [IO.File]::WriteAllText($p, $n, (New-Object Text.UTF8Encoding($false))); Write-Host '  - Telegram bot token written to [telegram].bot_token'"
if not %errorlevel%==0 echo  - WARNING: could not write the token; edit [telegram] bot_token in config.toml manually.

:: --- Autostart at logon via Task Scheduler --------------------------------
:autostart
echo.
echo ===== Autostart
schtasks /Query /TN "MoneyPrinterTurboBot" >nul 2>&1
if %errorlevel%==0 (
    echo  - scheduled task "MoneyPrinterTurboBot" already registered.
    goto finish
)
set "AUTOSTART="
set /p AUTOSTART="Start the Telegram bot automatically at logon? [y/N]: "
if /i not "%AUTOSTART%"=="y" (
    echo  - skipping autostart. Run run-bot.bat manually to start the bot.
    goto finish
)
schtasks /Create /TN "MoneyPrinterTurboBot" /TR "\"%~dp0run-bot.bat\"" /SC ONLOGON /F
if not %errorlevel%==0 (
    echo  - WARNING: failed to register the scheduled task. You can start the
    echo            bot manually any time with run-bot.bat.
) else (
    echo  - registered. The bot will start at your next logon.
)

:finish
echo.
echo ============================================================
echo  Done. Starting the bot now in a new window...
echo  Control everything from Telegram: /make ^<topic^>, /news, /status.
echo ============================================================
start "MoneyPrinterTurbo Bot" "%~dp0run-bot.bat"
echo.
pause
