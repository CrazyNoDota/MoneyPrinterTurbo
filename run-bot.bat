@echo off
cd /d "%~dp0"

:: MoneyPrinterTurbo Telegram bot worker, with an auto-restart loop so a crash
:: (or a transient network drop) never leaves the bot offline. Registered to run
:: at logon by install.bat via Task Scheduler. stdout+stderr are appended to
:: storage\logs\bot.log.

if not exist "storage\logs" mkdir "storage\logs"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run install.bat first.>>"storage\logs\bot.log"
    echo ERROR: .venv not found. Run install.bat first.
    exit /b 1
)

:loop
echo [%date% %time%] starting bot worker (python -m app.bot)>>"storage\logs\bot.log"
".venv\Scripts\python.exe" -m app.bot >>"storage\logs\bot.log" 2>&1
echo [%date% %time%] bot exited; restarting in 5s>>"storage\logs\bot.log"
timeout /t 5 /nobreak >nul
goto loop
