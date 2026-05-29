@echo off
cd /d "%~dp0"

echo ============================================================
echo  Hyperframes motion-graphics video setup
echo  Scaffolds a local composition project (.hyperframes), fetches
echo  a headless Chrome, and installs a static ffmpeg+ffprobe so the
echo  app can render HTML/GSAP compositions to MP4 -- all offline
echo  after this one-time setup.
echo ============================================================

set HF_VERSION=0.6.58
set PROJ=.hyperframes

:: --- Node.js (>=22) is required by the hyperframes CLI ---------------------
where node >nul 2>&1
if not %errorlevel%==0 (
    echo ERROR: Node.js not found. Install Node.js 22+ from https://nodejs.org first.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version') do echo  - Node.js %%v detected

:: --- Scaffold the composition project --------------------------------------
if exist "%PROJ%\index.html" (
    echo  - %PROJ% already exists; keeping it. Delete the folder to re-scaffold.
) else (
    echo  - scaffolding %PROJ% (blank, portrait 1080x1920)
    call npx --yes hyperframes@%HF_VERSION% init %PROJ% --example blank --resolution portrait --non-interactive --skip-skills
    if not %errorlevel%==0 (
        echo ERROR: hyperframes init failed.
        pause
        exit /b 1
    )
)

:: --- Headless Chrome (cached under %USERPROFILE%\.cache\hyperframes) --------
echo  - ensuring headless Chrome for rendering
call npx --yes hyperframes@%HF_VERSION% browser ensure

:: --- ffmpeg + ffprobe -------------------------------------------------------
:: hyperframes needs BOTH on PATH; the app's bundled imageio-ffmpeg has no
:: ffprobe, so install a static build into %PROJ%\bin (render.py prepends it
:: to PATH). Skip if already present there or on the system PATH.
if exist "%PROJ%\bin\ffprobe.exe" (
    echo  - ffmpeg already installed in %PROJ%\bin
    goto done
)
where ffprobe >nul 2>&1
if %errorlevel%==0 (
    echo  - ffprobe found on PATH; skipping bundled ffmpeg download
    goto done
)

echo  - downloading static ffmpeg+ffprobe from GitHub (one-time, ~100MB)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';" ^
  "$u='https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip';" ^
  "$z=Join-Path $env:TEMP 'hf-ffmpeg.zip'; $d=Join-Path $env:TEMP 'hf-ffmpeg';" ^
  "Invoke-WebRequest -Uri $u -OutFile $z;" ^
  "Expand-Archive -Path $z -DestinationPath $d -Force;" ^
  "$bin=(Get-ChildItem -Recurse -Filter 'ffprobe.exe' $d | Select-Object -First 1).Directory.FullName;" ^
  "New-Item -ItemType Directory -Force '%PROJ%\bin' | Out-Null;" ^
  "Copy-Item (Join-Path $bin '*.exe') '%PROJ%\bin' -Force;" ^
  "Remove-Item $z -Force -ErrorAction SilentlyContinue;"
if not exist "%PROJ%\bin\ffprobe.exe" (
    echo ERROR: ffmpeg install failed. Install ffmpeg manually and add it to PATH.
    pause
    exit /b 1
)
echo  - ffmpeg installed to %PROJ%\bin

:done
echo.
echo Done. In the app, tick "Generate with Hyperframes (motion graphics)" to use it.
pause
