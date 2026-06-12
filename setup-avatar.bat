@echo off
cd /d "%~dp0"

echo ============================================================
echo  Wav2Lip avatar fallback setup
echo  Creates an isolated environment (.venv-wav2lip), clones the
echo  Wav2Lip inference repo, installs its deps there, and downloads
echo  wav2lip_gan.pth. Azure TTS Avatar needs no setup here.
echo ============================================================

set WAV2LIP_DIR=.wav2lip\Wav2Lip
set WEIGHTS_DIR=models\wav2lip
set WEIGHTS=%WEIGHTS_DIR%\wav2lip_gan.pth
set WEIGHTS_URL=https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth
set MIN_BYTES=350000000
set MAX_BYTES=600000000
set EXPECTED_SHA256=

:: locate uv (same logic as start.bat)
if exist ".uv\uv.exe" (
    set UV=".uv\uv.exe"
    goto setup
)
where uv >nul 2>&1
if %errorlevel%==0 (
    set UV=uv
    goto setup
)
where python >nul 2>&1
if %errorlevel%==0 (
    python -m pip install uv --quiet
    set UV=python -m uv
    goto setup
)
echo ERROR: Python/uv not found. Install Python 3.11 first.
pause
exit /b 1

:setup
if exist ".venv-wav2lip\Scripts\python.exe" (
    echo  - .venv-wav2lip already exists; keeping it. Delete it to reinstall.
) else (
    %UV% venv .venv-wav2lip --python 3.11
)

if exist "%WAV2LIP_DIR%\inference.py" (
    echo  - Wav2Lip repo already exists; keeping it.
) else (
    where git >nul 2>&1
    if not %errorlevel%==0 (
        echo ERROR: git not found. Install Git for Windows first.
        pause
        exit /b 1
    )
    mkdir ".wav2lip" 2>nul
    git clone https://github.com/Rudrabha/Wav2Lip.git "%WAV2LIP_DIR%"
    if not %errorlevel%==0 (
        echo ERROR: failed to clone Wav2Lip.
        pause
        exit /b 1
    )
)

echo  - installing Wav2Lip Python dependencies in .venv-wav2lip
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo    NVIDIA GPU detected: installing CUDA torch
    %UV% pip install --python ".venv-wav2lip\Scripts\python.exe" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo    No NVIDIA GPU detected: installing CPU torch
    %UV% pip install --python ".venv-wav2lip\Scripts\python.exe" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)
%UV% pip install --python ".venv-wav2lip\Scripts\python.exe" -r "%WAV2LIP_DIR%\requirements.txt"
%UV% pip install --python ".venv-wav2lip\Scripts\python.exe" librosa==0.10.2.post1 opencv-python==4.10.0.84

if exist "%WEIGHTS%" (
    echo  - wav2lip_gan.pth already exists; verifying size.
) else (
    echo  - downloading wav2lip_gan.pth (~400MB)
    mkdir "%WEIGHTS_DIR%" 2>nul
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';" ^
      "Invoke-WebRequest -Uri '%WEIGHTS_URL%' -OutFile '%WEIGHTS%';"
    if not exist "%WEIGHTS%" (
        echo ERROR: failed to download %WEIGHTS%.
        pause
        exit /b 1
    )
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$p='%WEIGHTS%'; $min=[int64]%MIN_BYTES%; $max=[int64]%MAX_BYTES%;" ^
  "$size=(Get-Item $p).Length;" ^
  "if ($size -lt $min -or $size -gt $max) { throw \"Unexpected Wav2Lip weight size: $size bytes\" }" ^
  "$expected='%EXPECTED_SHA256%';" ^
  "if ($expected) { $actual=(Get-FileHash -Algorithm SHA256 $p).Hash.ToLowerInvariant(); if ($actual -ne $expected.ToLowerInvariant()) { throw \"SHA256 mismatch: $actual\" } }" ^
  "Write-Host \"  - weights verified: $size bytes\";"
if not %errorlevel%==0 (
    echo ERROR: weight verification failed. Delete %WEIGHTS% and run again.
    pause
    exit /b 1
)

echo.
echo Done. Set avatar_provider="wav2lip" or "auto" and avatar_portrait in config.toml.
pause
