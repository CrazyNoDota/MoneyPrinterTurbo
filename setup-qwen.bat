@echo off
cd /d "%~dp0"

echo ============================================================
echo  Qwen3-TTS local voice setup
echo  Creates an isolated environment (.venv-qwen) so the heavy
echo  Qwen dependencies never touch the main app.
echo  First run downloads a few GB; the ~1.8GB voice model is
echo  fetched the first time you actually synthesize.
echo ============================================================

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
if exist ".venv-qwen\Scripts\python.exe" (
    echo .venv-qwen already exists. Delete it to reinstall.
    goto done
)

%UV% venv .venv-qwen --python 3.11

:: GPU-aware torch: CUDA build when an NVIDIA GPU is detected (e.g. GTX 1070 Ti),
:: otherwise the CPU build. The worker falls back to CPU automatically anyway.
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo  - NVIDIA GPU detected: installing CUDA build of torch
    %UV% pip install --python ".venv-qwen\Scripts\python.exe" torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo  - No NVIDIA GPU detected: installing CPU build of torch
    %UV% pip install --python ".venv-qwen\Scripts\python.exe" torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)

%UV% pip install --python ".venv-qwen\Scripts\python.exe" -r requirements-qwen.txt

:done
echo.
echo Done. In the app, choose TTS Server = "Qwen3-TTS (local)" and pick a voice.
pause
