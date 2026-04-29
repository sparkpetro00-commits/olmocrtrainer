@echo off
setlocal EnableDelayedExpansion

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=olmocr_trainer"

cd /d "%~dp0"

set "CONDA_BAT="
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\anaconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\anaconda3\condabin\conda.bat"

if not defined CONDA_BAT (
  echo ERROR: Could not find conda.bat.
  pause
  exit /b 1
)

echo Ensuring trainer env: %ENV_NAME%
call "%CONDA_BAT%" run -n "%ENV_NAME%" python --version >nul 2>&1
if errorlevel 1 (
  echo Creating env %ENV_NAME% ...
  call "%CONDA_BAT%" create -n "%ENV_NAME%" python=3.11 tk -y
  if errorlevel 1 (
    echo Failed to create environment.
    pause
    exit /b 1
  )
)

echo Installing trainer dependencies...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --upgrade pip wheel "setuptools<82"
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128"
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install transformers==4.57.3 accelerate huggingface-hub compressed-tensors pypdf pdf2image pypdfium2 pillow

echo Verifying GPU + PDF toolchain...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -c "import torch,shutil; print('Torch:',torch.__version__,'CUDA:',torch.version.cuda,'available:',torch.cuda.is_available(),'GPU:',torch.cuda.device_count()); print('pdfinfo:',shutil.which('pdfinfo')); print('pdftoppm:',shutil.which('pdftoppm'))"
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -c "import transformers,torch,torchvision,compressed_tensors; from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; print('transformers',transformers.__version__,'qwen2.5-vl import OK')"
if errorlevel 1 (
  echo.
  echo ERROR: Qwen2.5-VL runtime import failed.
  echo Try upgrading transformers in this env:
  echo   conda run -n %ENV_NAME% python -m pip install --upgrade "transformers>=4.57.3"
  echo Then rerun this launcher.
  pause
  exit /b 1
)

set "POPPLER_BIN="
if exist "C:\Users\admin\Tools\poppler-25.12.0\Library\bin\pdfinfo.exe" set "POPPLER_BIN=C:\Users\admin\Tools\poppler-25.12.0\Library\bin"
if not defined POPPLER_BIN if exist "C:\Users\admin\Tools\poppler-25.12.0\poppler-25.12.0\Library\bin\pdfinfo.exe" set "POPPLER_BIN=C:\Users\admin\Tools\poppler-25.12.0\poppler-25.12.0\Library\bin"
if not defined POPPLER_BIN if exist "%USERPROFILE%\Tools\poppler-25.12.0\Library\bin\pdfinfo.exe" set "POPPLER_BIN=%USERPROFILE%\Tools\poppler-25.12.0\Library\bin"
if not defined POPPLER_BIN if exist "%USERPROFILE%\Tools\poppler-25.12.0\poppler-25.12.0\Library\bin\pdfinfo.exe" set "POPPLER_BIN=%USERPROFILE%\Tools\poppler-25.12.0\poppler-25.12.0\Library\bin"
if not defined POPPLER_BIN if exist "%USERPROFILE%\Downloads\poppler-25.12.0\Library\bin\pdfinfo.exe" set "POPPLER_BIN=%USERPROFILE%\Downloads\poppler-25.12.0\Library\bin"
if not defined POPPLER_BIN if exist "%USERPROFILE%\Downloads\poppler-25.12.0\poppler-25.12.0\Library\bin\pdfinfo.exe" set "POPPLER_BIN=%USERPROFILE%\Downloads\poppler-25.12.0\poppler-25.12.0\Library\bin"
if defined POPPLER_BIN (
  echo Detected Poppler at: %POPPLER_BIN%
  set "PATH=%POPPLER_BIN%;%PATH%"
) else (
  echo Poppler auto-detect did not find a known path.
)

where pdfinfo >nul 2>&1
if errorlevel 1 (
  echo.
  echo WARNING: Poppler binaries not found in PATH ^(pdfinfo/pdftoppm^).
  echo Install with: winget install -e --id oschwartz10612.poppler
  echo Then restart terminal and rerun this launcher.
  echo.
)

echo Starting olmocr_trainer.py ...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python "olmocr_trainer.py"

endlocal
