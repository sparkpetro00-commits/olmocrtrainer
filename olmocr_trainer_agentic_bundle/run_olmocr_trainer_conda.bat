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
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -m pip install transformers==4.57.3 accelerate huggingface-hub pypdf pdf2image pypdfium2 pillow

echo Verifying GPU + PDF toolchain...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -c "import torch,shutil; print('Torch:',torch.__version__,'CUDA:',torch.version.cuda,'available:',torch.cuda.is_available(),'GPU:',torch.cuda.device_count()); print('pdfinfo:',shutil.which('pdfinfo')); print('pdftoppm:',shutil.which('pdftoppm'))"

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
