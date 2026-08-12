@echo off
cd /d "%~dp0"
title STARXION
echo ==========================================
echo          STARXION
echo   AI COMPUTE INTEGRITY LAYER
echo ==========================================
echo.
python -c "import torch" >nul 2>&1
if errorlevel 1 (
  echo PyTorch is not installed.
  echo Installing it now...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Installation failed.
    pause
    exit /b 1
  )
)
echo.
echo Running STARXION benchmark...
python starxion.py
if errorlevel 1 (
  echo.
  echo STARXION failed.
  pause
  exit /b 1
)
echo.
echo Opening report...
start "" "starxion_report.html"
echo.
echo STARXION completed successfully.
pause
