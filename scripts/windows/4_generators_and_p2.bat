@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0\..\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PY=vqaenv\Scripts\python.exe
if not exist %PY% (
  echo vqaenv not found in %CD% - see README section 3.
  pause
  exit /b 1
)
if not exist runs mkdir runs
rem 4) Stages 4-6: synthetic defects for the primary datasets (training-free diffusion
rem    generator + copy-paste, N = 1000 each), scoring by every detector and seed, the
rem    headline sweep with stress tests, and protocol P2. Needs 3_run_full_pipeline first.
rem    The first run downloads the inpainting model (~5 GB) from Hugging Face.
echo Generators + P2 started %DATE% %TIME% - log: runs\generators_p2_log.txt
%PY% scripts\run_pipeline.py --skip-train --generate training_free,copy_paste --p2 --wait-for-gpu >> runs\generators_p2_log.txt 2>&1
echo Finished %DATE% %TIME%. Look at data\generated\*\*\_audit_sheet.png, then 5_run_notebook.bat
pause
