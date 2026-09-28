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
rem 1) One-time setup: extras, notebook kernel, readiness check, correctness tests.
echo === 1/4 install the extras into vqaenv (diffusers, pytest, ...) ===
%PY% -m pip install -r requirements.txt
echo === 2/4 register the notebook kernel "Python (vision_qa_xai)" ===
%PY% -m ipykernel install --user --name vision_qa_xai --display-name "Python (vision_qa_xai)"
echo === 3/4 readiness: GPU + sm_120, VRAM, RAM, long paths, power, datasets, splits ===
%PY% scripts\check_ready.py > runs\setup_check.log 2>&1
type runs\setup_check.log
echo === 4/4 correctness tests (no GPU) ===
%PY% -m pytest -q tests >> runs\setup_check.log 2>&1
%PY% -m pytest -q tests
echo.
echo Log: runs\setup_check.log   Next: 2_smoke_test.bat
pause
