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
rem 2) Every pipeline stage on a tiny copy of the data (about 10-15 minutes on the GPU).
rem    Touches nothing in data\ or runs\; works in runs_smoke\.
echo Running the smoke test - log: runs\smoke_log.txt
%PY% scripts\smoke_test.py --device 0 > runs\smoke_log.txt 2>&1
findstr /L /C:"[ok]" /C:"[FAIL]" /C:"SMOKE TEST" runs\smoke_log.txt
echo.
echo Full log: runs\smoke_log.txt   Next (if PASSED): 3_run_full_pipeline.bat
pause
