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
rem 5) Rebuild and run the notebook headless; the executed copy opens in VS Code / Jupyter.
%PY% scripts\build_certified_notebook.py
%PY% scripts\execute_notebook.py
echo Executed copy: notebook\VisionQA_CertifiedInspection_executed.ipynb
where code >nul 2>&1 && code notebook\VisionQA_CertifiedInspection_executed.ipynb
pause
