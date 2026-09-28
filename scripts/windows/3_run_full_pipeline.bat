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
rem 3) Stage 3 onwards: prepare data, train RT-DETR-L + YOLOv8s x 5 datasets x 3 seeds
rem    (primary datasets first), cache detections, Stage 1 go / no-go, sweeps, summary,
rem    notebook. One to two days on an RTX 5060. Resumable: if it stops, run it again.
rem    The PC is kept awake while it runs; keep the laptop plugged in.
echo Full run started %DATE% %TIME% - log: runs\pipeline_log.txt
echo (watch progress with:  powershell Get-Content runs\pipeline_log.txt -Wait -Tail 30)
%PY% scripts\run_pipeline.py --wait-for-gpu >> runs\pipeline_log.txt 2>&1
echo Finished %DATE% %TIME%. Next: 4_generators_and_p2.bat, then 5_run_notebook.bat
pause
