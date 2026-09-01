#!/usr/bin/env bash
# Full project run: training -> analysis -> notebook.
#
# Ordered so a COMPLETE deliverable lands early rather than only at the end.
# The previous attempt ran dataset-major within each seed, so when it was
# stopped at hour 1 it had produced no kolektor model at all and left the
# notebook thinner than before it started. This runs seed 0 across everything
# first, then re-runs analysis and the notebook, and only then spends the
# remaining ~12 h on seeds 1 and 2.
#
# Every stage is resumable: train_baselines skips (dataset, model, encoding,
# seed) combinations already present in results_*.json, and refuses to record a
# run that covered less than a third of its schedule.
set -u
cd /c/Users/nirma/Desktop/vision_qa_xai
PY=./vqaenv/Scripts/python.exe
DS=neu,magnetic_tile,kolektor

banner () { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

train_seed () {
  local SEED=$1
  banner "SEED $SEED : baseline"
  $PY -u -m src.evaluation.train_baselines --model yolov8s.pt \
      --datasets $DS --seeds "$SEED" --results runs/results_yolov8s.json
  banner "SEED $SEED : CCE"
  $PY -u -m src.evaluation.train_baselines --model yolov8s.pt --encoding cce \
      --yolo_dir data/yolo_cce --datasets $DS --seeds "$SEED" \
      --results runs/results_yolov8s.json
}

analyse_and_build () {
  banner "ANALYSIS (predictions -> risk / drift / triage)"
  $PY -u scripts/run_all_experiments.py
  banner "SEED SUMMARY"
  $PY -u -m src.evaluation.seed_analysis
  banner "NOTEBOOK build + execute"
  $PY -u scripts/build_certified_notebook.py
  $PY -u - <<'PYEOF'
import nbformat, io, os, time
from nbclient import NotebookClient
t0 = time.time()
nb = nbformat.read(io.open('notebook/VisionQA_CertifiedInspection.ipynb', encoding='utf-8'), as_version=4)
c = NotebookClient(nb, timeout=1800, kernel_name='vision_qa_xai',
                   resources={'metadata': {'path': os.getcwd()}}, allow_errors=True)
c.execute()
nbformat.write(nb, io.open('notebook/VisionQA_CertifiedInspection_executed.ipynb', 'w', encoding='utf-8'))
code = [x for x in nb.cells if x.cell_type == 'code']
figs = sum(1 for x in code for o in x.get('outputs', []) if 'image/png' in o.get('data', {}))
errs = [(i, o) for i, x in enumerate(nb.cells) if x.cell_type == 'code'
        for o in x.get('outputs', []) if o.output_type == 'error']
print(f"NOTEBOOK cells={len(code)} figures={figs} errors={len(errs)} in {(time.time()-t0)/60:.1f} min")
for i, o in errs[:8]:
    print(f"  cell {i}: {o.get('ename')}: {str(o.get('evalue'))[:150]}")
PYEOF
}

# ---- phase 1: complete seed 0, then produce a full deliverable -------------
train_seed 0
analyse_and_build
banner "PHASE 1 COMPLETE - notebook is current with a full single-seed picture"

# ---- phase 2: the remaining seeds, then rebuild with error bars ------------
train_seed 1
train_seed 2
analyse_and_build
banner "FULL RUN COMPLETE"
