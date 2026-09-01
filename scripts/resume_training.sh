#!/usr/bin/env bash
# Resume the 3-seed sweep exactly where it stopped.
#
# Safe to run any number of times. train_baselines.py skips any
# (model, dataset, encoding, seed) already present in runs/results_*.json, and
# refuses to record a run that covered less than a third of its schedule - so an
# interrupted run is simply redone rather than silently kept as a partial result.
#
# Granularity of resume is one RUN, not one epoch: whatever was mid-training when
# you stopped starts again from epoch 1. Everything already finished is kept.
cd "$(dirname "$0")/.."
PY=./vqaenv/Scripts/python.exe
DS=neu,magnetic_tile,kolektor

for SEED in 0 1 2; do
  echo "########## SEED $SEED : baseline ##########"
  $PY -u -m src.evaluation.train_baselines --model yolov8s.pt --datasets $DS \
      --seeds $SEED --results runs/results_yolov8s.json
  echo "########## SEED $SEED : CCE ##########"
  $PY -u -m src.evaluation.train_baselines --model yolov8s.pt --encoding cce \
      --yolo_dir data/yolo_cce --datasets $DS --seeds $SEED \
      --results runs/results_yolov8s.json
done
echo "########## TRAINING COMPLETE ##########"
