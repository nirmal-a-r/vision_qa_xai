"""Generate the reproducible certificate-quality audit used in the notebook."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.rigor import audit_prediction_directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="runs/preds")
    ap.add_argument("--out", default="runs/rigor_metrics.json")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report = audit_prediction_directory(args.pred_dir, args.out, n_boot=args.bootstrap)
    print(f"wrote {args.out}: {len(report['pairs'])} prediction pairs audited")


if __name__ == "__main__":
    main()
