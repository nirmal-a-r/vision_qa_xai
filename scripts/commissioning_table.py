"""
commissioning_table.py - how many real defects does a new line need? (project
document Section 8.4; RQ4). Needs no data: the SPERC Tier H cap depends only on
m, N, alpha and beta (SPI Theorem 3.5), so a plant can plan before it collects a
single defect image.

    python scripts/commissioning_table.py               # prints, writes runs/commissioning_table.csv
    python scripts/commissioning_table.py --out my.csv
"""

import argparse
import csv
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.risk.sperc import commissioning_table   # noqa: E402


def crc_text(m, alpha):
    j = int((alpha * (m + 1)) + 1e-9)
    if j == 0:
        return "refuses"
    names = {1: "only the lowest real score"}
    return names.get(j, f"up to the {j}{'nd' if j == 2 else 'rd' if j == 3 else 'th'}-lowest real score")


def main(argv=None):
    try:
        from src.utils.winenv import setup_console
        setup_console()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "commissioning_table.csv"))
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)["commissioning"]
    rows = []
    for al in cfg["alphas"]:
        rows += commissioning_table(cfg["m_values"], cfg["N"], al, cfg["betas"])
    N = cfg["N"]
    print(f"N = {N} synthetic defects\n")
    print(f"{'m':>4} {'alpha':>6} {'beta':>5} {'Tier H cap':>11} {'Tier H floor':>13} {'Tier N bound':>13} {'CRC can issue?':>15}")
    for r in rows:
        print(f"{r['m']:>4} {r['alpha']:>6.2f} {r['beta']:>5.2f} {r['tier_h_cap']:>11.3f}"
              f" {r['tier_h_floor']:>13.3f} {r['tier_n_bound']:>13.2f} {str(r['crc_issuable_at_alpha']):>15}")
    # the document's Section 8.4 layout
    get = {(r["m"], r["alpha"], r["beta"]): r["tier_h_cap"] for r in rows}
    print("\nSection 8.4 layout:")
    print(f"{'m':>4} | {'CRC at alpha = 0.10':<30} | {'a0.05 b0.05':>11} | {'a0.05 b0.20':>11} | {'a0.10 b0.05':>11}")
    for m in cfg["m_values"]:
        cells = [get.get((m, 0.05, 0.05)), get.get((m, 0.05, 0.20)), get.get((m, 0.10, 0.05))]
        print(f"{m:>4} | {crc_text(m, 0.10):<30} | " + " | ".join(
            f"{c:>11.3f}" if c is not None else f"{'-':>11}" for c in cells))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
