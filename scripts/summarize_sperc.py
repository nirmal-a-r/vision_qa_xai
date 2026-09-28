"""
summarize_sperc.py - pool the cold-start sweeps of every detector and seed into
the tables the paper reports (project document, Sections 3 and 12.2-12.3).

Reads runs/coldstart/coldstart_<tag>.json, _mstar.csv and _alignment.csv for
every tag present and writes, in runs/coldstart/summary/:

  seeds_summary.csv   per (model, dataset, kind, source, m, method, beta):
                      mean and sd ACROSS SEEDS of escape / review / issuance,
                      with the number of seeds and draws per seed (Section 12.3)
  mstar_summary.csv   m-star per seed, and mean / min / max across seeds (RQ1, RQ4)
  validity.csv        RQ2: every SPERC row's realised escape vs its Tier H cap,
                      stress sources flagged; the fraction beyond 3 SE must be ~0
  planned_vs_realised.csv  RQ4: commissioning (planned) cap vs worst realised escape
  alignment_vs_mstar.csv   RQ3: KS distance real-vs-synthetic per generator source
                      vs SPERC's m-star, with a Spearman correlation per model
  summary.json        headline numbers

    python scripts/summarize_sperc.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_coldstart_sweep import write_csv   # noqa: E402


def _read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def base_model(tag):
    import re
    return re.sub(r"_s\d+$", "", tag)


def load(out_dir):
    rows, mstar, align = [], [], []
    for p in sorted(glob.glob(os.path.join(out_dir, "coldstart_*.json"))):
        with open(p) as f:
            rows += json.load(f)
        stem = p[:-5]
        if os.path.exists(stem + "_mstar.csv"):
            mstar += _read_csv(stem + "_mstar.csv")
        if os.path.exists(stem + "_alignment.csv"):
            align += _read_csv(stem + "_alignment.csv")
    return rows, mstar, align


def seeds_summary(rows):
    g = defaultdict(list)
    for r in rows:
        k = (base_model(r["model"]), r["dataset"], r["kind"], r["source"], r["m"], r["method"], r["beta"])
        g[k].append(r)
    out = []
    for k, rs in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        def ms(key):
            v = np.array([x[key] for x in rs if x.get(key) is not None], float)
            return (float(v.mean()) if v.size else None,
                    float(v.std(ddof=1)) if v.size > 1 else None)
        e, esd = ms("escape")
        rv, rsd = ms("review")
        iss, isd = ms("issued")
        out.append({"model": k[0], "dataset": k[1], "kind": k[2], "source": k[3], "m": k[4],
                    "method": k[5], "beta": k[6], "n_seeds": len({x.get("seed", 0) for x in rs}),
                    "draws_per_seed": int(np.median([x["draws"] for x in rs])),
                    "escape_mean": e, "escape_sd_seeds": esd, "review_mean": rv,
                    "review_sd_seeds": rsd, "issued_mean": iss, "issued_sd_seeds": isd,
                    "tier_h_cap": rs[0].get("tier_h_cap"), "target": rs[0].get("target"),
                    "any_hard_bound_violation": any(bool(x.get("exceeds_hard_bound")) for x in rs)})
    return out


def mstar_summary(mstar):
    g = defaultdict(list)
    for r in mstar:
        k = (base_model(r["model"]), r["dataset"], r["kind"], r["source"], r["method"], r["beta"], r["alpha_max"])
        g[k].append(r)
    out = []
    for k, rs in sorted(g.items()):
        vals = [_num(x["m_star"]) for x in rs]
        got = [v for v in vals if v is not None]
        out.append({"model": k[0], "dataset": k[1], "kind": k[2], "source": k[3], "method": k[4],
                    "beta": k[5], "alpha_max": k[6], "n_seeds": len(rs),
                    "m_star_per_seed": ";".join("-" if v is None else str(int(v)) for v in vals),
                    "reached_in_seeds": len(got),
                    "m_star_mean": float(np.mean(got)) if got else None,
                    "m_star_min": min(got) if got else None, "m_star_max": max(got) if got else None,
                    "valid_guarantee": rs[0]["valid_guarantee"]})
    return out


HARD_CAP_METHODS = ("crc", "crc_at_cap", "sperc_tierN", "sperc_nominal")


def holm(pvals, level=0.05):
    """Holm-Bonferroni: boolean 'rejected' per p-value at family-wise level."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    rej = np.zeros(p.size, bool)
    for k, i in enumerate(order):
        if p[i] <= level / (p.size - k):
            rej[i] = True
        else:
            break
    return rej


def validity(rows, level=0.05):
    """RQ2: every row that carries a hard escape bound, with a one-sided test of
    'mean escape > cap' and a Holm correction over all of them (Section 12, Stage 5:
    'realised escape above Tier H beyond sampling error = the claim fails')."""
    from scipy.stats import norm
    out = []
    for r in rows:
        if r["method"] not in HARD_CAP_METHODS or r.get("tier_h_cap") is None or r.get("escape") is None:
            continue
        se = r.get("escape_se") or 0.0
        z = (r["escape"] - r["tier_h_cap"]) / se if se > 0 else (np.inf if r["escape"] > r["tier_h_cap"] + 1e-12 else -np.inf)
        out.append({"model": r["model"], "dataset": r["dataset"], "kind": r["kind"],
                    "source": r["source"], "source_role": r.get("source_role"), "m": r["m"],
                    "method": r["method"], "beta": r["beta"], "escape": r["escape"],
                    "escape_se": se, "tier_h_cap": r["tier_h_cap"],
                    "margin_to_cap": r["tier_h_cap"] - r["escape"], "z_over_cap": float(z),
                    "p_one_sided": float(norm.sf(z)),
                    "exceeds_hard_bound": r["exceeds_hard_bound"]})
    if out:
        rej = holm([v["p_one_sided"] for v in out], level)
        for v, j in zip(out, rej):
            v["holm_violation"] = bool(j)
    return out


def planned_vs_realised(rows):
    """RQ4: the cap a plant plans with vs the escape actually realised.

    For every SPERC row (Tier N = target) the planned cap is the Tier H formula at
    that row's (m, N, alpha, beta) - computable before any score is seen - and the
    realised value is its mean escape. Per (detector, dataset, kind, m, beta) the
    source with the largest excess over its own plan is reported; the plan and the
    outcome agree when that excess is within 3 standard errors. The N = 1000 plan
    of the commissioning table is shown alongside for reference.
    """
    from src.risk.spi_exact import tier_h_cap
    g = defaultdict(list)
    for r in rows:
        if r["method"] == "sperc_tierN" and r.get("tier_h_cap") is not None:
            g[(base_model(r["model"]), r["dataset"], r["kind"], r["m"], r["beta"])].append(r)
    out = []
    for k, rs in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        worst = max(rs, key=lambda x: x["escape"] - x["tier_h_cap"])
        out.append({"model": k[0], "dataset": k[1], "kind": k[2], "m": k[3], "beta": k[4],
                    "alpha": worst["alpha"],
                    "planned_cap_N1000": tier_h_cap(int(k[3]), 1000, worst["alpha"], k[4]),
                    "worst_source": worst["source"], "N_of_worst": worst.get("N_mean"),
                    "planned_cap_of_worst": worst["tier_h_cap"],
                    "realised_escape_of_worst": worst["escape"], "se": worst["escape_se"],
                    "agree": bool(worst["escape"] - 3 * (worst["escape_se"] or 0)
                                  <= worst["tier_h_cap"] + 1e-12)})
    return out


def alignment_vs_mstar(align, mstar, alpha_max, beta):
    """RQ3: does real-vs-synthetic KS distance predict m-star?"""
    from scipy.stats import spearmanr
    ms = {(r["model"], r["dataset"], r["kind"], r["source"]): _num(r["m_star"]) for r in mstar
          if r["method"] == "sperc_tierN" and _num(r["beta"]) == beta and _num(r["alpha_max"]) == alpha_max}
    out = []
    for a in align:
        if a["source"] == "real_as_synthetic":
            continue
        k = (a["model"], a["dataset"], a["kind"], a["source"])
        out.append({"model": a["model"], "dataset": a["dataset"], "kind": a["kind"],
                    "source": a["source"], "source_role": a["source_role"],
                    "ks_full_pool": _num(a["ks_full_pool"]), "N": a["N"],
                    "m_star": ms.get(k), "alpha_max": alpha_max, "beta": beta})
    corr = {}
    for mod in sorted({base_model(r["model"]) for r in out}):
        pts = [(r["ks_full_pool"], r["m_star"] if r["m_star"] is not None else 1e6) for r in out
               if base_model(r["model"]) == mod and r["kind"] == "part"]
        if len(pts) >= 4 and len({p[0] for p in pts}) > 1:
            rho, p = spearmanr([x for x, _ in pts], [y for _, y in pts])
            corr[mod] = {"spearman_rho": float(rho), "p_value": float(p), "n_points": len(pts),
                         "note": "m-star not reached counted as +infinity (rank-based)"}
    return out, corr


def main(argv=None):
    try:
        from src.utils.winenv import setup_console
        setup_console()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "coldstart"))
    ap.add_argument("--alpha_max", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=0.05)
    a = ap.parse_args(argv)
    rows, mstar, align = load(a.out_dir)
    if not rows:
        print(f"no sweep results in {a.out_dir}")
        return 1
    dst = os.path.join(a.out_dir, "summary")
    os.makedirs(dst, exist_ok=True)
    ss, msum, val, pvr = seeds_summary(rows), mstar_summary(mstar), validity(rows), planned_vs_realised(rows)
    avm, corr = alignment_vs_mstar(align, mstar, a.alpha_max, a.beta)
    for name, data in (("seeds_summary", ss), ("mstar_summary", msum), ("validity", val),
                       ("planned_vs_realised", pvr), ("alignment_vs_mstar", avm)):
        if data:
            write_csv(data, os.path.join(dst, name + ".csv"))
    viol = [v for v in val if v["exceeds_hard_bound"]]
    holm_v = [v for v in val if v.get("holm_violation")]
    head = {"tags": sorted({r["model"] for r in rows}), "rows": len(rows),
            "hard_cap_rows": len(val), "sperc_rows": sum(1 for v in val if v["method"].startswith("sperc")),
            "rows_above_cap_by_3se": len(viol),
            "expected_by_chance_3se": round(len(val) * 0.00135, 2),
            "holm_significant_violations": len(holm_v),
            "rq2_verdict": "claim holds" if not holm_v else "CLAIM FAILS",
            "stress_rows": sum(1 for v in val if v["source_role"] == "stress"),
            "planned_vs_realised_disagreements": sum(1 for p in pvr if not p["agree"]),
            "alignment_correlation": corr}
    with open(os.path.join(dst, "summary.json"), "w") as f:
        json.dump(head, f, indent=1)
    print(json.dumps(head, indent=1))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
