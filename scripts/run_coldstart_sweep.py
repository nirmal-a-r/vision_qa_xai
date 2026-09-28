"""
run_coldstart_sweep.py - the headline experiment (cold-start protocol P1, project
document Sections 9.2 and 12; also P2 via --model <tag>).

For each detector (model x seed), dataset, escape kind, synthetic source and
number of REAL defective calibration images m, the same random draws of m real
defects are scored by every method (paired comparison, Section 12.1):

  crc            1. conformal risk control at the target, real data only
  naive_pooled   2. real + synthetic pooled into CRC - no valid guarantee
  legacy         3. the old spi.py heuristic - never beats CRC
  acq_uncorrected 4. defect-seeking acquisition (screen on detector score until m
                     defects are labelled), CRC without bias correction
  crc @ m=full   5. the oracle: CRC on the full real calibration pool
  sperc_tierN    6. SPERC at alpha = target - beta: Tier N bound equals the target
  sperc_nominal     SPERC at alpha = target: Tier N = target + beta
  crc_at_cap        CRC at sperc_tierN's Tier H cap: the SAME hard guarantee (C3)

Metrics (Section 12.2), all on held-out images the detector never saw: realised
escape on defective parts (mean over draws, standard error, 95% interval and a
pooled Wilson interval), issuance rate (with Wilson interval), review load on all
held-out parts, parts labelled to obtain the m defects, Tier H cap / Tier N bound,
KS alignment between real and synthetic escape scores, and paired review-load gains
of SPERC over CRC. Refusals count as escape 0 / review 1 and are never dropped
(Section 12.3). Each draw re-partitions the held-out calibration + test images
(same sizes), so the mean over draws estimates exactly what the theorems bound.
Ties are broken at random (Section 8.5, assumption 4). A hard-bound violation is
flagged only when the mean exceeds the Tier H cap by more than 3 standard errors.
SPERC rows are computed WITHOUT the plant's alpha_max; every analysis (m-star,
Stage 1) applies "refuse if Tier H cap > alpha_max" to the recorded cap, so one
sweep serves every tolerance in coldstart.mstar_alpha_max_values.

Synthetic sources
  real_as_synthetic  unused real calibration defects stand in (Stage 1, best case)
  uniform_noise      N random scores over the observed score range (Stage 5 stress)
  cross:<dataset>    another line's real defects - misaligned data (Stage 5 stress)
  <name>             generator output scored by src/evaluation/score_synthetic.py:
                     <preds_dir>/<dataset>_<tag>_syn_<name>.json

Outputs (in <out_dir>), one set per detector tag (<model> or <model>_s<seed>)
  coldstart_<tag>.json / .csv        one row per (dataset, kind, source, m, method, beta)
  coldstart_<tag>_mstar.csv          real defects needed to operate at the target
  coldstart_<tag>_alignment.csv      dataset-level KS distance per synthetic source

    python scripts/run_coldstart_sweep.py                          # models x seeds from config
    python scripts/run_coldstart_sweep.py --datasets gc10 --sources real_as_synthetic --n_draws 200
    python scripts/run_coldstart_sweep.py --model yolov8s_p2_training_free   # one explicit tag
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import yaml
from scipy.stats import ks_2samp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.experiments import load_preds                      # noqa: E402
from src.evaluation.rigor import wilson_interval                        # noqa: E402
from src.risk.baselines import (legacy_nearest_rank_threshold,         # noqa: E402
                                naive_pooled_crc_threshold)
from src.risk.escape import break_ties, escape_score, escape_scores     # noqa: E402
from src.risk.sperc import certify_crc, certify_sperc                   # noqa: E402
from src.risk.spi_exact import tier_n_interval                          # noqa: E402

NOISE_N = 1000
TIE_SCALE = 1e-7


def _find_cache(preds_dir, dataset, tag, split):
    for p in (os.path.join(preds_dir, f"{dataset}_{tag}_{split}.json"),
              os.path.join(preds_dir, f"preds_{dataset}_{tag}_{split}.json"),
              os.path.join(os.path.dirname(preds_dir), f"preds_{dataset}_{tag}_{split}.json")):
        if os.path.exists(p):
            return p
    return None


def _max_score(r):
    s = r.get("pred_scores", [])
    return float(max(s)) if len(s) else float("-inf")


def _finite(x):
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), x, -1.0)


def _crc_at(real, level):
    """CRC threshold at an arbitrary hard level (0 -> refuse, >= 1 -> accept all)."""
    if level <= 0:
        return float("-inf")
    if level >= 1:
        return float("inf")
    return certify_crc(real, level).threshold


def _summary(values):
    v = np.asarray(values, float)
    if v.size == 0:
        return None, None, None, None
    mean = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def seed_of_tag(tag: str) -> int:
    m = re.search(r"_s(\d+)$", tag)
    return int(m.group(1)) if m else 0


def detector_tags(models, seeds):
    """Cache tags written by dump_ultralytics_preds: seed 0 -> '<model>', else '<model>_s<k>'."""
    return [m if int(s) == 0 else f"{m}_s{int(s)}" for m in models for s in seeds]


def source_role(src, stress_names=()):
    if src == "real_as_synthetic":
        return "best_case"
    if src == "uniform_noise" or src.startswith("cross:") or src in stress_names or "undertrained" in src:
        return "stress"
    return "generator"


def _take_until(order, is_def, m):
    """First m defective indices along ``order`` and the number of parts labelled."""
    picked, labelled = [], 0
    for i in order:
        labelled += 1
        if is_def[i]:
            picked.append(i)
            if len(picked) == m:
                break
    return np.asarray(picked, int), labelled


def sweep_one(c_all, part_all, n_cal_rec, syn_fixed, source, m_values, n_draws, target,
              betas, alpha_max, rng, resample=True, tie_break=True, acq_gamma=3.0,
              stress_names=()):
    """Long-format rows for one (dataset, kind, source).

    c_all / part_all hold the escape score (nan for clean parts) and the highest
    detection score of every held-out image (calibration block first, then test
    block; n_cal_rec = size of the calibration block). With resample=True each
    draw re-partitions these held-out images into a calibration part and a test
    part of the original sizes before drawing m real defects.
    """
    rows = []
    n_all = c_all.size
    is_def_all = ~np.isnan(c_all)
    base_cal = int(is_def_all[:n_cal_rec].sum())
    fin = c_all[np.isfinite(c_all)]
    lo_s, hi_s = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    for m in m_values:
        label_m = base_cal if m == "full" else int(m)
        if label_m > base_cal or label_m < 1:
            continue
        acc = {}

        def add(method, beta, thr, issued, c_test, part_test, cap=None, tier_n=None,
                ks=None, alpha=None, N=None, labelled=None):
            a = acc.setdefault((method, beta), {"esc": [], "rev": [], "iss": [], "cap": [],
                                                "tier_n": tier_n, "ks": [], "alpha": alpha,
                                                "n_esc": 0, "n_def": 0, "N": [], "lab": []})
            if cap is not None:
                a["cap"].append(cap)
            esc = c_test < thr
            a["esc"].append(float(esc.mean()) if c_test.size else float("nan"))
            a["n_esc"] += int(esc.sum())
            a["n_def"] += int(c_test.size)
            a["rev"].append(float(np.mean(part_test >= thr)))
            a["iss"].append(bool(issued))
            if ks is not None:
                a["ks"].append(ks)
            if N is not None:
                a["N"].append(N)
            if labelled is not None:
                a["lab"].append(labelled)

        for _ in range(n_draws):
            if tie_break:
                u = rng.uniform(0.0, TIE_SCALE, size=n_all)
                c_d = break_ties(c_all, rng, u=u)
                part_d = break_ties(part_all, rng, u=u)
            else:
                c_d, part_d = c_all, part_all
            order = rng.permutation(n_all) if resample else np.arange(n_all)
            cal_part, test_part = order[:n_cal_rec], order[n_cal_rec:]
            is_def_cal = is_def_all[cal_part]
            pool_idx = cal_part[is_def_cal]                     # defective calibration images
            c_test = c_d[test_part][~np.isnan(c_d[test_part])]
            part_test = part_d[test_part]
            if pool_idx.size < 1 or c_test.size == 0:
                continue
            m_int = pool_idx.size if m == "full" else min(label_m, pool_idx.size)

            # random labelling: walk the calibration part in random order until m defects
            pos, labelled = _take_until(rng.permutation(cal_part.size), is_def_cal, m_int)
            real_idx = cal_part[pos]
            real = c_d[real_idx]
            c = certify_crc(real, target)
            add("crc", None, c.threshold, c.issued, c_test, part_test, cap=target,
                alpha=target, labelled=labelled)

            # baseline 4: defect-seeking acquisition without correction
            if m != "full" and acq_gamma is not None:
                s = np.clip(part_d[cal_part], 0.0, None) + 1e-3
                p = s ** acq_gamma
                p = p / p.sum()
                acq_order = rng.choice(cal_part.size, size=cal_part.size, replace=False, p=p)
                apos, alab = _take_until(acq_order, is_def_cal, m_int)
                ta = certify_crc(c_d[cal_part[apos]], target)
                add("acq_uncorrected", None, ta.threshold, ta.issued, c_test, part_test,
                    alpha=target, labelled=alab)

            if source == "real_as_synthetic":
                syn = c_d[np.setdiff1d(pool_idx, real_idx)]
            elif syn_fixed is None:
                syn = None
            else:
                syn = break_ties(syn_fixed, rng) if tie_break else syn_fixed
            if syn is None or syn.size < 2:
                continue
            N = int(syn.size)
            ks = float(ks_2samp(_finite(real), _finite(syn)).statistic)
            t = naive_pooled_crc_threshold(real, syn, target)
            add("naive_pooled", None, t, np.isfinite(t), c_test, part_test, alpha=target, N=N)
            t = legacy_nearest_rank_threshold(real, syn, target)
            add("legacy", None, t, np.isfinite(t), c_test, part_test, alpha=target, N=N)
            for b in betas:
                if target - b > 0:
                    # alpha_max is NOT applied here: whether a plant would refuse depends on
                    # its own tolerance, so m-star / Stage 1 apply it to the recorded cap.
                    s_ = certify_sperc(real, syn, target - b, beta=b)
                    add("sperc_tierN", b, s_.threshold, s_.issued, c_test, part_test,
                        s_.tier_h_cap, s_.tier_n_bound, ks, target - b, N, labelled)
                    cap = s_.tier_h_cap if s_.tier_h_cap is not None else 0.0
                    h = _crc_at(real, cap)
                    add("crc_at_cap", b, h, np.isfinite(h), c_test, part_test, cap,
                        alpha=cap, labelled=labelled)
                sn = certify_sperc(real, syn, target, beta=b)
                add("sperc_nominal", b, sn.threshold, sn.issued, c_test, part_test,
                    sn.tier_h_cap, sn.tier_n_bound, ks, target, N, labelled)

        ref = acc.get(("crc", None))
        for (method, beta), a in acc.items():
            em, ese, elo, ehi = _summary(a["esc"])
            rm, rse, _, _ = _summary(a["rev"])
            cap = float(np.mean(a["cap"])) if a["cap"] else None
            cap_max = float(np.max(a["cap"])) if a["cap"] else None
            wl, wh = wilson_interval(a["n_esc"], a["n_def"])
            il, ih = wilson_interval(int(np.sum(a["iss"])), len(a["iss"]))
            Nm = float(np.mean(a["N"])) if a["N"] else None
            tn_lo = (tier_n_interval(a["alpha"], beta, int(round(Nm)))[0]
                     if (method.startswith("sperc") and Nm) else None)
            row = {"source": source, "source_role": source_role(source, stress_names),
                   "m": label_m, "method": method, "beta": beta, "alpha": a["alpha"],
                   "draws": len(a["esc"]), "target": target, "resampled": resample,
                   "tie_break": tie_break, "N_mean": Nm,
                   "issued": float(np.mean(a["iss"])), "issued_wilson_low": il,
                   "issued_wilson_high": ih,
                   "escape": em, "escape_se": ese, "escape_ci_low": elo, "escape_ci_high": ehi,
                   "escape_wilson_low": wl, "escape_wilson_high": wh,
                   "review": rm, "review_se": rse,
                   "labelled_parts": float(np.mean(a["lab"])) if a["lab"] else None,
                   "tier_h_cap": cap, "tier_h_cap_max": cap_max,
                   "tier_n_bound": a["tier_n"], "tier_n_low": tn_lo,
                   "exceeds_hard_bound": (bool(em - 3 * ese > cap)
                                          if (cap is not None and em is not None) else None),
                   # z of the excess over the hard cap; summarize_sperc.py turns these into a
                   # multiplicity-corrected (Holm) verdict - with ~1000 rows, a per-row 3 SE flag
                   # fires about once by chance alone.
                   "z_over_cap": ((em - cap) / ese if (cap is not None and em is not None and ese)
                                  else None),
                   "inside_tier_n": (bool(tn_lo - 3 * ese <= em <= a["tier_n"] + 3 * ese)
                                     if (tn_lo is not None and em is not None) else None),
                   "ks_real_vs_syn": float(np.mean(a["ks"])) if a["ks"] else None}
            # paired review-load gains (positive = fewer parts sent to review)
            for name, other in (("crc", ref), ("crc_at_cap", acc.get(("crc_at_cap", beta)))):
                if (method.startswith("sperc") and other is not None
                        and len(other["rev"]) == len(a["rev"])):
                    d = np.asarray(other["rev"]) - np.asarray(a["rev"])
                    row[f"review_gain_vs_{name}"] = float(d.mean())
                    row[f"review_gain_vs_{name}_se"] = (float(d.std(ddof=1) / np.sqrt(d.size))
                                                        if d.size > 1 else 0.0)
            rows.append(row)
    return rows


def m_star(rows, target, alpha_max, review_tol=1.1, min_issued=0.8, review_target=None):
    """Real defects a method needs to reach a usable operating point (Section 12.2).

    m-star = smallest m at which a method (i) issues in >= min_issued of draws,
    (ii) keeps mean realised escape at or below the target (allowing 3 standard
    errors of resampling noise), (iii) reaches a review load at most
    ``review_target`` (absolute) or, if that is None, within review_tol x that of
    the oracle - CRC with the whole calibration pool - and (iv) for SPERC, has a
    Tier H cap <= alpha_max (the fixed hard guarantee the plant accepts).
    """
    oracle = {}
    for r in rows:
        if r["method"] == "crc":
            k = (r["dataset"], r["kind"], r["source"])
            if k not in oracle or r["m"] > oracle[k]["m"]:
                oracle[k] = r
    out, groups = [], {}
    for r in rows:
        groups.setdefault((r["dataset"], r["kind"], r["source"], r["method"], r["beta"]), []).append(r)
    for key, g in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        g = sorted(g, key=lambda r: r["m"])
        orc = oracle.get(key[:3])
        hit = None
        for r in g:
            if orc is None or r["escape"] is None or r["m"] >= orc["m"]:
                continue
            rev_ok = (r["review"] <= review_target if review_target is not None
                      else r["review"] <= review_tol * orc["review"])
            ok = (r["issued"] >= min_issued
                  and r["escape"] <= target + 3 * (r["escape_se"] or 0) and rev_ok)
            cap = r.get("tier_h_cap_max", r["tier_h_cap"])
            if r["method"].startswith("sperc") and (cap is None or cap > alpha_max + 1e-12):
                ok = False              # the plant would refuse this certificate
            if ok:
                hit = r
                break
        out.append({"dataset": key[0], "kind": key[1], "source": key[2], "method": key[3],
                    "beta": key[4], "alpha_max": alpha_max,
                    "valid_guarantee": key[3] not in ("naive_pooled", "acq_uncorrected"),
                    "review_rule": (f"<= {review_target}" if review_target is not None
                                    else f"<= {review_tol} x oracle"),
                    "m_star": hit["m"] if hit else None,
                    "escape_at_m_star": hit["escape"] if hit else None,
                    "review_at_m_star": hit["review"] if hit else None,
                    "labelled_at_m_star": hit.get("labelled_parts") if hit else None,
                    "tier_h_cap_at_m_star": hit["tier_h_cap"] if hit else None,
                    "oracle_m": orc["m"] if orc else None,
                    "oracle_review": orc["review"] if orc else None,
                    "oracle_escape": orc["escape"] if orc else None})
    return out


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


_write = write_csv     # backwards-compatible name


def run_tag(tag, cfg, datasets, base_sources, auto_sources, n_draws, preds_dir, out_dir,
            m_values=None):
    """Full sweep for one detector tag. Returns the rows (also written to disk)."""
    cc, cert = cfg["coldstart"], cfg["certificate"]
    betas = cert.get("betas") or [cert["beta"]]
    target, alpha_max = cert["target_escape"], cert["alpha_max"]
    seed = seed_of_tag(tag)
    rng = np.random.default_rng(int(cc["seed"]) + 1000 * seed)
    iou = cfg["iou_thresh"]
    m_values = m_values or cc["m_values"]
    stress_names = tuple(cfg.get("stress_generators", []))
    acq_gamma = cfg.get("baselines", {}).get("acquisition_gamma", 3.0)
    all_rows, align = [], []
    for ds in datasets:
        pc, pt = (_find_cache(preds_dir, ds, tag, s) for s in ("cal", "test"))
        if not (pc and pt):
            print(f"[{tag}/{ds}] no cached predictions in {preds_dir} - run the pipeline first")
            continue
        cal, test = load_preds(pc), load_preds(pt)
        held_out = cal + test                    # calibration block first
        part_all = np.array([_max_score(r) for r in held_out])
        sources = list(base_sources)
        if auto_sources:                          # also pick up generator output already scored
            prefix = f"{ds}_{tag}_syn_"
            for f in sorted(os.listdir(preds_dir)):
                if f.startswith(prefix) and f.endswith(".json") and f[len(prefix):-5] not in sources:
                    sources.append(f[len(prefix):-5])
        for kind in cfg["escape_kinds"]:
            c_all = np.array([escape_score(r, kind, iou) for r in held_out], dtype=float)
            real_pool = c_all[~np.isnan(c_all)]
            for src in sources:
                syn = None
                if src == "uniform_noise":
                    fin = real_pool[np.isfinite(real_pool)]
                    lo, hi = (fin.min(), fin.max()) if fin.size else (0.0, 1.0)
                    syn = np.random.default_rng(int(cc["seed"]) + 7).uniform(lo, hi, NOISE_N)
                elif src.startswith("cross:"):
                    other = src.split(":", 1)[1]
                    po = _find_cache(preds_dir, other, tag, "cal")
                    if other == ds or po is None:
                        continue
                    syn = escape_scores(load_preds(po), kind, iou)
                elif src != "real_as_synthetic":
                    ps = os.path.join(preds_dir, f"{ds}_{tag}_syn_{src}.json")
                    if not os.path.exists(ps):
                        print(f"[{tag}/{ds}/{kind}] synthetic source '{src}' not scored yet; skipping")
                        continue
                    syn = escape_scores(load_preds(ps), kind, iou)
                if syn is not None:
                    align.append({"model": tag, "seed": seed, "dataset": ds, "kind": kind,
                                  "source": src, "source_role": source_role(src, stress_names),
                                  "n_real": int(real_pool.size), "N": int(syn.size),
                                  "ks_full_pool": float(ks_2samp(_finite(real_pool), _finite(syn)).statistic),
                                  "real_missed_frac": float(np.mean(~np.isfinite(real_pool))),
                                  "syn_missed_frac": float(np.mean(~np.isfinite(syn)))})
                rows = sweep_one(c_all, part_all, len(cal), syn, src, m_values, n_draws,
                                 target, betas, alpha_max, rng, cc.get("resample_cal_test", True),
                                 cc.get("tie_break", True), acq_gamma, stress_names)
                for r in rows:
                    r.update(dataset=ds, kind=kind, model=tag, seed=seed,
                             primary=ds in cfg.get("primary_datasets", []))
                fmt = lambda v: "  -  " if v is None else f"{v:.3f}"   # noqa: E731
                for m in sorted({r["m"] for r in rows}):
                    rm = {(r["method"], r["beta"]): r for r in rows if r["m"] == m}
                    c = rm.get(("crc", None))
                    s = rm.get(("sperc_nominal", betas[0]))
                    t = rm.get(("sperc_tierN", betas[0]))
                    print(f"[{tag}/{ds}/{kind}/{src}] m={m:>4}  CRC esc {fmt(c and c['escape'])} rev {fmt(c and c['review'])}"
                          f" | SPERC@target esc {fmt(s and s['escape'])} rev {fmt(s and s['review'])}"
                          f" | SPERC tierN esc {fmt(t and t['escape'])} capH {fmt(t and t['tier_h_cap'])}", flush=True)
                all_rows += rows

    if not all_rows:
        return []
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"coldstart_{tag}")
    with open(base + ".json", "w") as f:
        json.dump(all_rows, f, indent=1)
    write_csv(all_rows, base + ".csv")
    ms = []
    for amax in cc.get("mstar_alpha_max_values", [alpha_max]):
        ms += m_star(all_rows, target, amax, cc.get("mstar_review_tolerance", 1.1),
                     cc.get("mstar_min_issued", 0.8), cc.get("mstar_review_target"))
    for r in ms:
        r.update(model=tag, seed=seed)
    write_csv(ms, base + "_mstar.csv")
    if align:
        write_csv(align, base + "_alignment.csv")
    n_viol = sum(1 for r in all_rows if r.get("exceeds_hard_bound"))
    n_cap = sum(1 for r in all_rows if r.get("tier_h_cap") is not None)
    print(f"\n[{tag}] wrote {base}.json/.csv ({len(all_rows)} rows), _mstar.csv, _alignment.csv;"
          f" rows above their hard cap by > 3 SE: {n_viol} of {n_cap}"
          f" (~{n_cap * 0.00135:.1f} expected by chance; scripts/summarize_sperc.py gives the"
          f" multiplicity-corrected verdict)")
    return all_rows


def main(argv=None):
    try:
        from src.utils.winenv import setup_console
        setup_console()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--datasets", default=None, help="comma list; default from config")
    ap.add_argument("--sources", default=None,
                    help="comma list; default = config sources + stress sources + any cached generator output")
    ap.add_argument("--model", default=None,
                    help="ONE explicit cache tag, e.g. rtdetr-l, yolov8s_s1, yolov8s_p2_training_free")
    ap.add_argument("--models", default=None, help="comma list of detectors; default from config")
    ap.add_argument("--seeds", default=None, help="comma list of seeds; default from config")
    ap.add_argument("--m_values", default=None, help="comma list, e.g. 5,10,15,25 (Stage 1)")
    ap.add_argument("--n_draws", type=int, default=None)
    ap.add_argument("--preds_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args(argv)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    preds_dir = a.preds_dir or os.path.join(ROOT, cfg["paths"]["preds_dir"])
    out_dir = a.out_dir or os.path.join(ROOT, cfg["paths"]["out_dir"])
    if a.model:
        tags = [a.model]
    else:
        models = a.models.split(",") if a.models else cfg.get("models", [cfg.get("model", "yolov8s")])
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg.get("seeds", [0])
        tags = detector_tags(models, seeds)
    datasets = a.datasets.split(",") if a.datasets else cfg["datasets"]
    base_sources = (a.sources.split(",") if a.sources
                    else list(cfg["synthetic_sources"]) + list(cfg.get("stress_sources", [])))
    # configured generator names are looked up per dataset; unscored ones are skipped quietly
    m_values = None
    if a.m_values:
        m_values = [v if v == "full" else int(v) for v in a.m_values.split(",")]
    n_draws = a.n_draws or cfg["coldstart"]["n_draws"]

    total = 0
    for tag in tags:
        total += len(run_tag(tag, cfg, datasets, base_sources, not a.sources, n_draws,
                             preds_dir, out_dir, m_values))
    if not total:
        print("nothing to write (no cached predictions for any requested detector)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
