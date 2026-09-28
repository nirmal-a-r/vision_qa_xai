"""
Tests for the decision / summary scripts: the Stage-1 go / no-go rule, the seed
summary, and the MVTec bridge's ranking metric - on synthetic rows, no GPU.
"""

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(m, gain, se=0.01, cap=0.3, esc=0.05, issued=1.0):
    return {"method": "sperc_tierN", "m": m, "beta": 0.05, "dataset": "gc10", "kind": "part",
            "tier_h_cap": cap, "tier_h_cap_max": cap, "issued": issued, "escape": esc,
            "escape_se": 0.005, "review": 0.9, "review_gain_vs_crc": gain,
            "review_gain_vs_crc_se": se}


def test_stage1_rule():
    s1 = _load("run_stage1")
    go = s1.decide([_row(5, 0.05, cap=0.33)], 0.10, [0.25, 0.35], [5, 10])
    assert go["verdict"] == "GO" and go["passing"][0]["alpha_max"] == 0.35
    assert s1.decide([_row(5, 0.05, cap=0.33)], 0.10, [0.25], [5])["verdict"] == "NO-GO"   # cap too loose
    assert s1.decide([_row(5, 0.01, se=0.01)], 0.10, [0.5], [5])["verdict"] == "NO-GO"     # within noise
    assert s1.decide([_row(5, 0.05, esc=0.2)], 0.10, [0.5], [5])["verdict"] == "NO-GO"     # invalid escape
    assert s1.decide([_row(5, 0.05, issued=0.5)], 0.10, [0.5], [5])["verdict"] == "NO-GO"  # rarely issues
    print("  GO needs cap <= alpha_max, valid escape, >= 80% issuance and a gain beyond 2 SE")


def test_summary_pools_seeds():
    sm = _load("summarize_sperc")
    rows = []
    for seed, tag in ((0, "yolov8s"), (1, "yolov8s_s1")):
        for meth, esc in (("crc", 0.08), ("sperc_tierN", 0.05)):
            rows.append({"model": tag, "seed": seed, "dataset": "toy", "kind": "part", "source": "gen",
                         "source_role": "generator", "m": 15, "method": meth,
                         "beta": 0.05 if meth != "crc" else None, "draws": 100, "escape": esc + 0.01 * seed,
                         "escape_se": 0.005, "review": 0.4, "issued": 1.0, "tier_h_cap": 0.19,
                         "tier_h_cap_max": 0.19, "target": 0.10, "alpha": 0.05, "N_mean": 1000.0,
                         "exceeds_hard_bound": False})
    ss = sm.seeds_summary(rows)
    r = next(x for x in ss if x["method"] == "sperc_tierN")
    assert r["n_seeds"] == 2 and abs(r["escape_mean"] - 0.055) < 1e-12 and r["model"] == "yolov8s"
    assert all(v["exceeds_hard_bound"] is False for v in sm.validity(rows))
    pv = sm.planned_vs_realised(rows)
    assert pv and pv[0]["agree"] and pv[0]["planned_cap_of_worst"] == 0.19
    print("  seeds pooled per base model; validity and planned-vs-realised tables built")


def test_image_auroc():
    br = _load("run_mvtec_bridge")
    recs = [{"defect_type": "good", "score": s} for s in (0.1, 0.2)] + \
           [{"defect_type": "crack", "score": s} for s in (0.3, 0.4)]
    assert br.image_auroc(recs) == 1.0
    recs[0]["score"] = 0.35
    assert abs(br.image_auroc(recs) - 0.75) < 1e-12
    print("  image-level AUROC by ranks")


if __name__ == "__main__":
    from _runner import run
    run(globals())
