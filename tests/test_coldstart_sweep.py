"""
End-to-end smoke test of scripts/run_coldstart_sweep.py on fake prediction caches,
so the headline script is known to run before any GPU time is spent.
"""

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _fake_records(n_def, n_clean, rng):
    recs = []
    for i in range(n_def):
        s = float(rng.beta(5, 2))
        recs.append({"image_id": f"d{i}", "gt_boxes": [[0, 0, 10, 10]],
                     "pred_boxes": [[0, 0, 10, 10]], "pred_scores": [s]})
    for i in range(n_clean):
        recs.append({"image_id": f"c{i}", "gt_boxes": [],
                     "pred_boxes": [[5, 5, 9, 9]], "pred_scores": [float(rng.beta(1, 6))]})
    return recs


def test_sweep_runs_on_fake_caches():
    spec = importlib.util.spec_from_file_location("sweep", os.path.join(ROOT, "scripts", "run_coldstart_sweep.py"))
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        preds, out = os.path.join(d, "preds"), os.path.join(d, "out")
        os.makedirs(preds)
        json.dump({"records": _fake_records(60, 200, rng)}, open(os.path.join(preds, "toy_yolov8s_cal.json"), "w"))
        json.dump({"records": _fake_records(80, 300, rng)}, open(os.path.join(preds, "toy_yolov8s_test.json"), "w"))
        json.dump({"records": _fake_records(500, 0, rng)}, open(os.path.join(preds, "toy_yolov8s_syn_gen.json"), "w"))
        json.dump({"records": _fake_records(90, 0, rng)}, open(os.path.join(preds, "other_yolov8s_cal.json"), "w"))
        cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml")))
        cfg.update(datasets=["toy"], primary_datasets=["toy"], synthetic_sources=["real_as_synthetic"],
                   stress_sources=["uniform_noise", "cross:other"], models=["yolov8s"], seeds=[0])
        cfg["coldstart"].update(m_values=[10, 25, "full"], n_draws=15)
        cfg["certificate"]["betas"] = [0.05, 0.10]
        cp = os.path.join(d, "cfg.yaml")
        yaml.safe_dump(cfg, open(cp, "w"))
        assert sweep.main(["--config", cp, "--preds_dir", preds, "--out_dir", out]) == 0
        rows = json.load(open(os.path.join(out, "coldstart_yolov8s.json")))
        mstar = open(os.path.join(out, "coldstart_yolov8s_mstar.csv")).read()
        align = open(os.path.join(out, "coldstart_yolov8s_alignment.csv")).read()
    sources = {r["source"] for r in rows}
    assert sources == {"real_as_synthetic", "uniform_noise", "cross:other", "gen"}, sources
    methods = {r["method"] for r in rows}
    assert methods == {"crc", "naive_pooled", "legacy", "acq_uncorrected", "sperc_tierN",
                       "sperc_nominal", "crc_at_cap"}
    # alpha = target - beta must stay positive: beta = 0.10 at target 0.10 has no Tier-N run
    assert {r["beta"] for r in rows if r["method"] == "sperc_tierN"} == {0.05}
    assert {r["beta"] for r in rows if r["method"] == "sperc_nominal"} == {0.05, 0.10}
    for r in rows:
        assert 0 <= r["escape"] <= 1 and 0 <= r["review"] <= 1 and r["escape_se"] >= 0
        assert r["escape_wilson_low"] <= r["escape"] + 1e-12 <= r["escape_wilson_high"] + 2e-12
        assert r["seed"] == 0 and r["model"] == "yolov8s"
    acq = [r for r in rows if r["method"] == "acq_uncorrected"]
    crc = {(r["source"], r["m"]): r for r in rows if r["method"] == "crc"}
    assert acq and all(r["labelled_parts"] <= crc[(r["source"], r["m"])]["labelled_parts"] for r in acq), \
        "screening must need fewer labelled parts than random labelling"
    assert "m_star" in mstar and "valid_guarantee" in mstar and "labelled_at_m_star" in mstar
    assert "ks_full_pool" in align and "cross:other" in align
    print(f"  {len(rows)} rows: 4 sources x 7 methods x betas x 3 m, plus m-star and alignment tables")


def test_sperc_hard_bound_holds_under_resampling():
    """Worst-case synthetic data on a fake line: SPERC must stay under Tier H."""
    spec = importlib.util.spec_from_file_location("sweep", os.path.join(ROOT, "scripts", "run_coldstart_sweep.py"))
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    rng = np.random.default_rng(1)
    n_cal, n_test = 400, 400
    c_all = np.concatenate([rng.beta(5, 2, n_cal), rng.beta(5, 2, n_test)])
    part_all = c_all.copy()
    syn = rng.beta(12, 1.5, 1000)                        # far too easy
    rows = sweep.sweep_one(c_all, part_all, n_cal, syn, "bad", [15], 300, 0.10, [0.05], 1.0, rng)
    r = next(x for x in rows if x["method"] == "sperc_tierN")
    assert r["escape"] <= r["tier_h_cap"] + 3 * r["escape_se"], r
    assert not r["exceeds_hard_bound"]
    print(f"  escape {r['escape']:.3f} vs Tier H cap {r['tier_h_cap']:.3f}")


if __name__ == "__main__":
    from _runner import run
    run(globals())
