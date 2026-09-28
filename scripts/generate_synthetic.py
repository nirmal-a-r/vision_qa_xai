"""
generate_synthetic.py - synthetic defects for the primary datasets (project
document Sections 7, 11.3 and 15: N = 1,000 per primary dataset, visually audited).

Everything reads references and clean backgrounds from the TRAIN split only.

    # in-repo generators
    python scripts/generate_synthetic.py --dataset kolektor --generator copy_paste --n 1000
    python scripts/generate_synthetic.py --dataset kolektor --generator training_free --n 1000

    # external generators
    python scripts/generate_synthetic.py --dataset kolektor --export-anomalydiffusion
    python scripts/generate_synthetic.py --dataset kolektor --export-tfidg --n 1000
    python scripts/generate_synthetic.py --dataset kolektor --collect <their output dir> --source anomalydiffusion

Output: data/generated/<dataset>/<source>/<stem>.png + <stem>_mask.png, generator.json
(provenance) and _audit_sheet.png (look at it before using the set). The pipeline's
synth stage then builds labelled sets and scores them with every frozen detector.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Hugging Face on Windows without Developer Mode cannot symlink its cache; it copies
# instead and warns on every load. The copy is fine; silence the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from src.utils.winenv import keep_awake, setup_console   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--generator", default=None, choices=["copy_paste", "training_free"])
    ap.add_argument("--source", default=None, help="output source name (default: generator name)")
    ap.add_argument("--n", type=int, default=None, help="default: config generators.n_synthetic")
    ap.add_argument("--k", type=int, default=None, help="reference defects (default: config)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out_root", default="data/generated")
    ap.add_argument("--collect", default=None, help="collect pairs from an external generator's output dir")
    ap.add_argument("--export-anomalydiffusion", dest="export_ad", action="store_true")
    ap.add_argument("--export-tfidg", dest="export_tfidg", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate even if the folder has pairs")
    a = ap.parse_args(argv)
    setup_console()

    with open(os.path.join(ROOT, "configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    g = cfg.get("generators", {})
    n = a.n or g.get("n_synthetic", 1000)
    k = a.k or g.get("n_reference", 5)
    seed = g.get("reference_seed", 0) if a.seed is None else a.seed

    from src.synth.generators import GENERATORS, audit_sheet, list_pairs
    from src.synth import external as X

    if a.export_ad:
        root = X.export_mvtec_layout(a.dataset, os.path.join("data", "external_inputs", "anomalydiffusion"),
                                     k, seed, processed_dir=a.processed_dir)
        print(f"wrote MVTec-style folder (train split only): {root}\n")
        print(X.anomalydiffusion_commands(os.path.abspath(os.path.dirname(root)),
                                          g.get("anomalydiffusion", {}).get("repo_dir", "external/anomalydiffusion")))
        return 0
    if a.export_tfidg:
        out = X.export_tfidg_inputs(a.dataset, os.path.join("data", "external_inputs", "tfidg", a.dataset),
                                    n, k, seed, a.processed_dir)
        print(f"wrote TF-IDG inputs (normal / reference / reference_mask / target_mask per sample): {out}")
        print("run the official repo's run_inference.py on them, then collect with --collect <its output> --source training_free")
        return 0
    if a.collect:
        source = a.source or "external"
        out = os.path.join(a.out_root, a.dataset, source)
        c = X.collect_pairs(a.collect, out, source, limit=n)
        print(f"collected {c} pairs into {out}; audit sheet: {audit_sheet(out)}")
        return 0
    if not a.generator:
        ap.error("give --generator, --collect, --export-anomalydiffusion or --export-tfidg")

    source = a.source or a.generator
    out = os.path.join(a.out_root, a.dataset, source)
    if os.path.isdir(out) and len(list_pairs(out)) >= n and not a.force:
        print(f"{out} already holds >= {n} pairs; use --force to regenerate")
        return 0
    kw = {}
    if a.generator == "training_free":
        tf = dict(g.get("training_free", {}))
        backend = tf.pop("backend", "builtin_sd_inpaint")
        if backend == "copy_paste":
            a.generator = "copy_paste"
        elif backend == "tfidg":
            print("config says backend: tfidg - use --export-tfidg, run the official repo, then --collect")
            return 1
        else:
            kw = {kk: tf[kk] for kk in ("model_id", "strength", "steps", "guidance", "prompt") if kk in tf}
            kw["device"] = a.device
    gen = GENERATORS[a.generator](a.dataset, k=k, seed=seed, processed_dir=a.processed_dir, **kw)
    gen.name = source
    with keep_awake(f"generating {source} defects for {a.dataset}"):
        made = gen.generate(n, out)
    print(f"[{a.dataset}/{source}] {made} pairs -> {out}; audit sheet: {audit_sheet(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
