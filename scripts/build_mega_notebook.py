"""
build_mega_notebook.py
======================
Fuse the whole project into ONE self-contained notebook.

Every `src/*.py` module is embedded as a string and registered into
`sys.modules` at run time, so the existing `from src.risk.conformal import ...`
lines keep working verbatim and the notebook has no `.py` dependency at all.

Why register modules rather than paste the function bodies at top level:

  * several modules define same-named helpers (`main`, `_calibrate`, `save`);
    flattening them into one namespace would silently shadow one with another
  * `if __name__ == "__main__"` blocks stay inert, because each module's
    `__name__` is its real dotted name, not `"__main__"`
  * the analysis cells that already exist need no rewriting, so the fusion
    cannot introduce a difference between what the modules do here and what
    they do on disk

Output: notebook/VisionQA_Complete.ipynb  — open, select the kernel, Run All.
"""

from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SRC_NB = "notebook/VisionQA_RiskControlled_Inspection.ipynb"
OUT_NB = "notebook/VisionQA_Complete.ipynb"

# Dependency order: a module must appear after everything it imports.
MODULES = [
    "src/paper/figures.py",
    "src/risk/conformal.py",
    "src/risk/adaptive.py",
    "src/risk/triage.py",
    "src/data/photometric.py",
    "src/xai/faithfulness.py",
    "src/xai/detector_saliency.py",
    "src/agentic/inspector.py",
    "src/agentic/llm_narrator.py",
    "src/data/prepare_datasets.py",
    "src/data/splits_and_yolo.py",
    "src/data/build_cce_dataset.py",
    "src/evaluation/train_baselines.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


LOADER = '''# =============================================================================
# EMBEDDED SOURCE  -  this notebook carries the entire project inside itself
# =============================================================================
# Each module below is registered into sys.modules under its real dotted name,
# so every `from src.x.y import z` later in this notebook resolves to the code
# embedded here. Nothing is read from disk; there is no .py dependency.
import sys, types, json, os

_INLINE_SOURCE = {}          # dotted name -> source text, filled by the next cells

def _register(name, source):
    """Materialise `source` as an importable module called `name`."""
    parts = name.split(".")
    for i in range(1, len(parts)):                 # create parent packages
        pkg = ".".join(parts[:i])
        if pkg not in sys.modules:
            p = types.ModuleType(pkg)
            p.__path__ = []                        # marks it as a package
            sys.modules[pkg] = p
    mod = types.ModuleType(name)
    mod.__name__ = name                            # keeps __main__ guards inert
    mod.__file__ = f"<inline:{name}>"
    sys.modules[name] = mod
    exec(compile(source, f"<inline:{name}>", "exec"), mod.__dict__)
    # bind as an attribute of its parent so `from src.risk import triage` works
    if len(parts) > 1:
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod

print("module loader ready")
'''

FINALISE = '''# Register every embedded module, in dependency order.
for _name, _src in _INLINE_SOURCE.items():
    _register(_name, _src)
print(f"registered {len(_INLINE_SOURCE)} modules:")
for _n in _INLINE_SOURCE:
    print("   ", _n)

# Smoke-test: the guarantee machinery must be live before anything below runs.
from src.risk.conformal import conformal_risk_control, escape_threshold_grid
import numpy as _np
_g = escape_threshold_grid(51)
_L = _np.tile(_np.linspace(0.6, 0.0, 51), (200, 1))
print("\\nself-test  lambda =", round(conformal_risk_control(_L, _g, 0.10), 4))
'''


def module_cell(path):
    name = path.replace("/", ".").replace("\\", ".")[:-3]     # src/risk/x.py -> src.risk.x
    source = io.open(path, encoding="utf-8").read()
    # A triple-quoted heredoc would collide with the docstrings inside every
    # module, so carry the text as a JSON literal instead - json.dumps escapes
    # quotes, backslashes and newlines unambiguously.
    payload = json.dumps(source)
    return code(f'# ---- {name} ----------------------------------------------------\n'
                f'_INLINE_SOURCE["{name}"] = json.loads({payload!r})\n'
                f'print("embedded {name}  ({len(source)} chars)")\n')


TRAIN_MD = '''## 5b · Training — optional, GPU

Set `TRAIN = True` below to retrain the detectors from scratch. It is `False` by
default so Run All completes in minutes off the cached artefacts.

Training is idempotent: each (dataset, model, encoding) already present in
`runs/results_*.json` is skipped, so an interrupted sweep resumes rather than
restarting. The full grid is 5 datasets × 2 detectors × 2 encodings; on one
8 GB card that is roughly a day, which is why the cached results ship with the
repo.
'''

TRAIN_CODE = '''TRAIN = False          # <-- flip to True to retrain (needs a free GPU)
TRAIN_DATASETS = "neu,magnetic_tile,kolektor,gc10,pcb"

if not TRAIN:
    print("TRAIN = False -> using cached detector results in runs/")
    print("set TRAIN = True to regenerate (hours, needs GPU)")
else:
    import subprocess, shlex
    from src.data import prepare_datasets as _prep
    from src.data import splits_and_yolo as _split
    from src.data import build_cce_dataset as _cce

    if not os.path.exists("data/yolo/neu/data.yaml"):
        print(">> preparing datasets")
        _prep.prepare_all("data/raw", "data/processed")
        import glob as _g
        for cp in sorted(_g.glob("data/processed/*_coco.json")):
            nm = os.path.basename(cp).replace("_coco.json", "")
            paths, cats = _split.write_splits(cp, "data/processed/splits")
            _split.coco_to_yolo(paths, cats, "data/yolo", nm)

    if not os.path.exists("data/yolo_cce/neu/data.yaml"):
        print(">> building CCE trees")
        for nm in TRAIN_DATASETS.split(","):
            src_root = os.path.join("data/yolo", nm.strip())
            if os.path.isdir(src_root):
                _cce.encode_tree(src_root, os.path.join("data/yolo_cce", nm.strip()))

    # Training runs as a subprocess: ultralytics installs signal handlers and
    # reconfigures logging, and doing that inside the kernel leaves the notebook
    # in a state where later cells emit no output.
    from src.evaluation.train_baselines import CONFIGS
    jobs = [("yolov8s.pt", "baseline", "data/yolo",     "runs/results_yolov8s.json"),
            ("yolov8s.pt", "cce",      "data/yolo_cce", "runs/results_yolov8s.json"),
            ("rtdetr-l.pt", "baseline", "data/yolo",     "runs/results_rtdetr.json"),
            ("rtdetr-l.pt", "cce",      "data/yolo_cce", "runs/results_rtdetr.json")]
    for model, enc, ydir, res in jobs:
        cmd = [sys.executable, "-m", "src.evaluation.train_baselines",
               "--model", model, "--encoding", enc, "--yolo_dir", ydir,
               "--datasets", TRAIN_DATASETS, "--results", res]
        print(f"\\n>> {model} / {enc}")
        subprocess.call(cmd)
    print("training complete")
'''

ANALYSIS_MD = '''## 5c · Analysis pipeline — optional

`RUN_ANALYSIS = True` recomputes the cached prediction dumps and every risk,
drift and triage result from the trained weights. It needs the weights but is
otherwise CPU-bound and takes a few minutes.
'''

ANALYSIS_CODE = '''RUN_ANALYSIS = False   # <-- flip to True to recompute risk / drift / triage

if not RUN_ANALYSIS:
    print("RUN_ANALYSIS = False -> using cached runs/*.json")
else:
    import subprocess
    subprocess.call([sys.executable, "scripts/run_all_experiments.py"])
    print("analysis complete; re-run the cells below to pick up new numbers")
'''


def main():
    nb = json.load(io.open(SRC_NB, encoding="utf-8"))
    cells = nb["cells"]

    # The loader must precede EVERY cell that touches `src`, including the
    # setup cell itself - that cell does `from src.paper import figures as F`,
    # so inserting after it raised ModuleNotFoundError on the very first run.
    # It is therefore self-sufficient (imports sys/types/json/os itself) and
    # goes before the first code cell, whatever that turns out to be.
    setup_idx = -1
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            setup_idx = i - 1
            break

    inline = [md("## 0b · Embedded project source\n\n"
                 "Everything under `src/` is carried inside this notebook and registered as\n"
                 "importable modules. The notebook is therefore the whole deliverable: it has\n"
                 "no `.py` dependency and can be run from a directory containing only itself\n"
                 "and `data/`.\n"),
              code(LOADER)]
    for p in MODULES:
        if os.path.exists(p):
            inline.append(module_cell(p))
        else:
            print(f"  WARNING: {p} missing, skipped")
    inline.append(code(FINALISE))

    # Training / analysis toggles go just before the detector-results section.
    train_at = len(cells)
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and "".join(c["source"]).strip().startswith("## 5"):
            train_at = i
            break
    extra = [md(TRAIN_MD), code(TRAIN_CODE), md(ANALYSIS_MD), code(ANALYSIS_CODE)]

    head = cells[:setup_idx + 1]
    body = cells[setup_idx + 1:train_at]
    new = head + inline + body + extra + cells[train_at:]
    nb["cells"] = new
    nb["metadata"]["kernelspec"] = {"display_name": "Python (vision_qa_xai)",
                                    "language": "python", "name": "vision_qa_xai"}
    json.dump(nb, io.open(OUT_NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    n_code = sum(1 for c in new if c["cell_type"] == "code")
    print(f"wrote {OUT_NB}")
    print(f"  {len(new)} cells ({n_code} code), {os.path.getsize(OUT_NB)//1024} KB")
    print(f"  embedded {len([p for p in MODULES if os.path.exists(p)])} modules")


if __name__ == "__main__":
    main()
