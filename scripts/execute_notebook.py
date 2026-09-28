"""
execute_notebook.py - run notebook/VisionQA_CertifiedInspection.ipynb top to bottom
with the current Python (the venv) and save an executed copy next to it.

    python scripts/execute_notebook.py                    # results in the repository
    python scripts/execute_notebook.py --work runs_smoke  # results in another folder

Exit code 1 if any cell raised. The notebook finds the code through VQA_ROOT and
the data / results through VQA_WORK, which this script sets.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=os.path.join(ROOT, "notebook", "VisionQA_CertifiedInspection.ipynb"))
    ap.add_argument("--out", default=None, help="default: <notebook>_executed.ipynb")
    ap.add_argument("--work", default=ROOT)
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per cell")
    ap.add_argument("--kernel", default="python3")
    a = ap.parse_args(argv)

    import asyncio
    import json
    import tempfile
    import nbformat
    from nbclient import NotebookClient

    sys.path.insert(0, ROOT)
    from src.utils.winenv import setup_console
    setup_console()
    if os.name == "nt":
        # pyzmq / jupyter_client need a selector event loop; Python's Windows default
        # (Proactor) makes kernel start-up flaky and noisy.
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    # Pin the kernel to THIS interpreter (the venv): a globally installed Jupyter
    # may define its own "python3" kernel pointing at another Python.
    if a.kernel == "python3":
        kdir = tempfile.mkdtemp(prefix="vqa_kernel_")
        os.makedirs(os.path.join(kdir, "kernels", "vqa_exec"))
        with open(os.path.join(kdir, "kernels", "vqa_exec", "kernel.json"), "w") as f:
            json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                       "display_name": "vqa_exec", "language": "python"}, f)
        os.environ["JUPYTER_PATH"] = kdir + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        a.kernel = "vqa_exec"

    os.environ["VQA_ROOT"] = ROOT
    os.environ["VQA_WORK"] = os.path.abspath(a.work)
    out = a.out or a.notebook.replace(".ipynb", "_executed.ipynb")
    nb = nbformat.read(io.open(a.notebook, encoding="utf-8"), as_version=4)
    client = NotebookClient(nb, timeout=a.timeout, kernel_name=a.kernel, allow_errors=True,
                            resources={"metadata": {"path": os.path.abspath(a.work)}})
    client.execute()
    nbformat.write(nb, io.open(out, "w", encoding="utf-8"))
    errs = [(i, o) for i, c in enumerate(nb.cells) if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error"]
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"executed {n_code} code cells -> {out}; {len(errs)} cell error(s)")
    for i, o in errs[:10]:
        print(f"  cell {i}: {o.get('ename')}: {str(o.get('evalue'))[:200]}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
