"""Reachability analysis: which src/ modules can the real entry points actually reach?

Grepping for a module name gives false positives (searching for `train` matches
`train_baselines`, `training`, `.train(`), so this walks the real import graph
from the notebook and the runnable scripts instead.
"""
import ast, glob, io, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Every runnable entry point: scripts, tests and the deliverable notebook.
ENTRIES = sorted(glob.glob("scripts/*.py") + glob.glob("tests/*.py")
                 + glob.glob("notebook/*.ipynb"))


def imported_src_modules(path):
    try:
        if path.endswith(".ipynb"):
            nb = json.load(io.open(path, encoding="utf-8"))
            src = "\n".join("".join(c["source"]) for c in nb["cells"]
                            if c["cell_type"] == "code")
            src = "\n".join(l for l in src.splitlines()
                            if not l.lstrip().startswith(("%", "!")))
        else:
            src = io.open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("src"):
            out.add(n.module)
            # `from src.agentic import llm_narrator` names the MODULE in
            # n.names, not n.module. Recording only n.module marks the leaf
            # unreachable and would delete live code.
            for al in n.names:
                out.add(f"{n.module}.{al.name}")
        elif isinstance(n, ast.Import):
            for al in n.names:
                if al.name.startswith("src."):
                    out.add(al.name)
    # Modules launched as subprocesses (`python -m src.x.y`) are never imported,
    # so the import graph cannot see them at all.
    for m in re.findall(r'"-m",\s*"(src\.[\w.]+)"', src):
        out.add(m)
    for m in re.findall(r'python -m (src\.[\w.]+)', src):
        out.add(m)
    # The notebook builder embeds modules by FILE PATH ("src/risk/sperc.py").
    for p in re.findall(r'"(src/[\w/]+)\.py"', src):
        out.add(p.replace("/", "."))
    return out


def module_path(m):
    p = os.path.join(*m.split(".")) + ".py"
    return p if os.path.exists(p) else None


def main():
    stack, seen = [], set()
    for e in ENTRIES:
        if os.path.exists(e):
            stack += list(imported_src_modules(e))
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        p = module_path(m)
        if p:
            stack += [x for x in imported_src_modules(p) if x not in seen]

    all_src = set()
    for root, _, files in os.walk("src"):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                all_src.add(os.path.join(root, f).replace("\\", "/"))

    reachable = {module_path(m).replace("\\", "/") for m in seen if module_path(m)}
    dead = sorted(all_src - reachable)

    print(f"reachable: {len(reachable)} modules")
    for r in sorted(reachable):
        print("  live  ", r)
    print(f"\nunreachable: {len(dead)} modules")
    for d in dead:
        print(f"  dead   {d}  ({os.path.getsize(d) // 1024} KB)")
    return dead


if __name__ == "__main__":
    main()
