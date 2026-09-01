"""Repair notebook cells whose string literals were split by a stray newline.

The notebook builders hold cell bodies in non-raw triple-quoted strings. A `\n`
written inside an emitted string literal is interpreted by Python when the
*builder* runs, so it becomes a real newline in the generated cell and cuts that
literal in half. Every occurrence produces `SyntaxError: unterminated string
literal` in the notebook, and hunting them individually through three layers of
escaping cost several rounds of edits.

This pass detects the failure and rejoins the two halves. Self-healing, so a new
one introduced later cannot slip through unnoticed.
"""

from __future__ import annotations


def repair_split_literals(cells, max_passes: int = 40) -> int:
    """Join literals split across lines. Returns how many joins were made."""
    fixed = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for _ in range(max_passes):
            src = "".join(cell["source"])
            try:
                compile(src, "<cell>", "exec")
                break
            except SyntaxError as e:
                if "unterminated string literal" not in (e.msg or ""):
                    break                      # a different problem; leave it
                lines = src.splitlines(keepends=True)
                i = (e.lineno or 1) - 1
                if i + 1 >= len(lines):
                    break
                head = lines[i].rstrip("\r\n")
                lines[i] = head + lines[i + 1].lstrip()
                del lines[i + 1]
                cell["source"] = lines
                fixed += 1
    return fixed


def check(cells) -> list:
    """Return [(index, message)] for every code cell that still fails to compile."""
    bad = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        try:
            compile("".join(cell["source"]), f"<cell{i}>", "exec")
        except SyntaxError as e:
            bad.append((i, f"line {e.lineno}: {e.msg}"))
    return bad
