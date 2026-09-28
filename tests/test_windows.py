"""
Tests for the Windows / 8 GB optimisations (project document, Sections 10.3-10.4):
the batch profile, OOM detection, portable paths and the winenv helpers. They run
on any OS; the Windows-only calls must be harmless no-ops elsewhere.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.train_baselines import (CONFIGS, RTDETR_OVERRIDES, _is_oom,  # noqa: E402
                                            _portable, batch_8gb, batch_for, resolve_profile)
from src.utils import winenv  # noqa: E402


def test_8gb_batch_profile_matches_document():
    # YOLOv8s: 8 at imgsz 256-384, 4 at 512-640, 2 at 800 (and above); RT-DETR-L half
    assert [batch_8gb("yolov8s.pt", s) for s in (256, 384, 512, 640, 800, 1024)] == [8, 8, 4, 4, 2, 2]
    assert [batch_8gb("rtdetr-l.pt", s) for s in (256, 384, 512, 640, 800)] == [4, 4, 2, 2, 1]
    cfg = dict(CONFIGS["kolektor"], **RTDETR_OVERRIDES["kolektor"])
    assert batch_for("rtdetr-l.pt", cfg, "8gb") == 2 and batch_for("rtdetr-l.pt", cfg, "full") == 4
    assert resolve_profile("8gb", 0) == "8gb" and resolve_profile("auto", "cpu") == "8gb"
    print("  8 GB profile = document Section 10.3; full profile keeps the larger batches")


def test_oom_detection_and_portable_paths():
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert _is_oom(type("OutOfMemoryError", (RuntimeError,), {})("x"))
    assert not _is_oom(ValueError("bad label file"))
    assert _portable(os.path.join(os.getcwd(), "runs", "w.pt")) == os.path.join("runs", "w.pt")
    outside = os.path.abspath(os.path.join(os.sep, "elsewhere", "w.pt"))
    assert _portable(outside) == outside
    print("  OOM recognised for a retry at half batch; weights stored relative when inside the project")


def test_winenv_helpers_are_safe_everywhere():
    winenv.setup_console()
    with winenv.keep_awake("test") as active:
        assert active in (True, False)
    assert winenv.in_cloud_sync_folder("C:/Users/x/OneDrive/Desktop/proj") == "OneDrive"
    assert winenv.in_cloud_sync_folder("C:/Users/x/Desktop/proj") is None
    if os.name != "nt":
        assert winenv.on_ac_power() is None and winenv.long_paths_enabled() is None
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sub", "f.txt")
    os.makedirs(os.path.dirname(p))
    open(p, "w").write("x")
    os.chmod(p, 0o444)                                   # read-only, as Windows tools leave them
    assert winenv.rmtree(d) and not os.path.exists(d)
    total, free = winenv.gpu_memory_mb(0)
    assert (total is None) == (free is None)
    print("  console, keep-awake, sync-folder check, robust delete and VRAM query all safe")


if __name__ == "__main__":
    from _runner import run
    run(globals())
