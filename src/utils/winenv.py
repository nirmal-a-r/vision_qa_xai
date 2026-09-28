"""
winenv.py
=========
Small, dependency-free helpers that make long runs robust on Windows (project
document, Section 10.4) while doing nothing harmful on Linux / macOS.

* ``setup_console``   UTF-8 stdout / stderr. Ultralytics prints emoji; redirected to a
                      log file under the default cp1252 code page that raises
                      UnicodeEncodeError and kills the run.
* ``keep_awake``      asks Windows not to sleep while THIS process runs
                      (SetThreadExecutionState). Nothing is changed in the power
                      settings; the request ends with the process.
* ``on_ac_power``     laptop on battery? (a multi-hour GPU run should not be)
* ``long_paths_enabled``  the 260-character path limit (Ultralytics and COCO trees
                      nest deeply)
* ``in_cloud_sync_folder``  OneDrive / Dropbox / Google Drive folders lock files mid-run
* ``gpu_memory_mb``   (total, free) VRAM of one GPU, via torch or nvidia-smi
* ``rmtree``          shutil.rmtree that survives read-only files and brief locks
                      (antivirus / indexer) on Windows
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import subprocess
import sys
import time

IS_WINDOWS = os.name == "nt"


def setup_console():
    """Make stdout / stderr UTF-8 with replacement, and pass that on to children."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ------------------------------------------------------------------ sleep
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake(reason: str = "long GPU run"):
    """Prevent system sleep for the duration of the block (Windows only).

    The display may still turn off; only the system stays awake. Released when the
    block ends or the process exits - no persistent power-setting change.
    """
    active = False
    if IS_WINDOWS:
        try:
            import ctypes
            active = bool(ctypes.windll.kernel32.SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))
            if active:
                print(f"  [windows] sleep blocked while this runs ({reason})", flush=True)
        except Exception:
            active = False
    try:
        yield active
    finally:
        if active:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
            except Exception:
                pass


# ------------------------------------------------------------------ checks
def on_ac_power():
    """True / False on Windows laptops, None when unknown (or not Windows)."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes

        class _SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = _SPS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
        return {0: False, 1: True}.get(s.ACLineStatus)
    except Exception:
        return None


def long_paths_enabled():
    """True / False on Windows (registry LongPathsEnabled), None elsewhere."""
    if not IS_WINDOWS:
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except Exception:
        return False


def in_cloud_sync_folder(path: str):
    """Name of the sync service if ``path`` lives in a synced folder, else None."""
    p = os.path.abspath(path).lower().replace("\\", "/")
    for key, name in (("/onedrive", "OneDrive"), ("/dropbox", "Dropbox"),
                      ("/google drive", "Google Drive"), ("/icloud", "iCloud")):
        if key in p:
            return name
    return None


def gpu_memory_mb(index: int = 0):
    """(total_mb, free_mb) of one GPU, or (None, None) if unknown."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(index)
            return int(total / 2**20), int(free / 2**20)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=20)
        t, f = (int(x) for x in out.decode().strip().splitlines()[0].split(","))
        return t, f
    except Exception:
        return None, None


def rmtree(path: str, retries: int = 5):
    """shutil.rmtree that clears read-only bits and retries brief Windows locks."""
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    for i in range(retries):
        if not os.path.exists(path):
            return True
        try:
            shutil.rmtree(path, onerror=_onerror)
        except Exception:
            pass
        if not os.path.exists(path):
            return True
        time.sleep(1.0 + i)
    return not os.path.exists(path)
