"""Environment detection, default folders and small runtime helpers.

The same notebooks run on Kaggle and on a local GPU machine; this module hides
the differences (paths, where the packed dataset lives, device info).
"""

import json
import os
import platform
import random
import shutil
import sys
from pathlib import Path

PACKED_MARKER = "agrivision_packed"


def environment():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").exists():
        return "kaggle"
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def default_dirs(env=None):
    """input: where datasets are read from; output: saved results; scratch: temporary work."""
    env = env or environment()
    if env == "kaggle":
        return {
            "input": Path("/kaggle/input"),
            "output": Path("/kaggle/working"),
            "scratch": Path("/tmp/agrivision_work"),
        }
    if env == "colab":
        base = Path("/content")
        return {"input": base / "data", "output": base / "outputs", "scratch": Path("/tmp/agrivision_work")}
    base = Path(os.environ.get("AGRIVISION_HOME", Path.cwd()))
    return {
        "input": Path(os.environ.get("AGRIVISION_INPUT", base / "data")),
        "output": Path(os.environ.get("AGRIVISION_OUTPUT", base / "outputs")),
        "scratch": Path(os.environ.get("AGRIVISION_SCRATCH", base / "_work")),
    }


def is_packed_root(path):
    meta = Path(path) / "meta.json"
    if not meta.is_file():
        return False
    try:
        return json.loads(meta.read_text()).get(PACKED_MARKER) == 1
    except (OSError, ValueError):
        return False


def find_packed_root(hint=None):
    """Locate a packed dataset: explicit hint, $AGRIVISION_DATA, then the usual input folders."""
    candidates = []
    for p in (hint, os.environ.get("AGRIVISION_DATA")):
        if p:
            candidates.append(Path(p))
    dirs = default_dirs()
    for base in (dirs["input"], dirs["output"]):
        if base.exists():
            candidates.append(base)
            for depth in ("*", "*/*", "*/*/*"):
                candidates.extend(sorted(base.glob(depth)))
    for c in candidates:
        if c.is_dir() and is_packed_root(c):
            return c
    raise FileNotFoundError(
        "No packed Agriculture-Vision dataset found. Run 01_prepare_data first, attach its output "
        "(Kaggle: Add Input), or set DATA_ROOT / $AGRIVISION_DATA to the folder containing meta.json."
    )


def find_files(pattern, roots=None):
    """Search output and input folders for files matching a glob (e.g. '**/best.pt')."""
    dirs = default_dirs()
    roots = roots or [dirs["output"], dirs["input"]]
    found = []
    for r in roots:
        r = Path(r)
        if r.exists():
            found.extend(r.glob(pattern))
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def apply_overrides(config, section):
    """Merge JSON overrides from $AGRIVISION_OVERRIDES (e.g. '{"train": {"epochs": 1}}')."""
    raw = os.environ.get("AGRIVISION_OVERRIDES")
    if raw:
        extra = json.loads(raw).get(section, {})
        if extra:
            print(f"[overrides] {section}: {extra}")
            config.update(extra)
    return config


def system_report():
    """Print what hardware this notebook is running on."""
    lines = [f"environment : {environment()}", f"python      : {platform.python_version()} ({platform.system()})"]
    try:
        import torch
        lines.append(f"torch       : {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                lines.append(f"gpu {i}       : {p.name}, {p.total_memory / 2**30:.1f} GB")
        else:
            lines.append("gpu         : none (CPU only)")
    except ImportError:
        lines.append("torch       : not installed")
    lines.append(f"cpu cores   : {os.cpu_count()}")
    try:
        import psutil
        lines.append(f"ram         : {psutil.virtual_memory().total / 2**30:.1f} GB")
    except ImportError:
        pass
    for name, path in default_dirs().items():
        p = Path(path)
        probe = p if p.exists() else p.parent
        if probe.exists():
            free = shutil.disk_usage(probe).free / 2**30
            lines.append(f"{name:<12}: {p}  ({free:.0f} GB free)")
    print("\n".join(lines))
