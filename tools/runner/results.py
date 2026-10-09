"""Pack a run's results (no checkpoints, no dataset) into a small zip for another computer."""

import io
import time
import zipfile
from pathlib import Path

SKIP_SUFFIXES = {".pt", ".bin", ".tmp"}
SKIP_DIRS = {"agrivision2021_packed", "_agrivision_src", "__pycache__", "notebooks", "kaggle"}


def iter_result_files(workspace):
    ws = Path(workspace)
    for base in ("outputs", "runner"):
        root = ws / base
        if not root.exists():
            continue
        for p in root.rglob("*"):
            rel = p.relative_to(ws)
            if p.is_file() and p.suffix not in SKIP_SUFFIXES and not (set(rel.parts) & SKIP_DIRS):
                yield p, rel.as_posix()
    for extra in ("notifications.log",):
        if (ws / extra).is_file():
            yield ws / extra, extra


def write_zip(workspace, fileobj):
    n = 0
    with zipfile.ZipFile(fileobj, "w", zipfile.ZIP_DEFLATED) as z:
        for path, arc in iter_result_files(workspace):
            try:
                z.write(path, f"results/{arc}")
                n += 1
            except OSError:  # file being written right now
                pass
        z.writestr("results/README.txt", (
            "AgriVision results (no checkpoints / dataset).\n"
            "Draw all charts on any computer with matplotlib:\n"
            "  python tools/make_charts.py <this zip or the unzipped results folder>\n"))
    return n


def export(workspace, dest=None):
    ws = Path(workspace)
    dest = Path(dest) if dest else ws / f"results_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    with open(dest, "wb") as f:
        n = write_zip(ws, f)
    print(f"wrote {dest} ({n} files, {dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def zip_bytes(workspace):
    buf = io.BytesIO()
    write_zip(workspace, buf)
    return buf.getvalue()
