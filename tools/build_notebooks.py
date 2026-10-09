"""Build runnable .ipynb notebooks from the recipes in notebooks_src/.

Each notebook is self-contained: the toolkit source (src/agrivision) is embedded
as %%writefile cells, so it runs on Kaggle or any machine without copying the repo.

Recipe format (percent format, as used by Jupytext / VS Code):
    # %%                      code cell
    # %% [markdown]           markdown cell ("# " prefixes are stripped)
    # %% [kaggle] / [local]   cell only included for that target
    # %% [markdown kaggle]    markdown cell only for Kaggle

Usage:
    python tools/build_notebooks.py                      # both targets
    python tools/build_notebooks.py --target kaggle --kaggle-user myname --kaggle-dataset myname/agrivision-2021-packed
    python tools/build_notebooks.py --target local --no-embed   # import from the repo instead of embedding
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "agrivision"
RECIPES = REPO / "notebooks_src"
BUILD = REPO / "build"

REQUIREMENTS = ["segmentation-models-pytorch>=0.5.0"]
LOCAL_REQUIREMENTS = REQUIREMENTS + ["opencv-python-headless", "pandas", "matplotlib", "psutil"]
SRC_DIR = {"kaggle": "/tmp/agrivision_src", "local": "_agrivision_src"}

# Kaggle kernel settings per notebook (used in kernel-metadata.json)
KAGGLE = {
    "01_prepare_data": {"gpu": False, "title": "AgriVision 01 Prepare Data"},
    "02_explore_data": {"gpu": False, "title": "AgriVision 02 Explore Data"},
    "03_train": {"gpu": True, "title": "AgriVision 03 Train"},
    "04_evaluate": {"gpu": True, "title": "AgriVision 04 Evaluate"},
}

HEADER = re.compile(r"^# %%\s*(?:\[(?P<tags>[^\]]*)\])?\s*(?P<title>.*)$")


def parse_recipe(path):
    cells, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADER.match(line)
        if m:
            if cur:
                cells.append(cur)
            tags = set((m.group("tags") or "").split())
            cur = {"type": "markdown" if "markdown" in tags else "code",
                   "targets": tags & {"kaggle", "local"} or {"kaggle", "local"}, "lines": []}
            continue
        if cur is None:
            continue
        cur["lines"].append(line)
    if cur:
        cells.append(cur)
    out = []
    for c in cells:
        lines = c["lines"]
        if c["type"] == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            out.append({**c, "source": "\n".join(lines)})
    return out


def source_files():
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def source_hash():
    h = hashlib.sha1()
    for p in source_files():
        h.update(p.relative_to(SRC).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def _cell(kind, source, hidden=False, tags=()):
    meta = {}
    if hidden:
        meta["jupyter"] = {"source_hidden": True}
    if tags:
        meta["tags"] = list(tags)
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": meta, "source": lines}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": lines}


def setup_cells(target, embed, nb_dir):
    version = re.search(r'__version__ = "([^"]+)"', (SRC / "__init__.py").read_text()).group(1)
    cells = [_cell("markdown", (
        "## 0 · Setup\n"
        f"Installs dependencies and loads the **AgriVision toolkit v{version}** (source hash `{source_hash()}`). "
        "The toolkit source is embedded below (collapsed cells), generated from `src/agrivision` by "
        "`tools/build_notebooks.py` — edit the repo, not these cells."))]
    reqs = REQUIREMENTS if target == "kaggle" else LOCAL_REQUIREMENTS
    if target == "kaggle":
        cells.append(_cell("code", (
            "import subprocess, sys\n"
            f"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', {', '.join(repr(r) for r in reqs)}], check=True)")))
    else:
        cells.append(_cell("code", (
            "# Install missing packages on a fresh machine (PyTorch with CUDA must be installed separately:\n"
            "# https://pytorch.org/get-started/locally/).\n"
            "INSTALL = False\n"
            "if INSTALL:\n"
            "    import subprocess, sys\n"
            f"    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', {', '.join(repr(r) for r in reqs)}], check=True)")))
    if embed:
        src_dir = SRC_DIR[target]
        subdirs = sorted({p.parent.relative_to(SRC).as_posix() for p in source_files()})
        mk = "import os\n" + "\n".join(
            f"os.makedirs({(src_dir + '/agrivision' + ('' if d == '.' else '/' + d))!r}, exist_ok=True)" for d in subdirs)
        cells.append(_cell("code", mk, hidden=True, tags=["toolkit-source"]))
        for p in source_files():
            rel = p.relative_to(SRC).as_posix()
            code = p.read_text(encoding="utf-8").rstrip("\n") + "\n"
            cells.append(_cell("code", f"%%writefile {src_dir}/agrivision/{rel}\n{code}", hidden=True, tags=["toolkit-source"]))
        path_expr = repr(src_dir)
    else:
        rel = os.path.relpath(REPO / "src", nb_dir).replace("\\", "/")
        path_expr = f"str((__import__('pathlib').Path.cwd() / {rel!r}).resolve())"
    cells.append(_cell("code", (
        "import importlib, sys\n"
        f"_src = {path_expr}\n"
        "if _src not in sys.path:\n"
        "    sys.path.insert(0, _src)\n"
        "for _m in [m for m in sys.modules if m == 'agrivision' or m.startswith('agrivision.')]:\n"
        "    del sys.modules[_m]  # pick up freshly written source when re-running\n"
        "importlib.invalidate_caches()\n"
        "import agrivision\n"
        "print('agrivision', agrivision.__version__, 'from', agrivision.__file__)")))
    return cells


def settings_cell(settings):
    """Bake run settings into the notebook: environment variables + CONFIG overrides."""
    env = dict(settings.get("env") or {})
    if settings.get("overrides"):
        env["AGRIVISION_OVERRIDES"] = json.dumps(settings["overrides"])
    lines = ["# Run settings injected by the runner (run.py). Delete this cell to use the defaults below.",
             "import os"]
    lines += [f"os.environ[{k!r}] = {v!r}" for k, v in env.items()]
    return _cell("code", "\n".join(lines), tags=["injected-settings"])


def build_notebook(recipe, target, embed=True, nb_dir=None, settings=None):
    cells = [c for c in parse_recipe(recipe) if target in c["targets"]]
    out = []
    # keep the first markdown cell (title) on top, then setup, then the rest
    if cells and cells[0]["type"] == "markdown":
        out.append(_cell("markdown", cells[0]["source"] + f"\n\n*Built for **{target}** on {time.strftime('%Y-%m-%d')}.*"))
        cells = cells[1:]
    out += setup_cells(target, embed, nb_dir or BUILD / target)
    if settings:
        out.append(settings_cell(settings))
    out += [_cell(c["type"], c["source"]) for c in cells]
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def kaggle_slug(name, suffix=""):
    title = KAGGLE.get(name, {"title": f"AgriVision {name}"})["title"] + (f" {suffix}" if suffix else "")
    return _slug(title), title


def build_one(name, target, nb_dir, embed=True, settings=None, kaggle_user=None, dataset_sources=(),
              kernel_sources=(), suffix=""):
    """Build one notebook (and kernel-metadata.json for Kaggle). Returns the .ipynb path."""
    recipe = RECIPES / f"{name}.py"
    nb_dir = Path(nb_dir)
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(recipe, target, embed, nb_dir, settings)
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"{name[:2]}-{i:03d}"
    path = nb_dir / f"{name}.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    if target == "kaggle":
        slug, title = kaggle_slug(name, suffix)
        meta = {
            "id": f"{kaggle_user or 'YOUR_KAGGLE_USERNAME'}/{slug}",
            "title": title,
            "code_file": path.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": KAGGLE.get(name, {}).get("gpu", False),
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": list(dataset_sources),
            "competition_sources": [],
            "kernel_sources": list(kernel_sources),
        }
        (nb_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return path


def build(target, kaggle_user=None, kaggle_dataset=None, embed=True, out_root=None):
    out_root = Path(out_root) if out_root else BUILD
    written = []
    for recipe in sorted(RECIPES.glob("*.py")):
        name = recipe.stem
        nb_dir = out_root / "kaggle" / name if target == "kaggle" else out_root / "local"
        sources = [kaggle_dataset] if (kaggle_dataset and not name.startswith("01")) else []
        written.append(build_one(name, target, nb_dir, embed, kaggle_user=kaggle_user, dataset_sources=sources))
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["kaggle", "local", "all"], default="all")
    ap.add_argument("--kaggle-user", help="your Kaggle username (for kernel-metadata.json)")
    ap.add_argument("--kaggle-dataset", help="packed dataset slug, e.g. myname/agrivision-2021-packed")
    ap.add_argument("--no-embed", action="store_true", help="local only: import the toolkit from the repo")
    a = ap.parse_args()
    targets = ["kaggle", "local"] if a.target == "all" else [a.target]
    for t in targets:
        paths = build(t, a.kaggle_user, a.kaggle_dataset, embed=not (a.no_embed and t == "local"))
        print(f"[{t}]", *[str(p) for p in paths], sep="\n  ")


if __name__ == "__main__":
    main()
