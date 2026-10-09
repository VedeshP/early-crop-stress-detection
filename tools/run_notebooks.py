"""Execute built notebooks headlessly (used to test them; also handy on a remote GPU box).

    python tools/run_notebooks.py build/local/03_train.ipynb [more.ipynb ...] [--out executed/] [--timeout 0]

Environment variables (AGRIVISION_DATA, AGRIVISION_OUTPUT, AGRIVISION_OVERRIDES, ...) are passed through,
so a notebook can be pointed at other data or given smaller settings without editing it.
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--out", default="executed", help="folder for executed copies (with outputs)")
    ap.add_argument("--timeout", type=int, default=0, help="per-cell timeout in seconds, 0 = none")
    a = ap.parse_args()
    import nbformat
    from nbclient import NotebookClient
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in map(Path, a.notebooks):
        nb = nbformat.read(path, as_version=4)
        print(f"running {path} ...", flush=True)
        client = NotebookClient(nb, timeout=a.timeout or None, kernel_name="python3",
                                resources={"metadata": {"path": str(path.parent)}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, out / path.name)
        print(f"  ok -> {out / path.name}", flush=True)


if __name__ == "__main__":
    main()
