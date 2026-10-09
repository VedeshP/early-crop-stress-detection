"""Draw every chart from saved results — run this on any computer with matplotlib.

    python tools/make_charts.py results_20261012-101500.zip        # zip from `run.py export-results`
    python tools/make_charts.py workspace                          # or a workspace / outputs folder
    python tools/make_charts.py results.zip --out charts

Charts written (PNG):
  class_shares_<split>.png, index_ndvi.png, index_gndvi.png        (from eda/)
  history_<run>.png, miou_all_runs.png                             (from runs/*/history.csv)
  comparison.png, per_class_iou.png                                (from runs/*/eval_*.json / .csv)
  confusion_<run>_<tag>.png                                        (from runs/*/confusion_*.csv)
"""

import argparse
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _save(fig, path):
    if fig is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("  ", path.name)


def best_eval(run_dir):
    """Prefer the full-split evaluation with flip TTA, then any other."""
    evals = [json.loads(p.read_text()) | {"_file": p} for p in run_dir.glob("eval_*.json")]
    if not evals:
        return None
    evals.sort(key=lambda e: (bool(re.search(r"_n\d+$", e["_file"].stem)), e.get("tta") != "flip", -e.get("tiles", 0)))
    return evals[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="results zip, workspace, or outputs folder")
    ap.add_argument("--out", help="output folder for PNGs (default: <source>_charts)")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from agrivision import viz
    from agrivision.classes import PALETTE

    src = Path(a.source)
    tmp = None
    if src.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory()
        zipfile.ZipFile(src).extractall(tmp.name)
        root = Path(tmp.name)
    else:
        root = src
    out = Path(a.out) if a.out else src.with_name(src.stem + "_charts")
    out.mkdir(parents=True, exist_ok=True)
    print(f"reading {src}\nwriting {out}")

    # ---- EDA
    for summary_csv in root.glob("**/eda/class_summary.csv"):
        eda = summary_csv.parent
        summary = pd.read_csv(summary_csv)
        for split in summary["split"].unique():
            _save(viz.plot_class_shares(summary, split), out / f"class_shares_{split}.png")
        for npz in eda.glob("index_*.npz"):
            ch = npz.stem.split("_", 1)[1]
            values = dict(np.load(npz))
            _save(viz.plot_distributions(values, ch.upper()), out / f"index_{ch}.png")
        break

    # ---- training runs
    histories = {h.parent.name: h for h in root.glob("**/runs/*/history.csv")}
    for run, h in sorted(histories.items()):
        _save(viz.plot_history(h, run), out / f"history_{run}.png")
    if histories:
        fig, ax = plt.subplots(figsize=(8, 4))
        for run, h in sorted(histories.items()):
            df = pd.read_csv(h)
            if "val_miou" in df:
                ax.plot(df["epoch"], df["val_miou"], marker="o", markersize=3, label=run)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation mIoU (quick subset)")
        ax.set_title("Validation mIoU — all runs")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, out / "miou_all_runs.png")

    # ---- final evaluation comparison
    run_dirs = sorted({p.parent for p in root.glob("**/runs/*/eval_*.json")})
    best = {d.name: best_eval(d) for d in run_dirs}
    best = {k: v for k, v in best.items() if v}
    if best:
        metrics = [("miou", "mIoU"), ("iou_drydown", "drydown"), ("iou_nutrient_deficiency", "nutrient def."),
                   ("iou_weed_cluster", "weed cluster")]
        runs = sorted(best, key=lambda r: best[r].get("miou", 0))
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(runs) * len(metrics) / 2 + 2))
        h = 0.8 / len(metrics)
        y = np.arange(len(runs))
        for i, (key, label) in enumerate(metrics):
            vals = [best[r].get(key, np.nan) for r in runs]
            bars = ax.barh(y + i * h, vals, h, label=label)
            for b, v in zip(bars, vals):
                if v == v:
                    ax.text(v, b.get_y() + h / 2, f" {v:.3f}", va="center", fontsize=7)
        ax.set_yticks(y + h * (len(metrics) - 1) / 2, runs, fontsize=8)
        ax.set_xlabel("IoU")
        ax.set_title("Final evaluation (val, flip TTA where available)")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        _save(fig, out / "comparison.png")
        pd.DataFrame([{"run": r, **{k: v for k, v in best[r].items() if k != "_file"}} for r in runs]) \
            .to_csv(out / "comparison.csv", index=False)

        tables = {}
        for r in runs:
            f = best[r]["_file"]
            csv = f.with_suffix(".csv")
            if csv.exists():
                tables[r] = pd.read_csv(csv, index_col=0)["IoU"]
        if tables:
            t = pd.DataFrame(tables)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            t.plot.bar(ax=ax, width=0.8)
            ax.set_ylabel("IoU")
            ax.set_title("Per-class IoU by run")
            ax.legend(fontsize=7)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
            fig.tight_layout()
            _save(fig, out / "per_class_iou.png")

    # ---- confusion matrices
    for cm_csv in sorted(root.glob("**/runs/*/confusion_*.csv")):
        run = cm_csv.parent.name
        cm = pd.read_csv(cm_csv, index_col=0)
        _save(viz.plot_confusion(cm, title=f"{run} · {cm_csv.stem[10:]}"), out / f"confusion_{run}_{cm_csv.stem[10:]}.png")

    # ---- copy image panels already rendered on the GPU machine
    for png in list(root.glob("**/eda/*.png")) + list(root.glob("**/runs/*/predictions_*.png")):
        name = png.name if "eda" in png.parts else f"{png.parent.name}_{png.name}"
        (out / name).write_bytes(png.read_bytes())
        print("  ", name, "(copied)")

    if tmp:
        tmp.cleanup()
    print("done")


if __name__ == "__main__":
    main()
