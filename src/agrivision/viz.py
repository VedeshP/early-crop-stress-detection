"""Charts and image panels for the notebooks.

matplotlib is optional: if it cannot be loaded (e.g. blocked by a Windows
application-control policy) every chart function prints a note and returns
None. The data behind each chart is saved separately (save_eda, history.csv,
eval_*.csv) so `tools/make_charts.py` can draw all charts later on another
computer. Image panels (sample tiles, predictions) use OpenCV and are always
written as PNG files.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .classes import AV_ID, INVALID, PALETTE, resolve_classes
from .dataset import TargetEncoder, build_channels

_PLT = {"checked": False, "mod": None}


def _plt():
    """matplotlib.pyplot, or None if it cannot be loaded."""
    if not _PLT["checked"]:
        _PLT["checked"] = True
        try:
            import matplotlib.pyplot as plt
            _PLT["mod"] = plt
        except Exception as e:  # ImportError, DLL blocked by policy, ...
            print(f"[viz] matplotlib unavailable ({type(e).__name__}: {str(e)[:120]}); charts are skipped, "
                  "their data is saved - draw them later with tools/make_charts.py")
    return _PLT["mod"]


def available():
    return _plt() is not None


def colorize(target, classes):
    classes = resolve_classes(classes)
    out = np.zeros(target.shape + (3,), np.uint8)
    for k, name in enumerate(classes):
        out[target == k] = PALETTE[name]
    out[target == 255] = (255, 255, 255)
    return out


# ---------------------------------------------------------------- data (no matplotlib)

def class_summary(tiles):
    """Pixel share and tile count per class for every split."""
    rows = []
    for split in tiles.splits:
        f = tiles.rows(split)
        valid = f["px_valid"].sum()
        for c in [c[3:] for c in f.columns if c.startswith("px_") and c not in ("px_valid",)]:
            rows.append({"split": split, "class": c, "pixel_share_%": 100 * f[f"px_{c}"].sum() / valid,
                         "tiles_with_class": int((f[f"px_{c}"] > 0).sum()), "tiles": len(f)})
    return pd.DataFrame(rows).round(3)


def index_distributions(tiles, frame, channel="ndvi", classes=("background", "drydown", "nutrient_deficiency", "weed_cluster"),
                        n_tiles=300, per_tile=400, seed=0):
    """Sample per-class values of a vegetation index (evidence that stress is spectrally separable)."""
    rng = np.random.default_rng(seed)
    values = {c: [] for c in classes}
    for rec in frame.sample(min(n_tiles, len(frame)), random_state=seed).to_dict("records"):
        rgb, nir, bits = tiles.read(rec)
        idx = build_channels(rgb, nir, (channel,))[0]
        valid = (bits & INVALID) == 0
        for c in classes:
            m = valid & (bits == 0) if c == "background" else valid & ((bits & (1 << AV_ID[c])) != 0)
            v = idx[m]
            if v.size:
                values[c].append(rng.choice(v, min(per_tile, v.size), replace=False))
    return {c: np.concatenate(v) if v else np.array([], np.float32) for c, v in values.items()}


def distribution_table(values):
    rows = []
    for c, v in values.items():
        if v.size:
            q = np.percentile(v, [10, 25, 50, 75, 90])
            rows.append({"class": c, "samples": v.size, "mean": v.mean(), "p10": q[0], "p25": q[1],
                         "median": q[2], "p75": q[3], "p90": q[4]})
    return pd.DataFrame(rows).round(4)


def save_eda(out_dir, summary, distributions):
    """Save EDA results: class_summary.csv, index_<ch>.npz + index_<ch>_summary.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "class_summary.csv", index=False)
    for ch, values in distributions.items():
        np.savez_compressed(out / f"index_{ch}.npz", **values)
        distribution_table(values).to_csv(out / f"index_{ch}_summary.csv", index=False)
    print(f"EDA data saved to {out}")
    return out


# ---------------------------------------------------------------- image panels (OpenCV, always available)

def _ndvi_rgb(ndvi, lo=-0.2, hi=0.9):
    """Red (low) -> yellow -> green (high), like matplotlib's RdYlGn."""
    t = np.clip((ndvi - lo) / (hi - lo), 0, 1)[..., None]
    red, yellow, green = np.array([215, 48, 39]), np.array([255, 255, 191]), np.array([26, 152, 80])
    out = np.where(t < 0.5, red + (yellow - red) * (t / 0.5), yellow + (green - yellow) * ((t - 0.5) / 0.5))
    return out.astype(np.uint8)


def _as_rgb(img):
    return np.repeat(img[..., None], 3, -1) if img.ndim == 2 else img


def _titled(img, title, size=256):
    img = cv2.resize(_as_rgb(img), (size, size), interpolation=cv2.INTER_NEAREST)
    bar = np.full((22, size, 3), 255, np.uint8)
    cv2.putText(bar, title[:40], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def _legend_strip(classes, width):
    strip = np.full((26, width, 3), 255, np.uint8)
    x = 4
    for c in classes:
        if x > width - 80:
            break
        cv2.rectangle(strip, (x, 7), (x + 12, 19), tuple(int(v) for v in PALETTE[c]), -1)
        cv2.putText(strip, c, (x + 16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        x += 22 + 7 * len(c)
    return strip


def save_panel(rows, path, classes):
    """rows: list of lists of (image, title). Writes one PNG grid."""
    grid = np.concatenate([np.concatenate([_titled(i, t) for i, t in r], 1) for r in rows], 0)
    grid = np.concatenate([grid, _legend_strip(resolve_classes(classes), grid.shape[1])], 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return path


def tile_rows(tiles, frame, n=4, classes="all9", seed=0):
    classes = resolve_classes(classes)
    enc = TargetEncoder(classes)
    rows = []
    for rec in frame.sample(min(n, len(frame)), random_state=seed).to_dict("records"):
        rgb, nir, bits = tiles.read(rec)
        target, _ = enc(bits)
        ndvi = build_channels(rgb, nir, ("ndvi",))[0]
        rows.append([(rgb, f"RGB {rec['id'][:22]}"), (nir, "NIR"), (_ndvi_rgb(ndvi), "NDVI"),
                     (colorize(target, classes), "label")])
    return rows


def show_tiles(tiles, frame, n=4, classes="all9", seed=0, save_to=None):
    """RGB | NIR | NDVI | label for n random tiles. Saves a PNG when save_to is given."""
    if len(frame) == 0:
        print("no tiles match this filter")
        return None
    rows = tile_rows(tiles, frame, n, classes, seed)
    if save_to:
        print("saved", save_panel(rows, save_to, classes))
    return _show_rows(rows, classes)


def prediction_rows(model, ckpt, tiles, frame, n=4, tta="flip", seed=0, device=None):
    import torch
    from .dataset import SegDataset
    from .models import predict_proba
    device = device or next(model.parameters()).device
    classes = ckpt["classes"]
    sub = frame.sample(min(n, len(frame)), random_state=seed).reset_index(drop=True)
    ds = SegDataset(tiles, sub, channels=ckpt["channels"], classes=classes)
    rows = []
    for i in range(len(sub)):
        x, target, _ = ds[i]
        rgb, nir, _ = ds.load(i)
        with torch.no_grad():
            pred = predict_proba(model, x[None].to(device), tta).argmax(1)[0].cpu().numpy()
        ndvi = build_channels(rgb, nir, ("ndvi",))[0]
        rows.append([(rgb, f"RGB {sub.loc[i, 'id'][:22]}"), (_ndvi_rgb(ndvi), "NDVI"),
                     (colorize(target.numpy(), classes), "ground truth"), (colorize(pred, classes), "prediction")])
    return rows


def show_predictions(model, ckpt, tiles, frame, n=4, tta="flip", seed=0, device=None, save_to=None):
    """RGB | NDVI | ground truth | prediction. Saves a PNG when save_to is given."""
    if len(frame) == 0:
        print("no tiles match this filter")
        return None
    rows = prediction_rows(model, ckpt, tiles, frame, n, tta, seed, device)
    if save_to:
        print("saved", save_panel(rows, save_to, ckpt["classes"]))
    return _show_rows(rows, ckpt["classes"])


def _show_rows(rows, classes):
    plt = _plt()
    if plt is None:
        return None
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(3 * len(rows[0]), 3 * len(rows)), squeeze=False)
    for axrow, row in zip(axes, rows):
        for ax, (img, title) in zip(axrow, row):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    _legend(fig, classes)
    fig.tight_layout()
    return fig


def _legend(fig, classes):
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=np.array(PALETTE[c]) / 255, label=c) for c in resolve_classes(classes)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.08)


# ---------------------------------------------------------------- charts (matplotlib, optional)

def plot_class_shares(summary, split="train"):
    plt = _plt()
    if plt is None:
        return None
    s = summary[(summary["split"] == split) & (summary["class"] != "background")].sort_values("pixel_share_%")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s["class"], s["pixel_share_%"], color=[np.array(PALETTE.get(c, (128, 128, 128))) / 255 for c in s["class"]])
    ax.set_xlabel("% of valid pixels")
    ax.set_title(f"Class pixel share ({split})")
    for y, v in enumerate(s["pixel_share_%"]):
        ax.text(v, y, f" {v:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_distributions(values, channel="NDVI"):
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, v in values.items():
        v = np.asarray(v)
        if v.size:
            ax.hist(v, bins=80, range=(-0.5, 1.0), density=True, histtype="step", linewidth=2,
                    color=np.array(PALETTE[c]) / 255, label=f"{c} (median {np.median(v):.2f})")
    ax.set_xlabel(channel)
    ax.set_ylabel("density")
    ax.set_title(f"{channel} distribution by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_history(history_csv, title=None):
    plt = _plt()
    if plt is None:
        return None
    h = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(h["epoch"], h["train_loss"], marker="o")
    ax[0].set_title("training loss")
    if "val_miou" in h:
        ax[1].plot(h["epoch"], h["val_miou"], marker="o", label="mIoU")
        for c in ("drydown", "nutrient_deficiency", "weed_cluster"):
            if f"iou_{c}" in h:
                ax[1].plot(h["epoch"], h[f"iou_{c}"], linestyle="--", label=c, color=np.array(PALETTE[c]) / 255)
        ax[1].set_title("validation IoU")
        ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("epoch")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def confusion_frame(metric):
    return pd.DataFrame(metric.cm.numpy(), index=metric.names, columns=metric.names)


def plot_confusion(cm, names=None, normalize=True, title=None):
    """cm: a SegMetric, a DataFrame (rows = true, columns = predicted) or an array with `names`."""
    plt = _plt()
    if plt is None:
        return None
    if hasattr(cm, "cm"):
        names, cm = cm.names, cm.cm.numpy()
    elif isinstance(cm, pd.DataFrame):
        names, cm = list(cm.columns), cm.to_numpy()
    cm = np.asarray(cm, float)
    if normalize:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
