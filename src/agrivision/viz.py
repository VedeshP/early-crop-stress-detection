"""Plots for the exploration and evaluation notebooks."""

import numpy as np
import pandas as pd

from .classes import INVALID, PALETTE, resolve_classes
from .dataset import TargetEncoder, build_channels


def _plt():
    import matplotlib.pyplot as plt
    return plt


def colorize(target, classes):
    classes = resolve_classes(classes)
    out = np.zeros(target.shape + (3,), np.uint8)
    for k, name in enumerate(classes):
        out[target == k] = PALETTE[name]
    out[target == 255] = (255, 255, 255)
    return out


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


def plot_class_shares(summary, split="train"):
    plt = _plt()
    s = summary[(summary["split"] == split) & (summary["class"] != "background")].sort_values("pixel_share_%")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s["class"], s["pixel_share_%"], color=[np.array(PALETTE[c]) / 255 for c in s["class"]])
    ax.set_xlabel("% of valid pixels")
    ax.set_title(f"Class pixel share ({split})")
    for y, v in enumerate(s["pixel_share_%"]):
        ax.text(v, y, f" {v:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def show_tiles(tiles, frame, n=4, classes="all9", seed=0):
    """RGB | NIR | NDVI | label for n random tiles."""
    plt = _plt()
    classes = resolve_classes(classes)
    enc = TargetEncoder(classes)
    if len(frame) == 0:
        print("no tiles match this filter")
        return None
    recs = frame.sample(min(n, len(frame)), random_state=seed).to_dict("records")
    fig, axes = plt.subplots(len(recs), 4, figsize=(12, 3 * len(recs)), squeeze=False)
    for row, rec in zip(axes, recs):
        rgb, nir, bits = tiles.read(rec)
        ndvi = build_channels(rgb, nir, ("ndvi",))[0]
        target, _ = enc(bits)
        for ax, img, title in zip(row, (rgb, nir, ndvi, colorize(target, classes)),
                                  ("RGB", "NIR", "NDVI", "label")):
            ax.imshow(img, cmap="gray" if title == "NIR" else ("RdYlGn" if title == "NDVI" else None),
                      vmin=-0.2 if title == "NDVI" else None, vmax=0.9 if title == "NDVI" else None)
            ax.set_title(f"{title} · {rec['id']}" if title == "RGB" else title, fontsize=9)
            ax.axis("off")
    _legend(fig, classes)
    fig.tight_layout()
    return fig


def _legend(fig, classes):
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=np.array(PALETTE[c]) / 255, label=c) for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.08)


def index_distributions(tiles, frame, channel="ndvi", classes=("background", "drydown", "nutrient_deficiency", "weed_cluster"),
                        n_tiles=300, per_tile=400, seed=0):
    """Sample per-class values of a vegetation index (evidence that stress is spectrally separable)."""
    from .classes import AV_ID
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
    return {c: np.concatenate(v) if v else np.array([]) for c, v in values.items()}


def plot_distributions(values, channel="NDVI"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, v in values.items():
        if v.size:
            ax.hist(v, bins=80, range=(-0.5, 1.0), density=True, histtype="step", linewidth=2,
                    color=np.array(PALETTE[c]) / 255, label=f"{c} (median {np.median(v):.2f})")
    ax.set_xlabel(channel)
    ax.set_ylabel("density")
    ax.set_title(f"{channel} distribution by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_history(history_csv):
    plt = _plt()
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
    fig.tight_layout()
    return fig


def plot_confusion(metric, normalize=True):
    plt = _plt()
    cm = metric.cm.numpy().astype(float)
    if normalize:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(metric.names)), metric.names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(metric.names)), metric.names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(metric.names)):
        for j in range(len(metric.names)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def show_predictions(model, ckpt, tiles, frame, n=4, tta="flip", seed=0, device=None):
    """RGB | NDVI | ground truth | prediction."""
    import torch
    from .dataset import SegDataset
    from .models import predict_proba
    plt = _plt()
    device = device or next(model.parameters()).device
    classes = ckpt["classes"]
    if len(frame) == 0:
        print("no tiles match this filter")
        return None
    sub = frame.sample(min(n, len(frame)), random_state=seed).reset_index(drop=True)
    ds = SegDataset(tiles, sub, channels=ckpt["channels"], classes=classes)
    fig, axes = plt.subplots(len(sub), 4, figsize=(12, 3 * len(sub)), squeeze=False)
    for i, row in enumerate(axes):
        x, target, _ = ds[i]
        rgb, nir, _ = ds.load(i)
        with torch.no_grad():
            pred = predict_proba(model, x[None].to(device), tta).argmax(1)[0].cpu().numpy()
        ndvi = build_channels(rgb, nir, ("ndvi",))[0]
        for ax, img, title in zip(row, (rgb, ndvi, colorize(target.numpy(), classes), colorize(pred, classes)),
                                  ("RGB", "NDVI", "ground truth", "prediction")):
            ax.imshow(img, cmap="RdYlGn" if title == "NDVI" else None,
                      vmin=-0.2 if title == "NDVI" else None, vmax=0.9 if title == "NDVI" else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    _legend(fig, classes)
    fig.tight_layout()
    return fig
