"""Reading packed tiles, building the multi-channel input, augmentation, targets."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .classes import AV_ID, DEFAULT_PRIORITY, INVALID, resolve_classes
from .packed import ShardReader

ALL_CHANNELS = ("r", "g", "b", "nir", "ndvi", "gndvi", "savi")
DEFAULT_CHANNELS = ("r", "g", "b", "nir", "ndvi", "gndvi")
_FALLBACK_STATS = {
    "r": (0.40, 0.15), "g": (0.40, 0.15), "b": (0.40, 0.15), "nir": (0.50, 0.20),
    "ndvi": (0.15, 0.30), "gndvi": (0.15, 0.30), "savi": (0.20, 0.40),
}


# ---------------------------------------------------------------- decoding

def decode_rgb(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_gray(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)


def decode_bits(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.uint16)


class PackedTiles:
    """The packed dataset written by prepare.run(): meta.json, index.csv, shards/."""

    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.index = pd.read_csv(self.root / "index.csv")
        self.reader = ShardReader(self.root / "shards")

    def rows(self, split):
        return self.index[self.index["split"] == split].reset_index(drop=True)

    @property
    def splits(self):
        return sorted(self.index["split"].unique())

    def read(self, rec):
        rgb = decode_rgb(self.reader.read(rec["rgb_shard"], rec["rgb_off"], rec["rgb_len"]))
        nir = decode_gray(self.reader.read(rec["nir_shard"], rec["nir_off"], rec["nir_len"]))
        bits = decode_bits(self.reader.read(rec["lbl_shard"], rec["lbl_off"], rec["lbl_len"]))
        return rgb, nir, bits

    def channel_stats(self, channels):
        stats = self.meta.get("channel_stats", {})
        mean = np.array([stats.get(c, {}).get("mean", _FALLBACK_STATS[c][0]) for c in channels], np.float32)
        std = np.array([stats.get(c, {}).get("std", _FALLBACK_STATS[c][1]) for c in channels], np.float32)
        return mean, np.maximum(std, 1e-3)


# ---------------------------------------------------------------- channels

def build_channels(rgb, nir, channels=DEFAULT_CHANNELS):
    """uint8 RGB (H,W,3) + NIR (H,W) -> float32 (C,H,W). Bands scaled to 0..1, indices in -1..1."""
    r, g, b = (rgb[..., i].astype(np.float32) / 255.0 for i in range(3))
    n = nir.astype(np.float32) / 255.0
    out = []
    for c in channels:
        if c == "r":
            out.append(r)
        elif c == "g":
            out.append(g)
        elif c == "b":
            out.append(b)
        elif c == "nir":
            out.append(n)
        elif c == "ndvi":
            out.append(_norm_diff(n, r))
        elif c == "gndvi":
            out.append(_norm_diff(n, g))
        elif c == "savi":
            out.append(1.5 * (n - r) / (n + r + 0.5))
        else:
            raise ValueError(f"unknown channel '{c}' (choose from {ALL_CHANNELS})")
    return np.stack(out, 0)


def _norm_diff(a, b, eps=1e-3):
    s = a + b
    return np.where(s > eps, (a - b) / np.maximum(s, eps), 0.0).astype(np.float32)


# ---------------------------------------------------------------- targets

class TargetEncoder:
    """Turn the uint16 label bits into model targets.

    Returns
      target: uint8 (H,W) single label per pixel (overlaps resolved by priority), 255 = ignore
      mbits : uint16 (H,W) multi-hot over model classes, used by the overlap-aware metric
    """

    def __init__(self, classes, priority=DEFAULT_PRIORITY):
        self.classes = resolve_classes(classes)
        self.bits = [(k, 1 << AV_ID[name]) for k, name in enumerate(self.classes) if k > 0]
        order = [p for p in priority if p in self.classes] + [c for c in self.classes[1:] if c not in priority]
        self.paint_order = [(self.classes.index(name), 1 << AV_ID[name]) for name in reversed(order)]

    def __call__(self, bits):
        valid = (bits & INVALID) == 0
        target = np.zeros(bits.shape, np.uint8)
        for k, b in self.paint_order:
            target[(bits & b) != 0] = k
        mbits = np.zeros(bits.shape, np.uint16)
        for k, b in self.bits:
            mbits |= ((bits & b) != 0).astype(np.uint16) << k
        mbits[mbits == 0] = 1  # background
        target[~valid] = 255
        mbits[~valid] = 0
        return target, mbits


# ---------------------------------------------------------------- augmentation

DEFAULT_AUGMENT = {"hflip": True, "vflip": True, "rot90": True, "scale": (0.75, 1.25), "brightness": 0.1}


def augment(rgb, nir, bits, cfg, rng):
    """Geometric + uniform-brightness augmentation. Never changes the NIR/red ratio."""
    if cfg.get("hflip") and rng.random() < 0.5:
        rgb, nir, bits = rgb[:, ::-1], nir[:, ::-1], bits[:, ::-1]
    if cfg.get("vflip") and rng.random() < 0.5:
        rgb, nir, bits = rgb[::-1], nir[::-1], bits[::-1]
    if cfg.get("rot90"):
        k = int(rng.integers(4))
        if k:
            rgb, nir, bits = np.rot90(rgb, k), np.rot90(nir, k), np.rot90(bits, k)
    scale = cfg.get("scale")
    if scale:
        s = rng.uniform(*scale)
        if abs(s - 1.0) > 0.02:
            rgb, nir, bits = _rescale_crop(rgb, nir, bits, s, rng)
    b = cfg.get("brightness") or 0
    if b:
        f = rng.uniform(1 - b, 1 + b)
        rgb = np.clip(rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)
        nir = np.clip(nir.astype(np.float32) * f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(nir), np.ascontiguousarray(bits)


def _rescale_crop(rgb, nir, bits, s, rng):
    h, w = bits.shape
    nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    rgb = cv2.resize(np.ascontiguousarray(rgb), (nw, nh), interpolation=cv2.INTER_LINEAR)
    nir = cv2.resize(np.ascontiguousarray(nir), (nw, nh), interpolation=cv2.INTER_LINEAR)
    bits = cv2.resize(np.ascontiguousarray(bits), (nw, nh), interpolation=cv2.INTER_NEAREST)
    if nh >= h:  # zoomed in: random crop back to the tile size
        y, x = int(rng.integers(nh - h + 1)), int(rng.integers(nw - w + 1))
        return rgb[y:y + h, x:x + w], nir[y:y + h, x:x + w], bits[y:y + h, x:x + w]
    # zoomed out: pad; padded pixels are marked invalid so the loss ignores them
    y, x = int(rng.integers(h - nh + 1)), int(rng.integers(w - nw + 1))
    out_rgb = np.zeros((h, w, 3), np.uint8)
    out_nir = np.zeros((h, w), np.uint8)
    out_bits = np.full((h, w), INVALID, np.uint16)
    out_rgb[y:y + nh, x:x + nw], out_nir[y:y + nh, x:x + nw], out_bits[y:y + nh, x:x + nw] = rgb, nir, bits
    return out_rgb, out_nir, out_bits


def mosaic(tiles):
    """Four tiles downscaled 2x and stitched 2x2 into one tile of the original size."""
    parts = []
    for rgb, nir, bits in tiles:
        h, w = bits.shape
        parts.append((
            cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
            cv2.resize(nir, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
            cv2.resize(bits, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
        ))
    top = [np.concatenate([parts[0][i], parts[1][i]], 1) for i in range(3)]
    bottom = [np.concatenate([parts[2][i], parts[3][i]], 1) for i in range(3)]
    return tuple(np.concatenate([top[i], bottom[i]], 0) for i in range(3))


# ---------------------------------------------------------------- torch dataset

def _torch():
    import torch
    return torch


class SegDataset:
    """torch-compatible dataset over one split of a PackedTiles."""

    def __init__(self, tiles, frame, channels=DEFAULT_CHANNELS, classes="all9",
                 priority=DEFAULT_PRIORITY, train=False, augment_cfg=None, mosaic_p=0.0):
        self.tiles = tiles
        self.recs = frame.to_dict("records")
        self.channels = tuple(channels)
        self.encoder = TargetEncoder(classes, priority)
        self.train = train
        self.augment_cfg = DEFAULT_AUGMENT if augment_cfg is None else augment_cfg
        self.mosaic_p = mosaic_p if train else 0.0
        self.mean, self.std = tiles.channel_stats(self.channels)
        self._rng = None

    def __len__(self):
        return len(self.recs)

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(_torch().initial_seed() % 2**32)
        return self._rng

    def load(self, i):
        return self.tiles.read(self.recs[i])

    def __getitem__(self, i):
        torch = _torch()
        if self.mosaic_p and self.rng.random() < self.mosaic_p:
            idx = [i] + list(self.rng.integers(len(self.recs), size=3))
            rgb, nir, bits = mosaic([self.load(j) for j in idx])
        else:
            rgb, nir, bits = self.load(i)
        if self.train:
            rgb, nir, bits = augment(rgb, nir, bits, self.augment_cfg, self.rng)
        x = build_channels(rgb, nir, self.channels)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        target, mbits = self.encoder(bits)
        return (torch.from_numpy(x), torch.from_numpy(target.astype(np.int64)),
                torch.from_numpy(mbits.astype(np.int32)))


# ---------------------------------------------------------------- class balance

def class_pixel_freq(frame, classes):
    """Approximate pixel frequency of each model class in a split (from index.csv counts)."""
    classes = resolve_classes(classes)
    valid = float(frame["px_valid"].sum()) or 1.0
    freq = []
    for c in classes:
        if c == "background":
            fg = sum(frame[f"px_{k}"].sum() for k in classes[1:])
            freq.append(max(valid - fg, 1.0) / valid)
        else:
            freq.append(max(float(frame[f"px_{c}"].sum()), 1.0) / valid)
    return np.array(freq)


def class_weights(frame, classes, power=0.5, clip=(0.1, 10.0)):
    """Median-frequency weights for the loss: rare classes count more."""
    f = class_pixel_freq(frame, classes)
    w = (np.median(f) / f) ** power
    return np.clip(w, *clip).astype(np.float32)


def rare_sampling_weights(frame, classes, power=0.5):
    """Per-tile sampling weight: tiles that contain rare classes are drawn more often."""
    classes = resolve_classes(classes)
    f = class_pixel_freq(frame, classes)
    boost = (np.median(f) / f) ** power
    w = np.ones(len(frame))
    for k, c in enumerate(classes):
        if k == 0:
            continue
        present = frame[f"px_{c}"].to_numpy() > 0
        w = np.where(present, np.maximum(w, boost[k]), w)
    return w
