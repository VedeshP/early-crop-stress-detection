"""Build the packed dataset straight from the official 21 GB archive.

The archive is one .tar.gz, so we cannot download only some folders. Instead
we stream it (no full copy is ever stored):

  pass 1  read only label files (boundaries, masks, labels/*)  -> scratch
  compose merge each tile's label files into one uint16 bit-PNG, count pixels
  select  keep tiles with target classes (+ a sample of the rest), all of val
  pass 2  stream again and keep RGB / NIR images of the selected tiles only

With mode="all" a single pass is enough. Works on Kaggle (URL) and locally
(URL or an already-downloaded .tar.gz path).
"""

import json
import os
import shutil
import tarfile
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import __version__
from .classes import AV_ID, IGNORED_LABELS, INVALID, LABEL_CLASSES, TARGET_CLASSES
from .dataset import ALL_CHANNELS, PackedTiles, build_channels
from .env import PACKED_MARKER
from .packed import ShardReader, ShardWriter
from .stream import DEFAULT_URL, open_source

SPLITS = ("train", "val", "test")


@dataclass
class PrepareConfig:
    out_dir: str = ""
    work_dir: str = ""
    source: str = DEFAULT_URL
    splits: tuple = ("train", "val")          # test has no labels
    mode: str = "relevant"                    # "relevant" or "all"
    targets: tuple = TARGET_CLASSES
    min_target_pixels: int = 500              # ~5 m^2 at 10 cm/px
    negative_fraction: float = 0.20           # share of non-target tiles kept (healthy + other classes)
    keep_all_splits: tuple = ("val",)         # keep full val so scores compare with published results
    max_output_gb: float = 19.0               # Kaggle saves at most 20 GB of output
    est_tile_mb: float = 0.21                 # measured RGB+NIR+label size per tile
    shard_gb: float = 1.0
    stats_tiles: int = 1500
    workers: int = 0                          # 0 = all CPU cores
    seed: int = 0
    keep_work: bool = False
    limit_bytes: int = 0                      # testing only: stop the stream early
    extra: dict = field(default_factory=dict)


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- archive walking

def parse_member(name):
    """'Agriculture-Vision-2021/train/labels/drydown/ID.png' -> dict, or None if not a tile file."""
    parts = name.split("/")
    if len(parts) < 4:
        return None
    split = next((p for p in parts if p in SPLITS), None)
    if split is None:
        return None
    rest = parts[parts.index(split) + 1:]
    stem = os.path.splitext(rest[-1])[0]
    if rest[0] in ("boundaries", "masks") and len(rest) == 2:
        return {"split": split, "kind": rest[0], "comp": rest[0], "id": stem}
    if rest[0] == "labels" and len(rest) == 3:
        return {"split": split, "kind": "labels", "comp": rest[1], "id": stem}
    if rest[0] == "images" and len(rest) == 3:
        return {"split": split, "kind": "images", "comp": rest[1], "id": stem}
    return None


def _walk(cfg, label):
    stream = open_source(cfg.source, limit_bytes=cfg.limit_bytes, log=_log, label=label)
    tf = tarfile.open(fileobj=stream, mode="r|gz")
    try:
        for m in tf:
            if not m.isfile():
                continue
            info = parse_member(m.name)
            if info and info["split"] in cfg.splits:
                yield info, tf, m
    except (tarfile.ReadError, EOFError, zlib.error) as e:
        if not cfg.limit_bytes:
            raise
        _log(f"[{label}] stream stopped at the test limit ({type(e).__name__})")
    finally:
        tf.close()
        stream.close()


# ---------------------------------------------------------------- label composition

_READER = None


def _compose_batch(args):
    global _READER
    comp_dir, items = args
    if _READER is None or _READER.dir != Path(comp_dir):
        _READER = ShardReader(comp_dir)
    out = []
    for tile_id, comps in items:
        valid = None
        for name in ("boundaries", "masks"):
            if name in comps:
                m = cv2.imdecode(np.frombuffer(_READER.read(*comps[name]), np.uint8), cv2.IMREAD_GRAYSCALE) > 0
                valid = m if valid is None else (valid & m)
        layers = {}
        for cls in LABEL_CLASSES:
            if cls in comps:
                layers[cls] = cv2.imdecode(np.frombuffer(_READER.read(*comps[cls]), np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        if valid is None:
            shape = next(iter(layers.values())).shape if layers else (512, 512)
            valid = np.ones(shape, bool)
        shape = valid.shape
        bits = np.zeros(shape, np.uint16)
        counts = {}
        for cls, m in layers.items():
            bits[m] |= np.uint16(1 << AV_ID[cls])
            counts[cls] = int((m & valid).sum())
        bits[~valid] |= np.uint16(INVALID)
        counts["valid"] = int(valid.sum())
        counts["background"] = int(((bits == 0)).sum())
        ok, png = cv2.imencode(".png", bits)
        out.append((tile_id, png.tobytes(), counts))
    return out


def _compose_all(comps, comp_dir, lbl_writer, workers):
    ids = sorted(comps)
    batches = [(str(comp_dir), [(i, comps[i]) for i in ids[s:s + 256]]) for s in range(0, len(ids), 256)]
    rows, index, done, t0 = {}, {}, 0, time.time()

    def consume(results):
        nonlocal done
        for tile_id, png, counts in results:
            index[tile_id] = lbl_writer.add(png)
            rows[tile_id] = counts
        done += len(results)
        if done % 5120 < 256 or done == len(ids):
            _log(f"[compose] {done}/{len(ids)} tiles ({done / max(time.time() - t0, 1e-6):.0f}/s)")

    if workers > 1 and len(batches) > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for res in ex.map(_compose_batch, batches):
                    consume(res)
            return rows, index
        except Exception as e:  # e.g. restricted environments without process pools
            _log(f"[compose] process pool failed ({e}); continuing in one process")
            rows, index, done = {}, {}, 0
    global _READER
    for b in batches:
        consume(_compose_batch(b))
    if _READER is not None:  # release the file so the scratch folder can be deleted (Windows)
        _READER.close()
        _READER = None
    return rows, index


# ---------------------------------------------------------------- selection

def select_tiles(df, cfg, rng):
    keep = np.zeros(len(df), bool)
    target_cols = [f"px_{t}" for t in cfg.targets]
    has_target = (df[target_cols].to_numpy() >= cfg.min_target_pixels).any(axis=1)
    for split in df["split"].unique():
        m = (df["split"] == split).to_numpy()
        if cfg.mode == "all" or split in cfg.keep_all_splits:
            keep |= m
            continue
        keep |= m & has_target
        neg = np.flatnonzero(m & ~has_target)
        n = int(round(len(neg) * cfg.negative_fraction))
        if n:
            keep[rng.choice(neg, n, replace=False)] = True

    est_gb = keep.sum() * cfg.est_tile_mb / 1000
    if cfg.max_output_gb and est_gb > cfg.max_output_gb:
        # drop random non-target tiles from non-protected splits until the estimate fits
        protected = df["split"].isin(cfg.keep_all_splits).to_numpy()
        droppable = np.flatnonzero(keep & ~protected & ~has_target)
        excess = int((est_gb - cfg.max_output_gb) * 1000 / cfg.est_tile_mb) + 1
        drop = rng.choice(droppable, min(excess, len(droppable)), replace=False)
        keep[drop] = False
        _log(f"[select] estimate {est_gb:.1f} GB > budget; dropped {len(drop)} non-target tiles")
    return keep


# ---------------------------------------------------------------- statistics

def compute_channel_stats(root, n_tiles=1500, seed=0, channels=ALL_CHANNELS):
    tiles = PackedTiles(root)
    split = "train" if "train" in tiles.splits else tiles.splits[0]
    frame = tiles.rows(split)
    frame = frame.sample(min(n_tiles, len(frame)), random_state=seed)
    s = np.zeros(len(channels))
    s2 = np.zeros(len(channels))
    n = 0
    for rec in frame.to_dict("records"):
        rgb, nir, bits = tiles.read(rec)
        x = build_channels(rgb, nir, channels)[:, ::4, ::4]
        v = ((bits & INVALID) == 0)[::4, ::4]
        if not v.any():
            continue
        vals = x[:, v].astype(np.float64)
        s += vals.sum(1)
        s2 += (vals ** 2).sum(1)
        n += vals.shape[1]
    mean = s / max(n, 1)
    std = np.sqrt(np.maximum(s2 / max(n, 1) - mean ** 2, 1e-8))
    return {c: {"mean": float(m), "std": float(d)} for c, m, d in zip(channels, mean, std)}


# ---------------------------------------------------------------- main entry

def run(cfg):
    if isinstance(cfg, dict):
        cfg = PrepareConfig(**cfg)
    if not cfg.out_dir or not cfg.work_dir:
        raise ValueError("set out_dir and work_dir")
    out, work = Path(cfg.out_dir), Path(cfg.work_dir)
    if (out / "meta.json").exists():
        raise FileExistsError(f"{out} already contains a packed dataset; delete it or choose another out_dir")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    workers = cfg.workers or os.cpu_count() or 1
    shard_bytes = int(cfg.shard_gb * 1e9)
    budget = cfg.max_output_gb * 1e9 if cfg.max_output_gb else float("inf")
    single_pass = cfg.mode == "all"
    t_start = time.time()
    _log(f"[prepare] source={cfg.source}\n          mode={cfg.mode} splits={cfg.splits} out={out}")

    comps = defaultdict(dict)
    split_of = {}
    images = defaultdict(dict)
    img_writers = {m: ShardWriter(out / "shards", m, shard_bytes) for m in ("rgb", "nir")}
    written = 0
    skipped_budget = 0

    def add_image(info, data):
        nonlocal written, skipped_budget
        tid, mod = info["id"], info["comp"]
        if mod not in img_writers:
            return
        if mod not in images.get(tid, {}) and written + len(data) > budget and not images.get(tid):
            skipped_budget += 1
            return
        images[tid][mod] = img_writers[mod].add(data)
        written += len(data)

    # ---- pass 1: labels (and images too in single-pass mode)
    with ShardWriter(work / "components", "comp", shard_bytes) as cw:
        n = 0
        for info, tf, m in _walk(cfg, "pass 1/2" if not single_pass else "pass 1/1"):
            kind = info["kind"]
            if kind == "images":
                if single_pass:
                    add_image(info, tf.extractfile(m).read())
                continue
            if info["comp"] in IGNORED_LABELS or (kind == "labels" and info["comp"] not in AV_ID):
                continue
            comps[info["id"]][info["comp"]] = cw.add(tf.extractfile(m).read())
            split_of[info["id"]] = info["split"]
            n += 1
    _log(f"[pass 1] {n} label files for {len(comps)} tiles")
    if not comps:
        raise RuntimeError("no label files found; check source / splits")

    # ---- compose labels
    with ShardWriter(work / "labels", "lbl", shard_bytes) as lw:
        counts, lbl_index = _compose_all(comps, work / "components", lw, workers)
    df = pd.DataFrame.from_dict(counts, orient="index").fillna(0).astype(np.int64)
    df.index.name = "id"
    df = df.reset_index()
    df.insert(1, "split", df["id"].map(split_of))
    for c in ("valid", "background") + tuple(LABEL_CLASSES):
        if c not in df:
            df[c] = 0
    df = df.rename(columns={c: f"px_{c}" for c in ("valid", "background") + tuple(LABEL_CLASSES)})

    # ---- select
    keep = select_tiles(df, cfg, rng)
    selected = set(df.loc[keep, "id"])
    for split in df["split"].unique():
        m = df["split"] == split
        _log(f"[select] {split}: keeping {int((keep & m).sum())} of {int(m.sum())} tiles")

    # ---- pass 2: images of the selected tiles
    if not single_pass:
        for info, tf, m in _walk(cfg, "pass 2/2"):
            if info["kind"] == "images" and info["id"] in selected:
                add_image(info, tf.extractfile(m).read())
    for w in img_writers.values():
        w.close()
    if skipped_budget:
        _log(f"[warn] output budget reached: {skipped_budget} images skipped")

    # ---- labels of kept tiles -> output shards
    rows = []
    src = ShardReader(work / "labels")
    with ShardWriter(out / "shards", "lbl", shard_bytes) as ow:
        for rec in df[keep].to_dict("records"):
            tid = rec["id"]
            im = images.get(tid, {})
            if "rgb" not in im or "nir" not in im or tid not in lbl_index:
                continue
            lbl = ow.add(src.read(*lbl_index[tid]))
            row = {"id": tid, "split": rec["split"]}
            for key, (name, off, length) in (("rgb", im["rgb"]), ("nir", im["nir"]), ("lbl", lbl)):
                row.update({f"{key}_shard": name, f"{key}_off": off, f"{key}_len": length})
            row.update({k: v for k, v in rec.items() if k.startswith("px_")})
            rows.append(row)
    src.close()
    index = pd.DataFrame(rows).sort_values(["split", "id"]).reset_index(drop=True)
    index.to_csv(out / "index.csv", index=False)

    # ---- meta + stats
    meta = {
        PACKED_MARKER: 1,
        "toolkit_version": __version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": cfg.source,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "tiles": {s: int((index["split"] == s).sum()) for s in index["split"].unique()},
        "label_bits": {"invalid_bit": 15, "class_bits": {c: AV_ID[c] for c in LABEL_CLASSES}},
        "pixels": {s: {c[3:]: int(index.loc[index["split"] == s, c].sum()) for c in index.columns if c.startswith("px_")}
                   for s in index["split"].unique()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    _log("[stats] computing channel mean/std")
    meta["channel_stats"] = compute_channel_stats(out, cfg.stats_tiles, cfg.seed)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "README.md").write_text(_readme(meta))

    if not cfg.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    size_gb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
    summary = {"out_dir": str(out), "tiles": meta["tiles"], "size_gb": round(size_gb, 2),
               "minutes": round((time.time() - t_start) / 60, 1)}
    _log(f"[done] {summary}")
    return summary


def _readme(meta):
    tiles = ", ".join(f"{k}: {v}" for k, v in meta["tiles"].items())
    return f"""# Agriculture-Vision 2021 — packed subset

Built by the AgriVision toolkit v{meta['toolkit_version']} on {meta['created']} from
{meta['source']}

Tiles: {tiles}. Selection: mode={meta['config']['mode']}, targets={meta['config']['targets']},
negative_fraction={meta['config']['negative_fraction']}, kept in full: {meta['config']['keep_all_splits']}.

## Format
- `index.csv`: one row per tile; `rgb_*`, `nir_*`, `lbl_*` give (shard, byte offset, length);
  `px_*` columns are per-class pixel counts inside the valid area.
- `shards/*.bin`: concatenated original JPEG (RGB, NIR) and label PNG bytes.
- Label PNG: uint16; bit i set = Agriculture-Vision class i
  ({', '.join(f'{c}={b}' for c, b in meta['label_bits']['class_bits'].items())});
  bit 15 = outside field boundary / invalid pixel.
- `meta.json`: config, tile counts, pixel totals, per-channel mean/std.

Source data: Agriculture-Vision (Chiu et al., CVPR 2020), IntelinAir. Use is subject to the
dataset's Terms of Use (included in the original S3 bucket).
"""
