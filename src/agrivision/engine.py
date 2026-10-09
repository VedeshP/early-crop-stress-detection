"""Training and evaluation loops with checkpoint/resume and a time budget.

Kaggle sessions stop after ~12 h, so training saves `last.pt` every epoch and
stops cleanly before `time_limit_hours`. Point `resume` at that file (attach the
previous run's output as input) to continue in a new session.
"""

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .classes import DEFAULT_PRIORITY, resolve_classes
from .dataset import DEFAULT_AUGMENT, DEFAULT_CHANNELS, PackedTiles, SegDataset, class_weights, rare_sampling_weights
from .env import default_dirs, find_files, find_packed_root, seed_everything
from .losses import SegLoss
from .metrics import SegMetric
from .models import DEFAULT_LR, build_model, count_parameters, predict_proba, unwrap


@dataclass
class TrainConfig:
    run_name: str = "segformer_b2_rgbn_idx"
    arch: str = "segformer"                   # unet | deeplabv3plus | segformer | upernet | fpn
    encoder: str = None                       # None = default for the arch (e.g. mit_b2)
    pretrained: bool = True
    channels: tuple = DEFAULT_CHANNELS        # any of r g b nir ndvi gndvi savi
    classes: object = "all9"                  # "all9" | "targets" | explicit list
    priority: tuple = DEFAULT_PRIORITY
    epochs: int = 30
    steps_per_epoch: int = None               # None = one pass over the training tiles
    batch_size: object = 8                    # int, or "auto" = largest batch that fits the GPU
    grad_accum: int = 1
    lr: float = None                          # None = default for the arch
    weight_decay: float = 0.01
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01
    amp: bool = True
    num_workers: object = 4                   # int, or "auto" = min(8, CPU cores - 2)
    augment: dict = field(default_factory=lambda: dict(DEFAULT_AUGMENT))
    mosaic_p: float = 0.25
    rare_sampling: bool = True
    rare_power: float = 0.5
    class_weight_power: float = 0.5           # 0 disables class weights
    dice_weight: float = 0.5
    train_fraction: float = 1.0
    val_max_tiles: int = 3000                 # quick per-epoch validation; full val at the end
    val_every: int = 1
    final_eval: bool = True
    final_tta: str = "flip"
    time_limit_hours: float = None            # e.g. 11.0 on Kaggle
    data_parallel: bool = True
    seed: int = 42
    output_dir: str = None                    # None = <output>/runs
    resume: str = "auto"                      # "auto" | path to last.pt | None
    log_every: int = 100


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(ds, batch_size, workers, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=workers > 0)


def _splits(tiles, cfg):
    train = tiles.rows("train") if "train" in tiles.splits else None
    val = tiles.rows("val") if "val" in tiles.splits else None
    if train is None and val is None:
        raise ValueError("dataset has neither train nor val split")
    if train is None or val is None:  # carve a validation set out of whatever exists
        frame = train if train is not None else val
        frame = frame.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
        cut = max(1, int(len(frame) * 0.1))
        val, train = frame.iloc[:cut].reset_index(drop=True), frame.iloc[cut:].reset_index(drop=True)
    if cfg.train_fraction < 1.0:
        train = train.sample(frac=cfg.train_fraction, random_state=cfg.seed).reset_index(drop=True)
    return train, val


@torch.no_grad()
def evaluate(model, loader, class_names, device, tta="none", amp=True, max_batches=None):
    model.eval()
    metric = SegMetric(class_names)
    use_amp = amp and device.type == "cuda"
    for b, (x, target, mbits) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            prob = predict_proba(model, x, tta)
        metric.update(prob.argmax(1).cpu(), target, mbits)
    return metric


def _save(path, payload):
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _resolve_resume(cfg, run_dir):
    if not cfg.resume:
        return None
    if cfg.resume != "auto":
        p = Path(cfg.resume)
        if not p.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {p}")
        return p
    if (run_dir / "last.pt").exists():
        return run_dir / "last.pt"
    found = [p for p in find_files(f"**/{cfg.run_name}/last.pt") if p != run_dir / "last.pt"]
    return found[0] if found else None


def find_batch_size(cfg, device, size=512, candidates=(48, 32, 24, 16, 12, 8, 6, 4, 2)):
    """Largest batch whose forward+backward fits in GPU memory, then one step smaller for headroom."""
    if device.type != "cuda":
        print("batch_size='auto' needs a GPU; using 4")
        return 4
    n_classes = len(resolve_classes(cfg.classes))
    model = build_model(cfg.arch, cfg.encoder, len(cfg.channels), n_classes, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = SegLoss(n_classes, dice_weight=cfg.dice_weight).to(device)
    fits = None
    for b in candidates:
        try:
            x = torch.randn(b, len(cfg.channels), size, size, device=device)
            y = torch.randint(0, n_classes, (b, size, size), device=device)
            with torch.autocast("cuda", enabled=cfg.amp):
                loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            fits = b
            break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    del model, opt
    torch.cuda.empty_cache()
    if fits is None:
        raise RuntimeError("even batch size 2 does not fit in GPU memory")
    smaller = [c for c in candidates if c < fits]
    chosen = smaller[0] if smaller else fits
    print(f"batch_size auto: {fits} fits, using {chosen} for headroom")
    return chosen


def train(cfg, data_root=None):
    if isinstance(cfg, dict):
        cfg = TrainConfig(**cfg)
    t_start = time.time()
    seed_everything(cfg.seed)
    device = _device()
    if cfg.num_workers == "auto":
        cfg.num_workers = max(0, min(8, (os.cpu_count() or 2) - 2))
    if cfg.batch_size == "auto":
        cfg.batch_size = find_batch_size(cfg, device)
    classes = resolve_classes(cfg.classes)
    root = find_packed_root(data_root)
    tiles = PackedTiles(root)
    train_df, val_df = _splits(tiles, cfg)

    out_root = Path(cfg.output_dir) if cfg.output_dir else default_dirs()["output"] / "runs"
    run_dir = out_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_dict["classes"] = list(classes)
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=list))

    common = dict(channels=cfg.channels, classes=classes, priority=cfg.priority)
    train_ds = SegDataset(tiles, train_df, train=True, augment_cfg=cfg.augment, mosaic_p=cfg.mosaic_p, **common)
    val_ds = SegDataset(tiles, val_df, train=False, **common)
    if cfg.val_max_tiles and len(val_df) > cfg.val_max_tiles:
        quick_val_ds = SegDataset(tiles, val_df.sample(cfg.val_max_tiles, random_state=cfg.seed), **common)
    else:
        quick_val_ds = val_ds

    steps = cfg.steps_per_epoch or math.ceil(len(train_ds) / cfg.batch_size)
    sampler = None
    if cfg.rare_sampling:
        w = rare_sampling_weights(train_df, classes, cfg.rare_power)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), steps * cfg.batch_size, replacement=True)
    elif cfg.steps_per_epoch:
        sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=steps * cfg.batch_size)
    train_dl = _loader(train_ds, cfg.batch_size, cfg.num_workers, shuffle=True, sampler=sampler, drop_last=True)
    val_dl = _loader(quick_val_ds, cfg.batch_size, cfg.num_workers)

    model = build_model(cfg.arch, cfg.encoder, len(cfg.channels), len(classes), cfg.pretrained).to(device)
    print(f"model {cfg.arch} ({cfg.encoder or 'default encoder'}): {count_parameters(model):.1f}M params | "
          f"train {len(train_ds)} tiles, val {len(val_ds)} (quick {len(quick_val_ds)}) | device {device}")
    if cfg.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"using DataParallel on {torch.cuda.device_count()} GPUs")

    cw = class_weights(train_df, classes, cfg.class_weight_power) if cfg.class_weight_power else None
    criterion = SegLoss(len(classes), cw, cfg.dice_weight).to(device)
    lr = cfg.lr or DEFAULT_LR.get(cfg.arch, 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.epochs * steps // cfg.grad_accum)

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best, history = 0, -1.0, []
    ckpt_path = _resolve_resume(cfg, run_dir)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best_miou"], ck.get("history", [])
        print(f"resumed from {ckpt_path} at epoch {start_epoch} (best mIoU {best:.4f})")

    def checkpoint(epoch):
        base = {"config": cfg_dict, "classes": list(classes), "channels": list(cfg.channels),
                "channel_stats": tiles.meta.get("channel_stats", {}), "epoch": epoch, "best_miou": best,
                "history": history}
        _save(run_dir / "last.pt", {**base, "model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()})

    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, target, _) in enumerate(train_dl):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(x), target) / cfg.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item() * cfg.grad_accum
            seen += 1
            if (step + 1) % cfg.log_every == 0:
                print(f"epoch {epoch + 1} step {step + 1}/{steps} loss {running / seen:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {seen * cfg.batch_size / (time.time() - t0):.1f} img/s",
                      flush=True)
            if cfg.time_limit_hours and time.time() - t_start > cfg.time_limit_hours * 3600:
                stop = True
                break
        row = {"epoch": epoch + 1, "train_loss": running / max(seen, 1), "lr": scheduler.get_last_lr()[0],
               "minutes": (time.time() - t0) / 60}
        if device.type == "cuda":
            row["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
        if not stop and ((epoch + 1) % cfg.val_every == 0 or epoch + 1 == cfg.epochs):
            m = evaluate(model, val_dl, classes, device, amp=cfg.amp).compute()
            row.update({"val_miou": m["miou"], **{f"iou_{k}": v for k, v in m["iou"].items()}})
            if m["miou"] > best:
                best = m["miou"]
                _save(run_dir / "best.pt", {"config": cfg_dict, "classes": list(classes), "channels": list(cfg.channels),
                                             "channel_stats": tiles.meta.get("channel_stats", {}), "epoch": epoch,
                                             "best_miou": best, "model": unwrap(model).state_dict()})
            print(f"== epoch {epoch + 1}: loss {row['train_loss']:.4f}  val mIoU {m['miou']:.4f}  best {best:.4f}  "
                  f"({row['minutes']:.1f} min)", flush=True)
        history.append(row)
        _write_history(run_dir / "history.csv", history)
        if stop:
            print(f"time limit reached; saved last.pt at epoch {epoch + 1}. Resume in a new session.")
            checkpoint(epoch - 1)  # the interrupted epoch is repeated on resume
            break
        checkpoint(epoch)

    result = {"run_dir": str(run_dir), "best_quick_val_miou": best, "epochs_done": len(history),
              "stopped_early": stop}
    if cfg.final_eval and not stop and (run_dir / "best.pt").exists():
        result["final"] = evaluate_checkpoint(run_dir / "best.pt", root, "val", cfg.final_tta,
                                              batch_size=cfg.batch_size, num_workers=cfg.num_workers)["summary"]
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _write_history(path, history):
    keys = sorted({k for r in history for k in r}, key=lambda k: (k != "epoch", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def load_checkpoint(path, device=None):
    device = device or _device()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    model = build_model(c["arch"], c.get("encoder"), len(ck["channels"]), len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def evaluate_checkpoint(path, data_root=None, split="val", tta="flip", max_tiles=None, batch_size=8, num_workers=4,
                        save=True):
    device = _device()
    model, ck = load_checkpoint(path, device)
    tiles = PackedTiles(find_packed_root(data_root))
    frame = tiles.rows(split)
    if max_tiles and len(frame) > max_tiles:
        frame = frame.sample(max_tiles, random_state=0)
    c = ck["config"]
    ds = SegDataset(tiles, frame, channels=ck["channels"], classes=ck["classes"], priority=c.get("priority", DEFAULT_PRIORITY))
    metric = evaluate(model, _loader(ds, batch_size, num_workers), ck["classes"], device, tta=tta)
    res = metric.compute()
    table = metric.table()
    summary = {"checkpoint": str(path), "split": split, "tiles": len(ds), "tta": tta, "miou": res["miou"],
               **{f"iou_{k}": v for k, v in res["iou"].items()}}
    if save:
        out = Path(path).parent
        tag = f"{split}_tta-{tta}" + (f"_n{max_tiles}" if max_tiles else "")
        table.to_csv(out / f"eval_{tag}.csv")
        pd.DataFrame(metric.cm.numpy(), index=metric.names, columns=metric.names).to_csv(out / f"confusion_{tag}.csv")
        (out / f"eval_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"{Path(path).name} on {split} ({len(ds)} tiles, TTA={tta}): mIoU {res['miou']:.4f}")
    return {"summary": summary, "table": table, "metric": metric}
