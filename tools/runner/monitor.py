"""Background watcher: GPU, disk, stalls, new training epochs, periodic heartbeat."""

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path

DEFAULT_WATCH = {
    "poll_seconds": 30,
    "epoch_every": 1,            # notify every N epochs (0 = never)
    "heartbeat_minutes": 120,    # periodic "still alive" summary (0 = never)
    "stall_minutes": 30,         # alert if no new output for this long while a step runs
    "gpu_temp_c": 87,            # alert above this GPU temperature
    "disk_free_gb": 5,           # alert below this free space in the workspace
    "gpu_idle_minutes": 20,      # alert if GPU utilisation stays ~0 during training
}


def gpu_stats():
    """List of dicts from nvidia-smi, or None when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run([exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None
        gpus.append({"name": p[0], "util_%": num(p[1]), "mem_used_mb": num(p[2]), "mem_total_mb": num(p[3]),
                     "temp_c": num(p[4]), "power_w": num(p[5])})
    return gpus or None


def read_history(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _f(v, digits=4):
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


class Monitor(threading.Thread):
    def __init__(self, state, notifier, workspace, watch=None):
        super().__init__(daemon=True)
        self.state, self.notifier, self.workspace = state, notifier, Path(workspace)
        self.cfg = {**DEFAULT_WATCH, **(watch or {})}
        self.stop_event = threading.Event()
        self.seen_epochs = {}
        self.last_heartbeat = time.time()
        self.gpu_busy_at = time.time()
        self.stalled = False

    def stop(self):
        self.stop_event.set()

    def run(self):
        self._scan_epochs(initial=True)
        while not self.stop_event.wait(self.cfg["poll_seconds"]):
            try:
                self.tick()
            except Exception as e:  # keep watching whatever happens
                print(f"[monitor] {type(e).__name__}: {e}", flush=True)

    def tick(self):
        snap = self.state.snapshot()
        running = snap["status"] == "running" and snap["current"] is not None
        step = snap["steps"][snap["current"]]["name"] if running else None

        gpus = gpu_stats()
        free = shutil.disk_usage(self.workspace if self.workspace.exists() else Path.cwd()).free / 2**30
        self.state.update(gpu=gpus, disk_free_gb=round(free, 1))

        if gpus:
            g = gpus[0]
            if g["temp_c"] and g["temp_c"] >= self.cfg["gpu_temp_c"]:
                self._alert("GPU hot", f"GPU at {g['temp_c']:.0f} °C", "high", "gpu_hot")
            if (g["util_%"] or 0) > 5:
                self.gpu_busy_at = time.time()
            elif running and step and step.startswith("train") and self.cfg["gpu_idle_minutes"] and \
                    time.time() - self.gpu_busy_at > self.cfg["gpu_idle_minutes"] * 60:
                self._alert("GPU idle", f"GPU utilisation ~0% for {self.cfg['gpu_idle_minutes']} min during {step}. "
                            "Data loading stuck or training finished?", "high", "gpu_idle")
        if free < self.cfg["disk_free_gb"]:
            self._alert("Low disk space", f"{free:.1f} GB free in {self.workspace}", "high", "disk")

        if running and self.cfg["stall_minutes"]:
            silent = snap["seconds_since_output"] / 60
            if silent > self.cfg["stall_minutes"] and not self.stalled:
                self.stalled = True
                self._alert("No progress", f"{step}: no new output for {silent:.0f} min "
                            "(may be normal for a long validation; check the dashboard).", "high", "stall")
            elif silent < 1:
                self.stalled = False

        self._scan_epochs()

        hb = self.cfg["heartbeat_minutes"]
        if hb and time.time() - self.last_heartbeat > hb * 60:
            self.last_heartbeat = time.time()
            self.notifier.send("Heartbeat", self.summary(), "low")

    def _alert(self, title, msg, priority, key):
        self.state.alert(f"{title}: {msg}")
        self.notifier.send(title, msg, priority, ("warning",), key=key)

    def _scan_epochs(self, initial=False):
        runs = self.workspace / "outputs" / "runs"
        if not runs.exists():
            return
        every = self.cfg["epoch_every"]
        epochs = {}
        for hist in runs.glob("*/history.csv"):
            rows = read_history(hist)
            run = hist.parent.name
            epochs[run] = rows[-1] if rows else {}
            n_prev = self.seen_epochs.get(run, 0)
            if len(rows) > n_prev:
                self.seen_epochs[run] = len(rows)
                if initial or not every:
                    continue
                for r in rows[n_prev:]:
                    if int(float(r.get("epoch", 0))) % every == 0:
                        self.notifier.send(f"{run} epoch {r.get('epoch')}", self._epoch_text(r), "low", ("chart_with_upwards_trend",))
        self.state.update(epochs=epochs)

    @staticmethod
    def _epoch_text(r):
        return (f"loss {_f(r.get('train_loss'))} | val mIoU {_f(r.get('val_miou'))}\n"
                f"drydown {_f(r.get('iou_drydown'), 3)} | nutrient {_f(r.get('iou_nutrient_deficiency'), 3)} | "
                f"weed {_f(r.get('iou_weed_cluster'), 3)}\n"
                f"{_f(r.get('minutes'), 1)} min/epoch, GPU peak {_f(r.get('gpu_peak_gb'), 1)} GB")

    def summary(self):
        s = self.state.snapshot()
        lines = [f"status: {s['status']} on {s['host']}"]
        for st in s["steps"]:
            lines.append(f"  {st['name']}: {st['status']}")
        for run, r in (s.get("epochs") or {}).items():
            if r:
                lines.append(f"{run}: epoch {r.get('epoch')} mIoU {_f(r.get('val_miou'))}")
        if s.get("gpu"):
            g = s["gpu"][0]
            lines.append(f"GPU {g['util_%']:.0f}% | {g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f} MB | {g['temp_c']:.0f} °C")
        lines.append(f"disk free: {s.get('disk_free_gb')} GB")
        if s.get("dashboard_url"):
            lines.append(f"dashboard: {s['dashboard_url']}")
        return "\n".join(lines)
