"""Run state shared by the runner, the monitor and the dashboard (one JSON file per run)."""

import json
import os
import platform
import socket
import threading
import time
from collections import deque
from pathlib import Path


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class RunState:
    def __init__(self, run_dir, name, target, steps):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.path = self.dir / "state.json"
        self.lock = threading.RLock()
        self.tail = deque(maxlen=400)          # last log lines, for the dashboard
        self.last_output = time.time()
        self.data = {
            "name": name, "target": target, "status": "starting", "started": now(), "ended": None,
            "host": socket.gethostname(), "platform": platform.platform(), "pid": os.getpid(),
            "steps": [{"name": s["name"], "notebook": s.get("notebook"), "status": "pending",
                       "started": None, "ended": None, "error": None, "log": None} for s in steps],
            "current": None, "gpu": None, "disk_free_gb": None, "epochs": {}, "alerts": [],
            "dashboard_url": None, "kaggle": {},
        }
        self.save()

    def update(self, **kw):
        with self.lock:
            self.data.update(kw)
            self.save()

    def step(self, index, **kw):
        with self.lock:
            self.data["steps"][index].update(kw)
            self.save()

    def alert(self, text):
        with self.lock:
            self.data["alerts"] = (self.data["alerts"] + [f"{now()} {text}"])[-50:]
            self.save()

    def log_line(self, line):
        with self.lock:
            self.tail.append(line)
            self.last_output = time.time()

    def snapshot(self):
        with self.lock:
            d = json.loads(json.dumps(self.data))
            d["log_tail"] = list(self.tail)
            d["seconds_since_output"] = round(time.time() - self.last_output)
            return d

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
