"""Self-test for run.py: full local pipeline on synthetic data with a fake ntfy server.

    python tools/runner_selftest.py

Checks: steps run in order, notifications are sent (ntfy format), the dashboard
answers with live state, a re-run skips finished steps, and a failing step is reported.
"""

import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
from smoke_test import make_archive  # noqa: E402

RECEIVED = []


class FakeNtfy(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        RECEIVED.append({"path": self.path, "title": self.headers.get("Title"),
                         "priority": self.headers.get("Priority"), "body": body})
        self.send_response(200)
        self.end_headers()


def main():
    work = REPO / "_rtest"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    archive = work / "Agriculture-Vision-2021.tar.gz"
    make_archive(archive)

    srv = ThreadingHTTPServer(("127.0.0.1", 8799), FakeNtfy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    notify = work / "notify.json"
    notify.write_text(json.dumps({"min_seconds_between_same": 0, "channels": {
        "ntfy": {"server": "http://127.0.0.1:8799", "topic": "selftest"},
        "file": {"path": str(work / "notifications.log")}}}))

    train_common = {"pretrained": False, "epochs": 2, "steps_per_epoch": 2, "batch_size": 2, "num_workers": 0,
                    "val_max_tiles": None, "warmup_steps": 1, "amp": False, "log_every": 1, "time_limit_hours": None}
    cfg = {
        "name": "selftest", "target": "local", "workspace": str(work / "ws"),
        "steps": ["data", "explore", "train", "evaluate"],
        "data": {"source": "prepare"},
        "prepare": {"source": str(archive), "negative_fraction": 0.5, "min_target_pixels": 10, "stats_tiles": 50},
        "train_common": train_common,
        "train_runs": [{"run_name": "unet_t", "arch": "unet", "encoder": "resnet18"},
                       {"run_name": "segformer_t", "arch": "segformer", "encoder": "mit_b0"}],
        "evaluate": {"batch_size": 2, "num_workers": 0},
        "watch": {"poll_seconds": 2, "heartbeat_minutes": 0.05, "epoch_every": 1},
    }
    cfg_path = work / "selftest.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    cmd = [sys.executable, str(REPO / "run.py"), "run", "--config", str(cfg_path), "--notify", str(notify),
           "--dashboard", "--port", "8790", "--token", "tok", "--ignore-checks"]
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dash_seen, lines = [], []

    def reader():
        for line in p.stdout:
            lines.append(line)
    threading.Thread(target=reader, daemon=True).start()
    while p.poll() is None:
        time.sleep(3)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8790/api/state?t=tok", timeout=5) as r:
                s = json.loads(r.read())
                dash_seen.append((s["status"], s.get("current"), len(s.get("log_tail", []))))
        except OSError:
            pass
    time.sleep(1)
    out = "".join(lines)
    print(out[-2500:])
    assert p.returncode == 0, f"runner exited with {p.returncode}"

    # forbidden without token
    state = json.loads(sorted((work / "ws" / "runner").glob("*/state.json"))[-1].read_text())
    assert [s["status"] for s in state["steps"]] == ["done"] * 6, state["steps"]
    assert any(c == "running" and n > 0 for c, _, n in dash_seen), f"dashboard never showed live output: {dash_seen[:5]}"
    titles = [m["title"] for m in RECEIVED]
    print("\nnotifications:", *titles, sep="\n  ")
    for want in ("Run started", "Step started", "Step finished", "epoch", "Heartbeat", "Run finished"):
        assert any(want in (t or "") for t in titles), f"missing notification: {want}"
    assert all(m["path"] == "/selftest" for m in RECEIVED)

    # ---- re-run: everything already done -> skipped
    p2 = subprocess.run(cmd[:-6] + ["--ignore-checks"], cwd=REPO, capture_output=True, text=True)
    assert p2.returncode == 0, p2.stdout[-2000:]
    s2 = json.loads(sorted((work / "ws" / "runner").glob("*/state.json"))[-1].read_text())
    st2 = {s["name"]: s["status"] for s in s2["steps"]}
    assert st2["data"] == "skipped" and st2["train:unet_t"] == "skipped", st2

    # ---- failing step is reported
    bad = dict(cfg, steps=["train"], train_runs=[{"run_name": "broken", "arch": "nope"}])
    bad_path = work / "bad.json"
    bad_path.write_text(json.dumps(bad))
    RECEIVED.clear()
    p3 = subprocess.run([sys.executable, str(REPO / "run.py"), "run", "--config", str(bad_path), "--notify", str(notify),
                         "--ignore-checks"], cwd=REPO, capture_output=True, text=True)
    assert p3.returncode == 1
    assert any("FAILED" in (m["title"] or "") and "unknown arch" in m["body"] for m in RECEIVED), RECEIVED
    srv.shutdown()
    print("\nRUNNER SELF-TEST PASSED")


if __name__ == "__main__":
    main()
