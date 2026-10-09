"""Run steps on Kaggle through the official Kaggle CLI (push, poll status, download output).

Needs `pip install kaggle` and an API token (Kaggle → Settings → API → Create New Token,
saved as ~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY environment variables).
"""

import re
import shutil
import subprocess
import time

STATUS = re.compile(r"(queued|running|complete|error|cancel\w*)", re.I)


def kaggle_exe():
    exe = shutil.which("kaggle")
    if not exe:
        raise FileNotFoundError("Kaggle CLI not found: pip install kaggle, then add your API token")
    return exe


def _run(args, timeout=600):
    p = subprocess.run([kaggle_exe()] + args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def push(kernel_dir):
    code, out = _run(["kernels", "push", "-p", str(kernel_dir)])
    if code != 0 or "error" in out.lower() and "successfully" not in out.lower():
        raise RuntimeError(f"kaggle kernels push failed:\n{out}")
    return out


def status(kernel_id):
    code, out = _run(["kernels", "status", kernel_id], timeout=120)
    m = STATUS.search(out)
    return (m.group(1).lower() if m else "unknown"), out


def download_output(kernel_id, dest):
    dest.mkdir(parents=True, exist_ok=True)
    code, out = _run(["kernels", "output", kernel_id, "-p", str(dest)], timeout=7200)
    if code != 0:
        raise RuntimeError(f"kaggle kernels output failed:\n{out}")
    return out


def wait(kernel_id, on_change, poll_seconds=120, max_hours=13):
    """Poll until the kernel finishes. Returns the final status string."""
    last, t0 = None, time.time()
    while time.time() - t0 < max_hours * 3600:
        try:
            st, raw = status(kernel_id)
        except Exception as e:  # transient network / CLI errors
            st, raw = "unknown", str(e)
        if st != last:
            on_change(st, raw)
            last = st
        if st.startswith(("complete", "error", "cancel")):
            return st
        time.sleep(poll_seconds)
    return "timeout"
