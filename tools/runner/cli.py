"""`python run.py ...` — run the whole pipeline unattended, locally or on Kaggle.

  python run.py run --config configs/rtx4000ada.json --target local --notify configs/notify.json --dashboard
  python run.py run --config configs/kaggle.json --target kaggle --notify configs/notify.json
  python run.py status                      # latest run in the workspace
  python run.py notify-init                 # create configs/notify.json with a private ntfy topic
  python run.py notify-test --notify configs/notify.json
  python run.py check                       # pre-flight check of this machine
  python run.py build                       # (re)build all notebooks into build/
"""

import argparse
import json
import os
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import build_notebooks as nbb  # noqa: E402
from runner import kaggle_target  # noqa: E402
from runner.dashboard import Dashboard, Tunnel  # noqa: E402
from runner.monitor import Monitor, gpu_stats  # noqa: E402
from runner.notify import Notifier  # noqa: E402
from runner.results import export as export_results  # noqa: E402
from runner.state import RunState, now  # noqa: E402

NOTEBOOK = {"data": "01_prepare_data", "explore": "02_explore_data", "train": "03_train", "evaluate": "04_evaluate"}
PACKED = "agrivision2021_packed"


# ---------------------------------------------------------------- helpers

def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("name", Path(path).stem)
    cfg.setdefault("steps", ["data", "explore", "train", "evaluate"])
    cfg.setdefault("data", {"source": "prepare"})
    cfg.setdefault("train_runs", [{}])
    return cfg


def keep_awake(on=True):
    """Stop Windows from sleeping during a long run (no-op elsewhere)."""
    if platform.system() != "Windows":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def find_packed(workspace, data_cfg):
    cands = [data_cfg.get("path"), workspace / "data" / PACKED, workspace / "outputs" / PACKED]
    for c in cands:
        if c and (Path(c) / "meta.json").exists():
            return Path(c)
    for meta in list((workspace / "data").glob("**/meta.json")) if (workspace / "data").exists() else []:
        return meta.parent
    return None


def plan_steps(cfg, steps):
    """Expand config steps into concrete jobs (one train/evaluate job per training run)."""
    jobs = []
    for s in steps:
        if s == "train":
            for run in cfg["train_runs"]:
                o = {**cfg.get("train_common", {}), **run}
                jobs.append({"name": f"train:{o.get('run_name', 'default')}", "kind": "train", "notebook": NOTEBOOK["train"],
                             "overrides": {"train": o}, "run_name": o.get("run_name")})
        elif s == "evaluate":
            for run in cfg["train_runs"]:
                o = {**cfg.get("train_common", {}), **run}
                jobs.append({"name": f"evaluate:{o.get('run_name', 'default')}", "kind": "evaluate",
                             "notebook": NOTEBOOK["evaluate"], "run_name": o.get("run_name"),
                             "overrides": {"evaluate": dict(cfg.get("evaluate", {}))}})
        elif s in NOTEBOOK:
            ov = {"prepare": cfg.get("prepare", {})} if s == "data" else {}
            jobs.append({"name": s, "kind": s, "notebook": NOTEBOOK[s], "overrides": ov})
        else:
            raise ValueError(f"unknown step '{s}' (choose from {list(NOTEBOOK)})")
    return jobs


def preflight(workspace, need_gpu=True, need_gb=0):
    issues, info = [], []
    info.append(f"python {platform.python_version()} on {platform.platform()}")
    try:
        import torch
        info.append(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.append(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
        elif need_gpu:
            issues.append("PyTorch cannot see a GPU - install the CUDA build of torch (see requirements.txt)")
    except ImportError:
        issues.append("PyTorch not installed")
    try:
        import torch
        import torchvision
        info.append(f"torchvision {torchvision.__version__}")
        tv_cuda = "+cu" in torchvision.__version__
        t_cuda = "+cu" in torch.__version__
        if t_cuda != tv_cuda or (t_cuda and torch.__version__.split("+")[1] != torchvision.__version__.split("+")[1]):
            issues.append(f"torch {torch.__version__} and torchvision {torchvision.__version__} are different builds; "
                          "reinstall both together: pip uninstall -y torch torchvision && pip install torch torchvision "
                          "--index-url https://download.pytorch.org/whl/cu124")
    except ImportError:
        pass
    except Exception as e:
        issues.append(f"torchvision is broken ({type(e).__name__}: {str(e)[:120]}). Usually torch/torchvision versions "
                      "do not match; reinstall both together: pip uninstall -y torch torchvision && pip install torch "
                      "torchvision --index-url https://download.pytorch.org/whl/cu124")
    for mod, pip in (("segmentation_models_pytorch", "segmentation-models-pytorch"), ("cv2", "opencv-python-headless"),
                     ("nbclient", "nbclient"), ("ipykernel", "ipykernel"), ("pandas", "pandas")):
        try:
            __import__(mod)
        except ModuleNotFoundError as e:
            if e.name == mod:
                issues.append(f"missing package: python -m pip install {pip}")
            else:
                issues.append(f"{mod} is installed but needs '{e.name}', which is missing: python -m pip install {e.name}")
        except ImportError as e:  # installed but cannot load, e.g. "DLL load failed" on Windows
            issues.append(f"{mod} is installed but fails to load: {str(e)[:200]}"
                          + (" -> blocked by a Windows application-control policy: ask IT / the owner to allow it"
                             if "policy" in str(e).lower() else
                             " -> install the Microsoft Visual C++ Redistributable (x64) and reboot"
                             if "DLL load failed" in str(e) else ""))
        except Exception as e:  # installed but fails to load (e.g. torch/torchvision mismatch)
            issues.append(f"{mod} fails to import: {type(e).__name__}: {str(e)[:160]}")
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        spec = KernelSpecManager().get_kernel_spec("python3")
        info.append(f"notebook kernel: {spec.argv[0]}")
        if Path(spec.argv[0]).name.lower().startswith("python") and Path(spec.argv[0]).resolve() != Path(sys.executable).resolve() \
                and spec.argv[0] not in ("python", "python3"):
            issues.append(f"the 'python3' notebook kernel uses {spec.argv[0]}, not this Python ({sys.executable}). "
                          f"Fix: {sys.executable} -m ipykernel install --user")
    except Exception as e:
        issues.append(f"no Jupyter kernel 'python3' ({e}); fix: python -m ipykernel install --user")
    try:
        import matplotlib.pyplot  # noqa: F401
        info.append("matplotlib: ok")
    except Exception as e:  # optional: charts are skipped, their data is saved
        why = str(e)[:160]
        hint = (" (blocked by a Windows application-control policy - ask IT/owner to allow it)"
                if "policy" in why.lower() else "")
        info.append(f"WARNING matplotlib unavailable: {why}{hint}. Not a blocker: charts are skipped and their "
                    "data is saved; draw them on another computer with tools/make_charts.py")
    g = gpu_stats()
    info.append(f"nvidia-smi: {g[0]['name'] if g else 'not found'}")
    workspace.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workspace).free / 2**30
    info.append(f"workspace {workspace} ({free:.0f} GB free)")
    if need_gb and free < need_gb:
        issues.append(f"only {free:.0f} GB free; this run needs about {need_gb} GB")
    return info, issues


# ---------------------------------------------------------------- local target

def run_local(cfg, jobs, state, notifier, workspace, args):
    from runner.executor import execute_notebook
    nb_dir, ex_dir = state.dir / "notebooks", state.dir / "executed"
    nb_dir.mkdir(exist_ok=True)
    ex_dir.mkdir(exist_ok=True)
    base_env = {"AGRIVISION_HOME": str(workspace), "MPLBACKEND": "Agg"}
    data_cfg = cfg["data"]

    for i, job in enumerate(jobs):
        state.update(current=i)
        log_path = state.dir / "logs" / f"{i:02d}_{job['name'].replace(':', '_')}.log"
        state.step(i, status="running", started=now(), log=str(log_path))
        env = dict(base_env)
        packed = find_packed(workspace, data_cfg)

        # ---- skip work that is already done (re-running the same command continues a run)
        if job["kind"] == "data" and packed and not args.force:
            state.step(i, status="skipped", ended=now(), error=f"dataset already present: {packed}")
            continue
        if job["kind"] == "train" and not args.force and \
                (workspace / "outputs" / "runs" / job["run_name"] / "result.json").exists():
            state.step(i, status="skipped", ended=now(), error="run already finished (use --force to retrain)")
            continue
        if packed:
            env["AGRIVISION_DATA"] = str(packed)

        # ---- data from a Kaggle dataset instead of preparing it here
        if job["kind"] == "data" and data_cfg.get("source") == "kaggle":
            notifier.send("Step started", f"{job['name']}: downloading {data_cfg['kaggle_dataset']}", "low")
            ok, err = _kaggle_download_dataset(data_cfg["kaggle_dataset"], workspace / "data" / PACKED, log_path, state)
        elif job["kind"] == "data" and data_cfg.get("source") == "existing":
            ok, err = (packed is not None), (None if packed else f"no packed dataset at {data_cfg.get('path')}")
        else:
            overrides = json.loads(json.dumps(job["overrides"]))
            if job["kind"] == "evaluate" and job.get("run_name"):
                ck = workspace / "outputs" / "runs" / job["run_name"] / "best.pt"
                overrides.setdefault("evaluate", {})["checkpoint"] = str(ck)
            nb = nbb.build_one(job["notebook"], "local", nb_dir, settings={"env": env, "overrides": overrides})
            target = nb_dir / f"{i:02d}_{job['name'].replace(':', '_')}.ipynb"
            shutil.move(str(nb), target)
            notifier.send("Step started", f"{job['name']} ({i + 1}/{len(jobs)})", "low", ("arrow_forward",))
            ok, err = execute_notebook(target, ex_dir / target.name, log_path, on_line=state.log_line,
                                       timeout=args.cell_timeout or None)

        if ok:
            state.step(i, status="done", ended=now())
            notifier.send("Step finished", _step_summary(job, workspace), "default", ("white_check_mark",))
        else:
            state.step(i, status="failed", ended=now(), error=err)
            notifier.send("Step FAILED", f"{job['name']}\n{(err or '')[-1500:]}", "urgent", ("rotating_light",))
            if not args.keep_going:
                return False
    return all(s["status"] in ("done", "skipped") for s in state.data["steps"])


def _kaggle_download_dataset(slug, dest, log_path, state):
    dest.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        p = subprocess.Popen([kaggle_target.kaggle_exe(), "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            state.log_line(line.rstrip())
        code = p.wait()
    if code != 0:
        return False, f"kaggle datasets download exited with {code}"
    metas = list(dest.glob("**/meta.json"))
    return (bool(metas), None if metas else "download finished but no meta.json found")


def _step_summary(job, workspace):
    if job["kind"] == "train":
        res = workspace / "outputs" / "runs" / job["run_name"] / "result.json"
        if res.exists():
            r = json.loads(res.read_text())
            f = r.get("final", {})
            return (f"{job['name']}: best quick-val mIoU {r.get('best_quick_val_miou', 0):.4f}"
                    + (f"\nfull val (TTA {f.get('tta')}): mIoU {f.get('miou', 0):.4f}, drydown {f.get('iou_drydown', 0):.3f}, "
                       f"nutrient {f.get('iou_nutrient_deficiency', 0):.3f}" if f else "")
                    + ("\n(stopped by time limit - re-run to resume)" if r.get("stopped_early") else ""))
    if job["kind"] == "data":
        metas = list((workspace).glob(f"**/{PACKED}/meta.json"))
        if metas:
            m = json.loads(metas[0].read_text())
            return f"data ready: {m.get('tiles')}"
    return f"{job['name']} done"


# ---------------------------------------------------------------- kaggle target

def run_kaggle(cfg, jobs, state, notifier, workspace, args):
    k = cfg.get("kaggle", {})
    user = k.get("user") or os.environ.get("KAGGLE_USERNAME")
    if not user:
        raise SystemExit("set kaggle.user in the config (your Kaggle username)")
    dataset = k.get("dataset") or (cfg["data"].get("kaggle_dataset") if cfg["data"].get("source") == "kaggle" else None)
    poll = int(k.get("poll_seconds", 120))
    download = set(k.get("download_outputs", ["train", "evaluate"]))
    prep_slug, _ = nbb.kaggle_slug(NOTEBOOK["data"])
    data_inputs = {"dataset_sources": [dataset]} if dataset else {"kernel_sources": [f"{user}/{prep_slug}"]}
    kdir = state.dir / "kaggle"

    for i, job in enumerate(jobs):
        state.update(current=i)
        if job["kind"] == "data" and dataset:
            state.step(i, status="skipped", ended=now(), error=f"using Kaggle dataset {dataset}")
            continue
        suffix = job.get("run_name") or ""
        slug, title = nbb.kaggle_slug(job["notebook"], suffix if job["kind"] in ("train", "evaluate") else "")
        sources = {"dataset_sources": [], "kernel_sources": []}
        if job["kind"] != "data":
            for key, v in data_inputs.items():
                sources[key] += v
        if job["kind"] == "evaluate":
            tslug, _ = nbb.kaggle_slug(NOTEBOOK["train"], suffix)
            sources["kernel_sources"].append(f"{user}/{tslug}")
        overrides = json.loads(json.dumps(job["overrides"]))
        nb_dir = kdir / slug
        nbb.build_one(job["notebook"], "kaggle", nb_dir, settings={"overrides": overrides} if overrides else None,
                      kaggle_user=user, suffix=suffix if job["kind"] in ("train", "evaluate") else "", **sources)
        kid = f"{user}/{slug}"
        state.step(i, status="running", started=now(), log=f"https://www.kaggle.com/code/{kid}")
        state.update(kaggle={**state.data.get("kaggle", {}), job["name"]: kid})
        out = kaggle_target.push(nb_dir)
        state.log_line(out)
        notifier.send("Kaggle step pushed", f"{job['name']} -> https://www.kaggle.com/code/{kid}", "low")

        def on_change(st, raw, job=job):
            state.log_line(f"{now()} {job['name']}: {st}")
            if st in ("running", "complete", "error") or st.startswith("cancel"):
                notifier.send(f"Kaggle {st}", f"{job['name']}: {raw[-300:]}", "urgent" if st == "error" else "default")
        final = kaggle_target.wait(kid, on_change, poll_seconds=poll)
        if not final.startswith("complete"):
            state.step(i, status="failed", ended=now(), error=f"Kaggle status: {final}")
            if not args.keep_going:
                return False
            continue
        if job["kind"] in download:
            dest = workspace / "kaggle_outputs" / slug
            try:
                kaggle_target.download_output(kid, dest)
                state.log_line(f"downloaded output to {dest}")
            except Exception as e:
                state.alert(f"output download failed for {kid}: {e}")
        state.step(i, status="done", ended=now())
    return True


# ---------------------------------------------------------------- commands

def cmd_run(args):
    cfg = load_config(args.config)
    target = args.target or cfg.get("target", "local")
    workspace = Path(args.workspace or cfg.get("workspace", "workspace")).resolve()
    steps = args.steps.split(",") if args.steps else cfg["steps"]
    jobs = plan_steps(cfg, steps)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{cfg['name']}_{target}"
    notifier = Notifier.from_file(args.notify, run_label=cfg["name"])

    if args.dry_run:
        print(json.dumps({"target": target, "workspace": str(workspace), "jobs": jobs}, indent=2))
        return 0
    if target == "local":
        info, issues = preflight(workspace, need_gpu=any(j["kind"] in ("train", "evaluate") for j in jobs),
                                 need_gb=25 if any(j["kind"] == "data" for j in jobs) else 0)
        print("\n".join(info))
        if issues:
            print("\nPRE-FLIGHT PROBLEMS:\n- " + "\n- ".join(issues))
            if not args.ignore_checks:
                notifier.send("Run not started", "\n".join(issues), "high")
                return 2

    state = RunState(workspace / "runner" / run_id, cfg["name"], target, jobs)
    state.update(workspace=str(workspace), config=cfg)
    monitor = Monitor(state, notifier, workspace, cfg.get("watch"))
    dash = tunnel = None
    if args.dashboard:
        dash = Dashboard(state, monitor, host=args.host, port=args.port, token=args.token)
        url = dash.start()
        print(f"dashboard (this machine): {url}")
        public = None
        if args.tunnel != "none":
            try:
                tunnel = Tunnel(args.tunnel, args.port, args.tunnel_exe)
                public = tunnel.start() + f"/?t={dash.token}"
                print(f"dashboard (public via {args.tunnel}): {public}")
            except Exception as e:
                print(f"[tunnel] {e}")
                state.alert(f"tunnel failed: {e}")
        state.update(dashboard_url=public or url)
        for ch in notifier.channels.values():
            if public and "topic" in ch:
                ch.setdefault("click", public)

    def on_signal(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    keep_awake(True)
    monitor.start()
    state.update(status="running")
    notifier.send("Run started", f"{len(jobs)} steps on {state.data['host']} ({target}): "
                  + ", ".join(j["name"] for j in jobs)
                  + (f"\ndashboard: {state.data['dashboard_url']}" if state.data.get("dashboard_url") else ""), "default",
                  ("rocket",))
    ok = False
    try:
        runner = run_local if target == "local" else run_kaggle
        ok = runner(cfg, jobs, state, notifier, workspace, args)
        state.update(status="done" if ok else "failed", ended=now(), current=None)
    except KeyboardInterrupt:
        state.update(status="cancelled", ended=now())
        notifier.send("Run cancelled", "stopped by user / system signal", "high")
    except Exception as e:
        state.update(status="crashed", ended=now())
        notifier.send("Runner CRASHED", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}", "urgent")
        raise
    finally:
        monitor.stop()
        keep_awake(False)
        try:
            zpath = export_results(workspace)
            state.update(results_zip=str(zpath))
        except Exception as e:
            print(f"[results] export failed: {e}")
        notifier.send("Run finished" if ok else "Run ended", monitor.summary(), "high" if not ok else "default",
                      ("checkered_flag",))
        if args.dashboard and args.linger_minutes and dash:
            print(f"keeping the dashboard up for {args.linger_minutes} min (Ctrl+C to quit)")
            try:
                time.sleep(args.linger_minutes * 60)
            except KeyboardInterrupt:
                pass
        if tunnel:
            tunnel.stop()
        if dash:
            dash.stop()
    print(f"\nrun folder: {state.dir}")
    return 0 if ok else 1


def cmd_export(args):
    export_results(Path(args.workspace or "workspace").resolve(), args.out)
    return 0


def cmd_status(args):
    ws = Path(args.workspace or "workspace").resolve()
    states = sorted((ws / "runner").glob("*/state.json"), key=lambda p: p.stat().st_mtime) if (ws / "runner").exists() else []
    if not states:
        print(f"no runs in {ws / 'runner'}")
        return 1
    s = json.loads(states[-1].read_text())
    print(f"{states[-1].parent.name}: {s['status']} (started {s['started']}, ended {s.get('ended')})")
    for st in s["steps"]:
        print(f"  {st['status']:<9} {st['name']}" + (f"  - {st['error'][:200]}" if st.get("error") else ""))
    for run, r in (s.get("epochs") or {}).items():
        if r:
            print(f"  {run}: epoch {r.get('epoch')} val mIoU {r.get('val_miou')}")
    return 0


def cmd_notify_init(args):
    path = Path(args.out)
    if path.exists():
        print(f"{path} already exists - edit it directly")
        return 1
    example = json.loads((REPO / "configs" / "notify.example.json").read_text())
    example["channels"]["ntfy"]["topic"] = "agrivision-" + secrets.token_hex(6)
    path.write_text(json.dumps(example, indent=2))
    t = example["channels"]["ntfy"]["topic"]
    print(f"wrote {path}\n\n1. Install the ntfy app (Android / iOS) or open https://ntfy.sh/{t}\n"
          f"2. Subscribe to topic: {t}\n3. Test: python run.py notify-test --notify {path}\n"
          "Keep the topic name private: anyone who knows it can read the messages.")
    return 0


def cmd_notify_test(args):
    n = Notifier.from_file(args.notify, "agrivision")
    n.send("Test notification", f"Hello from {platform.node()} at {now()}. If you see this, notifications work.")
    if n.errors:
        print("some channels failed:\n" + "\n".join(n.errors))
        return 1
    print("sent to: " + ", ".join(n.channels) if n.channels else "no channels enabled")
    return 0


def cmd_check(args):
    info, issues = preflight(Path(args.workspace or "workspace").resolve())
    print("\n".join(info))
    print("\nOK - ready to run" if not issues else "\nProblems:\n- " + "\n- ".join(issues))
    return 0 if not issues else 2


def cmd_build(args):
    for t in ("kaggle", "local"):
        for p in nbb.build(t):
            print(p)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--config", required=True, help="pipeline config JSON (see configs/)")
    r.add_argument("--target", choices=["local", "kaggle"], help="overrides the config's target")
    r.add_argument("--steps", help="comma list from: data,explore,train,evaluate (default: config)")
    r.add_argument("--workspace", help="folder for data, outputs and run logs (default: config / ./workspace)")
    r.add_argument("--notify", help="notification config JSON (see configs/notify.example.json)")
    r.add_argument("--dashboard", action="store_true", help="serve a read-only status page")
    r.add_argument("--host", default="127.0.0.1", help="dashboard bind address (0.0.0.0 = whole LAN)")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--token", help="dashboard access token (default: random)")
    r.add_argument("--tunnel", choices=["none", "cloudflared", "ngrok"], default="none",
                   help="publish the dashboard through a tunnel (only where you are allowed to)")
    r.add_argument("--tunnel-exe", help="path to cloudflared.exe / ngrok.exe if it is not on PATH")
    r.add_argument("--linger-minutes", type=int, default=0, help="keep the dashboard up after the run ends")
    r.add_argument("--force", action="store_true", help="redo steps that are already finished")
    r.add_argument("--keep-going", action="store_true", help="continue with later steps after a failure")
    r.add_argument("--cell-timeout", type=int, default=0, help="per-cell timeout in seconds (0 = none)")
    r.add_argument("--ignore-checks", action="store_true", help="start even if pre-flight checks fail")
    r.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="show the latest run")
    s.add_argument("--workspace")
    s.set_defaults(func=cmd_status)

    ni = sub.add_parser("notify-init", help="create a notify config with a private ntfy topic")
    ni.add_argument("--out", default=str(REPO / "configs" / "notify.json"))
    ni.set_defaults(func=cmd_notify_init)

    nt = sub.add_parser("notify-test", help="send a test notification")
    nt.add_argument("--notify", required=True)
    nt.set_defaults(func=cmd_notify_test)

    c = sub.add_parser("check", help="pre-flight check of this machine")
    c.add_argument("--workspace")
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("export-results", help="zip results (no checkpoints/data) to copy to another computer")
    e.add_argument("--workspace")
    e.add_argument("--out", help="zip path (default: workspace/results_<time>.zip)")
    e.set_defaults(func=cmd_export)

    b = sub.add_parser("build", help="rebuild notebooks into build/")
    b.set_defaults(func=cmd_build)

    args = ap.parse_args(argv)
    return args.func(args)
