# early-crop-stress-detection

Minor project: early crop stress detection (drought, nutrient deficiency, weed pressure) from RGB +
near-infrared drone imagery. The **AgriVision toolkit** in this repo prepares the Agriculture-Vision 2021
data, trains segmentation models (U-Net, DeepLabV3+, SegFormer, …) and evaluates them with the official
overlap-aware mIoU. Everything runs as **notebooks**, on **Kaggle** or a **local GPU machine**, from the
same source, and `run.py` runs the whole pipeline unattended with notifications.

## Quick start on a GPU machine (Windows or Linux)

```bash
git clone https://github.com/VedeshP/early-crop-stress-detection.git
cd early-crop-stress-detection
python -m venv .venv
.venv\Scripts\activate                 # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m ipykernel install --user --name python3
python run.py check                    # must end with "OK - ready to run"
python run.py notify-init              # then subscribe to the printed topic in the ntfy app
python run.py notify-test --notify configs/notify.json
python run.py run --config configs/rtx4000ada.json --notify configs/notify.json --dashboard --tunnel cloudflared
```

For `--tunnel cloudflared`, put `cloudflared.exe` (from https://github.com/cloudflare/cloudflared/releases,
renamed) next to `run.py`, or leave the option out. The `data` step streams ~21 GB from AWS twice
and needs ~25 GB free disk. Re-run the same command after any interruption: finished steps are skipped and
training resumes.

```
src/agrivision/        the toolkit (Python package) — edit code here
notebooks_src/         notebook recipes (plain .py, "# %%" cells)
tools/build_notebooks.py   recipes + toolkit -> self-contained .ipynb for kaggle / local
tools/run_notebooks.py     run notebooks headlessly (e.g. on a remote GPU box)
tools/smoke_test.py        end-to-end test on tiny synthetic data (CPU, ~1 min)
build/kaggle/<nb>/     generated Kaggle notebooks (+ kernel-metadata.json for the Kaggle CLI)
build/local/           generated local notebooks
```

The notebooks embed the toolkit source in collapsed setup cells, so a single `.ipynb` file is all you
need to copy to Kaggle or another computer.

## The four notebooks

| # | Notebook | Needs GPU | What it does |
|---|---|---|---|
| 01 | `01_prepare_data` | no | Streams the 21 GB archive from S3, keeps relevant tiles, writes a packed dataset |
| 02 | `02_explore_data` | no | Class balance, sample tiles (RGB / NIR / NDVI / labels), NDVI per class |
| 03 | `03_train` | yes | Trains one model per run (change `CONFIG`), checkpoints every epoch, resumes |
| 04 | `04_evaluate` | yes | mIoU with/without TTA, per-class table, confusion matrix, predictions, run comparison |

## Data: what gets downloaded

The 2021 supervised set is a single `Agriculture-Vision-2021.tar.gz` (21 GB) on AWS Open Data
(`s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/`). No AWS account or
CLI is needed: notebook 01 reads it over HTTPS and never stores the full archive.

- Pass 1 reads only labels → picks tiles. Pass 2 keeps only the images of picked tiles.
- Default selection (`mode="relevant"`): every **train** tile with ≥500 px of drydown, nutrient deficiency
  or weed cluster, plus 20% of the other train tiles; **val** kept in full (comparable with published
  results); **test** skipped (no labels).
- Output (`agrivision2021_packed/`): `index.csv`, `meta.json`, `shards/*.bin`. Budgeted below Kaggle's
  20 GB output limit (`max_output_gb`). All 8 classes are stored, so you can still train on all 9 classes
  (`classes="all9"`) or only the targets (`classes="targets"`).

## Workflow A — Kaggle

Kaggle needs a **phone-verified account** for Internet and GPU. GPU quota is about 30 h/week.

1. Build notebooks (on this machine):
   ```bash
   python tools/build_notebooks.py --target kaggle
   ```
2. Kaggle → **Create → New Notebook → File → Import Notebook** → `build/kaggle/01_prepare_data/01_prepare_data.ipynb`.
   Settings: Internet **On**, Accelerator **None** → **Save Version → Save & Run All**.
3. When the version finishes: **Output → New Dataset** → name it `agrivision-2021-packed`.
4. Import `02`, `03`, `04` the same way. In each: **Add Input → Datasets → agrivision-2021-packed**.
   For `03`/`04`: Accelerator **GPU**, Internet **On**. Use *Save & Run All* for long training.
5. Long training: `time_limit_hours=11` saves `last.pt` and stops before the 12 h limit. To continue, make a
   new version, **Add Input** → the previous version's output, run again (`resume="auto"` finds it).
6. For `04`, add the training notebook's output as input too (it contains `runs/<run_name>/best.pt`).

Optional, with the Kaggle CLI (`pip install kaggle`, API token in `~/.kaggle/kaggle.json`):
```bash
python tools/build_notebooks.py --target kaggle --kaggle-user YOUR_NAME --kaggle-dataset YOUR_NAME/agrivision-2021-packed
kaggle kernels push -p build/kaggle/03_train
```

## Workflow B — local GPU machine

1. Copy this folder (or just the `.ipynb` files from `build/local/`) to the machine.
2. Install PyTorch with CUDA (see `requirements.txt`), then `pip install -r requirements.txt`.
3. Data — pick one:
   - run `01_prepare_data` locally (streams 21 GB; set `source` to a local `.tar.gz` if you already have it), or
   - download the packed dataset you built on Kaggle: `kaggle datasets download -d YOUR_NAME/agrivision-2021-packed --unzip -p data/agrivision2021_packed`.
4. Open the notebooks in Jupyter / VS Code. Data is found automatically under `./data` or `./outputs`,
   or set `DATA_ROOT` in the notebook or the `AGRIVISION_DATA` environment variable.
5. No remote desktop? Run headlessly and collect executed copies:
   ```bash
   python tools/run_notebooks.py build/local/03_train.ipynb --out executed
   ```
   Settings can be overridden without editing the notebook:
   `AGRIVISION_OVERRIDES='{"train": {"epochs": 50, "batch_size": 16}}'`.

Folders on a local machine default to the current directory (`./data`, `./outputs`, `./_work`);
change them with `AGRIVISION_HOME`, or individually with `AGRIVISION_INPUT` / `AGRIVISION_OUTPUT` / `AGRIVISION_SCRATCH`.

## Unattended runs with one command (`run.py`)

`run.py` runs the whole pipeline (data → explore → every training run → evaluation) without anyone
at the machine, sends notifications, and serves a read-only status page.

```bash
python run.py check                                   # is this machine ready? (GPU, packages, disk)
python run.py notify-init                             # creates configs/notify.json with a private ntfy topic
python run.py notify-test --notify configs/notify.json
python run.py run --config configs/rtx4000ada.json --target local --notify configs/notify.json --dashboard
python run.py status                                  # what the latest run is doing
```

- **Configs** (`configs/*.json`) list the steps, data source, and one entry per training run (the
  RTX 4000 Ada config runs the full baseline + SegFormer + ablation suite). `--steps train,evaluate`
  runs a subset. `--dry-run` prints the plan.
- **Restart-safe:** run the same command again after a crash or reboot. Finished steps are skipped and
  training resumes from `last.pt`. `--force` redoes finished steps.
- **Logs:** `workspace/runner/<run>/logs/*.log` (live), executed notebooks in `.../executed/`.
- **Windows** is kept awake while a run is active.
- `--target kaggle` pushes the same steps to Kaggle with the Kaggle CLI (`pip install kaggle` + API token),
  polls their status, notifies you, and downloads outputs. Set `kaggle.user` in `configs/kaggle.json`.

### Notifications
Configured in `configs/notify.json` (never commit it). Easiest is **ntfy**: install the ntfy app,
subscribe to the topic printed by `notify-init`, done — no account. Telegram, Discord, Slack and email are
also supported. You get: run started/finished, each step started/finished/**failed** (with the error),
each epoch's mIoU and drydown / nutrient IoU, a periodic heartbeat, and alerts for no progress, GPU idle,
GPU too hot, and low disk. If heartbeats stop arriving, the machine or the runner is down.

### Seeing the run from elsewhere
`--dashboard` serves a read-only page (status, steps, GPU, mIoU curves, live log) on
`http://127.0.0.1:8765/?t=<token>`; the link with its token is included in the notifications.

| Option | How | Needs |
|---|---|---|
| Same network | `--host 0.0.0.0`, open `http://<machine-ip>:8765/?t=...` | firewall allows the port |
| **Tailscale** (recommended) | install Tailscale on the GPU machine and your phone, `--host 0.0.0.0`, open `http://<tailscale-ip>:8765/?t=...` | free account; private, nothing public |
| Cloudflare quick tunnel | install `cloudflared`, add `--tunnel cloudflared` | no account; public URL protected only by the token |
| ngrok | install ngrok + `ngrok config add-authtoken ...`, add `--tunnel ngrok` | free account; public URL |

The page cannot run commands. Only expose it (especially through a tunnel) on a machine where the owner / IT
allows it.

## Experiments (notebook 03 `CONFIG`)

| Run | `arch` / `encoder` | `channels` |
|---|---|---|
| Baseline | `unet` / `resnet34` | r g b nir ndvi gndvi |
| Main | `segformer` / `mit_b2` | r g b nir ndvi gndvi |
| Ablation RGB | `segformer` / `mit_b2` | r g b |
| Ablation +NIR | `segformer` / `mit_b2` | r g b nir |
| CNN comparison | `deeplabv3plus` / `resnet50` | r g b nir ndvi gndvi |

Give each run its own `run_name`; notebook 04 lists all evaluated runs side by side.

## Changing the code

Edit `src/agrivision/*.py` or `notebooks_src/*.py`, then rebuild and re-test:
```bash
python tools/smoke_test.py
python tools/build_notebooks.py
```
