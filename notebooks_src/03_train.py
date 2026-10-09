# %% [markdown]
# # 03 · Train a segmentation model
#
# One notebook for every experiment: change `CONFIG` and `run_name`, run again.
#
# | Experiment | `arch` | `channels` |
# |---|---|---|
# | Baseline | `unet` | `("r","g","b","nir","ndvi","gndvi")` |
# | Main model | `segformer` (MiT-B2) | `("r","g","b","nir","ndvi","gndvi")` |
# | Ablation: RGB only | `segformer` | `("r","g","b")` |
# | Ablation: + NIR | `segformer` | `("r","g","b","nir")` |
# | CNN comparison | `deeplabv3plus` | 6 channels |
#
# Checkpoints: `runs/<run_name>/last.pt` (every epoch, for resuming) and `best.pt` (best validation mIoU).

# %% [markdown kaggle]
# ### Kaggle settings
# - **Accelerator: GPU** (P100 or T4 x2). **Internet: On** (pretrained weights + pip).
# - *Add Input* → `agrivision-2021-packed`.
# - Use **Save Version → Save & Run All** for long runs: it keeps going after you close the browser.
# - Sessions stop at 12 h, so `time_limit_hours=11` saves and stops cleanly. To **continue**: create a new
#   version, *Add Input* → the previous version's output, and run again (resume is automatic).

# %% [markdown local]
# ### Local settings
# Install a CUDA build of PyTorch first (https://pytorch.org/get-started/locally/), then the packages in
# `requirements.txt`. Set `DATA_ROOT` (folder with `meta.json`) or `$AGRIVISION_DATA`.
# Raise `batch_size` and `num_workers` to match the GPU memory and CPU cores; on Windows, if the data
# loader hangs, set `num_workers=0`.

# %%
from agrivision import env, engine

env.system_report()
DATA_ROOT = None

# %% [markdown]
# ## Configuration
# Key options (all fields: `engine.TrainConfig`):
# - `classes`: `"all9"` (benchmark-comparable) or `"targets"` (background, drydown, nutrient_deficiency, weed_cluster)
# - `batch_size`: 8 fits SegFormer-B2 at 512 px on a 16 GB GPU with mixed precision
# - `steps_per_epoch`: fixes epoch length (with rare-class sampling the epoch is a number of draws)
# - `val_max_tiles`: quick validation on a subset each epoch; full validation with TTA at the end

# %%
CONFIG = dict(
    run_name="segformer_b2_rgbn_idx",
    arch="segformer",
    encoder="mit_b2",
    channels=("r", "g", "b", "nir", "ndvi", "gndvi"),
    classes="all9",
    epochs=30,
    steps_per_epoch=2000,
    batch_size=8,
    lr=6e-5,
    num_workers=4,
    mosaic_p=0.25,
    rare_sampling=True,
    val_max_tiles=3000,
    final_tta="flip",
    time_limit_hours=11.0,
    resume="auto",
)
env.apply_overrides(CONFIG, "train")
CONFIG

# %% [markdown]
# ## Train

# %%
result = engine.train(CONFIG, data_root=DATA_ROOT)
result

# %% [markdown]
# ## Training curves

# %%
from pathlib import Path
from agrivision import viz

run_dir = Path(result["run_dir"])
viz.plot_history(run_dir / "history.csv");

# %%
import pandas as pd
pd.read_csv(run_dir / "history.csv").tail(10)
