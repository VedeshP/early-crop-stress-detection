# %% [markdown]
# # 01 · Prepare the Agriculture-Vision 2021 dataset
#
# Streams the official **21 GB** archive straight from AWS S3 (no AWS account, no AWS CLI needed)
# and writes a compact **packed** copy containing only what we need:
#
# - **train**: every tile with drydown, nutrient deficiency or weed cluster, plus a random share of
#   the other tiles (healthy fields and other classes, so the model also learns "not stressed")
# - **val**: kept in full, so our scores are comparable with published results
# - **test**: skipped (it has no labels)
#
# The archive is a single `.tar.gz`, so it is read twice: pass 1 reads only the labels and picks the tiles,
# pass 2 saves only the images of those tiles. Nothing larger than the final output is ever stored.

# %% [markdown kaggle]
# ### How to run on Kaggle
# 1. **Settings → Internet: On.** Accelerator: **None** (CPU is enough, saves GPU quota).
# 2. Click **Save Version → Save & Run All (Commit)**. The run continues even if you close the browser.
#    Expect roughly 30–60 minutes.
# 3. When it finishes, open the version → **Output** tab → **New Dataset**, e.g. `agrivision-2021-packed`.
#    (Alternatively, later notebooks can attach this notebook's output directly via *Add Input → Notebooks*.)
#
# Kaggle saves at most **20 GB** of output; the budget below keeps us under it.

# %% [markdown local]
# ### How to run on a local machine
# - Needs Python with the packages in `requirements.txt` (no GPU needed for this step) and ~25 GB free disk.
# - With a fast connection leave `source` as the S3 URL. If you already downloaded the archive
#   (`aws s3 cp ... --no-sign-request`), set `source` in `CONFIG` to the local `.tar.gz` path instead.
# - Faster alternative: build the packed dataset once on Kaggle, then download it with
#   `kaggle datasets download -d <user>/agrivision-2021-packed --unzip`.

# %%
from agrivision import env, prepare
from agrivision.stream import DEFAULT_URL

env.system_report()

# %% [markdown]
# ## Settings
# | Setting | Meaning |
# |---|---|
# | `mode` | `"relevant"` (recommended) keeps target tiles + a sample of others; `"all"` keeps every tile |
# | `targets` | classes that make a training tile "relevant" |
# | `min_target_pixels` | minimum target pixels in a tile (500 px ≈ 5 m² at 10 cm/px) |
# | `negative_fraction` | share of non-target training tiles kept |
# | `keep_all_splits` | splits kept in full regardless of content |
# | `max_output_gb` | hard budget for the output folder |

# %%
CONFIG = dict(
    source=DEFAULT_URL,          # or a local path to Agriculture-Vision-2021.tar.gz
    mode="relevant",
    splits=("train", "val"),
    targets=("drydown", "nutrient_deficiency", "weed_cluster"),
    min_target_pixels=500,
    negative_fraction=0.20,
    keep_all_splits=("val",),
    max_output_gb=19.0,
)
env.apply_overrides(CONFIG, "prepare")

dirs = env.default_dirs()
OUT_DIR = dirs["output"] / "agrivision2021_packed"
WORK_DIR = dirs["scratch"] / "prepare"
print("output :", OUT_DIR, "\nscratch:", WORK_DIR)

# %% [markdown]
# ## Run
# Progress is printed every 30 s (download speed and ETA per pass).

# %%
summary = prepare.run(prepare.PrepareConfig(out_dir=str(OUT_DIR), work_dir=str(WORK_DIR), **CONFIG))
summary

# %% [markdown]
# ## Check the result

# %%
import json
from agrivision.dataset import PackedTiles
from agrivision.viz import class_summary

tiles = PackedTiles(OUT_DIR)
print(json.dumps(tiles.meta["tiles"], indent=2))
class_summary(tiles)

# %% [markdown]
# **Next:** turn this output into a dataset (Kaggle: Output → New Dataset) and open `02_explore_data`.
