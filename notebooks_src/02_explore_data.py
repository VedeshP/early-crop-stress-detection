# %% [markdown]
# # 02 · Explore the data (EDA)
#
# - How many tiles and pixels each class has (class imbalance)
# - What tiles look like in RGB, NIR and NDVI next to their labels
# - **NDVI / GNDVI distributions per class** — first evidence that stress is spectrally separable
#
# Everything is also saved to `<output>/eda/` (CSV, NPZ, PNG panels). If matplotlib is unavailable the
# charts are skipped; draw them later with `python tools/make_charts.py <results folder>`.

# %% [markdown kaggle]
# **Kaggle:** *Add Input* → your `agrivision-2021-packed` dataset (or the output of notebook 01).
# Accelerator: None. Internet: On (only needed for the `pip install` in setup).

# %% [markdown local]
# **Local:** set `DATA_ROOT` below to the folder that contains `meta.json`, or set the environment
# variable `AGRIVISION_DATA`. Leave it as `None` to search `./data` and `./outputs` automatically.

# %%
from agrivision import env
from agrivision.dataset import PackedTiles
from agrivision import viz

DATA_ROOT = None
root = env.find_packed_root(DATA_ROOT)
tiles = PackedTiles(root)
EDA_DIR = env.default_dirs()["output"] / "eda"
print("dataset:", root)
print("tiles per split:", tiles.meta["tiles"])
print("saving EDA results to:", EDA_DIR, "| matplotlib available:", viz.available())

# %% [markdown]
# ## Class balance

# %%
summary = viz.class_summary(tiles)
summary.pivot(index="class", columns="split", values="pixel_share_%")

# %%
viz.plot_class_shares(summary, "train" if "train" in tiles.splits else tiles.splits[0]);

# %%
summary.pivot(index="class", columns="split", values="tiles_with_class")

# %% [markdown]
# ## Sample tiles
# Columns: RGB, near-infrared, NDVI (green = healthy, red = low vegetation), and the label map.

# %%
split = "train" if "train" in tiles.splits else tiles.splits[0]
frame = tiles.rows(split)
viz.show_tiles(tiles, frame, n=4, seed=1, save_to=EDA_DIR / "sample_tiles.png");

# %% [markdown]
# Tiles that contain **nutrient deficiency**:

# %%
viz.show_tiles(tiles, frame[frame["px_nutrient_deficiency"] > 2000], n=4, seed=2,
               save_to=EDA_DIR / "sample_tiles_nutrient_deficiency.png");

# %% [markdown]
# Tiles that contain **drydown**:

# %%
viz.show_tiles(tiles, frame[frame["px_drydown"] > 2000], n=4, seed=3, save_to=EDA_DIR / "sample_tiles_drydown.png");

# %% [markdown]
# ## Vegetation-index distributions per class
# If the stressed classes sit at lower NDVI / GNDVI than background (healthy crop), the near-infrared
# band carries a stress signal. This is the first piece of evidence for the early-detection claim.

# %%
values = viz.index_distributions(tiles, frame, "ndvi", n_tiles=300)
viz.plot_distributions(values, "NDVI");
viz.distribution_table(values)

# %%
values_g = viz.index_distributions(tiles, frame, "gndvi", n_tiles=300)
viz.plot_distributions(values_g, "GNDVI");
viz.distribution_table(values_g)

# %% [markdown]
# ## Save EDA results

# %%
viz.save_eda(EDA_DIR, summary, {"ndvi": values, "gndvi": values_g})

# %% [markdown]
# ## Channel statistics used for normalization

# %%
import pandas as pd
pd.DataFrame(tiles.meta.get("channel_stats", {})).T.round(4)
