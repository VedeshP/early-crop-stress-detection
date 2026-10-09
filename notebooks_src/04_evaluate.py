# %% [markdown]
# # 04 · Evaluate a trained model
#
# - Overlap-aware **mIoU** (the official Agriculture-Vision metric) and per-class IoU / precision / recall
# - Effect of **test-time augmentation** (TTA): none vs flips
# - Confusion matrix and side-by-side predictions
# - Comparison table across all runs found

# %% [markdown kaggle]
# **Kaggle:** GPU on. *Add Input* → `agrivision-2021-packed` **and** the output of the training notebook
# (it contains `runs/<run_name>/best.pt`).

# %% [markdown local]
# **Local:** set `checkpoint` in `EVAL` to a `best.pt` path, or leave `None` to use the newest one under `./outputs`.

# %%
from agrivision import env, engine, viz
from agrivision.dataset import PackedTiles

EVAL = dict(
    data_root=None,
    checkpoint=None,         # e.g. "outputs/runs/segformer_b2_rgbn_idx/best.pt"; None = newest best.pt
    split="val",
    max_tiles=None,          # e.g. 2000 for a quick check; None = full split
    batch_size=8,
    num_workers=4,
)
env.apply_overrides(EVAL, "evaluate")
DATA_ROOT, SPLIT, MAX_TILES = EVAL["data_root"], EVAL["split"], EVAL["max_tiles"]
kw = dict(max_tiles=MAX_TILES, batch_size=EVAL["batch_size"], num_workers=EVAL["num_workers"])

found = env.find_files("**/best.pt")
print("checkpoints found:", *found, sep="\n  ")
ckpt_path = EVAL["checkpoint"] or found[0]
print("using:", ckpt_path)

# %% [markdown]
# ## Scores without and with TTA

# %%
plain = engine.evaluate_checkpoint(ckpt_path, DATA_ROOT, SPLIT, tta="none", **kw)
flip = engine.evaluate_checkpoint(ckpt_path, DATA_ROOT, SPLIT, tta="flip", **kw)
import pandas as pd
pd.DataFrame([plain["summary"], flip["summary"]]).set_index("tta").drop(columns=["checkpoint"]).T.round(4)

# %% [markdown]
# ## Per-class results (with TTA)

# %%
flip["table"]

# %%
viz.plot_confusion(flip["metric"]);

# %% [markdown]
# ## Predictions

# %%
from pathlib import Path
model, ck = engine.load_checkpoint(ckpt_path)
tiles = PackedTiles(env.find_packed_root(DATA_ROOT))
frame = tiles.rows(SPLIT)
stressed = frame[frame["px_drydown"] + frame["px_nutrient_deficiency"] > 5000]
run_dir = Path(ckpt_path).parent
viz.show_predictions(model, ck, tiles, stressed if len(stressed) else frame, n=6,
                     save_to=run_dir / f"predictions_{SPLIT}.png");

# %% [markdown]
# ## All runs side by side
# Every `eval_*.json` written next to a checkpoint, so ablations line up in one table.

# %%
import json
rows = [json.loads(p.read_text()) | {"run": p.parent.name, "file": p.name} for p in env.find_files("**/eval_*.json")]
cols = ["run", "split", "tta", "tiles", "miou", "iou_drydown", "iou_nutrient_deficiency", "iou_weed_cluster"]
comparison = pd.DataFrame(rows)[[c for c in cols if rows and c in rows[0]]].sort_values("miou", ascending=False).round(4) if rows else pd.DataFrame()
comparison.to_csv(env.default_dirs()["output"] / "comparison.csv", index=False)
comparison
