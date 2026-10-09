"""End-to-end smoke test on a tiny synthetic archive (CPU, ~1-3 minutes).

Builds a fake Agriculture-Vision-2021.tar.gz with the real folder layout, then
runs: prepare (2-pass streaming + selection) -> dataset -> train 1 short epoch
(U-Net and SegFormer) -> resume -> evaluate with TTA -> build notebooks.

    python tools/smoke_test.py            # keeps outputs in ./_smoke for inspection
"""

import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from agrivision import engine, prepare  # noqa: E402
from agrivision.classes import INVALID, LABEL_CLASSES  # noqa: E402
from agrivision.dataset import PackedTiles, SegDataset  # noqa: E402

SIZE = 64


def _png(arr):
    return cv2.imencode(".png", arr)[1].tobytes()


def _jpg(arr):
    return cv2.imencode(".jpg", arr)[1].tobytes()


def make_archive(path, n_train=24, n_val=8, seed=0):
    rng = np.random.default_rng(seed)
    files = {}  # folder -> list of (name, bytes); written folder by folder like the real archive
    truth = {}
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            tid = f"F{split[0].upper()}{i:03d}_0-0-{SIZE}-{SIZE}"
            base = rng.integers(60, 120, (SIZE, SIZE, 3), dtype=np.uint8)
            nir = np.full((SIZE, SIZE), 170, np.uint8)
            labels = {}
            if i % 3 == 0:  # drydown blob: lower NIR
                m = np.zeros((SIZE, SIZE), np.uint8)
                cv2.circle(m, (20, 20), 12, 255, -1)
                labels["drydown"] = m
                nir[m > 0] = 90
            if i % 4 == 1:  # nutrient deficiency with overlap on drydown
                m = np.zeros((SIZE, SIZE), np.uint8)
                cv2.rectangle(m, (30, 30), (55, 55), 255, -1)
                labels["nutrient_deficiency"] = m
                nir[m > 0] = 120
            if i % 5 == 2:
                m = np.zeros((SIZE, SIZE), np.uint8)
                m[5:12, :] = 255
                labels["planter_skip"] = m
            boundary = np.full((SIZE, SIZE), 255, np.uint8)
            boundary[:, -4:] = 0
            mask = np.full((SIZE, SIZE), 255, np.uint8)
            root = f"Agriculture-Vision-2021/{split}"
            files.setdefault(f"{root}/masks", []).append((f"{tid}.png", _png(mask)))
            files.setdefault(f"{root}/boundaries", []).append((f"{tid}.png", _png(boundary)))
            files.setdefault(f"{root}/images/rgb", []).append((f"{tid}.jpg", _jpg(base)))
            files.setdefault(f"{root}/images/nir", []).append((f"{tid}.jpg", _jpg(nir)))
            for cls in LABEL_CLASSES + ["storm_damage"]:
                files.setdefault(f"{root}/labels/{cls}", []).append(
                    (f"{tid}.png", _png(labels.get(cls, np.zeros((SIZE, SIZE), np.uint8)))))
            truth[tid] = labels
    with tarfile.open(path, "w:gz") as tf:
        for folder, items in files.items():
            for name, data in items:
                ti = tarfile.TarInfo(f"{folder}/{name}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return truth


def main():
    work = REPO / "_smoke"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    archive = work / "Agriculture-Vision-2021.tar.gz"
    truth = make_archive(archive)
    print("archive:", archive.stat().st_size // 1024, "KB")

    # ---- prepare (2-pass, relevant mode)
    summary = prepare.run(prepare.PrepareConfig(
        source=str(archive), out_dir=str(work / "packed"), work_dir=str(work / "scratch"),
        mode="relevant", negative_fraction=0.5, min_target_pixels=10, workers=2, stats_tiles=50))
    tiles = PackedTiles(work / "packed")
    train, val = tiles.rows("train"), tiles.rows("val")
    n_target = sum(1 for t, l in truth.items() if t.startswith("FT") and ({"drydown", "nutrient_deficiency"} & set(l)))
    n_other = sum(1 for t in truth if t.startswith("FT")) - n_target
    assert len(val) == 8, "val must be kept in full"
    assert len(train) == n_target + round(n_other * 0.5), (len(train), n_target, n_other)
    assert not (work / "scratch").exists(), "scratch should be cleaned"

    # ---- label bits round-trip
    rec = train[train["px_drydown"] > 0].iloc[0].to_dict()
    rgb, nir, bits = tiles.read(rec)
    assert rgb.shape == (SIZE, SIZE, 3) and nir.shape == (SIZE, SIZE) and bits.dtype == np.uint16
    dry = truth[rec["id"]]["drydown"] > 0
    assert ((bits & (1 << 2)) > 0)[dry][:, ].mean() > 0.95, "drydown bit mismatch"
    assert ((bits & INVALID) > 0)[:, -4:].all(), "outside-boundary pixels must be invalid"
    assert rec["px_drydown"] == int((dry[:, :-4]).sum()), "pixel counts must exclude invalid area"
    ds = SegDataset(tiles, train, classes="all9")
    x, target, mbits = ds[0]
    assert x.shape == (6, SIZE, SIZE) and target.max() <= 255 and mbits.dtype.is_floating_point is False
    print("prepare + dataset OK:", summary)

    # ---- train (unet), then resume one more epoch, then segformer
    base = dict(epochs=1, steps_per_epoch=3, batch_size=4, num_workers=0, pretrained=False, val_max_tiles=None,
                warmup_steps=1, output_dir=str(work / "runs"), log_every=1, final_tta="flip", amp=False)
    r1 = engine.train(dict(base, run_name="unet_test", arch="unet", encoder="resnet18"), data_root=work / "packed")
    assert Path(r1["run_dir"], "best.pt").exists() and "final" in r1
    r2 = engine.train(dict(base, run_name="unet_test", arch="unet", encoder="resnet18", epochs=2),
                      data_root=work / "packed")
    assert r2["epochs_done"] == 2, r2
    r3 = engine.train(dict(base, run_name="segformer_test", arch="segformer", encoder="mit_b0",
                           classes="targets", channels=("r", "g", "b", "nir")), data_root=work / "packed")
    print("training OK:", json.dumps(r3["final"], indent=1)[:300])

    # ---- time-limit stop
    r4 = engine.train(dict(base, run_name="timelimit", arch="unet", encoder="resnet18", epochs=3,
                           time_limit_hours=1e-9), data_root=work / "packed")
    assert r4["stopped_early"] and Path(r4["run_dir"], "last.pt").exists()

    # ---- evaluate with d4 TTA
    ev = engine.evaluate_checkpoint(Path(r1["run_dir"]) / "best.pt", work / "packed", "val", tta="d4", num_workers=0)
    assert 0.0 <= ev["summary"]["miou"] <= 1.0

    # ---- notebooks
    import build_notebooks
    for t in ("kaggle", "local"):
        paths = build_notebooks.build(t, "tester", "tester/agrivision-2021-packed", out_root=work / "build")
        for p in paths:
            nb = json.loads(p.read_text(encoding="utf-8"))
            try:
                import nbformat
                nbformat.validate(nbformat.reads(json.dumps(nb), as_version=4))
            except ImportError:
                pass
            for c in nb["cells"]:
                if c["cell_type"] != "code":
                    continue
                src = "".join(c["source"])
                if src.startswith("%%writefile"):
                    src = src.split("\n", 1)[1]
                compile(src, f"{p.name}", "exec")
    print("notebooks OK")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
