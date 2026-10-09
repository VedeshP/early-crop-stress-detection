"""Overlap-aware mIoU, as used by the Agriculture-Vision challenge.

Where a pixel carries several true labels, a prediction matching any of them
counts as correct. IoU_c = TP / (TP + FP + FN); mIoU = mean over classes.
"""

import numpy as np
import torch


class SegMetric:
    def __init__(self, class_names):
        self.names = list(class_names)
        self.n = len(self.names)
        self.cm = torch.zeros(self.n, self.n, dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred, target, mbits=None):
        """pred, target: (B,H,W) int; mbits: (B,H,W) multi-hot over model classes (optional)."""
        valid = target != 255
        gt = target
        if mbits is not None:
            matched = ((mbits.to(torch.int32) >> pred.to(torch.int32)) & 1).bool()
            gt = torch.where(matched, pred, target)
        idx = gt[valid].long() * self.n + pred[valid].long()
        self.cm += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n).cpu()

    def compute(self):
        cm = self.cm.double()
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = (tp / (tp + fp + fn)).numpy()
            precision = (tp / (tp + fp)).numpy()
            recall = (tp / (tp + fn)).numpy()
        support = cm.sum(1).numpy()
        present = support > 0
        return {
            "miou": float(np.nanmean(np.where(present, iou, np.nan))),
            "pixel_acc": float(tp.sum() / max(cm.sum(), 1)),
            "iou": dict(zip(self.names, map(float, iou))),
            "precision": dict(zip(self.names, map(float, precision))),
            "recall": dict(zip(self.names, map(float, recall))),
            "support": dict(zip(self.names, map(int, support))),
        }

    def table(self):
        import pandas as pd
        r = self.compute()
        return pd.DataFrame({
            "IoU": r["iou"], "precision": r["precision"], "recall": r["recall"], "pixels": r["support"],
        }).round(4)
