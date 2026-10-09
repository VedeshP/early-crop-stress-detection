"""Cross-entropy (optionally class-weighted) + soft Dice, both ignoring label 255."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, dice_weight=0.5, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", w, persistent=False)

    def forward(self, logits, target):
        logits = logits.float()
        loss = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss

    def dice(self, logits, target):
        valid = (target != self.ignore_index).unsqueeze(1)
        t = target.clone()
        t[t == self.ignore_index] = 0
        onehot = F.one_hot(t, self.num_classes).permute(0, 3, 1, 2).float() * valid
        probs = torch.softmax(logits, 1) * valid
        inter = (probs * onehot).sum((0, 2, 3))
        card = (probs + onehot).sum((0, 2, 3))
        dice = 1 - (2 * inter + 1) / (card + 1)
        present = onehot.sum((0, 2, 3)) > 0
        return dice[present].mean() if present.any() else dice.mean() * 0
