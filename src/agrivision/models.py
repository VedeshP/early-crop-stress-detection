"""Model factory: U-Net, DeepLabV3+, SegFormer, UPerNet, FPN (segmentation_models_pytorch).

Pretrained ImageNet encoders expect 3 channels; smp adapts the first layer to
our 6 channels by reusing the RGB filters for the extra channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = {
    "unet": ("Unet", "resnet34"),
    "deeplabv3plus": ("DeepLabV3Plus", "resnet50"),
    "segformer": ("Segformer", "mit_b2"),
    "upernet": ("UPerNet", "tu-convnext_tiny"),
    "fpn": ("FPN", "resnet34"),
}

# sensible starting learning rates (AdamW)
DEFAULT_LR = {"unet": 3e-4, "deeplabv3plus": 1e-4, "segformer": 6e-5, "upernet": 1e-4, "fpn": 3e-4}


class SegModel(nn.Module):
    """Thin wrapper that always returns logits at the input resolution."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y


def build_model(arch="segformer", encoder=None, in_channels=6, num_classes=9, pretrained=True):
    import segmentation_models_pytorch as smp
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}', choose from {list(ARCHS)}")
    cls_name, default_encoder = ARCHS[arch]
    Model = getattr(smp, cls_name, None)
    if Model is None:
        raise ImportError(f"segmentation_models_pytorch {smp.__version__} has no {cls_name}; upgrade to >=0.5")
    net = Model(
        encoder_name=encoder or default_encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
    )
    return SegModel(net)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def predict_proba(model, x, tta="none"):
    """Softmax probabilities, optionally averaged over test-time augmentations.

    tta: "none" | "flip" (4 flips) | "d4" (4 flips x 2 transposes = 8 views)
    """
    views = [(False, False, False)]
    if tta in ("flip", "d4"):
        views = [(h, v, False) for h in (False, True) for v in (False, True)]
    if tta == "d4":
        views += [(h, v, True) for h in (False, True) for v in (False, True)]
    total = None
    for h, v, t in views:
        xi = x
        if t:
            xi = xi.transpose(-1, -2)
        if h:
            xi = xi.flip(-1)
        if v:
            xi = xi.flip(-2)
        p = torch.softmax(model(xi).float(), 1)
        if v:
            p = p.flip(-2)
        if h:
            p = p.flip(-1)
        if t:
            p = p.transpose(-1, -2)
        total = p if total is None else total + p
    return total / len(views)
