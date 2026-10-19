"""Models.

SpectralSpatialFCN
  spectral stem   two 1x1 convolutions (a per-pixel MLP over the bands)
  encoder         residual 3x3 stages at 1, 1/2, 1/4 and 1/8 resolution (widths w, 1.5w, 2w, 2w), strided downsampling
  global context  image-level average of the deepest feature, projected and added back
  decoder         top-down pathway: 1x1 lateral projections, upsample-and-add, 3x3 smoothing
  classifier      dropout and a 1x1 convolution at full resolution
Batch normalisation statistics are pooled over all pixels of a batch of crops, so inference on a whole scene or on
tiles of any size uses the same (running) statistics.

SpectralMLP: the same stem followed by 1x1 layers only, i.e. no spatial context (ablation / spectral baseline).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv(ci, co, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, stride, k // 2, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                                  nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co))
        self.short = nn.Sequential(nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co)) if ci != co else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x) + self.short(x))


def stem(bands, width):
    return nn.Sequential(conv(bands, width, 1), conv(width, width, 1))


class SpectralSpatialFCN(nn.Module):
    def __init__(self, bands, num_classes, width=64, dropout=0.1):
        super().__init__()
        c = [width, width * 3 // 2, 2 * width, 2 * width]
        self.stem = stem(bands, c[0])
        self.enc = nn.ModuleList([Residual(c[0], c[0])] +
                                 [nn.Sequential(conv(c[i - 1], c[i], 3, 2), Residual(c[i], c[i])) for i in range(1, 4)])
        self.glob = nn.Sequential(nn.Conv2d(c[3], c[3], 1), nn.ReLU(inplace=True))
        self.lat = nn.ModuleList([nn.Conv2d(ci, width, 1) for ci in c])
        self.smooth = nn.ModuleList([conv(width, width) for _ in range(3)])
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(width, num_classes, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        x = F.pad(x, (0, (-W) % 8, 0, (-H) % 8), mode="replicate")
        feats = []; h = self.stem(x)
        for e in self.enc:
            h = e(h); feats.append(h)
        feats[3] = feats[3] + self.glob(feats[3].mean((2, 3), keepdim=True))
        p = self.lat[3](feats[3])
        for i in (2, 1, 0):
            p = self.smooth[i](F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](feats[i]))
        return self.head(p)[..., :H, :W]


class SpectralMLP(nn.Module):
    def __init__(self, bands, num_classes, width=64, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(stem(bands, width), conv(width, 2 * width, 1), conv(2 * width, width, 1))
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(width, num_classes, 1))

    def forward(self, x):
        return self.head(self.body(x))


MODELS = {"fcn": SpectralSpatialFCN, "mlp": SpectralMLP}


def build_model(kind, bands, num_classes, width=64):
    return MODELS[kind](bands, num_classes, width)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_checkpoint(path, device="cpu"):
    """Returns (model in eval mode, checkpoint dict with classes, bands and preprocessing)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = build_model(cfg["model"], len(ck["bands"]), len(ck["classes"]), cfg["width"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck
