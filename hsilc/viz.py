"""Colours and figures for class maps.

Classes 1-8 take eight well-separated hues; classes 9-16 take the same hues at a darker step (hue + lightness
encoding), so maps with up to 16 classes never use generated hues. Error maps use two colours only.
"""
from __future__ import annotations

import numpy as np

HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BACKGROUND = "#f4f3f0"
CORRECT, WRONG = "#bdbcb6", "#d03b3b"


def _rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


def class_colors(n):
    """(n, 3) uint8 colours for classes 1..n (n <= 16)."""
    if n > 16:
        raise ValueError("class maps support at most 16 classes")
    cols = [_rgb(h) for h in HUES] + [_rgb(h) * 0.55 for h in HUES]
    return np.clip(np.round(cols[:n]), 0, 255).astype(np.uint8)


def colorize(labels, n):
    """RGB image of a 1-based label map; 0 (unlabelled) is drawn in the background colour."""
    lut = np.vstack([_rgb(BACKGROUND)[None].astype(np.uint8), class_colors(n)])
    return lut[labels]


def error_image(pred, reference):
    """Correct and misclassified reference pixels; pixels without a reference label are background."""
    img = np.empty(reference.shape + (3,), np.uint8); img[:] = _rgb(BACKGROUND).astype(np.uint8)
    m = reference > 0
    img[m & (pred == reference)] = _rgb(CORRECT).astype(np.uint8)
    img[m & (pred != reference)] = _rgb(WRONG).astype(np.uint8)
    return img


def legend_handles(classes):
    from matplotlib.patches import Patch
    cols = class_colors(len(classes)) / 255.0
    return [Patch(facecolor=c, edgecolor="none", label=f"{i + 1} {name}") for i, (c, name) in enumerate(zip(cols, classes))]


def save_class_map(path, labels, classes, title=None):
    """PNG of a class map with a legend of the class names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = labels.shape; h = 6.0; w = h * W / H
    fig, ax = plt.subplots(figsize=(w + 3.2, h), dpi=150)
    ax.imshow(colorize(labels, len(classes)), interpolation="nearest"); ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, color="#0b0b0b")
    ax.legend(handles=legend_handles(classes), loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7,
              labelcolor="#52514e")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)
