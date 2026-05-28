"""Heatmap colorization. Mirrors the original ConceptAttention behavior:
normalize globally across concepts, then apply matplotlib colormap.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def colorize_heatmaps(
    heatmaps: np.ndarray,       # [num_concepts, H, W], float
    cmap: str = "plasma",
    upscale: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """Normalize across all concepts to the same min/max, color via matplotlib,
    return PIL images. If `upscale` is given, resize each to that (W, H)."""
    import matplotlib.pyplot as plt

    lo, hi = float(heatmaps.min()), float(heatmaps.max())
    rng = hi - lo if hi > lo else 1.0
    out: list[Image.Image] = []
    for hm in heatmaps:
        normed = (hm - lo) / rng
        colored = plt.get_cmap(cmap)(normed)              # [H, W, 4]
        rgb = (colored[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        if upscale is not None:
            img = img.resize(upscale, Image.BILINEAR)
        out.append(img)
    return out
