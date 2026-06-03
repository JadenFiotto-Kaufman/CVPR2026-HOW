"""Visualization helpers for the CVPR 2026 HOW companion notebook.

Pulled out of the notebook itself so the cells stay focused on the
nnsight / NDIF / Workbench parts of each section. Everything here is
plain matplotlib + pandas + numpy + hashlib — no nnsight dependency.
"""

import hashlib

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


# LLaVA-1.5 constants used by the Section 3 helpers.
LLAVA_IMG_TOKEN_ID = 32000
LLAVA_IMAGE_GRID = 24  # 24 × 24 = 576 image patches


# ---------------------------------------------------------------------------
# Section 1 — Attention ablation
# ---------------------------------------------------------------------------


def show_ablation_comparison(baseline_image, ablated_image, layers_to_ablate):
    """Side-by-side baseline vs ablated image."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(baseline_image)
    axes[0].set_title("Baseline")
    axes[0].axis("off")
    axes[1].imshow(ablated_image)
    axes[1].set_title(f"Ablated layers {layers_to_ablate}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def show_per_layer_ablation_grid(baseline_image, per_layer_images, n_cols=4):
    """Grid of ablation sweep: baseline + one image per ablated layer.

    `per_layer_images[i]` is the result of ablating cross-attention
    layer `i` and nothing else. Tiles are laid out row-major; the first
    tile is the unmodified baseline for reference.
    """
    tiles = [("baseline", baseline_image)] + [
        (f"layer {i}", img) for i, img in enumerate(per_layer_images)
    ]
    n_rows = -(-len(tiles) // n_cols)  # ceil-div
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows),
    )
    for ax, (title, img) in zip(axes.flat, tiles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    # Hide any unused tiles in the last row.
    for ax in axes.flat[len(tiles):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 2 — Concept attention
# ---------------------------------------------------------------------------


def show_concept_heatmaps(
    image, score_acc, n_accumulated, concepts,
    image_size=1024, patch_px=16,
):
    """Generated image + per-concept heatmap overlays (one column each)."""
    grid = image_size // patch_px
    heatmaps = (
        (score_acc / n_accumulated)
        .unflatten(-1, (grid, grid))[0]
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        1, len(concepts) + 1, figsize=(4 * (len(concepts) + 1), 4)
    )
    axes[0].imshow(image)
    axes[0].set_title("Generated image")
    axes[0].axis("off")
    for i, (c, hm) in enumerate(zip(concepts, heatmaps)):
        # Resize heatmap to image resolution by nearest-neighbour for crispness.
        hm_resized = np.kron(hm, np.ones((image_size // grid, image_size // grid)))
        axes[i + 1].imshow(image)
        axes[i + 1].imshow(hm_resized, cmap="plasma", alpha=0.55)
        axes[i + 1].set_title(c)
        axes[i + 1].axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 3 — VLM logit lens
# ---------------------------------------------------------------------------


def build_position_labels(
    tokenizer, prompt,
    img_token_id=LLAVA_IMG_TOKEN_ID, image_grid=LLAVA_IMAGE_GRID,
):
    """Expand the single `<image>` placeholder into `<IMG001>..<IMGNNN>`."""
    num_image_tokens = image_grid * image_grid
    input_ids = tokenizer.encode(prompt)
    labels: list[str] = []
    for tok_id in input_ids:
        if tok_id == img_token_id:
            labels.extend(f"<IMG{(i + 1):03d}>" for i in range(num_image_tokens))
        else:
            labels.append(tokenizer.decode([tok_id]))
    return labels


def show_lens_table(
    position_labels, top1_per_layer, tokenizer, sample_layers=None,
):
    """Pandas table of top-1 token per (text position, sampled layer)."""
    if sample_layers is None:
        n_layers = len(top1_per_layer)
        sample_layers = list(range(0, n_layers, 4)) + [n_layers - 1]

    text_positions = [
        i for i, lbl in enumerate(position_labels) if not lbl.startswith("<IMG")
    ]
    rows = []
    for pos in text_positions:
        row = {"position": pos, "token": repr(position_labels[pos])}
        for L in sample_layers:
            row[f"L{L}"] = repr(tokenizer.decode([top1_per_layer[L][0, pos].item()]))
        rows.append(row)

    pd.set_option("display.max_colwidth", 30)
    display(pd.DataFrame(rows))


def show_patch_segmentation(
    image, position_labels, top1_per_layer, tokenizer, layer,
    image_grid=LLAVA_IMAGE_GRID, top_n=12, patch_px=21,
):
    """Color the image-grid by top-1 token at the chosen layer (+ legend)."""
    def token_color(token: str):
        h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
        hue = (h % 360) / 360.0
        return matplotlib.colors.hsv_to_rgb((hue, 0.7, 0.95)).tolist() + [0.6]

    img_positions = [
        i for i, lbl in enumerate(position_labels) if lbl.startswith("<IMG")
    ]
    patch_tokens = [
        tokenizer.decode([top1_per_layer[layer][0, p].item()]) for p in img_positions
    ]
    unique_tokens = sorted(set(patch_tokens), key=patch_tokens.count, reverse=True)[:top_n]
    color_lookup = {t: token_color(t) for t in unique_tokens}

    overlay = np.zeros((image_grid, image_grid, 4))
    for idx, tok in enumerate(patch_tokens):
        if tok in color_lookup:
            r, c = divmod(idx, image_grid)
            overlay[r, c] = color_lookup[tok]

    disp_size = image_grid * patch_px
    overlay_resized = np.kron(overlay, np.ones((patch_px, patch_px, 1)))
    image_resized = image.resize((disp_size, disp_size))

    fig, (ax_img, ax_legend) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax_img.imshow(image_resized)
    ax_img.imshow(overlay_resized)
    ax_img.set_title(f"Layer {layer} top-1 token per patch")
    ax_img.axis("off")

    counts = {t: patch_tokens.count(t) for t in unique_tokens}
    ax_legend.axis("off")
    ax_legend.set_title("Top tokens (count)")
    for i, tok in enumerate(unique_tokens):
        ax_legend.add_patch(
            plt.Rectangle(
                (0, i), 0.15, 0.7,
                color=color_lookup[tok], transform=ax_legend.transData,
            )
        )
        ax_legend.text(
            0.2, i + 0.35,
            f"{tok!r:>12} ({counts[tok]:>3})",
            va="center", family="monospace",
            transform=ax_legend.transData,
        )
    ax_legend.set_xlim(0, 2)
    ax_legend.set_ylim(-1, len(unique_tokens) + 1)
    ax_legend.invert_yaxis()
    plt.tight_layout()
    plt.show()
