"""Side-by-side baseline / ablated renderer."""

import os

import matplotlib.pyplot as plt
import PIL.Image


def save_side_by_side(
    baseline: PIL.Image.Image,
    ablated: PIL.Image.Image,
    layers_to_ablate: list[int],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    baseline.save(os.path.join(output_dir, "baseline.png"))
    ablated.save(os.path.join(output_dir, "ablated.png"))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(baseline)
    axes[0].set_title("Baseline")
    axes[0].axis("off")
    axes[1].imshow(ablated)
    axes[1].set_title(f"Ablated layers {layers_to_ablate}")
    axes[1].axis("off")
    plt.tight_layout()
    path = os.path.join(output_dir, "side_by_side.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
