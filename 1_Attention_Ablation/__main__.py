"""Cross-attention ablation on Stable Diffusion 1.4.

Usage:
    python 1_Attention_Ablation/__main__.py
    python 1_Attention_Ablation/__main__.py --prompt "A van gogh starry night"
    python 1_Attention_Ablation/__main__.py --layers-to-ablate 0 5 7
"""

import argparse

from model import list_cross_attentions, load_sd
from ablate import generate_ablated, generate_baseline
from display import save_side_by_side


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", default="Starry Night")
    ap.add_argument("--layers-to-ablate", type=int, nargs="+", default=[5],
                    help="cross-attention indices (see printed table) to zero out")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--output-dir", default="./out")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load Stable Diffusion 1.4 wrapped for tracing.
    sd = load_sd()

    # 2. Enumerate cross-attention layers (one per `attn2` in the UNet).
    cross_attentions = list_cross_attentions(sd)
    print(f"Found {len(cross_attentions)} cross-attentions:")
    for i, (name, _) in enumerate(cross_attentions):
        marker = "  <-- ablated" if i in args.layers_to_ablate else ""
        print(f"  [{i:2d}] {name}{marker}")
    cross_attention_envoys = [envoy for _, envoy in cross_attentions]

    # 3. Generate the reference image.
    print(f"\nGenerating baseline: {args.prompt!r} (seed={args.seed}, steps={args.num_inference_steps})")
    baseline = generate_baseline(sd, args.prompt, args.seed, args.num_inference_steps)

    # 4. Generate the same prompt with selected cross-attentions ablated.
    print(f"Generating ablated (layers {args.layers_to_ablate})")
    ablated = generate_ablated(
        sd, args.prompt, args.seed, args.num_inference_steps,
        cross_attention_envoys, args.layers_to_ablate,
    )

    # 5. Save individual images + a side-by-side comparison.
    save_side_by_side(baseline, ablated, args.layers_to_ablate, args.output_dir)


if __name__ == "__main__":
    main()
