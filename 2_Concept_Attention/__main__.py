"""Concept attention on FLUX.1 / FLUX.2 via nnsight (local or remote on NDIF).

Usage:
    python 2_Concept_Attention/__main__.py
    python 2_Concept_Attention/__main__.py --model flux1
    python 2_Concept_Attention/__main__.py --remote
    python 2_Concept_Attention/__main__.py \\
        --prompt "A cat on a beach" --concepts cat sand sky water
"""

import argparse
import os

from nnsight import CONFIG

from pipeline import ConceptAttentionFlux2Pipeline
from pipeline_flux1 import ConceptAttentionFluxPipeline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["flux1", "flux2"], default="flux2",
                    help="flux2 = FLUX.2-klein-4B (Qwen3); flux1 = FLUX.1-schnell (T5+CLIP)")
    ap.add_argument("--prompt", default="A cat in a park on the grass by a tree")
    ap.add_argument("--concepts", nargs="+", default=["cat", "grass", "sky", "tree"])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--num-inference-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="./out")
    ap.add_argument("--remote", action="store_true",
                    help="encode + generate on NDIF instead of loading weights locally")
    ap.add_argument("--ndif-host", default=os.environ.get("NDIF_HOST"),
                    help="NDIF endpoint (only used with --remote); falls back to NDIF_HOST env or nnsight default")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.remote and args.ndif_host:
        CONFIG.API.HOST = args.ndif_host
    if args.remote:
        print(f"using NDIF at {CONFIG.API.HOST}")

    # 1. Pick the right pipeline for the chosen model family.
    Pipeline = (
        ConceptAttentionFlux2Pipeline if args.model == "flux2"
        else ConceptAttentionFluxPipeline
    )
    pipe = Pipeline(remote=args.remote)

    # 2. Run the technique end-to-end: text encoding, attention-mask
    #    intervention, in-trace score accumulation, heatmap colorisation.
    out = pipe.generate_image(
        prompt=args.prompt,
        concepts=args.concepts,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )

    # 3. Save the generated image + one PNG per concept heatmap.
    os.makedirs(args.output_dir, exist_ok=True)
    out.image.save(os.path.join(args.output_dir, "image.png"))
    for concept, heatmap in zip(args.concepts, out.concept_heatmaps):
        heatmap.save(os.path.join(args.output_dir, f"{concept}.png"))
    print(f"wrote {len(out.concept_heatmaps) + 1} files to {args.output_dir}/")
    print(f"  metadata: {out.metadata}")


if __name__ == "__main__":
    main()
