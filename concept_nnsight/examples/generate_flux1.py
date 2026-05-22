"""Smoke test for the FLUX.1-schnell concept-attention pipeline.

Run with: cd CVPR2026-HOW && python -m concept_nnsight.examples.generate_flux1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concept_nnsight import ConceptAttentionFluxPipeline


def main() -> None:
    pipe = ConceptAttentionFluxPipeline()

    prompt = "A cat in a park on the grass by a tree"
    concepts = ["cat", "grass", "sky", "tree"]

    out = pipe.generate_image(
        prompt=prompt,
        concepts=concepts,
        width=1024,
        height=1024,
        num_inference_steps=4,
        seed=0,
        layer_indices=[15, 16, 17, 18],  # paper's default for FLUX.1
    )

    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results_flux1")
    os.makedirs(results_dir, exist_ok=True)

    out.image.save(os.path.join(results_dir, "image.png"))
    for concept, hm in zip(concepts, out.concept_heatmaps):
        hm.save(os.path.join(results_dir, f"{concept}.png"))

    print(f"wrote {len(out.concept_heatmaps) + 1} files to {results_dir}/")
    print(f"  metadata: {out.metadata}")


if __name__ == "__main__":
    main()
