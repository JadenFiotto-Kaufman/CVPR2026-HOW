"""Smoke test for the nnsight-based concept attention pipeline.

Run with: cd CVPR2026-HOW && python -m concept_nnsight.examples.generate
"""

import os
import sys

# Make `import concept_nnsight` work when run as `python examples/generate.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concept_nnsight import ConceptAttentionFlux2Pipeline


def main() -> None:
    pipe = ConceptAttentionFlux2Pipeline()

    prompt = "A cat in a park on the grass by a tree"
    concepts = ["cat", "grass", "sky", "tree"]

    out = pipe.generate_image(
        prompt=prompt,
        concepts=concepts,
        width=1024,
        height=1024,
        num_inference_steps=4,
        seed=0,
    )

    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    out.image.save(os.path.join(results_dir, "image.png"))
    for concept, hm in zip(concepts, out.concept_heatmaps):
        hm.save(os.path.join(results_dir, f"{concept}.png"))

    print(f"wrote {len(out.concept_heatmaps) + 1} files to {results_dir}/")
    print(f"  metadata: {out.metadata}")


if __name__ == "__main__":
    main()
