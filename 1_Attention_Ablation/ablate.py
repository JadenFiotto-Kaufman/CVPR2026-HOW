"""Cross-attention ablation traces for Stable Diffusion 1.4.

Two entry points, both run one `model.generate(...)` trace and return a PIL
image:

  * `generate_baseline`  — unmodified forward, the reference image.
  * `generate_ablated`   — at every denoising step, zero the input to
                           `to_out[0]` for the chosen cross-attention
                           envoys. Attention is still computed; its
                           contribution just doesn't get added back into
                           the image stream.
"""

import PIL.Image
from nnsight import DiffusionModel


def generate_baseline(
    sd: DiffusionModel,
    prompt: str,
    seed: int,
    num_inference_steps: int,
) -> PIL.Image.Image:
    with sd.generate(prompt, num_inference_steps=num_inference_steps, seed=seed) as tracer:
        out = tracer.result.save()
    return out.images[0]


def generate_ablated(
    sd: DiffusionModel,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    cross_attention_envoys: list,
    layers_to_ablate: list[int],
) -> PIL.Image.Image:
    # Sort to access in forward order — nnsight's one-shot hooks are
    # order-sensitive within a single forward pass.
    sorted_layers = sorted(layers_to_ablate)
    with sd.generate(prompt, num_inference_steps=num_inference_steps, seed=seed) as tracer:
        for _step in tracer.iter[:]:
            for idx in sorted_layers:
                cross_attention_envoys[idx].to_out[0].input[:] = 0
        out = tracer.result.save()
    return out.images[0]
