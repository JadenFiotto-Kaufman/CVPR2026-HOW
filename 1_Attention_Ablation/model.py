"""Load Stable Diffusion 1.4 wrapped for nnsight tracing."""

# nnsight MUST be imported before torch in this env or Python segfaults
# during nnsight's lazy-import binding setup.
from nnsight import DiffusionModel  # noqa: I001

import torch  # noqa: E402


def load_sd(
    model_name: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> DiffusionModel:
    """Dispatch SD 1.4 with the safety checker disabled (we read raw model output)."""
    return DiffusionModel(
        model_name,
        torch_dtype=torch_dtype,
        safety_checker=None,
        dispatch=True,
        device_map=device,
    )


def list_cross_attentions(sd: DiffusionModel):
    """Return [(name, envoy)] for every `.attn2` cross-attention in the UNet,
    sorted into a stable forward-pass-friendly order (down → mid → up)."""
    pairs = [
        (name, envoy)
        for name, envoy in sd.unet.named_modules()
        if name.endswith(".attn2")
    ]
    return sorted(pairs, key=lambda x: x[0])
