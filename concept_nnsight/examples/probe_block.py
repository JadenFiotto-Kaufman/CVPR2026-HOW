"""Probe: load FLUX.2-klein-4B via nnsight and inspect its transformer blocks.

Goal: confirm the exact attribute paths we need in our pipeline (Flux2Attention
projection layer names, the per-block Modulation interface, the rotary-emb
threading) before we write the real concept-stream code.
"""

# nnsight MUST be imported before torch in this env or Python segfaults
# (something to do with how nnsight's lazy loader rewrites import bindings).
from nnsight import DiffusionModel
import torch  # noqa: E402

REPO = "black-forest-labs/FLUX.2-klein-4B"


def main() -> None:
    print(f"loading {REPO}...")
    model = DiffusionModel(
        REPO,
        dispatch=True,
        device_map="balanced",
        torch_dtype=torch.float16,
    )
    print("loaded")

    # Raw pipeline for structural inspection (read weights, shapes, etc.).
    raw_t = model._model.pipeline.transformer
    print(f"transformer: {type(raw_t).__name__}")
    print(f"  double blocks: {len(raw_t.transformer_blocks)}")
    print(f"  single blocks: {len(raw_t.single_transformer_blocks)}")

    blk = raw_t.transformer_blocks[0]
    attn = blk.attn
    print(f"  block[0] type: {type(blk).__name__}, attn: {type(attn).__name__}")
    print(f"    image qkv: to_q={tuple(attn.to_q.weight.shape)} "
          f"to_k={tuple(attn.to_k.weight.shape)} to_v={tuple(attn.to_v.weight.shape)}")
    print(f"    text  qkv: add_q={tuple(attn.add_q_proj.weight.shape)} "
          f"add_k={tuple(attn.add_k_proj.weight.shape)} add_v={tuple(attn.add_v_proj.weight.shape)}")
    print(f"    heads={attn.heads}, head_dim={attn.head_dim}, inner_dim={attn.inner_dim}")
    print(f"    to_out: {type(attn.to_out[0]).__name__} "
          f"to_add_out: {type(attn.to_add_out).__name__}")

    # nnsight envoy for tracing
    blk0_env = model.transformer.transformer_blocks[0]
    print("\nrunning 1-step trace...")
    with model.trace("a cat") as tracer:
        block0_out = blk0_env.output.save()
        result = tracer.result.save()

    print(f"block0 output: type={type(block0_out).__name__}, "
          f"len={len(block0_out) if hasattr(block0_out, '__len__') else 'n/a'}")
    if isinstance(block0_out, tuple):
        txt_out, img_out = block0_out
        print(f"  txt shape: {tuple(txt_out.shape)}, img shape: {tuple(img_out.shape)}")
    else:
        print(f"  shape: {tuple(block0_out.shape)}")
    print(f"final result: type={type(result).__name__}")
    if hasattr(result, "images"):
        print(f"  images: {len(result.images)}, first size: {result.images[0].size}")


if __name__ == "__main__":
    main()
