"""Concept-attention for FLUX.2 via nnsight.

Public API mirrors `concept_attention.ConceptAttentionFluxPipeline` from the
original repo but is implemented entirely on top of nnsight + diffusers'
stock `Flux2KleinPipeline`. We don't monkey-patch the DoubleStreamBlock; we
trace a normal generate, capture per-block tensors, and recreate the concept
stream's math externally (see `concept_stream.py`).
"""

# nnsight MUST be imported before torch in this env or Python segfaults
# during nnsight's lazy-import binding setup. Order matters everywhere this
# module is loaded.
from nnsight import DiffusionModel  # noqa: I001  (intentional order)

import torch  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from .concept_stream import (  # noqa: E402
    BlockCapture,
    compute_heatmaps,
    evolve_concept_stream,
)
from .heatmap import colorize_heatmaps  # noqa: E402


@dataclass
class ConceptAttentionOutput:
    image: PIL.Image.Image
    concept_heatmaps: list[PIL.Image.Image]
    # Raw cached vectors (only if cache_vectors=True). Shapes:
    #   image_vectors:   [T, L, B, num_image_patches, D]
    #   concept_vectors: [T, L, B, num_concepts,     D]
    image_vectors: torch.Tensor | None = None
    concept_vectors: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)


class ConceptAttentionFlux2Pipeline:
    """Concept-attention wrapper for FLUX.2-klein-4B (and other FLUX.2 variants).

    Usage:
        pipe = ConceptAttentionFlux2Pipeline()
        out = pipe.generate_image(
            prompt="a cat in a park",
            concepts=["cat", "grass", "sky", "tree"],
            num_inference_steps=4,
        )
        out.image.save("image.png")
        for c, hm in zip(["cat","grass","sky","tree"], out.concept_heatmaps):
            hm.save(f"{c}.png")
    """

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.2-klein-4B",
        device_map: str = "balanced",
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_name = model_name
        self.model = DiffusionModel(
            model_name,
            dispatch=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self._pipeline = self.model._model.pipeline

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def _encode_concepts(self, concepts: list[str], target_device: torch.device) -> torch.Tensor:
        """Encode concept strings through the pipeline's text encoder.

        FLUX.2 uses a Qwen3 causal LM as its text encoder. We re-use the
        pipeline's own `encode_prompt` machinery and concatenate the
        per-concept embeddings so we get one row per concept word.

        Returns: [1, num_concepts, joint_attention_dim] on `target_device`.
        """
        pipe = self._pipeline
        # encode_prompt API differs across diffusers versions; FLUX.2 uses
        # _get_qwen_prompt_embeds(prompt) or encode_prompt(prompt) depending
        # on the release. Try the public path first.
        embeds_per_concept: list[torch.Tensor] = []
        for concept in concepts:
            # encode_prompt returns (prompt_embeds, ...); shape [B, L, D]
            out = pipe.encode_prompt(prompt=concept, device=target_device, num_images_per_prompt=1)
            emb = out[0] if isinstance(out, tuple) else out
            # Take the mean over the per-concept token positions to get one
            # vector per concept word. Simplest pooling; matches the spirit
            # of the original repo's `embed_concepts` which averages tokens.
            embeds_per_concept.append(emb.mean(dim=1, keepdim=True))  # [1, 1, D]
        return torch.cat(embeds_per_concept, dim=1)  # [1, num_concepts, D]

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        concepts: list[str],
        *,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int = 0,
        layer_indices: list[int] | None = None,
        timestep_indices: list[int] | None = None,
        softmax: bool = True,
        cmap: str = "plasma",
        return_pil_heatmaps: bool = True,
        cache_vectors: bool = False,
    ) -> ConceptAttentionOutput:
        assert width == height, "Square output only (matches the original repo)."
        assert width % 16 == 0, "Width must be divisible by 16 (patch_size*2)."

        n_double_blocks = len(self._pipeline.transformer.transformer_blocks)
        if layer_indices is None:
            layer_indices = list(range(n_double_blocks))
        if timestep_indices is None:
            timestep_indices = list(range(num_inference_steps))

        # ----- 1. trace generate, capturing per-(step, block) tensors -----
        n_blocks = n_double_blocks
        with self.model.generate(
            prompt,
            width=width, height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
        ) as tracer:
            captures = list().save()
            for _step in tracer.iter[:]:
                step_caps = list()
                for blk_env in self.model.transformer.transformer_blocks:
                    attn_env = blk_env.attn
                    # Access modules in forward-pass order — nnsight's one-shot
                    # hooks demand it. Block .inputs fire first, then to_q/to_k/
                    # to_v inside the processor, then finally to_out[0].
                    blk_kwargs = blk_env.inputs[1]
                    hs   = blk_kwargs["hidden_states"].clone()
                    enc  = blk_kwargs["encoder_hidden_states"].clone()
                    mtxt = blk_kwargs["temb_mod_txt"].clone()
                    rope = blk_kwargs["image_rotary_emb"]  # tensor or (cos, sin)
                    img_k = attn_env.to_k.output.clone()
                    img_v = attn_env.to_v.output.clone()
                    # `to_out[0].input` is the pre-projection image attn output
                    # (flattened SDPA result, image portion). This lives in the
                    # same inner-dim space as the concept-attn output (also
                    # pre-projection) — so the eventual einsum is meaningful.
                    img_attn_pre_proj = attn_env.to_out[0].inputs[0][0].clone()
                    step_caps.append({
                        "hidden_states":         hs,
                        "encoder_hidden_states": enc,
                        "temb_mod_txt":          mtxt,
                        "image_rotary_emb":      rope,
                        "img_attn_out":          img_attn_pre_proj,
                        "img_k":                 img_k,
                        "img_v":                 img_v,
                    })
                captures.append(step_caps)
            result = tracer.result.save()

        image = result.images[0]

        # ----- 2. encode concepts (outside the trace) -----
        encoder = self._pipeline.text_encoder
        target_device = next(encoder.parameters()).device
        raw_concept_embeds = self._encode_concepts(concepts, target_device)  # [1, C, joint_dim]
        # Project to the transformer's internal hidden_size via context_embedder
        # (the same input projection the model uses for encoder_hidden_states).
        ctx_embedder = self._pipeline.transformer.context_embedder
        ctx_device = next(ctx_embedder.parameters()).device
        ctx_dtype = next(ctx_embedder.parameters()).dtype
        concept_embeds = ctx_embedder(
            raw_concept_embeds.to(ctx_device, dtype=ctx_dtype)
        )                                                                   # [1, C, hidden_size]

        # ----- 3. externally evolve the concept stream per timestep -----
        blocks = self._pipeline.transformer.transformer_blocks
        per_step_concept: list[list[torch.Tensor]] = []   # T × L of [B, C, D]
        per_step_image:   list[list[torch.Tensor]] = []   # T × L of [B, P, D]

        for step_idx, step_caps in enumerate(captures):
            cap_objs = [
                BlockCapture(
                    hidden_states=raw["hidden_states"],
                    encoder_hidden_states=raw["encoder_hidden_states"],
                    temb_mod_txt=raw["temb_mod_txt"],
                    image_rotary_emb=raw["image_rotary_emb"],
                    img_attn_out=raw["img_attn_out"],
                    img_k=raw["img_k"],
                    img_v=raw["img_v"],
                )
                for raw in step_caps
            ]
            # Reset concept state at every timestep (original paper averages
            # across timesteps anyway; carrying through timesteps mixes
            # information across noise levels in an undefined way).
            concept_state = concept_embeds.clone().to(cap_objs[0].hidden_states.device)
            _, concept_outs, image_outs = evolve_concept_stream(blocks, cap_objs, concept_state)
            per_step_concept.append(concept_outs)
            per_step_image.append(image_outs)

        # Stack to [T, L, B, *, D]
        concept_tensor = torch.stack(
            [torch.stack(layer_outs, dim=0) for layer_outs in per_step_concept], dim=0
        )
        image_tensor = torch.stack(
            [torch.stack(layer_outs, dim=0) for layer_outs in per_step_image], dim=0
        )

        # ----- 4. heatmaps -----
        grid = width // 16  # 64 for 1024
        heatmaps = compute_heatmaps(
            image_tensor,
            concept_tensor,
            layer_indices=layer_indices,
            timestep_indices=timestep_indices,
            softmax=softmax,
            grid=grid,
        )                                                              # [B, C, grid, grid]
        heatmaps_np = heatmaps[0].to(torch.float32).cpu().numpy()      # [C, grid, grid]

        if return_pil_heatmaps:
            heatmap_imgs = colorize_heatmaps(heatmaps_np, cmap=cmap, upscale=(width, height))
        else:
            heatmap_imgs = [PIL.Image.fromarray(np.uint8(hm * 255)) for hm in heatmaps_np]

        return ConceptAttentionOutput(
            image=image,
            concept_heatmaps=heatmap_imgs,
            image_vectors=image_tensor if cache_vectors else None,
            concept_vectors=concept_tensor if cache_vectors else None,
            metadata={
                "model_name": self.model_name,
                "prompt": prompt,
                "concepts": concepts,
                "num_inference_steps": num_inference_steps,
                "layer_indices": layer_indices,
                "timestep_indices": timestep_indices,
                "softmax": softmax,
                "grid": grid,
            },
        )
