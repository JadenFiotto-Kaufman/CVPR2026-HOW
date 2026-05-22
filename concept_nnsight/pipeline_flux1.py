"""Concept-attention for FLUX.1 (Schnell / Dev) via nnsight, mask-based.

Same idea as `ConceptAttentionFlux2Pipeline`: pre-encode prompt + concepts,
splice the concept rows into `prompt_embeds`, install an attention mask
that isolates concepts from prompt + image streams, accumulate the
concept↔image scores inside the trace. Image stream is bit-identical to
a vanilla forward.

Two FLUX.1-specific deltas vs. our FLUX.2 implementation:

  * FLUX.1's pipeline uses T5 (sequence) + CLIP (pooled) encoders, so
    we have to pass both `prompt_embeds` AND `pooled_prompt_embeds`.
  * FLUX.1's mask kwarg is `joint_attention_kwargs` (not
    `attention_kwargs`).
  * FLUX.1 derives `text_ids` as plain `torch.zeros(L, 3)` inline in
    `encode_prompt`, so splicing extra rows into `prompt_embeds`
    automatically gives concept tokens zero positional coords. No
    txt_ids intercept needed (FLUX.2 needs one because its
    `_prepare_text_ids` uses the L index).
"""

# nnsight before torch — env-specific segfault avoidance.
from nnsight import DiffusionModel  # noqa: I001

import torch  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402

from .heatmap import colorize_heatmaps  # noqa: E402
from .pipeline import ConceptAttentionOutput  # noqa: E402


class ConceptAttentionFluxPipeline:
    """Concept-attention wrapper for FLUX.1-{schnell,dev}.

    Usage:
        pipe = ConceptAttentionFluxPipeline()  # defaults to FLUX.1-schnell
        out = pipe.generate_image(
            prompt="a cat in a park",
            concepts=["cat", "grass", "sky", "tree"],
            num_inference_steps=4,
        )
    """

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
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
    def _encode_concept_one(self, text: str, target_device: torch.device) -> torch.Tensor:
        """Encode a single concept through the T5 encoder, return FIRST token.

        Matches the original ConceptAttention repo's `embed_concepts` (see
        concept_attention/utils.py): per-concept T5 output, take the first
        token's contextual embedding. Mean-pool over T5's padded sequence
        produced very weak / uniform concept vectors that aligned poorly.
        """
        # encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids).
        out = self._pipeline.encode_prompt(
            prompt=text, prompt_2=text, device=target_device, num_images_per_prompt=1
        )
        return out[0][:, :1, :]                                       # [1, 1, D_t5]

    @torch.no_grad()
    def _encode_concepts(self, concepts: list[str], target_device: torch.device) -> torch.Tensor:
        rows = [self._encode_concept_one(c, target_device) for c in concepts]
        return torch.cat(rows, dim=1)                                # [1, num_concepts, D_t5]

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
        guidance_scale: float = 0.0,
        layer_indices: list[int] | None = None,
        timestep_indices: list[int] | None = None,
        softmax: bool = True,
        cmap: str = "plasma",
        return_pil_heatmaps: bool = True,
        cache_vectors: bool = False,
    ) -> ConceptAttentionOutput:
        assert width == height, "Square output only."
        assert width % 16 == 0, "Width must be divisible by 16 (patch_size * 2)."

        n_double_blocks = len(self._pipeline.transformer.transformer_blocks)
        if layer_indices is None:
            layer_indices = list(range(n_double_blocks))
        if timestep_indices is None:
            timestep_indices = list(range(num_inference_steps))

        # ------------------------------------------------------------------
        # 1) Pre-encode prompt (T5 + CLIP) and concepts (T5 only, mean-pooled).
        # ------------------------------------------------------------------
        device = next(self._pipeline.text_encoder.parameters()).device
        prompt_embeds, pooled_prompt_embeds, _ = self._pipeline.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device, num_images_per_prompt=1,
        )
        concept_embeds = self._encode_concepts(concepts, device)     # [1, L_c, D_t5]
        prompt_embeds_full = torch.cat(
            [prompt_embeds, concept_embeds.to(prompt_embeds.dtype)], dim=1
        )                                                            # [1, L_txt+L_c, D_t5]
        L_txt = prompt_embeds.shape[1]
        L_c = concept_embeds.shape[1]

        # ------------------------------------------------------------------
        # 2) Build attention mask. True = allowed; two-way isolation between
        #    the concept side-channel and the rest of the joint stream.
        # ------------------------------------------------------------------
        L_img = (width // 16) ** 2
        L_tot = L_txt + L_c + L_img
        allow = torch.ones(L_tot, L_tot, dtype=torch.bool, device=prompt_embeds.device)
        c_start, c_end = L_txt, L_txt + L_c
        allow[:c_start, c_start:c_end] = False     # prompt  → concept blocked
        allow[c_end:,  c_start:c_end] = False      # image   → concept blocked
        allow[c_start:c_end, :c_start] = False     # concept → prompt  blocked

        # ------------------------------------------------------------------
        # 3) Trace generate. Inline accumulation of per-(step, layer)
        #    concept↔image scores; layer / timestep selection via cheap
        #    Python `if x not in set: continue` filters inside the loop.
        # ------------------------------------------------------------------
        acc_device = next(self._pipeline.transformer.parameters()).device
        timestep_set = set(timestep_indices)
        layer_set = set(layer_indices)

        with self.model.generate(
            prompt_embeds=prompt_embeds_full,
            pooled_prompt_embeds=pooled_prompt_embeds,
            joint_attention_kwargs={"attention_mask": allow},
            width=width, height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        ) as tracer:
            score_acc = torch.zeros(
                prompt_embeds_full.shape[0], L_c, L_img,
                dtype=torch.float32, device=acc_device,
            ).save()
            img_steps = list().save() if cache_vectors else None
            concept_steps = list().save() if cache_vectors else None

            for step_idx, _step in enumerate(tracer.iter[:]):
                if step_idx not in timestep_set:
                    continue

                if cache_vectors:
                    step_img: list = list()
                    step_concept: list = list()

                for layer_idx, blk_env in enumerate(self.model.transformer.transformer_blocks):
                    # FLUX.1's FluxAttnProcessor calls to_out[0] BEFORE
                    # to_add_out (FLUX.2 was the opposite). Access in
                    # forward-pass order for nnsight's one-shot hooks.
                    img_attn_pre = blk_env.attn.to_out[0].inputs[0][0]
                    enc_attn_pre = blk_env.attn.to_add_out.inputs[0][0]
                    if layer_idx not in layer_set:
                        if cache_vectors:
                            step_concept.append(enc_attn_pre[:, L_txt:].clone())
                            step_img.append(img_attn_pre.clone())
                        continue

                    concept_pre = enc_attn_pre[:, L_txt:]
                    scores = torch.einsum(
                        "bpd,bcd->bcp",
                        img_attn_pre.float(),
                        concept_pre.float(),
                    )                                                # [B, C, L_img]
                    if softmax:
                        scores = scores.softmax(dim=-2)
                    score_acc.add_(scores.to(acc_device))

                    if cache_vectors:
                        step_concept.append(concept_pre.clone())
                        step_img.append(img_attn_pre.clone())

                if cache_vectors:
                    concept_steps.append(step_concept)
                    img_steps.append(step_img)

            result = tracer.result.save()

        # ------------------------------------------------------------------
        # 4) Mean, reshape, colorize.
        # ------------------------------------------------------------------
        n_accumulated = len(timestep_indices) * len(layer_indices)
        grid = width // 16
        heatmaps = (score_acc / n_accumulated).unflatten(-1, (grid, grid))
        heatmaps_np = heatmaps[0].cpu().numpy()

        if return_pil_heatmaps:
            heatmap_imgs = colorize_heatmaps(heatmaps_np, cmap=cmap, upscale=(width, height))
        else:
            heatmap_imgs = [PIL.Image.fromarray(np.uint8(hm * 255)) for hm in heatmaps_np]

        image_tensor = concept_tensor = None
        if cache_vectors:
            image_tensor = torch.stack(
                [torch.stack(layer_outs, dim=0) for layer_outs in img_steps], dim=0
            )
            concept_tensor = torch.stack(
                [torch.stack(layer_outs, dim=0) for layer_outs in concept_steps], dim=0
            )

        return ConceptAttentionOutput(
            image=result.images[0],
            concept_heatmaps=heatmap_imgs,
            image_vectors=image_tensor,
            concept_vectors=concept_tensor,
            metadata={
                "model_name": self.model_name,
                "prompt": prompt,
                "concepts": concepts,
                "num_inference_steps": num_inference_steps,
                "layer_indices": layer_indices,
                "timestep_indices": timestep_indices,
                "softmax": softmax,
                "grid": grid,
                "L_c": L_c,
            },
        )
