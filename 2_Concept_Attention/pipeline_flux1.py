"""Concept-attention for FLUX.1-{schnell, dev} via nnsight.

Inherits the shared trace + accumulator + heatmap pipeline from `_base`,
implementing only the FLUX.1-specific bits:

  * Encode concepts via T5 (bidirectional, no chat template), taking
    the FIRST token's embedding per concept (the concept word itself).
  * FLUX.1's pipeline takes BOTH `prompt_embeds` (T5 sequence) and
    `pooled_prompt_embeds` (CLIP pooled), so both have to be forwarded.
  * Attention mask kwarg is `joint_attention_kwargs` (not
    `attention_kwargs` like FLUX.2).
  * No txt_ids intercept needed: FLUX.1 derives them inline as
    `torch.zeros(L, 3)`, so spliced concept rows automatically get
    zero coords (FLUX.2 needs an intercept because its
    `_prepare_text_ids` uses the L index).
  * In FluxAttnProcessor, `to_out[0]` (image) fires BEFORE `to_add_out`
    (encoder) — opposite of FLUX.2.
"""

from _base import ConceptAttentionOutput, ConceptAttentionPipeline  # noqa: I001

import torch  # noqa: E402


class ConceptAttentionFluxPipeline(ConceptAttentionPipeline):
    """ConceptAttention pipeline for FLUX.1-schnell / FLUX.1-dev (T5+CLIP + Flux1 DiT).

    Usage:
        pipe = ConceptAttentionFluxPipeline()                 # defaults to schnell
        out = pipe.generate_image(
            prompt="a cat in a park",
            concepts=["cat", "grass", "sky", "tree"],
            num_inference_steps=4,
            layer_indices=[15, 16, 17, 18],                   # paper's default
        )
    """

    kind = "flux1"

    # T5 has no chat-template prefix — the actual concept word lives at
    # position 0 (followed by '</s>' at position 1, padding from 2..511).
    _T5_CONCEPT_POS = 0

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
        device_map: str = "balanced",
        torch_dtype: torch.dtype = torch.float16,
        remote: bool = False,
    ) -> None:
        super().__init__(model_name, device_map=device_map,
                         torch_dtype=torch_dtype, remote=remote)

    # ------------------------------------------------------------------ hooks

    @torch.no_grad()
    def _encode_prompt_and_concepts(self, prompt, concepts):
        p = self._T5_CONCEPT_POS

        if self.remote:
            # Bind model to a local var so the session payload doesn't
            # capture `self` (abc.ABC in the class chain isn't on NDIF's
            # whitelist). One session covers all text-encoder forwards.
            flux = self.model
            with flux.session(remote=True):
                out = flux.pipeline.encode_prompt(
                    prompt=prompt, prompt_2=prompt,
                    device="cuda", num_images_per_prompt=1,
                )
                prompt_embeds = out[0].save()
                pooled_prompt_embeds = out[1].save()
                concept_rows = list().save()
                for c in concepts:
                    emb = flux.pipeline.encode_prompt(
                        prompt=c, prompt_2=c,
                        device="cuda", num_images_per_prompt=1,
                    )[0]
                    concept_rows.append(emb[:, p : p + 1, :])
        else:
            pipe = self._pipeline
            device = next(pipe.text_encoder.parameters()).device

            prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt,
                device=device, num_images_per_prompt=1,
            )
            concept_rows = []
            for c in concepts:
                emb, *_ = pipe.encode_prompt(
                    prompt=c, prompt_2=c,
                    device=device, num_images_per_prompt=1,
                )
                concept_rows.append(emb[:, p : p + 1, :])

        concept_embeds = torch.cat(concept_rows, dim=1)                       # [1, L_c, D_t5]
        prompt_embeds_full = torch.cat(
            [prompt_embeds, concept_embeds.to(prompt_embeds.dtype)], dim=1,
        )
        L_txt = prompt_embeds.shape[1]
        L_c = concept_embeds.shape[1]
        return (
            prompt_embeds_full,
            {"pooled_prompt_embeds": pooled_prompt_embeds},
            L_txt, L_c,
        )

    def _pipeline_call_kwargs(self, prompt_embeds_full, extra, attention_mask):
        return {
            "prompt_embeds": prompt_embeds_full,
            "pooled_prompt_embeds": extra["pooled_prompt_embeds"],
            "joint_attention_kwargs": {"attention_mask": attention_mask},
            # Schnell is a distilled model; guidance must be 0.
            "guidance_scale": 0.0,
        }

    # FLUX.1 needs no per-step intervention (its `_prepare_text_ids`
    # already gives `torch.zeros(L, 3)`). Per-block attention capture is
    # inlined in `_base.generate_image`, branching on `self.kind`.


__all__ = ["ConceptAttentionFluxPipeline", "ConceptAttentionOutput"]
