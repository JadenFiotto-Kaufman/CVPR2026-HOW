"""Concept-attention for FLUX.2-klein-4B via nnsight.

Inherits the shared trace + accumulator + heatmap pipeline from `_base`,
implementing only the FLUX.2-specific bits:

  * Encode concepts via Qwen3 (chat-template wrapped), taking the
    position-3 embedding (the concept-word slot) per concept.
  * Pass the joint prompt+concept embeds via `prompt_embeds=` and the
    attention mask via `attention_kwargs={"attention_mask": ...}`.
  * Override the pipeline-derived `txt_ids` per step so concept rows
    have zero positional coords (the paper's `concept_pe = 0`); FLUX.2
    derives them as `[(0,0,0,L) for L in arange(L_total)]` otherwise.
  * In Flux2AttnProcessor, `to_add_out` (encoder) is called BEFORE
    `to_out[0]` (image), so capture in that forward order.
"""

from ._base import ConceptAttentionOutput, ConceptAttentionPipeline  # noqa: I001

import torch  # noqa: E402


class ConceptAttentionFlux2Pipeline(ConceptAttentionPipeline):
    """ConceptAttention pipeline for FLUX.2-klein-4B (Qwen3 + Flux2 DiT).

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

    # Position of the concept word in the Qwen3-chat-templated sequence:
    #   tokens[0..2]    = <|im_start|>user\n     (chat-prefix)
    #   tokens[3]       = {concept word}          <-- the useful position
    #   tokens[4..12]   = chat-suffix
    #   tokens[13..511] = <pad>
    _QWEN_CHAT_CONCEPT_POS = 3

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.2-klein-4B",
        device_map: str = "balanced",
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(model_name, device_map=device_map, torch_dtype=torch_dtype)
        # Populated by `_pipeline_call_kwargs`; read by `_per_step_intervention`.
        self._cached_text_ids: torch.Tensor | None = None

    # ------------------------------------------------------------------ hooks

    @torch.no_grad()
    def _encode_prompt_and_concepts(self, prompt, concepts):
        pipe = self._pipeline
        device = next(pipe.text_encoder.parameters()).device

        # Prompt: full Qwen3 sequence (chat-templated, padded to 512).
        out = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)
        prompt_embeds = out[0] if isinstance(out, tuple) else out             # [1, L_txt, D]

        # Concepts: one position-3 embedding per concept.
        p = self._QWEN_CHAT_CONCEPT_POS
        concept_rows = []
        for c in concepts:
            out = pipe.encode_prompt(prompt=c, device=device, num_images_per_prompt=1)
            emb = out[0] if isinstance(out, tuple) else out                   # [1, L_tok, D]
            concept_rows.append(emb[:, p : p + 1, :])
        concept_embeds = torch.cat(concept_rows, dim=1)                       # [1, L_c, D]

        prompt_embeds_full = torch.cat(
            [prompt_embeds, concept_embeds.to(prompt_embeds.dtype)], dim=1,
        )
        L_txt = prompt_embeds.shape[1]
        L_c = concept_embeds.shape[1]

        # Pre-build the txt_ids tensor that the per-step intervention will
        # install (concept rows all-zero == concept_pe = 0 per the paper).
        text_ids = torch.zeros(
            prompt_embeds_full.shape[0], L_txt + L_c, 4,
            dtype=torch.long, device=prompt_embeds_full.device,
        )
        text_ids[:, :L_txt, 3] = torch.arange(L_txt, device=text_ids.device)
        return prompt_embeds_full, {"text_ids": text_ids}, L_txt, L_c

    def _pipeline_call_kwargs(self, prompt_embeds_full, extra, attention_mask):
        # FLUX.2 only needs the sequence embeddings (no pooled vector) and
        # uses `attention_kwargs` (not `joint_attention_kwargs`).
        # Stash text_ids on self so _per_step_intervention can install it.
        self._cached_text_ids = extra["text_ids"]
        return {
            "prompt_embeds": prompt_embeds_full,
            "attention_kwargs": {"attention_mask": attention_mask},
        }

    def _per_step_intervention(self, t_env, *, L_c: int) -> None:
        # FLUX.2's `_prepare_text_ids` numbers positions on the L axis,
        # so concept rows would otherwise get nonzero RoPE coords. Override
        # the txt_ids the transformer receives this step with our pre-built
        # zero-position version.
        new_kwargs = dict(t_env.inputs[1])
        new_kwargs["txt_ids"] = self._cached_text_ids.to(new_kwargs["txt_ids"].device)
        t_env.inputs = (t_env.inputs[0], new_kwargs)

    def _capture_block_attention(self, blk_env):
        # In Flux2AttnProcessor, `to_add_out` (encoder portion of the joint
        # SDPA output) fires BEFORE `to_out[0]` (image portion). Access in
        # forward-pass order for nnsight's one-shot hooks.
        enc_attn_pre = blk_env.attn.to_add_out.inputs[0][0]
        img_attn_pre = blk_env.attn.to_out[0].inputs[0][0]
        return img_attn_pre, enc_attn_pre


__all__ = ["ConceptAttentionFlux2Pipeline", "ConceptAttentionOutput"]
