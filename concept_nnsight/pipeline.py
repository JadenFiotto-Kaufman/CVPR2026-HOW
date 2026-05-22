"""Concept-attention for FLUX.2 via nnsight, mask-based implementation.

We pre-concatenate the encoded concept tokens onto the encoded prompt
(ONCE, outside the trace) and pass the combined embedding to the pipeline
via its standard `prompt_embeds=` kwarg. We pre-build a per-block attention
mask that achieves two-way isolation between the concept side-channel and
the rest of the joint stream, and ship it through `attention_kwargs=`. We
monkey-patch `_prepare_text_ids` for the duration of the call so the
pipeline-generated `txt_ids` give concept rows a zero-position (matches the
paper's `concept_pe = 0`). Then `model.generate(...)` is a normal trace
whose only job is to capture per-block pre-projection attention outputs.

No per-step intervention, no external math replay — one pass, two
heatmap ingredients fall out of the same forward.
"""

# nnsight MUST be imported before torch in this env or Python segfaults
# during nnsight's lazy-import binding setup.
from nnsight import DiffusionModel  # noqa: I001

import torch  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from .heatmap import colorize_heatmaps  # noqa: E402


@dataclass
class ConceptAttentionOutput:
    image: PIL.Image.Image
    concept_heatmaps: list[PIL.Image.Image]
    image_vectors: torch.Tensor | None = None
    concept_vectors: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)


class ConceptAttentionFlux2Pipeline:
    """Concept-attention wrapper for FLUX.2-klein-4B.

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
    def _encode_one(self, text: str, target_device: torch.device) -> torch.Tensor:
        """Encode a single string through the pipeline's Qwen3 text encoder.

        Returns [1, L_tok, joint_attention_dim] on `target_device`.
        """
        out = self._pipeline.encode_prompt(
            prompt=text, device=target_device, num_images_per_prompt=1
        )
        return out[0] if isinstance(out, tuple) else out

    # Position of the actual concept word in the Qwen3-chat-templated
    # sequence. The diffusers FLUX.2 pipeline wraps each input via:
    #   <|im_start|>user\n{concept}<|im_end|>\n<|im_start|>assistant\n<think>...
    # which tokenises as positions [0..2] = chat-prefix, [3] = concept word,
    # [4..12] = chat-suffix, [13..511] = pad. So position 3 is the only
    # position carrying concept content. Mean-pool over all 512 padded
    # positions dilutes the signal and produces semantically scrambled
    # heatmaps. The FLUX.1 sibling pipeline uses position 0 because T5 has
    # no chat-template prefix.
    _QWEN_CHAT_CONCEPT_POS = 3

    @torch.no_grad()
    def _encode_concepts(self, concepts: list[str], target_device: torch.device) -> torch.Tensor:
        """Encode each concept via Qwen3 and take the embedding at the
        chat-templated concept-word position (see _QWEN_CHAT_CONCEPT_POS).

        Returns [1, num_concepts, joint_attention_dim] on `target_device`.
        """
        p = self._QWEN_CHAT_CONCEPT_POS
        rows: list[torch.Tensor] = []
        for c in concepts:
            rows.append(self._encode_one(c, target_device)[:, p : p + 1, :])
        return torch.cat(rows, dim=1)

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
        assert width % 16 == 0, "Width must be divisible by 16 (patch_size * 2)."

        n_double_blocks = len(self._pipeline.transformer.transformer_blocks)
        if layer_indices is None:
            layer_indices = list(range(n_double_blocks))
        if timestep_indices is None:
            timestep_indices = list(range(num_inference_steps))

        # ------------------------------------------------------------------
        # 1) Pre-encode prompt + concepts together (ONCE).
        # ------------------------------------------------------------------
        text_encoder = self._pipeline.text_encoder
        encoder_device = next(text_encoder.parameters()).device
        prompt_embeds = self._encode_one(prompt, encoder_device)
        concept_embeds = self._encode_concepts(concepts, encoder_device)
        prompt_embeds_full = torch.cat(
            [prompt_embeds, concept_embeds.to(prompt_embeds.dtype)], dim=1
        )                                                              # [1, L_txt + L_c, joint_dim]
        L_txt = prompt_embeds.shape[1]
        L_c = concept_embeds.shape[1]

        # ------------------------------------------------------------------
        # 2) Build text_ids that gives concept rows a zero position
        #    (matches the paper's concept_pe = 0). The model's
        #    `_prepare_text_ids` would otherwise number them L_txt..L_txt+L_c-1.
        # ------------------------------------------------------------------
        text_ids = torch.zeros(
            prompt_embeds_full.shape[0], L_txt + L_c, 4,
            dtype=torch.long, device=prompt_embeds_full.device,
        )
        text_ids[:, :L_txt, 3] = torch.arange(L_txt, device=text_ids.device)

        # ------------------------------------------------------------------
        # 3) Pre-build the attention mask. True = allowed.
        #    Two-way isolation between the concept side-channel and the
        #    rest of the joint stream:
        #      * non-concept queries ↛ concept keys  (image stream + prompt
        #        stream stay bit-identical to a vanilla forward).
        #      * concept queries ↛ prompt keys       (concept stream attends
        #        only to {concept, image}, matching the paper's separate-
        #        attention setup).
        # ------------------------------------------------------------------
        L_img = (width // 16) ** 2
        L_tot = L_txt + L_c + L_img
        allow = torch.ones(L_tot, L_tot, dtype=torch.bool, device=prompt_embeds_full.device)
        c_start, c_end = L_txt, L_txt + L_c
        allow[:c_start, c_start:c_end] = False        # prompt  → concept blocked
        allow[c_end:,  c_start:c_end] = False         # image   → concept blocked
        allow[c_start:c_end, :c_start] = False        # concept → prompt  blocked

        # ------------------------------------------------------------------
        # 4) Trace generate. Inside the trace we accumulate the per-(step,
        #    layer) concept-image score tensor in-place — no need to keep
        #    the full image/concept post-attention tensors around (which
        #    would be ~500 MB at L_img × D × n_layers × n_steps × fp16).
        #    Layer / timestep selection happens inline via cheap Python
        #    `if x not in indices: continue` filters.
        # ------------------------------------------------------------------
        L_img = (width // 16) ** 2  # already known
        acc_device = next(self._pipeline.transformer.parameters()).device
        timestep_set = set(timestep_indices)
        layer_set = set(layer_indices)

        with self.model.generate(
            prompt_embeds=prompt_embeds_full,
            attention_kwargs={"attention_mask": allow},
            width=width, height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
        ) as tracer:
            # Score accumulator on the transformer's first-param device,
            # fp32 for safe summation.
            score_acc = torch.zeros(
                prompt_embeds_full.shape[0], L_c, L_img,
                dtype=torch.float32, device=acc_device,
            ).save()
            # Optional debug-mode raw vector cache (only allocated when
            # cache_vectors=True so the default fast path stays small).
            img_steps = list().save() if cache_vectors else None
            concept_steps = list().save() if cache_vectors else None

            for step_idx, _step in enumerate(tracer.iter[:]):
                if step_idx not in timestep_set:
                    continue

                # Override the pipeline-derived txt_ids with our pre-built
                # version (concept rows zeroed). One intervention per step.
                t_env = self.model.transformer
                new_kwargs = dict(t_env.inputs[1])
                new_kwargs["txt_ids"] = text_ids.to(new_kwargs["txt_ids"].device)
                t_env.inputs = (t_env.inputs[0], new_kwargs)

                if cache_vectors:
                    step_img: list = list()
                    step_concept: list = list()

                for layer_idx, blk_env in enumerate(t_env.transformer_blocks):
                    # In Flux2AttnProcessor: to_add_out (encoder) is called
                    # BEFORE to_out[0] (image) — access in forward order.
                    enc_attn_pre = blk_env.attn.to_add_out.inputs[0][0]
                    img_attn_pre = blk_env.attn.to_out[0].inputs[0][0]
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
                    )                                                  # [B, C, L_img]
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
        # 5) Mean and reshape to heatmap grid.
        # ------------------------------------------------------------------
        n_accumulated = len(timestep_indices) * len(layer_indices)
        grid = width // 16
        heatmaps = (score_acc / n_accumulated).unflatten(-1, (grid, grid))
        heatmaps_np = heatmaps[0].cpu().numpy()                       # [C, H, W]

        if return_pil_heatmaps:
            heatmap_imgs = colorize_heatmaps(heatmaps_np, cmap=cmap, upscale=(width, height))
        else:
            heatmap_imgs = [PIL.Image.fromarray(np.uint8(hm * 255)) for hm in heatmaps_np]

        # If cache_vectors=True, stack the per-(step, layer) cached raw
        # vectors into [T_sel, L_sel, B, *, D] tensors for debugging.
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
