"""Concept-attention for FLUX.2 via nnsight, mask-based implementation.

We extend the model's joint stream with extra "concept" tokens spliced into
`encoder_hidden_states` (with zero `txt_ids` so concepts have no positional
encoding). A per-block attention mask in `joint_attention_kwargs` blocks
every non-concept query from attending to concept keys — so the image and
prompt streams' attention outputs are bitwise identical to a vanilla
forward, while concept queries can still attend to the image. We read
both concept and image post-attention vectors out of the same forward
pass and einsum them into per-concept heatmaps.

This replaces the earlier capture-and-replay implementation
(see git history): no external math, all weights used in their native
context, image stream provably unperturbed.
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
    # Raw cached vectors (only if cache_vectors=True). Shapes:
    #   image_vectors:   [T, L, B, num_image_patches, D]
    #   concept_vectors: [T, L, B, num_concepts,     D]
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
    def _encode_concepts(self, concepts: list[str], target_device: torch.device) -> torch.Tensor:
        """Encode concept strings through the pipeline's text encoder, one row per concept.

        Mean-pools the Qwen tokens for each concept word. Returns
        [1, num_concepts, joint_attention_dim] on `target_device`.
        """
        pipe = self._pipeline
        rows: list[torch.Tensor] = []
        for concept in concepts:
            out = pipe.encode_prompt(prompt=concept, device=target_device, num_images_per_prompt=1)
            emb = out[0] if isinstance(out, tuple) else out
            rows.append(emb.mean(dim=1, keepdim=True))
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
        # 1) Pre-compute concept embeddings via the pipeline's text encoder.
        #    Returned at joint_attention_dim (pre `context_embedder`) — we
        #    splice them into encoder_hidden_states BEFORE the transformer
        #    runs, so the model's own context_embedder projects both prompt
        #    and concept rows together in the same forward.
        # ------------------------------------------------------------------
        text_encoder = self._pipeline.text_encoder
        encoder_device = next(text_encoder.parameters()).device
        concept_embeds = self._encode_concepts(concepts, encoder_device)  # [1, L_c, joint_dim]
        L_c = concept_embeds.shape[1]

        # ------------------------------------------------------------------
        # 2) Trace generate. Each denoising step: splice concept embeddings
        #    into encoder_hidden_states + txt_ids, install attention mask
        #    that blocks non-concept queries from attending to concept keys,
        #    capture pre-projection attention outputs for image and concept
        #    streams from every double block.
        # ------------------------------------------------------------------
        with self.model.generate(
            prompt,
            width=width, height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
        ) as tracer:
            img_steps = list().save()       # T × L of [B, L_img, D]
            concept_steps = list().save()   # T × L of [B, L_c,   D]

            for _step in tracer.iter[:]:
                t_env = self.model.transformer

                # Pull originals, build extended versions.
                orig_kwargs = t_env.inputs[1]
                eh   = orig_kwargs["encoder_hidden_states"]
                hs   = orig_kwargs["hidden_states"]
                tids = orig_kwargs["txt_ids"]
                L_txt = eh.shape[1]
                L_img = hs.shape[1]
                L_tot = L_txt + L_c + L_img

                # Extend encoder_hidden_states with the concept rows.
                cembs = concept_embeds.to(eh.device, dtype=eh.dtype)
                new_eh = torch.cat([eh, cembs], dim=1)

                # Extend txt_ids with zeros (concepts have no spatial position).
                if tids.ndim == 3:
                    zero_ids = tids.new_zeros(tids.shape[0], L_c, tids.shape[-1])
                    new_tids = torch.cat([tids, zero_ids], dim=1)
                else:
                    zero_ids = tids.new_zeros(L_c, tids.shape[-1])
                    new_tids = torch.cat([tids, zero_ids], dim=0)

                # Attention mask: True = allowed.
                # Two-way isolation between the concept side-channel and the
                # rest of the joint stream, matching the original paper's
                # SEPARATE-attention setup (text+image joint AND a parallel
                # concept+image joint):
                #   * non-concept queries ↛ concept keys  (image stream stays
                #     vanilla; prompt unaffected by concepts).
                #   * concept queries ↛ prompt keys       (concept stream
                #     attends only to {concept, image}, just like the paper's
                #     ModifiedDoubleStreamBlock).
                allow = torch.ones(L_tot, L_tot, dtype=torch.bool, device=eh.device)
                c_start, c_end = L_txt, L_txt + L_c
                allow[:c_start, c_start:c_end] = False     # prompt  → concept blocked
                allow[c_end:,  c_start:c_end] = False      # image   → concept blocked
                allow[c_start:c_end, :c_start] = False     # concept → prompt  blocked

                # Inject into joint_attention_kwargs (propagates to every block).
                jak = dict(orig_kwargs.get("joint_attention_kwargs") or {})
                jak["attention_mask"] = allow

                new_kwargs = dict(orig_kwargs)
                new_kwargs["encoder_hidden_states"] = new_eh
                new_kwargs["txt_ids"] = new_tids
                new_kwargs["joint_attention_kwargs"] = jak

                t_env.inputs = (t_env.inputs[0], new_kwargs)

                # ------------------------------------------------------------
                # Capture per-block PRE-projection attention outputs.
                # In Flux2AttnProcessor, to_add_out (encoder) is called BEFORE
                # to_out[0] (image), so access them in that order.
                # ------------------------------------------------------------
                step_img: list = list()
                step_concept: list = list()
                for blk_env in t_env.transformer_blocks:
                    enc_attn_pre = blk_env.attn.to_add_out.inputs[0][0]   # [B, L_txt+L_c, D]
                    img_attn_pre = blk_env.attn.to_out[0].inputs[0][0]    # [B, L_img,     D]
                    concept_pre = enc_attn_pre[:, L_txt:].clone()         # [B, L_c, D]
                    step_concept.append(concept_pre)
                    step_img.append(img_attn_pre.clone())

                concept_steps.append(step_concept)
                img_steps.append(step_img)

            result = tracer.result.save()

        # ------------------------------------------------------------------
        # 3) Stack to [T, L, B, *, D], compute heatmaps.
        # ------------------------------------------------------------------
        image_tensor = torch.stack(
            [torch.stack(layer_outs, dim=0) for layer_outs in img_steps], dim=0
        )
        concept_tensor = torch.stack(
            [torch.stack(layer_outs, dim=0) for layer_outs in concept_steps], dim=0
        )

        # einsum image_vec . concept_vec across the D axis; result is
        # [B, num_concepts, num_image_patches]. Average over selected
        # timesteps + layers, optionally softmax over concepts per patch.
        img_sel = image_tensor[timestep_indices][:, layer_indices]
        con_sel = concept_tensor[timestep_indices][:, layer_indices]
        scores = torch.einsum("tlbpd,tlbcd->tlbcp", img_sel, con_sel)
        if softmax:
            scores = scores.softmax(dim=-2)
        scores = scores.mean(dim=(0, 1))                                  # [B, C, P]
        grid = width // 16
        heatmaps = scores.unflatten(-1, (grid, grid))                     # [B, C, H, W]
        heatmaps_np = heatmaps[0].to(torch.float32).cpu().numpy()         # [C, H, W]

        if return_pil_heatmaps:
            heatmap_imgs = colorize_heatmaps(heatmaps_np, cmap=cmap, upscale=(width, height))
        else:
            heatmap_imgs = [PIL.Image.fromarray(np.uint8(hm * 255)) for hm in heatmaps_np]

        return ConceptAttentionOutput(
            image=result.images[0],
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
                "L_c": L_c,
            },
        )
