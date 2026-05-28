"""Shared base for concept-attention pipelines.

Both ConceptAttentionFlux2Pipeline (Qwen3 text encoder) and the FLUX.1
sibling (T5 + CLIP) share the same recipe:

  Step 1. Encode prompt + concepts into a single ``prompt_embeds_full``
          tensor, with one row per concept appended after the prompt's
          rows. (How "one row per concept" is computed differs per model
          — T5's first token vs Qwen3's chat-template position 3.)
  Step 2. Build an attention mask that achieves two-way isolation
          between the concept side-channel and the rest of the joint
          stream so the image stream stays bit-identical to a vanilla
          forward.
  Step 3. Trace ``model.generate(...)`` with the pre-built embeddings
          and mask passed through the pipeline's normal kwargs.
          Inside the trace, accumulate per-(step, layer) concept-image
          softmax scores in-place — never materialize the full
          [T, L, B, P, D] tensors.
  Step 4. Mean across selected (step, layer) and reshape to a grid.
  Step 5. Colorize and package the result.

Only Step 1 and a few cross-pipeline kwarg names actually differ
between the two model families. Subclasses override the small hooks
below; the orchestrating ``generate_image`` lives here.
"""

# nnsight MUST be imported before torch in this env or Python segfaults
# during nnsight's lazy-import binding setup.
from nnsight import DiffusionModel  # noqa: I001

import abc  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import torch  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402

from heatmap import colorize_heatmaps  # noqa: E402


# ---------------------------------------------------------------------------
# Public output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConceptAttentionOutput:
    image: PIL.Image.Image
    concept_heatmaps: list[PIL.Image.Image]
    image_vectors: torch.Tensor | None = None
    concept_vectors: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers — usable inside a trace because everything is torch ops on
# real tensors. No nnsight intervention semantics.
# ---------------------------------------------------------------------------


def build_attention_mask(
    L_txt: int, L_c: int, L_img: int, device: torch.device
) -> torch.Tensor:
    """Per-block joint-attention mask. ``True = allow attention``.

    Two-way isolation between the concept side-channel and the rest of
    the joint stream:

      * non-concept queries cannot attend to concept keys  -> image
        stream stays bit-identical to a vanilla forward.
      * concept  queries cannot attend to prompt  keys     -> concept
        stream attends only to {concept, image}, matching the paper's
        separate-attention setup.

    Shape: [L_tot, L_tot] (broadcasts across batch + heads inside SDPA).
    """
    L_tot = L_txt + L_c + L_img
    allow = torch.ones(L_tot, L_tot, dtype=torch.bool, device=device)
    c_start, c_end = L_txt, L_txt + L_c
    allow[:c_start, c_start:c_end] = False     # prompt  -> concept blocked
    allow[c_end:,  c_start:c_end] = False      # image   -> concept blocked
    allow[c_start:c_end, :c_start] = False     # concept -> prompt  blocked
    return allow


def score_concept_image(
    img_attn_pre: torch.Tensor,        # [B, L_img, D]
    concept_attn_pre: torch.Tensor,    # [B, L_c,   D]
    softmax: bool = True,
) -> torch.Tensor:
    """Per-(layer) concept-image score: ``einsum(image_vec, concept_vec)``.

    Both inputs are the PRE-projection attention outputs (i.e. the
    flattened SDPA output's image and concept rows respectively). They
    live in the same inner-dim space so the dot product is meaningful.

    Returns [B, num_concepts, num_image_patches]. Safe to call inside a
    trace — pure torch ops on real tensors.
    """
    scores = torch.einsum(
        "bpd,bcd->bcp",
        img_attn_pre.float(),
        concept_attn_pre.float(),
    )
    if softmax:
        scores = scores.softmax(dim=-2)
    return scores


def colorize_or_raw(
    heatmaps_np: np.ndarray,       # [C, H, W]
    cmap: str,
    upscale: tuple[int, int],
    return_pil: bool,
) -> list[PIL.Image.Image]:
    """Color-map and upscale (if return_pil) or just normalise."""
    if return_pil:
        return colorize_heatmaps(heatmaps_np, cmap=cmap, upscale=upscale)
    return [PIL.Image.fromarray(np.uint8(hm * 255)) for hm in heatmaps_np]


# ---------------------------------------------------------------------------
# Base pipeline
# ---------------------------------------------------------------------------


class ConceptAttentionPipeline(abc.ABC):
    """Shared orchestration for the FLUX.1 / FLUX.2 concept-attention pipelines.

    Subclasses implement model-specific hooks:

      * ``_encode_prompt_and_concepts`` — return the joint
        ``prompt_embeds_full`` + the model-specific extras (e.g. CLIP
        pooled embed for FLUX.1) + the prompt/concept lengths.
      * ``_pipeline_call_kwargs`` — pick the right pipeline kwargs
        (e.g. ``joint_attention_kwargs`` vs ``attention_kwargs``,
        whether to pass ``pooled_prompt_embeds``).
      * ``_per_step_intervention`` — inject any per-denoising-step
        overrides on the transformer's call (e.g. FLUX.2's ``txt_ids``).
      * ``_capture_block_attention`` — read the pre-projection image
        and encoder attention outputs in the correct forward-pass
        order for the model's attention processor.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        model_name: str,
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

    # ------------------------------------------------------------------ hooks
    @abc.abstractmethod
    def _encode_prompt_and_concepts(
        self,
        prompt: str,
        concepts: list[str],
    ) -> tuple[torch.Tensor, dict, int, int]:
        """Return ``(prompt_embeds_full, extra_pipeline_kwargs, L_txt, L_c)``.

        ``prompt_embeds_full`` is the concatenated [prompt | concepts]
        sequence embedding the pipeline will see. ``extra_pipeline_kwargs``
        bundles any other model-specific tensors the pipeline needs
        (e.g. ``pooled_prompt_embeds`` for FLUX.1).
        """
        ...

    @abc.abstractmethod
    def _pipeline_call_kwargs(
        self,
        prompt_embeds_full: torch.Tensor,
        extra: dict,
        attention_mask: torch.Tensor,
    ) -> dict:
        """Return the kwargs to pass to ``model.generate(**kwargs)``."""
        ...

    def _per_step_intervention(self, t_env, *, L_c: int) -> None:
        """Optional per-step transformer-input override. Defaults to no-op.

        Called inside the ``tracer.iter[:]`` loop with ``t_env`` being
        the nnsight Envoy for ``model.transformer``.
        """

    @abc.abstractmethod
    def _capture_block_attention(self, blk_env) -> tuple:
        """Return ``(img_attn_pre, encoder_attn_pre)`` from one double block.

        Both tensors are pre-projection (input to ``to_out[0]`` /
        ``to_add_out``). Access order matters for nnsight's one-shot
        hooks — the subclass must read them in forward-pass order
        (FLUX.1 calls ``to_out[0]`` before ``to_add_out``; FLUX.2 is
        the opposite).
        """
        ...

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
        **extra_generate_kwargs,
    ) -> ConceptAttentionOutput:
        assert width == height, "Square output only."
        assert width % 16 == 0, "Width must be divisible by 16 (patch_size * 2)."

        # ------------------------------------------------------------------
        # Step 1. Encode prompt + concepts into one joint embedding.
        # ------------------------------------------------------------------
        prompt_embeds_full, extra, L_txt, L_c = self._encode_prompt_and_concepts(
            prompt, concepts
        )

        # ------------------------------------------------------------------
        # Step 2. Build the joint-attention isolation mask.
        # ------------------------------------------------------------------
        L_img = (width // 16) ** 2
        mask = build_attention_mask(L_txt, L_c, L_img, prompt_embeds_full.device)

        # ------------------------------------------------------------------
        # Step 3. Trace generate. Inside the trace, accumulate the
        #         per-(step, layer) concept-image score in-place — no
        #         materialization of the full image/concept tensors.
        # ------------------------------------------------------------------
        n_blocks = len(self._pipeline.transformer.transformer_blocks)
        if layer_indices is None:
            layer_indices = list(range(n_blocks))
        if timestep_indices is None:
            timestep_indices = list(range(num_inference_steps))
        timestep_set = set(timestep_indices)
        layer_set = set(layer_indices)

        gen_kwargs = self._pipeline_call_kwargs(prompt_embeds_full, extra, mask)
        gen_kwargs.update(
            width=width, height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            **extra_generate_kwargs,
        )
        acc_device = next(self._pipeline.transformer.parameters()).device

        with self.model.generate(**gen_kwargs) as tracer:
            # Score accumulator on the transformer's device, fp32 for
            # safe summation across (step × layer × softmax) terms.
            score_acc = torch.zeros(
                prompt_embeds_full.shape[0], L_c, L_img,
                dtype=torch.float32, device=acc_device,
            ).save()
            # Optional raw-vector caches for debugging.
            img_steps = list().save() if cache_vectors else None
            concept_steps = list().save() if cache_vectors else None

            for step_idx, _step in enumerate(tracer.iter[:]):
                if step_idx not in timestep_set:
                    continue

                # Hook: override transformer kwargs if needed (FLUX.2 only).
                self._per_step_intervention(self.model.transformer, L_c=L_c)

                if cache_vectors:
                    step_img: list = list()
                    step_concept: list = list()

                for layer_idx, blk_env in enumerate(
                    self.model.transformer.transformer_blocks
                ):
                    img_attn_pre, enc_attn_pre = self._capture_block_attention(blk_env)
                    if layer_idx not in layer_set:
                        if cache_vectors:
                            step_concept.append(enc_attn_pre[:, L_txt:].clone())
                            step_img.append(img_attn_pre.clone())
                        continue

                    concept_pre = enc_attn_pre[:, L_txt:]
                    scores = score_concept_image(img_attn_pre, concept_pre, softmax)
                    score_acc.add_(scores.to(acc_device))

                    if cache_vectors:
                        step_concept.append(concept_pre.clone())
                        step_img.append(img_attn_pre.clone())

                if cache_vectors:
                    concept_steps.append(step_concept)
                    img_steps.append(step_img)

            result = tracer.result.save()

        # ------------------------------------------------------------------
        # Step 4. Mean across (step, layer), reshape to the patch grid.
        # ------------------------------------------------------------------
        n_accumulated = len(timestep_indices) * len(layer_indices)
        grid = width // 16
        heatmaps = (score_acc / n_accumulated).unflatten(-1, (grid, grid))
        heatmaps_np = heatmaps[0].cpu().numpy()                       # [C, H, W]

        # ------------------------------------------------------------------
        # Step 5. Colorize, package and return.
        # ------------------------------------------------------------------
        heatmap_imgs = colorize_or_raw(
            heatmaps_np, cmap=cmap, upscale=(width, height),
            return_pil=return_pil_heatmaps,
        )

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
