"""External concept-stream evolution for FLUX.2's double-stream blocks.

We capture per-block tensors during a normal nnsight trace and then re-run
the concept stream's math here, without modifying the model. The concept
stream uses the same projections as the model's text stream
(`add_q_proj`, `add_k_proj`, `add_v_proj`, `norm_added_q`, `norm_added_k`,
`norm1_context`, `norm2_context`, `to_add_out`, `ff_context`), and
participates in a joint concept-image attention against the captured image
keys/values. Concepts evolve through layers via their own residual; they
never feed back into the image stream — which is exactly what the original
ModifiedDoubleStreamBlock does (cf. CVPR2026-HOW/concept-attention/...).

This module is pure PyTorch — no nnsight dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BlockCapture:
    """Tensors captured from one Flux2TransformerBlock during one denoising step."""

    # Block-level inputs (read off `blk.inputs`).
    hidden_states: torch.Tensor          # [B, L_img, D]   image residual stream pre-block
    encoder_hidden_states: torch.Tensor  # [B, L_txt, D]   text residual stream pre-block
    temb_mod_txt: torch.Tensor           # [B, 6*D]        Modulation params for the text stream
    # Joint-stream RoPE — diffusers' Flux2 passes a (cos, sin) tuple of shape
    # ([L_txt + L_img, head_dim], [L_txt + L_img, head_dim]). We slice off the
    # image portion to rotate img_k for the concept-image attention.
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor]

    # Captured directly from the model's attention submodules.
    img_attn_out: torch.Tensor           # [B, L_img, D]   joint-attn img portion, PRE `to_out[0]`
                                         #                  (lives in same inner-dim space as
                                         #                  the externally-computed concept attn)
    img_k: torch.Tensor                  # [B, L_img, D]   to_k(norm_h) pre-unflatten/norm
    img_v: torch.Tensor                  # [B, L_img, D]   to_v(norm_h)


def _split_modulation(temb_mod: torch.Tensor, dim: int) -> tuple:
    """Mirrors diffusers' `Flux2Modulation.split(temb_mod, 2)`.

    temb_mod is layout `[B, 6 * dim]` for a 2-set block — six params per
    set (shift, scale, gate) × 2. Returns ((shift_msa, scale_msa, gate_msa),
    (shift_mlp, scale_mlp, gate_mlp)), each with shape [B, 1, dim].
    """
    # diffusers actually does temb_mod.unflatten(-1, (2, 3, dim)).split(...).
    six = temb_mod.unflatten(-1, (2, 3, dim))  # [B, 2, 3, D]
    out = []
    for set_idx in range(six.shape[-3]):
        s = six[..., set_idx, :, :]              # [B, 3, D]
        shift, scale, gate = s[..., 0, :], s[..., 1, :], s[..., 2, :]
        out.append((shift.unsqueeze(-2), scale.unsqueeze(-2), gate.unsqueeze(-2)))
    return out[0], out[1]


def _apply_rope(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Diffusers `apply_rotary_emb` for FLUX.2 (real-valued, (cos, sin) tuple).

    `x` shape: [B, L, H, D] (when sequence_dim=1). `freqs` = (cos, sin) each
    [L, D]. Same math as diffusers/models/embeddings.py:apply_rotary_emb with
    use_real=True, use_real_unbind_dim=-1.
    """
    cos, sin = freqs
    # Reshape cos/sin to broadcast: [..., L, 1, D] when sequence_dim=1.
    shape = [1] * x.dim()
    shape[sequence_dim] = cos.shape[0]
    shape[-1] = cos.shape[-1]
    cos = cos.reshape(shape).to(x.dtype)
    sin = sin.reshape(shape).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    x = x.float() * torch.rsqrt(var + eps)
    if weight is not None:
        x = x * weight.float()
    return x.to(weight.dtype if weight is not None else torch.float32)


def evolve_concept_stream(
    blocks: list[torch.nn.Module],         # diffusers' Flux2TransformerBlock list
    captures: list[BlockCapture],          # one entry per block (this timestep)
    concept_state: torch.Tensor,           # [B, L_c, D] — running concept residual
    *,
    apply_rope: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Evolve `concept_state` through every block at one denoising step.

    Returns:
        new_concept_state: evolved concept residual after the last block.
        concept_output_vectors: per-layer post-attention concept tensors
            (pre-gate, post `to_add_out`). Each is [B, L_c, D].
        image_output_vectors: per-layer captured image-attention outputs
            (the model's actual `to_out[0]` result, pre-gate). Each is
            [B, L_img, D]. These pair with concept_output_vectors for the
            heatmap einsum downstream.

    Implementation mirrors `Flux2TransformerBlock.forward` + the txt path of
    `Flux2AttnProcessor.__call__`. Concept stream uses the same submodules
    as the text stream but consumes its own concept tokens.
    """
    concept_outs: list[torch.Tensor] = []
    image_outs: list[torch.Tensor] = []

    for blk, cap in zip(blocks, captures):
        attn = blk.attn
        D = attn.inner_dim
        H = attn.heads
        d_head = attn.head_dim

        # Modulation for text stream (concepts reuse text modulation by convention).
        ((c_shift_msa, c_scale_msa, c_gate_msa),
         (c_shift_mlp, c_scale_mlp, c_gate_mlp)) = _split_modulation(cap.temb_mod_txt, D)

        # 1) norm1_context + modulation
        norm_c = blk.norm1_context(concept_state)
        norm_c = (1 + c_scale_msa) * norm_c + c_shift_msa

        # 2) add_*_proj -> q/k/v for concepts, then norm_added_*
        c_q = attn.add_q_proj(norm_c).unflatten(-1, (H, d_head))   # [B, L_c, H, D]
        c_k = attn.add_k_proj(norm_c).unflatten(-1, (H, d_head))
        c_v = attn.add_v_proj(norm_c).unflatten(-1, (H, d_head))
        c_q = attn.norm_added_q(c_q)
        c_k = attn.norm_added_k(c_k)

        # 3) Image keys/values from captured img_k, img_v + per-head shape + norm_k
        img_k = cap.img_k.unflatten(-1, (H, d_head))               # [B, L_img, H, D]
        img_v = cap.img_v.unflatten(-1, (H, d_head))
        img_k = attn.norm_k(img_k)

        # 4) Apply rotary embedding only on the image portion. Concepts have
        #    no spatial position; the original modified block also leaves a
        #    concept_pe of zeros, equivalent to skipping rope for concepts.
        if apply_rope:
            # image_rotary_emb is (cos, sin) for the joint [text|image] stream;
            # slice off the image portion (last L_img positions).
            cos_full, sin_full = cap.image_rotary_emb
            L_img = img_k.shape[1]
            img_freqs = (cos_full[-L_img:], sin_full[-L_img:])
            img_k = _apply_rope(img_k, img_freqs, sequence_dim=1)

        # 5) Joint concept-image attention via SDPA on the concatenated stream.
        q = torch.cat([c_q, c_q.new_zeros(c_q.shape[0], 0, H, d_head)], dim=1)  # placeholder
        q = c_q  # concept-only queries
        k = torch.cat([c_k, img_k], dim=1)
        v = torch.cat([c_v, img_v], dim=1)
        # Permute to [B, H, L, D] for SDPA.
        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k, v))
        attn_out = F.scaled_dot_product_attention(q_, k_, v_)       # [B, H, L_c, D]
        attn_out = attn_out.transpose(1, 2).flatten(2, 3)            # [B, L_c, D]

        # 6) Cache PRE-projection attention output for heatmap einsum (same
        #    inner-dim space as cap.img_attn_out, which is also pre-projection).
        concept_outs.append(attn_out)
        image_outs.append(cap.img_attn_out)

        # 7) Apply output projection for the concept residual update.
        c_attn_proj = attn.to_add_out(attn_out.to(attn.to_add_out.weight.dtype))
        concept_state = concept_state + c_gate_msa * c_attn_proj
        norm_c2 = blk.norm2_context(concept_state)
        norm_c2 = norm_c2 * (1 + c_scale_mlp) + c_shift_mlp
        ff_out = blk.ff_context(norm_c2)
        concept_state = concept_state + c_gate_mlp * ff_out
        if concept_state.dtype == torch.float16:
            concept_state = concept_state.clip(-65504, 65504)

    return concept_state, concept_outs, image_outs


def compute_heatmaps(
    image_vectors: torch.Tensor,    # [T, L, B, num_img, D]
    concept_vectors: torch.Tensor,  # [T, L, B, num_concepts, D]
    layer_indices: list[int],
    timestep_indices: list[int],
    *,
    softmax: bool = True,
    grid: int = 64,
) -> torch.Tensor:
    """Einsum image @ concept^T -> [B, num_concepts, grid, grid]."""
    image_vectors = image_vectors[timestep_indices][:, layer_indices]
    concept_vectors = concept_vectors[timestep_indices][:, layer_indices]
    scores = torch.einsum("tlbpd,tlbcd->tlbcp", image_vectors, concept_vectors)
    if softmax:
        scores = scores.softmax(dim=-2)
    scores = scores.mean(dim=(0, 1))                          # [B, C, P]
    return scores.unflatten(-1, (grid, grid))                 # [B, C, H, W]
