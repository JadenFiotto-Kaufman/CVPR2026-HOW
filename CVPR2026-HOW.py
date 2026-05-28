# -*- coding: utf-8 -*-
"""# CVPR 2026 HOW Workshop — Toward a Shared Stack for Vision Interpretability

Companion notebook for the talk at the **CVPR 2026 HOW Workshop**.

**Contents**

- **Section 1 — Attention ablation (a tour of `nnsight`)**
  What does each cross-attention layer in Stable Diffusion actually
  contribute? Zero one (or several) and watch the image change. The
  walkthrough doubles as the `nnsight` API intro — `.trace()`,
  `.input` / `.output`, `.save()`, `tracer.iter[:]`.
- **Section 2 — Concept attention (using NDIF)**
  Interpretable per-concept heatmaps for FLUX.2. The whole 4B-parameter
  pipeline runs on NDIF — your Colab only does tokenisation and a
  handful of tiny tensor ops.
- **Section 3 — VLM logit lens (using Workbench)**
  Read per-layer next-token predictions out of LLaVA, including over the
  576 image-patch positions, and render an interactive heatmap.
"""

"""## Install

`nnsight` from the `dev` branch (this notebook uses APIs newer than the
last PyPI release). `diffusers` / `transformers` / `accelerate` are
needed for the model wrappers and pipelines.
"""

# Commented-out shell commands for non-Colab linters; uncomment to run.
from IPython.display import clear_output, display

# !pip install -q git+https://github.com/ndif-team/nnsight.git@dev
# !pip install -q diffusers transformers accelerate

clear_output()

"""# Section 1 — Attention ablation (a tour of `nnsight`)

Stable Diffusion 1.4's UNet has **16 cross-attention layers**
(every `transformer_block.attn2`) sprinkled across its down-blocks,
mid-block, and up-blocks. Cross-attention is the only place where the
text prompt directly steers the image stream — so a natural
interpretability question is: *what does each individual cross-attention
layer contribute to the final image?*

The recipe with `nnsight`: open a `model.generate(...)` trace, iterate
over the denoising steps with `tracer.iter[:]`, and inside each step
zero the output of whichever cross-attention layers you want to ablate.
The image stream's forward pass everywhere else is untouched.

This is also the first time `nnsight` shows up in the notebook, so the
walkthrough below introduces every part of the core API as it gets used.

The library reduces to one core pattern:

1. Wrap any PyTorch model with `Envoy` (or, for tighter HuggingFace
   integration, use one of the model-family wrappers — `LanguageModel`,
   `VisionLanguageModel`, `DiffusionModel`, ...).
2. Open a `with model.trace(input):` (or `model.generate(input):`) block.
3. Inside the block, write normal Python that **references** the
   activations you want to read or change — accessing `module.output`,
   `module.input`, doing arithmetic on them, slicing them, replacing
   them. Use `.save()` on any variable to access it after the with block.
4. Exit the block. The model executes, your interventions are applied,
   and saved activations are accessible as real torch tensors.

Key surface:

| What | Returns |
|---|---|
| `module.output` | The output of a module's forward pass. |
| `module.input`  | The first positional arg the module receives. |
| `module.inputs` | A `(args, kwargs)` pair — what the module is *actually* called with. |
| `.save()`       | Keeps a value alive past the trace exit. Without it, values get filtered out on cleanup. |

Original work: [github.com/JadenFiotto-Kaufman/thesis](https://github.com/JadenFiotto-Kaufman/thesis).
"""

"""## Load the model

`DiffusionModel(...)` wraps a HuggingFace `DiffusionPipeline` so every
sub-module (the UNet, its attention blocks, the text encoder, ...) is
accessible as an `Envoy` — the object you reference inside a trace to
read activations or install interventions. `dispatch=True` materializes
the weights on the chosen device right now; `dispatch=False` would keep
them on `meta` for remote execution against NDIF (see Section 2).
"""

import torch
import matplotlib.pyplot as plt
from nnsight import DiffusionModel

sd = DiffusionModel(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None,
    dispatch=True,
    device_map="cuda",
)
clear_output()

PROMPT = "Starry Night"
SEED = 43
NUM_INFERENCE_STEPS = 50

"""## Baseline (no ablation)

Generate the reference image first so we have something to diff
against. `with sd.generate(prompt) as tracer:` is the diffusion analogue
of `model.trace(...)` — open the block, write Python that references
activations or final outputs, exit. `tracer.result` is the pipeline's
own return value; calling `.save()` on it keeps it accessible after the
block exits (without `.save()`, intermediates are cleaned up).
"""

with sd.generate(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED) as tracer:
    baseline = tracer.result.save()

baseline_image = baseline.images[0]
display(baseline_image.resize((256, 256)))

"""## List the cross-attention layers

`named_modules` returns every sub-Envoy in the UNet; we keep the ones
whose path ends in `.attn2` (cross-attention), sorted into a stable
forward-pass-friendly order (down → mid → up).
"""

cross_attentions = sorted(
    [
        (name, envoy)
        for name, envoy in sd.unet.named_modules()
        if name.endswith(".attn2")
    ],
    key=lambda x: x[0],
)
cross_attention_envoys = [envoy for _, envoy in cross_attentions]

for i, (name, _) in enumerate(cross_attentions):
    print(f"  [{i:2d}] {name}")

"""## Ablate the chosen layers

Edit `LAYERS_TO_ABLATE` below — each index corresponds to a row in the
list above. Defaults to `[5]`, which on the `"Starry Night"` prompt
strips the painterly Van-Gogh association out of the generation and
leaves a generic star-filled night sky behind. No other single cross-
attention layer has this effect — suggesting layer 5 is where the
specific *painting* (vs. the literal concept "stars at night")
gets bound.

Three pieces of `nnsight` come together here:

* `tracer.iter[:]` — diffusion runs the UNet once per denoising step,
  so we need an intervention that fires *every step*, not just on the
  first forward pass.
* `envoy.to_out[0].input` — `.input` is the tensor passed *into* a
  module's forward. `to_out[0]` is the cross-attention's output linear
  projection, so its input is the post-SDPA, pre-projection activation.
* `... [:] = 0` — slice-assignment on a captured tensor is an
  intervention: downstream code (the projection and everything after)
  sees zeros. The model still runs and attention is still computed; its
  result just doesn't get added back into the image stream.
"""

LAYERS_TO_ABLATE = [5]  # Edit me — e.g. [0, 7, 15] to ablate several at once.

with sd.generate(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED) as tracer:
    for _step in tracer.iter[:]:
        # Sort to access in forward order — nnsight's one-shot hooks
        # are order-sensitive within a single forward pass.
        for idx in sorted(LAYERS_TO_ABLATE):
            cross_attention_envoys[idx].to_out[0].input[:] = 0
    ablated = tracer.result.save()

ablated_image = ablated.images[0]

# Side-by-side comparison.
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(baseline_image)
axes[0].set_title("Baseline")
axes[0].axis("off")
axes[1].imshow(ablated_image)
axes[1].set_title(f"Ablated layers {LAYERS_TO_ABLATE}")
axes[1].axis("off")
plt.tight_layout()
plt.show()

"""# Section 2 — Concept attention (remote on NDIF)

The model in this section is **FLUX.2-klein-4B** — too big to fit in a
Colab T4's 16 GB. Instead of loading weights locally, we instantiate
it with `dispatch=False`, which gives us only the meta-tensor skeleton
(the module graph + names) needed to write `nnsight` interventions.
The actual forward pass runs on **NDIF**, the workshop-hosted
inference cluster, by passing `remote=True` to each `.trace()` /
`.generate()` call.

**The technique** ([ConceptAttention, Helbling et al. CVPR 2025](https://arxiv.org/abs/2502.04320)):
augment the joint text+image attention with a third "concept" stream.
Encode a list of concept words (e.g. `["cat", "grass", "sky", "tree"]`)
through the same text encoder as the prompt, splice their per-token
embeddings into `encoder_hidden_states`, and install an attention mask
that two-way-isolates them:

  * non-concept queries cannot attend to concept keys  → the image
    stream stays bit-identical to a vanilla forward (the concepts are
    invisible to it).
  * concept queries cannot attend to prompt  keys      → the concept
    stream attends only to {concept, image}, matching the paper's
    separate-attention setup.

For each double-stream block, take the inner product
`einsum(image_post_attn, concept_post_attn)`, softmax across concepts,
and average over a few selected layers + timesteps. The result is a
per-concept spatial heatmap that tells you which patches "belong to"
each concept.

All of the steps below run remotely — the Colab kernel only does
tokenisation, a handful of tensor concats, and the final colorisation.
"""

import os
from nnsight import CONFIG

# Point at the workshop's NDIF host. No API key needed for the CVPR
# deployment; the cell uses the env var if set, otherwise falls back to
# a local cloudflare tunnel (replace with the URL from the talk).
CONFIG.API.HOST = os.environ.get("NDIF_HOST", "http://localhost:5001")
print(f"using NDIF at {CONFIG.API.HOST}")

# `dispatch=False` -> weights stay on meta locally; the full model lives
# on NDIF. The module tree + tokenizer still load eagerly so we can
# tokenize and write interventions against the right paths.
flux = DiffusionModel("black-forest-labs/FLUX.2-klein-4B", dispatch=False)
clear_output()

PROMPT_2 = "A cat in a park on the grass by a tree"
CONCEPTS = ["cat", "grass", "sky", "tree"]
NUM_INFERENCE_STEPS_2 = 4
SEED_2 = 0

"""## Encode prompt + concepts on NDIF

We need the joint `[prompt | concepts]` sequence embedding the
transformer will receive. A `model.session(remote=True)` ships an
arbitrary block of Python to NDIF as a single payload — inside it,
we just call `flux.pipeline.encode_prompt(...)` directly, the same
way we would locally, but the underlying text-encoder forward runs
remotely. No diffusion forward, no inner `.trace(...)` per text —
just five calls to the pipeline's own encoder helper.

For Qwen3 chat-templated inputs the actual concept word lives at
position 3 (positions 0..2 are `<|im_start|>user\\n`, 4..12 are the
chat-suffix, 13..511 are padding), so we slice that one position out
of each concept's encoding.
"""

with flux.session(remote=True):
    # Encode the prompt — return value is (prompt_embeds, text_ids).
    prompt_embeds = flux.pipeline.encode_prompt(
        prompt=PROMPT_2,
        device="cuda",
        num_images_per_prompt=1,
    )[0].save()

    # Encode each concept and keep just the position-3 (concept word) row.
    concept_rows = list().save()
    for c in CONCEPTS:
        emb = flux.pipeline.encode_prompt(
            prompt=c,
            device="cuda",
            num_images_per_prompt=1,
        )[0]
        concept_rows.append(emb[:, 3:4, :])

print("prompt_embeds shape: ", tuple(prompt_embeds.shape))
print(
    "concept rows:        ", len(concept_rows), "× shape", tuple(concept_rows[0].shape)
)

concept_embeds = torch.cat(concept_rows, dim=1)  # [1, L_c, D]

# Splice: prompt first, concepts at the tail of the encoder sequence.
prompt_embeds_full = torch.cat(
    [prompt_embeds, concept_embeds.to(prompt_embeds.dtype)],
    dim=1,
)
L_txt = prompt_embeds.shape[1]
L_c = concept_embeds.shape[1]
L_img = (1024 // 16) ** 2  # 64 × 64 = 4096

"""## Build attention mask + zero-position txt_ids

The two off-diagonal blocks of the mask are what makes this work — see
the markdown above for what each one buys us.

FLUX.2's `_prepare_text_ids` would otherwise number concept positions
1..L_c on the RoPE L axis. The paper specifies `concept_pe = 0`, so
we pre-build the txt_ids with concept rows zeroed and override the
pipeline-generated version per denoising step.
"""

allow = torch.ones(L_txt + L_c + L_img, L_txt + L_c + L_img, dtype=torch.bool)
c_start, c_end = L_txt, L_txt + L_c
allow[:c_start, c_start:c_end] = False  # prompt  → concept blocked
allow[c_end:, c_start:c_end] = False  # image   → concept blocked
allow[c_start:c_end, :c_start] = False  # concept → prompt  blocked

text_ids = torch.zeros(1, L_txt + L_c, 4, dtype=torch.long)
text_ids[:, :L_txt, 3] = torch.arange(L_txt)

"""## Trace the generation, capture per-block scores

Inside the trace we accumulate the per-(step, layer) concept-image
softmax score in-place — `score_acc` is a saved tensor we add to at
every double block at every selected denoising step.

The accumulator never materialises the full
`[T, L, B, num_patches, D]` activation tensor; it stays at
`[1, num_concepts, num_image_patches]` regardless of how many steps or
layers we average over.
"""

with flux.generate(
    prompt_embeds=prompt_embeds_full,
    attention_kwargs={"attention_mask": allow},
    width=1024,
    height=1024,
    num_inference_steps=NUM_INFERENCE_STEPS_2,
    seed=SEED_2,
    remote=True,
) as tracer:
    # CPU accumulator — works whether captured tensors come back from
    # NDIF on cuda:0 (remote GPU) or end up local; we just `.cpu()` each
    # score before adding. Pay one device-transfer per (step × layer),
    # not per residual.
    score_acc = torch.zeros(1, L_c, L_img, dtype=torch.float32).save()

    for _step in tracer.iter[:]:
        # Override the pipeline-derived txt_ids with our zero-position version.
        new_kwargs = dict(flux.transformer.inputs[1])
        new_kwargs["txt_ids"] = text_ids
        flux.transformer.inputs = (flux.transformer.inputs[0], new_kwargs)

        # Capture per-block PRE-projection attention outputs in forward order:
        # `to_add_out` (encoder) is called BEFORE `to_out[0]` (image) in
        # Flux2AttnProcessor — access in that order for nnsight's one-shot hooks.
        for blk in flux.transformer.transformer_blocks:
            enc_attn_pre = blk.attn.to_add_out.inputs[0][0]  # [1, L_txt+L_c, D]
            img_attn_pre = blk.attn.to_out[0].inputs[0][0]  # [1, L_img,     D]
            concept_pre = enc_attn_pre[:, L_txt:]  # [1, L_c, D]
            scores = torch.einsum(
                "bpd,bcd->bcp",
                img_attn_pre.float(),
                concept_pre.float(),
            ).softmax(
                dim=-2
            )  # [1, L_c, L_img]
            score_acc.add_(scores.cpu())

    result = tracer.result.save()

"""## Display the image and per-concept heatmaps"""

import numpy as np

n_blocks = len(flux.transformer.transformer_blocks)
n_accumulated = NUM_INFERENCE_STEPS_2 * n_blocks
grid = 1024 // 16
heatmaps = (score_acc / n_accumulated).unflatten(-1, (grid, grid))[0].cpu().numpy()
image = result.images[0]

fig, axes = plt.subplots(1, len(CONCEPTS) + 1, figsize=(4 * (len(CONCEPTS) + 1), 4))
axes[0].imshow(image)
axes[0].set_title("Generated image")
axes[0].axis("off")
for i, (c, hm) in enumerate(zip(CONCEPTS, heatmaps)):
    # Resize heatmap to image resolution by nearest-neighbour for crispness.
    hm_resized = np.kron(hm, np.ones((1024 // grid, 1024 // grid)))
    axes[i + 1].imshow(image)
    axes[i + 1].imshow(hm_resized, cmap="plasma", alpha=0.55)
    axes[i + 1].set_title(c)
    axes[i + 1].axis("off")
plt.tight_layout()
plt.show()

"""# Section 3 — VLM logit lens (and Workbench)

The previous two sections show how cheap it is to add interpretability
ops to existing PyTorch pipelines with `nnsight` + NDIF. But each
research-quality visualisation still wants its own UI: click-through
patches, layer sliders, color legends, the lot. Re-implementing those
per paper is exactly the duplicated-effort the talk argues against.

**Workbench** ([github.com/ndif-team/workbench](https://github.com/ndif-team/workbench))
is an interpretability UI built directly on `nnsight` + NDIF. Each
"tool" is a thin backend route that runs the trace on NDIF plus a
React widget that renders the result. Adding a tool is on the order
of: write the trace in nnsight, define the result schema, drop in a
visualization component. The talk includes a live demo of adding the
**VLM Logit Lens** tool — a per-layer next-token decoder over the
LLaVA image-patch positions — to the workbench in under an hour.

> **Follow along live:** open the CVPR-deployed Workbench at
> `WORKBENCH_URL` (replace with the URL from the talk), pick
> "VLM Logit Lens" in the workspace sidebar, upload an image, and
> watch the heatmaps stream in.

The rest of this section reproduces the lens technique inline so you
can run it on Colab without leaving the notebook — same model
(LLaVA-1.5-7b-hf), same trace shape, just rendered with matplotlib
instead of the polished React widget.
"""

# Set this from the URL the talk provides for the workshop instance.
WORKBENCH_URL = "https://workbench.ndif.us"  # FIXME: workshop URL

"""## The technique

LLaVA-1.5 is a decoder-only Llama backbone fed image-patch tokens
(produced by a CLIP vision tower + small projector) interleaved with
the text prompt. The **logit lens** trick: at each of the 32 decoder
layers, apply the model's final RMSNorm + unembedding (`lm_head`) to
the residual stream as if it were the last layer, and read off the
top-1 token at every position.

For the image-token positions specifically, this becomes a 24 × 24
grid (576 patches) of "what would the model say each patch is, if we
stopped at layer L?". Plot it and you get a crude but interpretable
per-patch semantic map per layer.

The whole thing is a single `model.trace(prompt, images=[image])`
call: capture each layer's `output`, feed it through `norm` and
`lm_head` inside the same trace (calling wrapped modules as functions
inside a trace dispatches to their `forward()` and bypasses nnsight's
interleaving hooks — i.e. it's just the linear math we want), and
save the top-1 token id per (layer, position).

Original work: [Towards Interpreting Visual Information Processing in
Vision-Language Models, Neo et al. 2024](https://arxiv.org/abs/2410.07149).
"""

import requests
import io
import PIL.Image
from nnsight import VisionLanguageModel

# Same NDIF host as Section 2 — set in the env or rewritten here.
CONFIG.API.HOST = os.environ.get("NDIF_HOST", "http://localhost:5001")
print(f"using NDIF at {CONFIG.API.HOST}")

llava = VisionLanguageModel("llava-hf/llava-1.5-7b-hf", dispatch=False)
clear_output()

"""## Load the demo image

Reproducible URL pointing at the same image we use throughout the
talk's accompanying code (`lens/images/img.jpg` in this repo).
"""

IMAGE_URL = "https://raw.githubusercontent.com/JadenFiotto-Kaufman/CVPR2026-HOW/master/lens/images/img.jpg"
PROMPT_3 = "USER: <image>\nDescribe the image. ASSISTANT:"

image_3 = PIL.Image.open(
    io.BytesIO(requests.get(IMAGE_URL, timeout=10).content)
).convert("RGB")
display(image_3.resize((256, 256)))

"""## Trace and capture top-1 tokens per (layer, position)

We capture just the argmax (the top-1 token id) at each layer × each
position rather than the full top-k probabilities — keeps the wire
payload from NDIF small (~32 × seq_len ints instead of ~32 × seq_len ×
32064 floats).
"""

with llava.trace(PROMPT_3, images=[image_3], remote=True) as tracer:
    top1_per_layer = list().save()
    for layer in llava.model.language_model.layers:
        # `layer.output` is the residual stream after this block.
        # `model.lm_head(model.model.language_model.norm(...))` is the same
        # final readout the model itself does at the end.
        logits = llava.lm_head(llava.model.language_model.norm(layer.output))
        top1_per_layer.append(logits.argmax(dim=-1))  # [B, seq]

print(f"captured {len(top1_per_layer)} layers")
print(f"per-layer top1 shape: {tuple(top1_per_layer[0].shape)}")

"""## Build per-position labels

The processor expands the single `<image>` placeholder into 576 image
tokens. We mirror that expansion so each row of the lens table is
labelled either as a text token or as `<IMGxxx>` for one of the 576
patches.
"""

IMG_TOKEN_ID = 32000
IMAGE_GRID = 24  # 24 × 24 = 576 patches
NUM_IMAGE_TOKENS = IMAGE_GRID * IMAGE_GRID

tokenizer = llava.tokenizer
input_ids = tokenizer.encode(PROMPT_3)
position_labels: list[str] = []
for tok_id in input_ids:
    if tok_id == IMG_TOKEN_ID:
        position_labels.extend([f"<IMG{(i + 1):03d}>" for i in range(NUM_IMAGE_TOKENS)])
    else:
        position_labels.append(tokenizer.decode([tok_id]))

assert (
    len(position_labels) == top1_per_layer[0].shape[1]
), f"label count {len(position_labels)} != seq len {top1_per_layer[0].shape[1]}"
print(
    f"sequence length: {len(position_labels)} (== 576 image tokens + {len(position_labels) - NUM_IMAGE_TOKENS} text tokens)"
)

"""## Compact lens table — text tokens only

Show what the model "decides" at each text-token position across a
sample of layers. The last-position prediction is the actual next
token the model would emit; earlier positions show what's at each
text token's slot.
"""

import pandas as pd

text_positions = [
    i for i, lbl in enumerate(position_labels) if not lbl.startswith("<IMG")
]
sample_layers = list(range(0, 32, 4)) + [31]  # every 4th + the final
rows = []
for pos in text_positions:
    row = {"position": pos, "token": repr(position_labels[pos])}
    for L in sample_layers:
        row[f"L{L}"] = repr(tokenizer.decode([top1_per_layer[L][0, pos].item()]))
    rows.append(row)
pd.set_option("display.max_colwidth", 30)
display(pd.DataFrame(rows))

"""## Per-patch segmentation at a chosen layer

Color the 24 × 24 grid by the top-1 token at each patch — same colour
for the same predicted token. Overlay on the image so you can see
which patches the model lumps together semantically at this layer.
Edit `SEG_LAYER` to scroll through the depth of the model.
"""

import hashlib
import matplotlib.colors

SEG_LAYER = 22  # Works best for this prompt


def token_color(token: str) -> tuple[float, float, float, float]:
    """Deterministic HSL-ish color per token string."""
    h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    hue = (h % 360) / 360.0
    return matplotlib.colors.hsv_to_rgb((hue, 0.7, 0.95)).tolist() + [0.6]


img_positions = [i for i, lbl in enumerate(position_labels) if lbl.startswith("<IMG")]
patch_tokens = [
    tokenizer.decode([top1_per_layer[SEG_LAYER][0, p].item()]) for p in img_positions
]
unique_tokens = sorted(set(patch_tokens), key=patch_tokens.count, reverse=True)[:12]
color_lookup = {t: token_color(t) for t in unique_tokens}

# Build a [grid, grid, 4] RGBA overlay; uncoloured patches are transparent.
overlay = np.zeros((IMAGE_GRID, IMAGE_GRID, 4))
for idx, tok in enumerate(patch_tokens):
    if tok in color_lookup:
        r, c = divmod(idx, IMAGE_GRID)
        overlay[r, c] = color_lookup[tok]

# Resize overlay to the image's display size — must be a multiple of
# IMAGE_GRID (24) so each patch is an integer number of pixels.
PATCH_PX = 21  # → 504 × 504 display
disp_size = IMAGE_GRID * PATCH_PX
overlay_resized = np.kron(overlay, np.ones((PATCH_PX, PATCH_PX, 1)))
image_resized = image_3.resize((disp_size, disp_size))

fig, (ax_img, ax_legend) = plt.subplots(
    1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]}
)
ax_img.imshow(image_resized)
ax_img.imshow(overlay_resized)
ax_img.set_title(f"Layer {SEG_LAYER} top-1 token per patch")
ax_img.axis("off")

# Legend: top-N tokens at this layer + their counts.
counts = {t: patch_tokens.count(t) for t in unique_tokens}
ax_legend.axis("off")
ax_legend.set_title("Top tokens (count)")
for i, tok in enumerate(unique_tokens):
    ax_legend.add_patch(
        plt.Rectangle(
            (0, i), 0.15, 0.7, color=color_lookup[tok], transform=ax_legend.transData
        )
    )
    ax_legend.text(
        0.2,
        i + 0.35,
        f"{tok!r:>12} ({counts[tok]:>3})",
        va="center",
        family="monospace",
        transform=ax_legend.transData,
    )
ax_legend.set_xlim(0, 2)
ax_legend.set_ylim(-1, len(unique_tokens) + 1)
ax_legend.invert_yaxis()
plt.tight_layout()
plt.show()

"""## Where to go from here

What we just did inline — one trace, capture per-layer top-1s, render
two views — is exactly the data path the full Workbench VLM Lens
tool runs. The Workbench instance adds:

  * the full top-5 (probabilities) per cell, surfaced via tooltip
  * a layer slider and a min-probability threshold slider for the
    segmentation widget
  * a click-to-recolour swatch per token in the legend
  * red bounding-outlines around all blobs of a hovered legend entry
  * click-to-lock-patch + auto-scroll the table to the locked row

…all of which is just a React widget on top of the same NDIF-served
trace. If you build a new vision interp method, that's the
template — wire the trace in `nnsight`, ship the result through
NDIF, render with whatever UI fits your audience.
"""
