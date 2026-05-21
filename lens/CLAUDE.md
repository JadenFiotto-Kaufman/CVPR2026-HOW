# lens/

Logit lens for LLaVA-1.5-7b, implemented on top of `nnsight`. Re-implementation
of the `llava-interp` paper's lens utility (Neo et al., 2024,
[arXiv:2410.07149](https://arxiv.org/abs/2410.07149)). Produces a single
self-contained interactive HTML per image that exposes layer-wise next-token
predictions over the model's residual stream — including the 576 image tokens.

## Layout

- `logit_lens.py` — everything: nnsight extraction + HTML render + CLI.
- `images/` — input images (committed for reproducibility).
- `outputs/` — generated HTML lenses (committed as artifacts).
- `llava-interp/` — original reference repo, **gitignored**. Read-only; do
  not import from it (we are intentionally standalone).

## Environment

`nn6` conda env. Python at `/disk/u/jadenfk/miniconda3/envs/nn6/bin/python`.
nnsight is installed from `/disk/u/jadenfk/wd/nnsight` (dev install, 0.7.x).
transformers 5.x is required for the current model paths.

## Methodology

### Logit lens

For a decoder-only transformer the residual stream at layer L is `h_L`. The
final-layer prediction is `lm_head(ln_f(h_final))`. The logit lens applies
that same head to every intermediate layer:

```
preds_L = softmax(lm_head(ln_f(h_L)))      for L = 0, 1, ..., 31
```

Early layers tend to produce noisy multilingual subword tokens (e.g.
"Portail", "archivi"); middle layers track syntactic role; the final layer
converges on the actual next-token distribution. We extract the top-5 tokens
and probabilities at every (layer, position) and dump them into the HTML.

### Why nnsight

The reference repo runs a full forward pass with `output_hidden_states=True`
and post-processes the returned hidden-state tuple. nnsight gives the same
information through a single `model.trace(prompt, images=[image])` block
without the bookkeeping — `block.output` on each decoder layer returns the
residual tensor directly (transformers >= 5), so the per-layer logits are
just `model.lm_head(model.model.language_model.norm(block.output))` inside
the trace context. The `.save()` calls keep the topk results past the trace
exit. See `compute_logit_lens` in `logit_lens.py`.

### Model layout (LLaVA-1.5-7b-hf, transformers 5.x)

```
LlavaForConditionalGeneration
└── model: LlavaModel
    ├── vision_tower: CLIPVisionModel  (336×336, 14px patches → 576 tokens)
    ├── multi_modal_projector
    └── language_model: LlamaModel
        ├── embed_tokens
        ├── layers[0..31]    ← residual stream taps
        └── norm             ← final RMSNorm
└── lm_head                  ← unembedding (vocab 32064)
```

nnsight paths used in `compute_logit_lens`:

| What | Path |
|---|---|
| Residual stream at layer L | `model.model.language_model.layers[L].output` |
| Final norm | `model.model.language_model.norm` |
| Unembedding | `model.lm_head` |

### Image-token expansion

The prompt contains a single `<image>` placeholder (token id `32000`). The
processor expands it to 576 image tokens before the model sees it, so the
runtime sequence length is `len(text_tokens) - 1 + 576`. We mirror that
expansion when building the token labels — the single `<image>` label becomes
`<IMG001>..<IMG576>` so each row of the lens table maps to a known patch.
A runtime check asserts `len(token_labels) == model_seq_len`.

Constants (currently hardcoded for llava-1.5):
- `IMG_TOKEN_ID = 32000`
- `DEFAULT_IMAGE_SIZE = 336`, `DEFAULT_PATCH_SIZE = 14` → 576 tokens, 24×24 grid

## HTML viewer features

All client-side, no server, no external CDNs. `data` is a `[layer][pos][rank]`
JSON array of `[token_str, "p.4f"]` pairs embedded in the page.

### 1. Image-patch hover widget (top of left column)

The original `llava-interp` widget. Hover the input image → red bounding box
on the hovered patch + tooltip with that patch's top-5 at layer 0 + auto-scroll
the corresponding row of the lens table into view. Click to lock; click image
or table to unlock.

### 2. Lens table (right column)

Sticky-header table, one row per token position, one column per layer. Each
cell shows the layer's top-1 token; hovering reveals the full top-5 with
probabilities.

### 3. Segmentation widget (bottom of left column)

Second copy of the image with a `<canvas>` overlay. Each of the 576 patches
is filled with a semi-transparent color (alpha 0.8) keyed by the top-1 token
at a slider-selected layer.

- **Layer slider** — picks which layer sources the per-patch labels.
- **Min-p slider** (default 0.10) — patches with top-1 prob below this go into
  a synthetic `<EMPTY>` token bucket that always renders white. Useful for
  filtering noisy early-layer predictions.
- **Colors** are deterministic per token string (djb2 hash → HSLA), so
  sliding through layers refines the segmentation rather than reshuffling.
  `colorOverrides[token]` can replace any color via the legend swatch
  picker; `<EMPTY>` is exempt.
- **Hover a patch** → same effect as the top widget (tooltip with that
  layer's top-5, red box on the top image, table row scrolled into view).
  Respects `isLocked`.

### 4. Segmentation legend

Scrollable list below the slider showing every unique top-1 token at the
current layer, sorted by patch count desc. Format: `[swatch] "token" (N)`.

- **Click a swatch** → native `<input type="color">` picker. Chosen color is
  stored as `rgba(..., 0.8)` (alpha preserved) and persists across layer
  changes via `colorOverrides`.
- **Hover a row** → red perimeter drawn around every patch whose effective
  token (post-threshold) matches. Algorithm: for each in-set patch, draw a
  line on each of its 4 edges where the neighbor is out-of-set or off-grid.
  Shared interior edges are never drawn, so each connected blob ends up
  with a single clean outline (no overdraw, no bbox).

## Running

```bash
/disk/u/jadenfk/miniconda3/envs/nn6/bin/python lens/logit_lens.py \
  --image-folder lens/images \
  --save-folder  lens/outputs \
  --num-images   1
```

Optional flags: `--model-id`, `--device` (default `cuda:0`), `--prompt`
(default `USER: <image>\nDescribe the image. ASSISTANT:`), `--top-k`
(default 5).

Loading the model is the long part (~5 s); per-image compute is ~3–4 s.
Output HTML is ~2 MB (1.5 MB lens-data JSON + 0.5 MB base64 image, twice).

## Gotchas

- **`<image>` placeholder is model-specific.** `IMG_TOKEN_ID = 32000` is
  llava-1.5 only. Other VLMs use different ids / placeholder strings and a
  different image-token count. The hard sequence-length assertion in
  `compute_logit_lens` will trip if you swap models without updating these.
- **transformers >= 5.x required.** Older versions wrap decoder block
  outputs in tuples; we index `block.output` directly assuming a tensor.
  If you see "tuple has no attribute" errors deep in the trace, that's why.
- **HTML grows with seq length.** 576 × 32 × 5 entries already produces a
  2 MB file; multiplying any of these (longer prompts, deeper models,
  larger top-k) will scale linearly.
- **Layer slider hover state.** When hovering a legend row, the per-layer
  count cache is recomputed on every render. Cheap at 576 patches but
  worth knowing if you scale up.
- **HTML template is one string literal inside `logit_lens.py`.** Edits to
  the viewer mean editing `_HTML_TEMPLATE` — there is no separate template
  file (intentional: keeps the project to a single source file).
