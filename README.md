# CVPR 2026 HOW Workshop — companion code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JadenFiotto-Kaufman/CVPR2026-HOW/blob/master/CVPR2026-HOW.ipynb)

Hands-on materials for the *"Vision Interpretability with `nnsight`"* talk at
the **CVPR 2026 HOW Workshop**. The notebook walks through three vision
interpretability methods — re-implemented end-to-end on top of
[`nnsight`](https://nnsight.net) — and this repo holds the same code as
runnable standalone CLIs, one directory per section.

The Colab notebook is the canonical entry point. Each directory here is the
same code, broken out so you can run a single technique without the notebook
scaffolding.

## Repo layout

```
.
├── CVPR2026-HOW.ipynb           ← talk notebook (open in Colab ↑)
├── CVPR2026-HOW.py              ← notebook source (Colab .py format)
├── 1_Attention_Ablation/        ← Section 1
├── 2_Concept_Attention/         ← Section 2
├── 3_VLM_Lens/                  ← Section 3
└── requirements.txt
```

Each numbered directory follows the same convention: a `__main__.py` with the
high-level orchestration (load → trace → save) and a few helper modules with
the implementation details.

## The three demos

### 1. Cross-attention ablation on Stable Diffusion 1.4

![attention ablation](docs/section1.png)

SD 1.4's UNet has 16 cross-attention layers — the only places the text prompt
directly steers the image stream. We open a `model.generate(...)` trace, walk
the denoising steps with `tracer.iter[:]`, and zero the output of whichever
cross-attentions we want to ablate. Side-by-side shows what each one was
contributing.

```bash
python 1_Attention_Ablation/__main__.py --prompt "Starry Night" --layers-to-ablate 5
```

### 2. Concept Attention on FLUX.1 / FLUX.2

![concept attention](docs/section2.png)

Re-implementation of [**ConceptAttention** (Helbling et al., CVPR 2025)](https://arxiv.org/abs/2502.04320)
on top of `nnsight`. Concept tokens are appended to the encoder hidden
states; a custom attention mask isolates them two-ways (image ↛ concept,
prompt ↛ concept, concept ↛ prompt) so they ride along on the diffusion
forward pass without changing the image. Per-block attention scores between
the image stream and the concept tokens are accumulated in-trace and
colorised into heatmaps.

```bash
python 2_Concept_Attention/__main__.py \
    --prompt "A cat in a park on the grass by a tree" \
    --concepts cat grass sky tree
```

Defaults to FLUX.2-klein-4B (`--model flux2`); pass `--model flux1` for
FLUX.1-schnell.

### 3. VLM logit lens on LLaVA-1.5

![vlm logit lens](docs/section3.png)

Re-implementation of [**Towards Interpreting Visual Information Processing
in Vision-Language Models** (Neo et al., 2024)](https://arxiv.org/abs/2410.07149).
Applies the final norm + lm_head to the residual stream at every intermediate
decoder layer, including the 576 image-token positions, and dumps the result
as a single self-contained interactive HTML — hover any patch to see its
per-layer top-5 next-token predictions, slide between layers to watch the
segmentation refine.

```bash
python 3_VLM_Lens/__main__.py --image-folder 3_VLM_Lens/images --save-folder ./out
```

## Setup

```bash
pip install -r requirements.txt
```

`nnsight` works locally (small models) or as a thin client against a remote
NDIF deployment (large models like FLUX.2-klein-4B and LLaVA-1.5-7B). The
Colab notebook configures NDIF automatically; for local CLI runs, point
`CONFIG.API.HOST` at your NDIF instance or remove the `remote=True` /
`session(remote=True)` calls to run the full model in-process.

## Citations

- Helbling et al. *ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features.* CVPR 2025. [arXiv:2502.04320](https://arxiv.org/abs/2502.04320)
- Neo et al. *Towards Interpreting Visual Information Processing in Vision-Language Models.* 2024. [arXiv:2410.07149](https://arxiv.org/abs/2410.07149)
- `nnsight` and the NDIF backend: [nnsight.net](https://nnsight.net)
