"""Logit lens extraction for LLaVA-1.5 via nnsight.

For a decoder-only LM the logit lens applies the final norm + lm_head to the
residual stream at every intermediate layer:

    preds_L = softmax(lm_head(ln_f(h_L)))    for L = 0..N-1

We run one `model.trace(prompt, images=[image])`, save the per-layer top-k
inside the trace, and expand the single `<image>` placeholder into
`<IMG001>..<IMG576>` so each table row maps to a known patch.
"""

# nnsight must be imported before torch; reversing the order segfaults.
from nnsight import VisionLanguageModel  # noqa: F401 — re-exported for callers


IMG_TOKEN_ID = 32000  # llava-hf/llava-1.5-7b-hf image placeholder id
DEFAULT_IMAGE_SIZE = 336
DEFAULT_PATCH_SIZE = 14


def _expanded_token_labels(tokenizer, prompt, num_image_tokens):
    raw_ids = tokenizer.encode(prompt)
    labels = []
    for tok_id in raw_ids:
        if tok_id == IMG_TOKEN_ID:
            labels.extend(f"<IMG{(i + 1):03d}>" for i in range(num_image_tokens))
        else:
            labels.append(tokenizer.decode([tok_id]))
    return labels


def compute_logit_lens(model, image, prompt, top_k=5,
                       image_size=DEFAULT_IMAGE_SIZE,
                       patch_size=DEFAULT_PATCH_SIZE,
                       remote=False):
    """One forward pass -> per-layer top-k tokens for every sequence position.

    Set ``remote=True`` to run the trace on NDIF (model must have been
    loaded with ``dispatch=False``).

    Returns
    -------
    token_labels : list[str]
    all_top_tokens : list[list[list[(str, str)]]]  # [layer][pos] -> [(tok, "p.4f"), ...]
    """
    tokenizer = model.tokenizer
    layers = model.model.language_model.layers
    norm = model.model.language_model.norm
    lm_head = model.lm_head

    num_image_tokens = (image_size // patch_size) ** 2  # 576
    token_labels = _expanded_token_labels(tokenizer, prompt, num_image_tokens)

    # ``list().save()`` is the trace-graph-friendly accumulator — works
    # whether the trace runs locally or is shipped to NDIF. A plain
    # Python list would be a client-side object that the remote backend
    # never sees, so it would come back empty.
    with model.trace(prompt, images=[image], remote=remote):
        all_values = list().save()
        all_indices = list().save()
        for layer in layers:
            hs = layer.output  # tensor in transformers>=5
            probs = lm_head(norm(hs)).softmax(dim=-1)
            top = probs.topk(k=top_k, dim=-1)
            all_values.append(top.values)
            all_indices.append(top.indices)

    seq_len = all_indices[0].shape[1]
    if seq_len != len(token_labels):
        raise RuntimeError(
            f"Token-label length {len(token_labels)} != model sequence length "
            f"{seq_len}. Check IMG_TOKEN_ID / patch count."
        )

    all_top_tokens = []
    for values, indices in zip(all_values, all_indices):
        layer_tokens = []
        for pos in range(seq_len):
            layer_tokens.append([
                (tokenizer.decode(idx.item()), f"{p.item():.4f}")
                for idx, p in zip(indices[0, pos], values[0, pos])
            ])
        all_top_tokens.append(layer_tokens)

    return token_labels, all_top_tokens
