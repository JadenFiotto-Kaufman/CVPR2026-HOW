"""Logit lens for LLaVA-1.5 via nnsight.

Re-implementation of `llava-interp/scripts/logit_lens/create_logit_lens.py`
using nnsight's VisionLanguageModel instead of forward hooks + full
`output_hidden_states` extraction. Per-layer residual stream -> final norm ->
lm_head -> top-k softmax, rendered as an interactive HTML grid (image on the
left, hoverable per-layer top-5 cells per token position on the right).
"""

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from nnsight import VisionLanguageModel


# ---------------------------------------------------------------------------
# Lens computation
# ---------------------------------------------------------------------------

IMG_TOKEN_ID = 32000  # llava-hf/llava-1.5-7b-hf image placeholder id
DEFAULT_IMAGE_SIZE = 336
DEFAULT_PATCH_SIZE = 14


def compute_logit_lens(model, image, prompt, top_k=5):
    """Run one forward pass and return per-layer top-k tokens for every position.

    Returns
    -------
    token_labels : list[str]
        One label per position in the model's input sequence. The single
        `<image>` placeholder is expanded to `<IMG001>...<IMG576>`.
    all_top_tokens : list[list[list[(str, str)]]]
        Indexed as `all_top_tokens[layer][position]` -> list of
        `(token_str, "%.4f" % prob)` pairs of length `top_k`.
    """
    tokenizer = model.tokenizer
    layers = model.model.language_model.layers
    norm = model.model.language_model.norm
    lm_head = model.lm_head

    img_patch_count = (DEFAULT_IMAGE_SIZE // DEFAULT_PATCH_SIZE) ** 2  # 576

    raw_ids = tokenizer.encode(prompt)
    token_labels = []
    for tok_id in raw_ids:
        if tok_id == IMG_TOKEN_ID:
            token_labels.extend(
                [f"<IMG{(i + 1):03d}>" for i in range(img_patch_count)]
            )
        else:
            token_labels.append(tokenizer.decode([tok_id]))

    saved = []
    with model.trace(prompt, images=[image]):
        for layer in layers:
            hs = layer.output  # tensor in transformers>=5
            logits = lm_head(norm(hs))
            probs = logits.softmax(dim=-1)
            top = probs.topk(k=top_k, dim=-1)
            saved.append((top.values.save(), top.indices.save()))

    seq_len = saved[0][1].shape[1]
    if seq_len != len(token_labels):
        raise RuntimeError(
            f"Token-label length {len(token_labels)} does not match model "
            f"sequence length {seq_len}. Check IMG_TOKEN_ID / patch count."
        )

    all_top_tokens = []
    for values, indices in saved:
        layer_tokens = []
        for pos in range(seq_len):
            pairs = [
                (tokenizer.decode(idx.item()), f"{p.item():.4f}")
                for idx, p in zip(indices[0, pos], values[0, pos])
            ]
            layer_tokens.append(pairs)
        all_top_tokens.append(layer_tokens)

    return token_labels, all_top_tokens


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Logit Lens</title>
    <style>
        body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
        .container { display: flex; }
        .image-container {
            flex: 0 0 auto;
            margin: 20px;
            position: relative;
            width: __IMAGE_SIZE__px;
        }
        .highlight-box {
            position: absolute;
            border: 2px solid red;
            pointer-events: none;
            display: none;
        }
        .table-container {
            flex: 1 1 auto;
            position: relative;
            max-height: 90vh;
            overflow: auto;
            margin: 20px;
        }
        table { border-collapse: separate; border-spacing: 0; }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            min-width: 80px;
        }
        th { background-color: #f2f2f2; font-weight: bold; }
        .corner-header { position: sticky; top: 0; left: 0; z-index: 3; background-color: #f2f2f2; }
        .row-header   { position: sticky; left: 0; z-index: 2; background-color: #f2f2f2; }
        .col-header   { position: sticky; top: 0;  z-index: 1; background-color: #f2f2f2; }
        #tooltip {
            display: none; position: fixed; background: white;
            border: 1px solid black; padding: 5px; z-index: 1000;
            pointer-events: none; max-width: 300px; font-size: 14px;
        }
        .highlighted-row { background-color: #ffff99; }
        .image-info { margin-top: 10px; font-size: 14px; width: 100%; word-wrap: break-word; }
        .prompt { font-weight: bold; margin-bottom: 5px; }
        .instructions { font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="image-container">
            <img src="data:image/png;base64,__IMAGE_B64__" alt="Input Image"
                 style="width: __IMAGE_SIZE__px; height: __IMAGE_SIZE__px;">
            <div class="highlight-box"></div>
            <div class="image-info">
                <p class="prompt">Prompt: "__PROMPT__"</p>
                <p class="instructions">Instructions: Click on image to lock the patch, click on image/table to unlock</p>
                <p>Info: __MISC__</p>
            </div>
        </div>
        <div class="table-container">
            <table id="logitLens"></table>
        </div>
    </div>
    <div id="tooltip"></div>
<script>
    const data = __DATA__;
    const tokenLabels = __TOKEN_LABELS__;
    const tooltip = document.getElementById('tooltip');
    const highlightBox = document.querySelector('.highlight-box');
    const image = document.querySelector('.image-container img');
    const table = document.getElementById('logitLens');

    const imageSize = __IMAGE_SIZE__;
    const patchSize = __PATCH_SIZE__;
    const gridSize = imageSize / patchSize;

    let isLocked = false;
    let highlightedRow = null;
    let lockedPatchIndex = null;

    const cornerHeader = table.createTHead().insertRow();
    cornerHeader.insertCell().textContent = 'Token/Layer';
    cornerHeader.cells[0].classList.add('corner-header');
    for (let i = 0; i < data.length; i++) {
        const th = document.createElement('th');
        th.textContent = `Layer ${i + 1}`;
        th.classList.add('col-header');
        cornerHeader.appendChild(th);
    }

    for (let pos = 0; pos < tokenLabels.length; pos++) {
        const row = table.insertRow();
        const rowHeader = row.insertCell();
        rowHeader.textContent = tokenLabels[pos];
        rowHeader.classList.add('row-header');
        for (let layer = 0; layer < data.length; layer++) {
            const cell = row.insertCell();
            cell.textContent = data[layer][pos][0][0];
            cell.addEventListener('mouseover', (e) => { if (!isLocked) showTooltip(e, layer, pos, false); });
            cell.addEventListener('mousemove', updateTooltipPosition);
            cell.addEventListener('mouseout', () => { if (!isLocked) hideTooltip(); });
        }
    }

    function showTooltip(e, layer, pos, shouldScroll = false) {
        tooltip.innerHTML = data[layer][pos].map(([t, p]) => `${t}: ${p}`).join('<br>');
        tooltip.style.display = 'block';
        updateTooltipPosition(e);
        if (tokenLabels[pos].startsWith('<IMG')) {
            const patchIndex = parseInt(tokenLabels[pos].slice(4, 7));
            highlightImagePatch(patchIndex);
            highlightTableRow(pos, shouldScroll);
        } else {
            highlightBox.style.display = 'none';
            unhighlightTableRow();
        }
    }
    function hideTooltip() {
        tooltip.style.display = 'none';
        if (!isLocked) { highlightBox.style.display = 'none'; unhighlightTableRow(); }
    }
    function updateTooltipPosition(e) {
        const r = tooltip.getBoundingClientRect();
        let x = e.clientX + 10, y = e.clientY + 10;
        if (x + r.width  > window.innerWidth)  x = e.clientX - r.width  - 10;
        if (y + r.height > window.innerHeight) y = e.clientY - r.height - 10;
        tooltip.style.left = Math.max(0, x) + 'px';
        tooltip.style.top  = Math.max(0, y) + 'px';
    }
    function highlightImagePatch(patchIndex) {
        const s = image.width / imageSize;
        const row = Math.floor((patchIndex - 1) / gridSize);
        const col = (patchIndex - 1) % gridSize;
        highlightBox.style.left   = `${col * patchSize * s}px`;
        highlightBox.style.top    = `${row * patchSize * s}px`;
        highlightBox.style.width  = `${patchSize * s}px`;
        highlightBox.style.height = `${patchSize * s}px`;
        highlightBox.style.display = 'block';
    }
    function highlightTableRow(rowIndex, shouldScroll = false) {
        if (highlightedRow) highlightedRow.classList.remove('highlighted-row');
        highlightedRow = table.rows[rowIndex + 1];
        highlightedRow.classList.add('highlighted-row');
        if (shouldScroll) highlightedRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    function unhighlightTableRow() {
        if (highlightedRow) { highlightedRow.classList.remove('highlighted-row'); highlightedRow = null; }
    }
    image.addEventListener('mousemove', (e) => {
        if (!isLocked) {
            const patchIndex = getPatchIndexFromMouseEvent(e);
            highlightImagePatch(patchIndex);
            const tokenIndex = getTokenIndexFromPatchIndex(patchIndex);
            if (tokenIndex !== -1) showTooltip(e, 0, tokenIndex, true);
        }
    });
    image.addEventListener('mouseout', () => { if (!isLocked) hideTooltip(); });
    image.addEventListener('click', (e) => {
        isLocked = !isLocked;
        if (isLocked) {
            lockedPatchIndex = getPatchIndexFromMouseEvent(e);
            highlightImagePatch(lockedPatchIndex);
            const tokenIndex = getTokenIndexFromPatchIndex(lockedPatchIndex);
            if (tokenIndex !== -1) highlightTableRow(tokenIndex, true);
        } else { lockedPatchIndex = null; hideTooltip(); }
    });
    table.addEventListener('click', () => { isLocked = false; lockedPatchIndex = null; hideTooltip(); });

    function getPatchIndexFromMouseEvent(e) {
        const r = image.getBoundingClientRect();
        const x = e.clientX - r.left, y = e.clientY - r.top;
        const px = Math.floor(x / (image.width  / gridSize));
        const py = Math.floor(y / (image.height / gridSize));
        return py * gridSize + px + 1;
    }
    function getTokenIndexFromPatchIndex(patchIndex) {
        return tokenLabels.findIndex(l => l === `<IMG${patchIndex.toString().padStart(3, '0')}>`);
    }
</script>
</body>
</html>
"""


def _image_to_base64(image, size):
    w, h = image.size
    m = min(w, h)
    cropped = image.crop(((w - m) / 2, (h - m) / 2, (w + m) / 2, (h + m) / 2))
    resized = cropped.resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    resized.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_html(
    token_labels,
    all_top_tokens,
    image,
    model_name,
    image_filename,
    prompt,
    save_folder,
    image_size=DEFAULT_IMAGE_SIZE,
    patch_size=DEFAULT_PATCH_SIZE,
    misc_text="",
):
    img_b64 = _image_to_base64(image, image_size)
    html = (
        _HTML_TEMPLATE
        .replace("__IMAGE_B64__", img_b64)
        .replace("__DATA__", json.dumps(all_top_tokens))
        .replace("__TOKEN_LABELS__", json.dumps(token_labels))
        .replace("__IMAGE_SIZE__", str(image_size))
        .replace("__PATCH_SIZE__", str(patch_size))
        .replace("__PROMPT__", prompt)
        .replace("__MISC__", misc_text)
    )

    out_path = Path(save_folder) / f"{model_name}_{Path(image_filename).stem}_logit_lens.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_IMG_EXTS = (".jpeg", ".jpg", ".png")


def _list_images(folder, limit=None):
    files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(_IMG_EXTS)
    )
    return files[:limit] if limit else files


def main():
    ap = argparse.ArgumentParser(description="LLaVA-1.5 logit lens via nnsight")
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-images", type=int, default=None)
    ap.add_argument("--prompt", default="USER: <image>\nDescribe the image. ASSISTANT:")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    model = VisionLanguageModel(
        args.model_id,
        device_map=args.device,
        dispatch=True,
        torch_dtype=torch.float16,
    )
    model_name = args.model_id.split("/")[-1]

    files = _list_images(args.image_folder, args.num_images)
    for fname in tqdm(files):
        path = os.path.join(args.image_folder, fname)
        try:
            image = Image.open(path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Skipping {path}: {e}")
            continue

        labels, lens = compute_logit_lens(model, image, args.prompt, top_k=args.top_k)
        render_html(
            labels, lens, image, model_name, path, args.prompt, args.save_folder
        )


if __name__ == "__main__":
    main()
