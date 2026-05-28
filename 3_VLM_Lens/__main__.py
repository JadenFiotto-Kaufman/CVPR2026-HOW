"""LLaVA-1.5 logit lens viewer.

Usage:
    python 3_VLM_Lens/__main__.py --image-folder ./images --save-folder ./out
    python 3_VLM_Lens/__main__.py --image-folder ./images --save-folder ./out --num-images 3
"""

# nnsight must be imported before torch; reversing the order segfaults.
from compute import compute_logit_lens
from render_html import render_html
from nnsight import VisionLanguageModel

import argparse
import os

import torch
from PIL import Image
from tqdm import tqdm


_IMG_EXTS = (".jpeg", ".jpg", ".png")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-images", type=int, default=None)
    ap.add_argument("--prompt", default="USER: <image>\nDescribe the image. ASSISTANT:")
    ap.add_argument("--top-k", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load LLaVA-1.5 wrapped for tracing.
    model = VisionLanguageModel(
        args.model_id,
        device_map=args.device,
        dispatch=True,
        torch_dtype=torch.float16,
    )
    model_name = args.model_id.split("/")[-1]

    # 2. Enumerate input images (sorted, optionally capped).
    files = sorted(f for f in os.listdir(args.image_folder) if f.lower().endswith(_IMG_EXTS))
    if args.num_images:
        files = files[: args.num_images]

    # 3. For each image: one trace -> per-layer top-k -> one HTML viewer.
    for fname in tqdm(files):
        path = os.path.join(args.image_folder, fname)
        try:
            image = Image.open(path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Skipping {path}: {e}")
            continue

        labels, lens = compute_logit_lens(model, image, args.prompt, top_k=args.top_k)
        render_html(labels, lens, image, model_name, path, args.prompt, args.save_folder)


if __name__ == "__main__":
    main()
