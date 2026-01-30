"""InsertAnything inference wrapper for OSInsert (libcom-style).

This is migrated from `osinsert/inference.py` so that all required logic lives
inside `libcom/os_insert` and does not depend on external repositories.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

from .ia_utils import (
    get_bbox_from_mask,
    expand_bbox,
    pad_to_square,
    box2squre,
    crop_back,
    expand_image_mask,
)


device = torch.device("cuda")
dtype = torch.bfloat16
size = (768, 768)


# ---------------------------------------------------------------------------
# 模型路径：对齐 libcom 的 `pretrained_models/` 目录约定。
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_ROOT = REPO_ROOT / "pretrained_models"

FLUX_FILL_PATH = Path(
    os.getenv(
        "FLUX_FILL_PATH",
        PRETRAINED_ROOT / "flux" / "FLUX.1-Fill-dev",
    )
)
FLUX_REDUX_PATH = Path(
    os.getenv(
        "FLUX_REDUX_PATH",
        PRETRAINED_ROOT / "flux" / "FLUX.1-Redux-dev",
    )
)
INSERTANYTHING_LORA_PATH = Path(
    os.getenv(
        "IA_LORA_PATH",
        PRETRAINED_ROOT
        / "insert_anything"
        / "20250321_steps5000_pytorch_lora_weights.safetensors",
    )
)


# Load the pre-trained model and LoRA weights
pipe = FluxFillPipeline.from_pretrained(
    str(FLUX_FILL_PATH),
    torch_dtype=dtype,
)
pipe.load_lora_weights(str(INSERTANYTHING_LORA_PATH))
redux = FluxPriorReduxPipeline.from_pretrained(str(FLUX_REDUX_PATH)).to(dtype=dtype)

# 26GB 显存配置
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
redux.enable_model_cpu_offload()


def run_insertanything(
    source_image_path: str,
    mask_image_path: str,
    ref_image_path: str,
    ref_mask_path: str,
    seeds=123,
    strength: float | None = None,
    save_path: str = "./result",
    filename_suffix: str = "",
):
    """Single-image InsertAnything inference following the original diptych pipeline."""

    if seeds is None:
        seeds = [666]

    # Load the images and masks
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        raise FileNotFoundError(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    tar_image = cv2.imread(source_image_path)
    if tar_image is None:
        raise FileNotFoundError(source_image_path)
    tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

    ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:, :, 0]
    tar_mask = (cv2.imread(mask_image_path) > 128).astype(np.uint8)[:, :, 0]
    tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

    # Remove the background information of the reference picture
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

    # Extract the box where the reference image is located, and place the reference
    # object at the center of the image
    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]
    ratio = 1.3
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)

    # Dilate the mask
    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

    # zoom in
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2)
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop

    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]

    H1, W1 = tar_image.shape[0], tar_image.shape[1]

    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)

    # Extract the features of the reference image
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    pipe_prior_output = redux(Image.fromarray(masked_ref_image))

    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2 = tar_image.shape[0], tar_image.shape[1]

    tar_image = cv2.resize(tar_image, size)
    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)

    tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
    mask_black = np.ones_like(tar_image) * 0
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)

    os.makedirs(save_path, exist_ok=True)

    for seed in seeds:
        generator = torch.Generator(device).manual_seed(seed)
        edited_image = pipe(
            image=diptych_ref_tar,
            mask_image=mask_diptych,
            height=mask_diptych.size[1],
            width=mask_diptych.size[0],
            max_sequence_length=512,
            generator=generator,
            strength=strength if strength is not None else 1.0,
            **pipe_prior_output,
        ).images[0]

        width, height = edited_image.size
        left = width // 2
        right = width
        top = 0
        bottom = height
        edited_image = edited_image.crop((left, top, right, bottom))

        edited_image = np.array(edited_image)
        edited_image = crop_back(
            edited_image,
            old_tar_image,
            np.array([H1, W1, H2, W2]),
            np.array(tar_box_yyxx_crop),
        )
        edited_image = Image.fromarray(edited_image)

        ref_with_ext = os.path.basename(ref_mask_path)
        tar_with_ext = os.path.basename(mask_image_path)
        ref_without_ext = os.path.splitext(ref_with_ext)[0]
        tar_without_ext = os.path.splitext(tar_with_ext)[0]

        suffix = filename_suffix if filename_suffix else ""
        edited_image_save_path = os.path.join(
            save_path,
            f"{ref_without_ext}_to_{tar_without_ext}_seed{seed}{suffix}.png",
        )
        edited_image.save(edited_image_save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Single-image InsertAnything inference using diptych pipeline",
    )
    parser.add_argument("--source_image", default="examples/source_image/1.png")
    parser.add_argument("--source_mask", default="examples/source_mask/1.png")
    parser.add_argument("--ref_image", default="examples/ref_image/1.png")
    parser.add_argument("--ref_mask", default="examples/ref_mask/1.png")
    parser.add_argument("--outdir", default="./result")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    run_insertanything(
        source_image_path=args.source_image,
        mask_image_path=args.source_mask,
        ref_image_path=args.ref_image,
        ref_mask_path=args.ref_mask,
        seeds=args.seeds,
        strength=args.strength,
        save_path=args.outdir,
    )


if __name__ == "__main__":
    main()
