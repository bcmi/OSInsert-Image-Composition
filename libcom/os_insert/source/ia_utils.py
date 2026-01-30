"""Local reimplementation of InsertAnything utility functions.

Moved from `osinsert/ia_utils.py` so that all OSInsert logic lives under
`libcom/os_insert` and does not depend on external repositories.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get (y1, y2, x1, x2) bounding box from a binary mask.

    The mask is expected to be 2D with values 0/1 or 0/255.
    """

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty, cannot compute bbox")
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    # +1 on max side to make the box inclusive-exclusive
    return y1, y2 + 1, x1, x2 + 1


def expand_bbox(img_or_mask: np.ndarray, box, ratio: float = 1.2):
    """Expand a bbox by a ratio while keeping it inside the image."""

    h, w = img_or_mask.shape[:2]
    y1, y2, x1, x2 = map(int, box)
    cy = (y1 + y2) / 2.0
    cx = (x1 + x2) / 2.0
    hh = (y2 - y1) * ratio / 2.0
    hw = (x2 - x1) * ratio / 2.0

    y1_new = int(max(0, np.floor(cy - hh)))
    y2_new = int(min(h, np.ceil(cy + hh)))
    x1_new = int(max(0, np.floor(cx - hw)))
    x2_new = int(min(w, np.ceil(cx + hw)))

    return y1_new, y2_new, x1_new, x2_new


def pad_to_square(img: np.ndarray, pad_value: int = 0, random: bool = False) -> np.ndarray:
    """Pad an image to a square with constant value.

    If `random` is False, pads symmetrically to keep the content centered.
    """

    h, w = img.shape[:2]
    if h == w:
        return img

    size = max(h, w)
    if img.ndim == 2:
        padded = np.full((size, size), pad_value, dtype=img.dtype)
    else:
        padded = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype)

    if random:
        if h < size:
            top = np.random.randint(0, size - h + 1)
        else:
            top = 0
        if w < size:
            left = np.random.randint(0, size - w + 1)
        else:
            left = 0
    else:
        top = (size - h) // 2
        left = (size - w) // 2

    padded[top : top + h, left : left + w, ...] = img
    return padded


def box2squre(img: np.ndarray, box) -> Tuple[int, int, int, int]:  # spelling kept for compatibility
    """Adjust a bbox to be square by expanding the shorter side."""

    h, w = img.shape[:2]
    y1, y2, x1, x2 = map(int, box)
    box_h = y2 - y1
    box_w = x2 - x1

    if box_h == box_w:
        return y1, y2, x1, x2

    if box_h > box_w:
        diff = box_h - box_w
        x1 -= diff // 2
        x2 += diff - diff // 2
    else:
        diff = box_w - box_h
        y1 -= diff // 2
        y2 += diff - diff // 2

    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(h, y2)
    x2 = min(w, x2)

    return y1, y2, x1, x2


def crop_back(edited: np.ndarray, old_tar: np.ndarray, hw_vec, box_crop) -> np.ndarray:
    """Paste the edited crop back into the original target image."""

    H1, W1, H2, W2 = map(int, hw_vec)
    y1, y2, x1, x2 = map(int, box_crop)

    edited_resized = cv2.resize(edited, (x2 - x1, y2 - y1))

    result = old_tar.copy()
    result[y1:y2, x1:x2] = edited_resized

    return result


def expand_image_mask(img: np.ndarray, mask: np.ndarray, ratio: float = 1.3):
    """Expand image and mask around the mask bbox by a ratio."""

    y1, y2, x1, x2 = get_bbox_from_mask(mask)
    h, w = img.shape[:2]

    cy = (y1 + y2) / 2.0
    cx = (x1 + x2) / 2.0
    hh = (y2 - y1) * ratio / 2.0
    hw = (x2 - x1) * ratio / 2.0

    y1_new = int(max(0, np.floor(cy - hh)))
    y2_new = int(min(h, np.ceil(cy + hh)))
    x1_new = int(max(0, np.floor(cx - hw)))
    x2_new = int(min(w, np.ceil(cx + hw)))

    img_crop = img[y1_new:y2_new, x1_new:x2_new]
    mask_crop = mask[y1_new:y2_new, x1_new:x2_new]

    return img_crop, mask_crop
