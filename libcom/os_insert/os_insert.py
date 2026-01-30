"""Core OSInsertModel implementation.

Currently only the conservative mode is implemented, which maps directly to the
existing InsertAnything pipeline by constructing a rectangular mask from the
provided bbox and calling :func:`run_insertanything`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Literal

import os

import cv2
import numpy as np

from .source.insertanything_infer import run_insertanything
from .source.objectstitch_infer import ObjectStitchConfig, run_objectstitch_single
from .source.sam_on_objectstitch import (
    SamOnObjectStitchConfig,
    run_sam_on_objectstitch_single,
)
from .source.utils import load_bbox_txt, make_rect_mask_from_bbox


@dataclass
class OSInsertConfig:
    model_dir: Path
    device: str = "cuda:0"


class OSInsertModel:
    """High-level OSInsert interface.

    Modes
    -----
    - ``aggressive``: ObjectStitch + SAM + InsertAnything (not yet implemented)
    - ``conservative``: bg + bbox -> mask -> InsertAnything
    """

    def __init__(self, model_dir: str | Path, device: str = "cuda:0") -> None:
        self.config = OSInsertConfig(model_dir=Path(model_dir), device=device)

    def __call__(
        self,
        background_path: str | Path,
        foreground_path: str | Path,
        foreground_mask_path: str | Path,
        bbox_txt_path: str | Path,
        result_dir: str | Path,
        mode: Literal["aggressive", "conservative"] = "conservative",
        cleanup_intermediate: bool = True,
        seed: int = 123,
        strength: float = 1.0,
    ) -> None:
        """Run a single OSInsert inference.

        Parameters
        ----------
        background_path:
            Path to the background image.
        foreground_path:
            Path to the foreground image used as the InsertAnything reference
            image.
        foreground_mask_path:
            Binary mask for the foreground image.
        bbox_txt_path:
            Text file containing ``x1 y1 x2 y2`` on a single line, specifying the
            insertion region on the background image.
        result_dir:
            Directory where the final composed image will be written.
        mode:
            - ``"conservative"``: background + bbox -> mask -> InsertAnything.
            - ``"aggressive"``: (planned) ObjectStitch + SAM -> combined
              source/mask -> InsertAnything. The public API does not require
              any extra paths; all three stages will be handled internally in
              future updates.
        cleanup_intermediate:
            Present for API compatibility; currently conservative mode does not
            write any intermediates so this flag has no effect.
        seed:
            Random seed for InsertAnything.
        strength:
            InsertAnything strength parameter.
        """

        if mode not in {"aggressive", "conservative"}:
            raise ValueError(f"Unsupported mode: {mode}")

        # ------------------------------------------------------------------
        # Path normalization and output directory.
        # ------------------------------------------------------------------
        background_path = Path(background_path)
        foreground_path = Path(foreground_path)
        foreground_mask_path = Path(foreground_mask_path)
        bbox_txt_path = Path(bbox_txt_path)
        result_dir = Path(result_dir)

        os.makedirs(result_dir, exist_ok=True)

        # InsertAnything expects a list of seeds.
        seeds = [seed]

        # Load background once; used by both modes.
        bg = cv2.imread(str(background_path))
        if bg is None:
            raise FileNotFoundError(background_path)
        h, w = bg.shape[:2]

        bbox = load_bbox_txt(bbox_txt_path)

        # ------------------------------------------------------------------
        # Aggressive mode: ObjectStitch + SAM + InsertAnything.
        # ------------------------------------------------------------------
        if mode == "aggressive":
            # 1) ObjectStitch coarse composite.
            os_cfg = ObjectStitchConfig(device=self.config.device)
            os_image_path = run_objectstitch_single(
                bg_path=background_path,
                fg_path=foreground_path,
                fg_mask_path=foreground_mask_path,
                bbox_xyxy=tuple(bbox),
                config=os_cfg,
                out_dir=result_dir,
            )

            # 2) SAM mask on top of ObjectStitch composite.
            sam_cfg = SamOnObjectStitchConfig(device=self.config.device)
            sam_mask_path = run_sam_on_objectstitch_single(
                os_image_path=os_image_path,
                bg_shape_hw=(h, w),
                bbox_xyxy_bg=tuple(bbox),
                config=sam_cfg,
            )

            # 3) Construct InsertAnything source & mask following
            #    exp/run_insertanything_strength_sweep_dispatch.py::make_source_and_mask
            bg_bgr = bg  # already read above
            os_bgr = cv2.imread(str(os_image_path))
            if os_bgr is None:
                raise FileNotFoundError(os_image_path)

            sam_mask = cv2.imread(str(sam_mask_path), cv2.IMREAD_GRAYSCALE)
            if sam_mask is None:
                raise FileNotFoundError(sam_mask_path)

            hh, ww = bg_bgr.shape[:2]
            os_bgr = cv2.resize(os_bgr, (ww, hh), interpolation=cv2.INTER_AREA)
            if sam_mask.shape[:2] != (hh, ww):
                sam_mask = cv2.resize(sam_mask, (ww, hh), interpolation=cv2.INTER_NEAREST)

            m = (sam_mask > 127).astype(np.float32)
            m3 = np.stack([m, m, m], axis=-1)

            src_bgr = bg_bgr.astype(np.float32) * (1.0 - m3) + os_bgr.astype(np.float32) * m3
            src_bgr = np.clip(src_bgr, 0, 255).astype(np.uint8)

            tmp_src_path = result_dir / "objectstitch_coarse_sam_blend.png"
            tmp_mask_path = result_dir / "objectstitch_coarse_sam_mask_resized.png"
            cv2.imwrite(str(tmp_src_path), src_bgr)
            cv2.imwrite(str(tmp_mask_path), sam_mask)

            # 4) InsertAnything refinement using blended source and resized mask.
            run_insertanything(
                source_image_path=str(tmp_src_path),
                mask_image_path=str(tmp_mask_path),
                ref_image_path=str(foreground_path),
                ref_mask_path=str(foreground_mask_path),
                seeds=seeds,
                strength=strength,
                save_path=str(result_dir),
                filename_suffix="",
            )

            if cleanup_intermediate:
                for p in (os_image_path, sam_mask_path, tmp_src_path, tmp_mask_path):
                    try:
                        if p.exists():
                            p.unlink()
                    except OSError:
                        # Best-effort cleanup; ignore failures.
                        pass

            return

        # ------------------------------------------------------------------
        # Conservative mode: background + bbox -> mask -> InsertAnything.
        # ------------------------------------------------------------------
        mask = make_rect_mask_from_bbox(h, w, bbox)

        tmp_mask_path = result_dir / "_tmp_bg_mask_from_bbox.png"
        cv2.imwrite(str(tmp_mask_path), mask)

        run_insertanything(
            source_image_path=str(background_path),
            mask_image_path=str(tmp_mask_path),
            ref_image_path=str(foreground_path),
            ref_mask_path=str(foreground_mask_path),
            seeds=seeds,
            strength=strength,
            save_path=str(result_dir),
            filename_suffix="",
        )

        if cleanup_intermediate and tmp_mask_path.exists():
            try:
                tmp_mask_path.unlink()
            except OSError:
                # Best-effort cleanup; ignore failures so that inference
                # results are not affected.
                pass
