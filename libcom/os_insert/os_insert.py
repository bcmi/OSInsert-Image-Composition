from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Literal

import os

import cv2
import numpy as np

from .source.insertanything_infer import InsertAnythingModel, insertanything_infer, run_insertanything
from .source.objectstitch_infer import (
    ObjectStitchConfig,
    run_objectstitch_single_image,
    run_objectstitch_single_image_from_images,
)
from .source.sam_on_objectstitch import SamOnObjectStitchConfig, run_sam_on_objectstitch
from .source.utils import load_bbox_txt, make_rect_mask_from_bbox


@dataclass
class OSInsertConfig:
    model_dir: Path
    device: str = "cuda:0"
    # Optional explicit checkpoint paths (override model_dir conventions)
    objectstitch_ckpt_path: Path | None = None
    objectstitch_config_path: Path | None = None
    objectstitch_clip_dir: Path | None = None
    sam_checkpoint: Path | None = None
    flux_fill_path: Path | None = None
    flux_redux_path: Path | None = None
    ia_lora_path: Path | None = None


class OSInsertModel:
    """High-level OSInsert interface.

    Modes
    -----
    - ``aggressive``: ObjectStitch + SAM + InsertAnything (not yet implemented)
    - ``conservative``: bg + bbox -> mask -> InsertAnything
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda:0",
        *,
        objectstitch_ckpt_path: str | Path | None = None,
        objectstitch_config_path: str | Path | None = None,
        objectstitch_clip_dir: str | Path | None = None,
        sam_checkpoint: str | Path | None = None,
        flux_fill_path: str | Path | None = None,
        flux_redux_path: str | Path | None = None,
        ia_lora_path: str | Path | None = None,
    ) -> None:
        self.config = OSInsertConfig(
            model_dir=Path(model_dir),
            device=device,
            objectstitch_ckpt_path=Path(objectstitch_ckpt_path) if objectstitch_ckpt_path is not None else None,
            objectstitch_config_path=Path(objectstitch_config_path) if objectstitch_config_path is not None else None,
            objectstitch_clip_dir=Path(objectstitch_clip_dir) if objectstitch_clip_dir is not None else None,
            sam_checkpoint=Path(sam_checkpoint) if sam_checkpoint is not None else None,
            flux_fill_path=Path(flux_fill_path) if flux_fill_path is not None else None,
            flux_redux_path=Path(flux_redux_path) if flux_redux_path is not None else None,
            ia_lora_path=Path(ia_lora_path) if ia_lora_path is not None else None,
        )

        self._ia_net = InsertAnythingModel(
            model_dir=self.config.model_dir,
            flux_fill_path=self.config.flux_fill_path,
            flux_redux_path=self.config.flux_redux_path,
            ia_lora_path=self.config.ia_lora_path,
            device=self.config.device,
        )

    def __call__(
        self,
        background_path: str | Path,
        foreground_path: str | Path,
        foreground_mask_path: str | Path,
        bbox_txt_path: str | Path,
        result_dir: str | Path,
        mode: Literal["aggressive", "conservative"] = "conservative",
        cleanup_intermediate: bool = True,
        verbose: bool = False,
        seed: int = 123,
        strength: float = 1.0,
        split_ratio: float = 0.5,
    ) -> np.ndarray | None:
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
            Deprecated. Present for backward compatibility.
        verbose:
            If True, save intermediate artifacts into ``result_dir/intermediates``.
            Default False (do not save intermediates).
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

        intermediates_dir = result_dir / "intermediates"
        if verbose:
            os.makedirs(intermediates_dir, exist_ok=True)

        # InsertAnything expects a list of seeds.
        seeds = [seed]

        # Load background once; used by both modes.
        bg = cv2.imread(str(background_path))
        if bg is None:
            raise FileNotFoundError(background_path)
        h, w = bg.shape[:2]

        fg = cv2.imread(str(foreground_path))
        if fg is None:
            raise FileNotFoundError(foreground_path)
        fg_mask = cv2.imread(str(foreground_mask_path))
        if fg_mask is None:
            raise FileNotFoundError(foreground_mask_path)
        if fg_mask.ndim == 3:
            fg_mask = fg_mask[:, :, 0]

        bbox = load_bbox_txt(bbox_txt_path)

        # ------------------------------------------------------------------
        # Aggressive mode: ObjectStitch + SAM + InsertAnything.
        # ------------------------------------------------------------------
        if mode == "aggressive":
            # 1) ObjectStitch coarse composite (in-memory).
            objectstitch_ckpt = self.config.objectstitch_ckpt_path
            if objectstitch_ckpt is None:
                objectstitch_ckpt = self.config.model_dir / "objectstitch" / "v1" / "model.ckpt"

            objectstitch_cfg = self.config.objectstitch_config_path
            if objectstitch_cfg is None:
                objectstitch_cfg = self.config.model_dir / "objectstitch" / "v1" / "configs" / "v1.yaml"

            os_cfg = ObjectStitchConfig(
                ckpt_path=objectstitch_ckpt,
                config_path=objectstitch_cfg,
                clip_dir=self.config.objectstitch_clip_dir,
                device=self.config.device,
            )
            os_rgb = run_objectstitch_single_image(
                background_path,
                foreground_path,
                foreground_mask_path,
                tuple(bbox),
                config=os_cfg,
                seed=seed,
            )

            # 2) SAM mask on top of ObjectStitch composite (in-memory).
            sam_ckpt = self.config.sam_checkpoint
            if sam_ckpt is None:
                sam_ckpt = self.config.model_dir / "sam" / "sam_vit_h_4b8939.pth"

            sam_cfg = SamOnObjectStitchConfig(
                sam_checkpoint=sam_ckpt,
                device=self.config.device,
            )
            sam_mask = run_sam_on_objectstitch(
                os_image=cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR),
                bg_shape_hw=(h, w),
                bbox_xyxy_bg=tuple(bbox),
                config=sam_cfg,
            )

            # 3) Construct InsertAnything source & mask following
            #    exp/run_insertanything_strength_sweep_dispatch.py::make_source_and_mask
            bg_bgr = bg  # already read above
            os_bgr = cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR)

            hh, ww = bg_bgr.shape[:2]
            os_bgr = cv2.resize(os_bgr, (ww, hh), interpolation=cv2.INTER_AREA)
            if sam_mask.shape[:2] != (hh, ww):
                sam_mask = cv2.resize(sam_mask, (ww, hh), interpolation=cv2.INTER_NEAREST)

            m = (sam_mask > 127).astype(np.float32)
            m3 = np.stack([m, m, m], axis=-1)

            src_bgr = bg_bgr.astype(np.float32) * (1.0 - m3) + os_bgr.astype(np.float32) * m3
            src_bgr = np.clip(src_bgr, 0, 255).astype(np.uint8)

            # BBox-based mask for InsertAnything second-stage (bbox mask)
            bbox_mask = make_rect_mask_from_bbox(h, w, bbox)

            if verbose:
                cv2.imwrite(str(intermediates_dir / "objectstitch_coarse_rgb.png"), cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(intermediates_dir / "sam_mask.png"), sam_mask)
                cv2.imwrite(str(intermediates_dir / "blended_source.png"), src_bgr)
                cv2.imwrite(str(intermediates_dir / "bbox_mask.png"), bbox_mask)

            # 4) InsertAnything refinement using blended source, bbox mask
            #    (for the second half of denoising) and SAM mask
            #    (for the first half, wired via sam_mask_path).
            result = insertanything_infer(
                source_image=cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB),
                mask_image=bbox_mask,
                ref_image=cv2.cvtColor(fg, cv2.COLOR_BGR2RGB),
                ref_mask=fg_mask,
                sam_mask=sam_mask,
                seeds=seeds,
                strength=strength,
                split_ratio=split_ratio,
                save_path=str(result_dir),
                filename_suffix="",
                net=self._ia_net,
                return_image=True,
            )

            return result

        # ------------------------------------------------------------------
        # Conservative mode: background + bbox -> mask -> InsertAnything.
        # ------------------------------------------------------------------
        mask = make_rect_mask_from_bbox(h, w, bbox)

        if verbose:
            cv2.imwrite(str(intermediates_dir / "bbox_mask.png"), mask)

        result = insertanything_infer(
            source_image=cv2.cvtColor(bg, cv2.COLOR_BGR2RGB),
            mask_image=mask,
            ref_image=cv2.cvtColor(fg, cv2.COLOR_BGR2RGB),
            ref_mask=fg_mask,
            seeds=seeds,
            strength=strength,
            split_ratio=split_ratio,
            save_path=str(result_dir),
            filename_suffix="",
            net=self._ia_net,
            return_image=True,
        )

        return result

    def infer_images(
        self,
        *,
        background: np.ndarray,
        foreground: np.ndarray,
        foreground_mask: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        mode: Literal["aggressive", "conservative"] = "conservative",
        verbose: bool = False,
        seed: int = 123,
        strength: float = 1.0,
        split_ratio: float = 0.5,
        save_path: str | Path | None = None,
        filename_suffix: str = "",
    ) -> np.ndarray | None:
        if background.ndim != 3:
            raise ValueError("background must be HxWx3")
        h, w = background.shape[:2]

        out_dir = str(save_path) if save_path is not None else "./result"

        if verbose and save_path is not None:
            inter_dir = Path(save_path) / "intermediates"
            os.makedirs(inter_dir, exist_ok=True)

        # InsertAnything expects a list of seeds.
        seeds = [seed]

        if mode == "aggressive":
            objectstitch_ckpt = self.config.objectstitch_ckpt_path
            if objectstitch_ckpt is None:
                objectstitch_ckpt = self.config.model_dir / "objectstitch" / "v1" / "model.ckpt"

            objectstitch_cfg = self.config.objectstitch_config_path
            if objectstitch_cfg is None:
                objectstitch_cfg = self.config.model_dir / "objectstitch" / "v1" / "configs" / "v1.yaml"

            os_cfg = ObjectStitchConfig(
                ckpt_path=objectstitch_ckpt,
                config_path=objectstitch_cfg,
                clip_dir=self.config.objectstitch_clip_dir,
                device=self.config.device,
            )
            os_rgb = run_objectstitch_single_image_from_images(
                background=background[:, :, ::-1],
                foreground=foreground[:, :, ::-1],
                foreground_mask=foreground_mask,
                bbox_xyxy=tuple(bbox_xyxy),
                config=os_cfg,
                seed=seed,
            )

            sam_ckpt = self.config.sam_checkpoint
            if sam_ckpt is None:
                sam_ckpt = self.config.model_dir / "sam" / "sam_vit_h_4b8939.pth"
            sam_cfg = SamOnObjectStitchConfig(
                sam_checkpoint=sam_ckpt,
                device=self.config.device,
            )

            sam_mask = run_sam_on_objectstitch(
                os_image=cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR),
                bg_shape_hw=(h, w),
                bbox_xyxy_bg=tuple(bbox_xyxy),
                config=sam_cfg,
            )

            bg_bgr = background
            os_bgr = cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR)
            hh, ww = bg_bgr.shape[:2]
            os_bgr = cv2.resize(os_bgr, (ww, hh), interpolation=cv2.INTER_AREA)
            if sam_mask.shape[:2] != (hh, ww):
                sam_mask = cv2.resize(sam_mask, (ww, hh), interpolation=cv2.INTER_NEAREST)

            m = (sam_mask > 127).astype(np.float32)
            m3 = np.stack([m, m, m], axis=-1)
            src_bgr = bg_bgr.astype(np.float32) * (1.0 - m3) + os_bgr.astype(np.float32) * m3
            src_bgr = np.clip(src_bgr, 0, 255).astype(np.uint8)

            bbox_mask = make_rect_mask_from_bbox(h, w, list(bbox_xyxy))

            if verbose and save_path is not None:
                inter_dir = Path(save_path) / "intermediates"
                cv2.imwrite(str(inter_dir / "objectstitch_coarse_rgb.png"), cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(inter_dir / "sam_mask.png"), sam_mask)
                cv2.imwrite(str(inter_dir / "blended_source.png"), src_bgr)
                cv2.imwrite(str(inter_dir / "bbox_mask.png"), bbox_mask)

            return insertanything_infer(
                source_image=cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB),
                mask_image=bbox_mask,
                ref_image=cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB),
                ref_mask=foreground_mask,
                sam_mask=sam_mask,
                seeds=seeds,
                strength=strength,
                split_ratio=split_ratio,
                save_path=out_dir,
                filename_suffix=filename_suffix,
                net=self._ia_net,
                return_image=True,
            )

        if mode != "conservative":
            raise ValueError(f"Unsupported mode: {mode}")

        mask = make_rect_mask_from_bbox(h, w, list(bbox_xyxy))

        return insertanything_infer(
            source_image=cv2.cvtColor(background, cv2.COLOR_BGR2RGB),
            mask_image=mask,
            ref_image=cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB),
            ref_mask=foreground_mask,
            seeds=seeds,
            strength=strength,
            split_ratio=split_ratio,
            save_path=out_dir,
            filename_suffix=filename_suffix,
            net=self._ia_net,
            return_image=True,
        )
