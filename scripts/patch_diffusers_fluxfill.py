import argparse
import importlib.util
import os
import shutil
from datetime import datetime


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _locate_pipeline_file() -> str:
    spec = importlib.util.find_spec("diffusers")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("diffusers is not importable in the current Python environment")

    diffusers_root = list(spec.submodule_search_locations)[0]
    candidate = os.path.join(diffusers_root, "pipelines", "flux", "pipeline_flux_fill.py")
    if not os.path.exists(candidate):
        raise RuntimeError(f"Cannot find FluxFillPipeline file at: {candidate}")
    return candidate


def _already_patched(text: str) -> bool:
    # Be strict: avoid false positives when the file has partial/manual edits.
    # We consider the patch applied only if both:
    # 1) the pipeline signature accepts sam_mask/bbox_mask/split_ratio
    # 2) the denoising loop switches masked latents (cur_masked_image_latents)
    # 3) transformer hidden_states concatenates cur_masked_image_latents
    has_sig = "sam_mask:" in text and "bbox_mask:" in text and "split_ratio" in text
    has_switch = "cur_masked_image_latents" in text and "split_index = int(len(timesteps) * split_ratio)" in text
    has_hidden = "hidden_states=torch.cat((latents, cur_masked_image_latents), dim=2)," in text
    # Also require that the additional SAM/BBox masked latents computation exists.
    # Otherwise the pipeline will accept sam_mask/bbox_mask but never use them.
    has_extra_latents = (
        "sam_mask_proc" in text
        or "bbox_mask_proc" in text
        or "sam_mask_latents = torch.cat" in text
        or "bbox_mask_latents = torch.cat" in text
    )
    return has_sig and has_switch and has_hidden and has_extra_latents


def _apply_patch(text: str) -> str:
    if _already_patched(text):
        return text

    anchor_sig = "max_sequence_length: int = 512," 
    if anchor_sig not in text:
        raise RuntimeError("Cannot find signature anchor 'max_sequence_length: int = 512,'")

    if "sam_mask:" not in text:
        text = text.replace(
            anchor_sig,
            anchor_sig
            + "\n        sam_mask: Optional[torch.FloatTensor] = None,\n"
            + "        bbox_mask: Optional[torch.FloatTensor] = None,\n"
            + "        split_ratio: float = 0.5,",
        )

    # 6. Prepare mask and masked image latents: add separate SAM/BBox masked latents
    mask_section_anchor = "# 6. Prepare mask and masked image latents"
    if mask_section_anchor not in text:
        raise RuntimeError("Cannot find mask section anchor '# 6. Prepare mask and masked image latents'")

    if "sam_mask_latents" not in text or "bbox_mask_latents" not in text:
        text = text.replace(
            mask_section_anchor,
            mask_section_anchor
            + "\n        sam_mask_latents = None\n"
            + "        bbox_mask_latents = None",
        )

    base_cat_anchors = [
        "masked_image_latents = torch.cat((base_masked_latents, base_mask), dim=-1)",
        "masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)",
    ]
    found_cat_anchor = next((a for a in base_cat_anchors if a in text), None)
    if found_cat_anchor is None:
        # Fallback: any cat line that constructs masked_image_latents with dim=-1.
        for line in text.splitlines():
            if "masked_image_latents = torch.cat(" in line and "dim=-1" in line:
                found_cat_anchor = line
                break

    if found_cat_anchor is not None and "sam_mask_proc" not in text:
        extra_latents_block = (
            "\n\n"
            "            if sam_mask is not None:\n"
            "                sam_mask_proc = self.mask_processor.preprocess(sam_mask, height=height, width=width)\n"
            "                sam_masked_image = init_image * (1 - sam_mask_proc)\n"
            "                sam_masked_image = sam_masked_image.to(device=device, dtype=prompt_embeds.dtype)\n\n"
            "                sam_mask_tensor, sam_masked_latents = self.prepare_mask_latents(\n"
            "                    sam_mask_proc,\n"
            "                    sam_masked_image,\n"
            "                    batch_size,\n"
            "                    num_channels_latents,\n"
            "                    num_images_per_prompt,\n"
            "                    height,\n"
            "                    width,\n"
            "                    prompt_embeds.dtype,\n"
            "                    device,\n"
            "                    generator,\n"
            "                )\n"
            "                sam_mask_latents = torch.cat((sam_masked_latents, sam_mask_tensor), dim=-1)\n\n"
            "            if bbox_mask is not None:\n"
            "                bbox_mask_proc = self.mask_processor.preprocess(bbox_mask, height=height, width=width)\n"
            "                bbox_masked_image = init_image * (1 - bbox_mask_proc)\n"
            "                bbox_masked_image = bbox_masked_image.to(device=device, dtype=prompt_embeds.dtype)\n\n"
            "                bbox_mask_tensor, bbox_masked_latents = self.prepare_mask_latents(\n"
            "                    bbox_mask_proc,\n"
            "                    bbox_masked_image,\n"
            "                    batch_size,\n"
            "                    num_channels_latents,\n"
            "                    num_images_per_prompt,\n"
            "                    height,\n"
            "                    width,\n"
            "                    prompt_embeds.dtype,\n"
            "                    device,\n"
            "                    generator,\n"
            "                )\n"
            "                bbox_mask_latents = torch.cat((bbox_masked_latents, bbox_mask_tensor), dim=-1)"
        )
        text = text.replace(found_cat_anchor, found_cat_anchor + extra_latents_block)

    if "split_index = int(len(timesteps) * split_ratio)" not in text:
        if "self._num_timesteps = len(timesteps)" not in text:
            raise RuntimeError("Cannot find anchor 'self._num_timesteps = len(timesteps)'")
        text = text.replace(
            "self._num_timesteps = len(timesteps)",
            "self._num_timesteps = len(timesteps)\n        split_index = int(len(timesteps) * split_ratio)",
        )

    # Denoising loop: choose which masked latents to use per timestep.
    if "cur_masked_image_latents" not in text:
        broadcast_anchor = "# broadcast to batch dimension in a way that's compatible with ONNX/Core ML"
        if broadcast_anchor not in text:
            raise RuntimeError("Cannot find broadcast anchor to insert mask switching")

        switch_block = (
            "# 这一 step 使用哪一份 mask latents\n"
            "                if sam_mask_latents is not None and bbox_mask_latents is not None:\n"
            "                    if i < split_index:\n"
            "                        cur_masked_image_latents = sam_mask_latents\n"
            "                    else:\n"
            "                        cur_masked_image_latents = bbox_mask_latents\n"
            "                else:\n"
            "                    cur_masked_image_latents = masked_image_latents\n\n"
            "                "
        )
        text = text.replace(broadcast_anchor, switch_block + broadcast_anchor)

    old_hidden = "hidden_states=torch.cat((latents, masked_image_latents), dim=2),"
    new_hidden = "hidden_states=torch.cat((latents, cur_masked_image_latents), dim=2),"
    if old_hidden not in text:
        if new_hidden in text:
            return text
        raise RuntimeError("Cannot find transformer hidden_states concat line")
    text = text.replace(old_hidden, new_hidden)

    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    file_path = args.file or _locate_pipeline_file()
    original = _read_text(file_path)

    if _already_patched(original):
        print(f"Already patched: {file_path}")
        return 0

    patched = _apply_patch(original)
    if patched == original:
        print(f"No changes needed: {file_path}")
        return 0

    backup_path = file_path + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Backing up to: {backup_path}")

    if args.dry_run:
        print("Dry run: not writing changes")
        return 0

    shutil.copy2(file_path, backup_path)
    _write_text(file_path, patched)
    print(f"Patched: {file_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
