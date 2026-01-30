# OSInsert-Image-Composition

OSInsert is a two-stage object insertion pipeline. This repository packages the
minimal inference code for ObjectStitch, SAM, and InsertAnything into the
`libcom/os_insert` module.

- **Stage 1 (ObjectStitch)**: generate a **coarse composite** on the target
  background image.  
- **Stage 2 (SAM + InsertAnything)**: apply SAM to obtain a foreground
  insertion mask, then combine the "original background + ObjectStitch output + SAM mask" into a source image and mask, and feed them into InsertAnything
  to obtain a **high-quality final insertion result**.

## 0. Example Results

The table below shows several samples at different stages (from left to right:
background, foreground, aggressive mode (ObjectStitch + SAM + InsertAnything),
and conservative mode (InsertAnything only)).

| Sample   | Background                                            | Foreground                                              | aggressive (OSInsert, full pipeline)                              | conservative (InsertAnything)                                     |
|----------|-------------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
| bottle   | ![](examples/results/bottle/bottle_bg_bbox.png)      | ![](examples/results/bottle/bottle_foreground.png)      | ![](examples/results/bottle/bottle_osinsert.png)                  | ![](examples/results/bottle/bottle_insertanything.png)            |
| box      | ![](examples/results/box/box_bg_bbox.png)            | ![](examples/results/box/box_foreground.png)            | ![](examples/results/box/box_osinsert.png)                        | ![](examples/results/box/box_insertanything.png)                  |
| bus      | ![](examples/results/bus/bus_bg_bbox.png)            | ![](examples/results/bus/bus_foreground.png)            | ![](examples/results/bus/bus_osinsert.png)                        | ![](examples/results/bus/bus_insertanything.png)                  |
| cake     | ![](examples/results/cake/cake_bg_bbox.png)          | ![](examples/results/cake/cake_foreground.png)          | ![](examples/results/cake/cake_osinsert.png)                      | ![](examples/results/cake/cake_insertanything.png)                |
| keyboard | ![](examples/results/keyboard/keyboard_bg_bbox.png)  | ![](examples/results/keyboard/keyboard_foreground.png)  | ![](examples/results/keyboard/keyboard_osinsert.png)              | ![](examples/results/keyboard/keyboard_insertanything.png)        |
| frame    | ![](examples/results/frame/frame_bg_bbox.png)        | ![](examples/results/frame/frame_foreground.png)        | ![](examples/results/frame/frame_osinsert.png)                    | ![](examples/results/frame/frame_insertanything.png)              |

---

## 1. Environment

Example environment configuration:

- OS: Linux
- Python 3.10
- PyTorch ≥ 2.6.0

Dependency installation example:

```bash
conda create -n osinsert python=3.10
conda activate osinsert
pip install -r requirements.txt
```

> Note: This repository **does not include any pretrained weights**. Checkpoints
> must be downloaded via the links below and configured via the local directory
> structure or environment variables.

---

## 2. Models and Directory Layout

This repository is **self-contained** and no longer depends on external
ObjectStitch / InsertAnything source repositories. All inference-related code
resides under `libcom/os_insert`. All checkpoints are organized under the
`pretrained_models/` directory:

```text
pretrained_models/
  flux/
    FLUX.1-Fill-dev/
    FLUX.1-Redux-dev/
  insert_anything/
    20250321_steps5000_pytorch_lora_weights.safetensors
  objectstitch/
    v1/
      model.ckpt                      # -> ObjectStitch.pth
      configs/
        v1.yaml
      openai-clip-vit-large-patch14/  # CLIP weights directory
  sam/
    sam_vit_h_4b8939.pth
```

### 2.1 Checkpoints

- **ObjectStitch checkpoint**:
  - openai-clip-vit-large-patch14  
    - HuggingFace: <https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/openai-clip-vit-large-patch14.zip>  
    - ModelScope: <https://www.modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/openai-clip-vit-large-patch14.zip>
  - ObjectStitch.pth  
    - HuggingFace: <https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/ObjectStitch.pth>  
    - ModelScope: <https://www.modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/ObjectStitch.pth>

- **SAM ViT-H**: `sam_vit_h_4b8939.pth`  
  - Official download: <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>

- **InsertAnything LoRA** (recommended):  
  - Direct download: <https://huggingface.co/WensongSong/Insert-Anything/resolve/main/20250321_steps5000_pytorch_lora_weights.safetensors>

- **FLUX.1-Fill-dev / FLUX.1-Redux-dev**:  
  - FLUX.1-Fill-dev: <https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors>  
  - FLUX.1-Redux-dev: <https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors>

After downloading, organize all files according to the directory structure
above. The following environment variables can override default paths:

- `FLUX_FILL_PATH`
- `FLUX_REDUX_PATH`
- `IA_LORA_PATH`

If these variables are not set, the defaults under `pretrained_models/...` are
used.

---

## 3. Data Format

The data format of OSInsert follows the original ObjectStitch convention:

- `background/{uniq}.png`
- `foreground/{uniq}.png`
- `foreground_mask/{uniq}.png`
- `bbox/{uniq}.txt` (content: `x1 y1 x2 y2`)

The TSV list file contains the following columns:

```text
uniq_id \t bg_path \t fg_path \t fg_mask_path
```

### 3.1 Built-in Demo Data

This repository provides a **minimal runnable demo**:

- `examples/samples_demo.tsv`
- `os_test_demo/background/Demo_0.png`
- `os_test_demo/foreground/Demo_0.png`
- `os_test_demo/foreground_mask/Demo_0.png`
- `os_test_demo/bbox/Demo_0.txt`

Typical usage:

- Directly reuse these demo files to verify the pipeline.  
- Replace the images with custom data while keeping the same filenames and
  directory structure.  
- Create a new TSV and `os_test` directory, and pass their paths via script
  arguments.

---

## 4. One-Click Demo: OSInsertModel

The main entry script is `tests/test_os_insert.py`, which calls
`libcom.os_insert.OSInsertModel`. Legacy multi-script pipelines such as
`osinsert/run_osinsert_full.py` are no longer required.

### 4.1 Demo Data

The repository includes a minimal demo under:

- `tests/source/background/Demo_0.png`
- `tests/source/foreground/Demo_0.png`
- `tests/source/foreground_mask/Demo_0.png`
- `tests/source/bbox/Demo_0.txt`

These files can be replaced (while keeping filenames unchanged) for quick
custom tests.

### 4.2 Running Conservative / Aggressive Modes

`tests/test_os_insert.py` exposes a `--mode` argument to select the run mode:

- `conservative`: use InsertAnything only, performing insertion within the bbox
  region on the background image.
- `aggressive`: full two-stage pipeline: ObjectStitch → SAM → InsertAnything.

Example commands:

```bash
conda activate osinsert
cd OSInsert-Image-Composition

# Conservative mode (default)
python -m tests.test_os_insert --mode conservative

# Aggressive mode (ObjectStitch + SAM + InsertAnything)
python -m tests.test_os_insert --mode aggressive
```

Outputs are written to:

- `tests/result_dir/osinsert_demo/`: conservative mode results.  
- `tests/result_dir/osinsert_demo_aggressive/`: aggressive mode results.

In aggressive mode, setting `cleanup_intermediate=False` additionally keeps the
following intermediate files:

- `objectstitch_coarse.png`: ObjectStitch coarse composite.  
- `objectstitch_coarse_sam_mask.png`: raw SAM mask on the coarse composite.  
- `objectstitch_coarse_sam_blend.png`: background and ObjectStitch composite
  blended by the SAM mask (source image).  
- `objectstitch_coarse_sam_mask_resized.png`: SAM mask resized to the
  background resolution, used as the final InsertAnything mask.

When `cleanup_intermediate=True`, these intermediate files are removed after
inference, and only the final outputs are kept.

### 4.3 OSInsertModel API Overview

The unified `OSInsertModel` is defined in `libcom/os_insert/os_insert.py`:

```python
from libcom.os_insert import OSInsertModel

model = OSInsertModel(model_dir="pretrained_models", device="cuda:0")

model(
    background_path="tests/source/background/Demo_0.png",
    foreground_path="tests/source/foreground/Demo_0.png",
    foreground_mask_path="tests/source/foreground_mask/Demo_0.png",
    bbox_txt_path="tests/source/bbox/Demo_0.txt",
    result_dir="tests/result_dir/osinsert_demo_aggressive",
    mode="aggressive",          # or "conservative"
    cleanup_intermediate=False,  # whether to keep intermediate files
    seed=123,
    strength=1.0,
)
```

The internal behavior is as follows:

- `conservative`:  
  - Use `background + bbox` to construct a rectangular mask.  
  - Call InsertAnything directly on this region.

- `aggressive`:  
  - ObjectStitch: generate a coarse composite `objectstitch_coarse.png` on the
    background.  
  - SAM: run SAM on the coarse composite with the bbox and obtain a binary
    mask.  
  - Blending: blend the original background and the coarse composite according
    to the SAM mask to form a new source image and mask (aligned to the
    original background resolution).  
  - InsertAnything: run InsertAnything on this region to obtain the final
    high-quality insertion result.
