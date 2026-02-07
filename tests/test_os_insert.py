"""One-click demo for OSInsertModel (conservative + aggressive modes).

This script assumes a single demo sample with the following layout:

    examples/background/{uniq}.png
    examples/foreground/{uniq}.png
    examples/foreground_mask/{uniq}.png
    examples/bbox/{uniq}.txt

It will run the OSInsert pipeline and write results to

    result_dir/
"""

import argparse
import sys
from pathlib import Path

try:
    from libcom.os_insert import OSInsertModel
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from libcom.os_insert import OSInsertModel


# ---------------------------------------------------------------------------
# CONFIG (edit here)
#
# If you rename/move checkpoints, edit the full paths below. If a value is set
# to None, OSInsert falls back to the default `model_dir` convention.
# ---------------------------------------------------------------------------

OSINSERT_MODEL_DIR = "model_dir"

# ObjectStitch
OBJECTSTITCH_CKPT_PATH = "model_dir/objectstitch/v1/model.ckpt"
OBJECTSTITCH_CONFIG_PATH = "model_dir/objectstitch/v1/configs/v1.yaml"
OBJECTSTITCH_CLIP_DIR = "model_dir/objectstitch/v1/openai-clip-vit-large-patch14"

# SAM
SAM_CHECKPOINT = "model_dir/sam/sam_vit_h_4b8939.pth"

# InsertAnything / FLUX
FLUX_FILL_PATH = "model_dir/flux/FLUX.1-Fill-dev"
FLUX_REDUX_PATH = "model_dir/flux/FLUX.1-Redux-dev"
IA_LORA_PATH = "model_dir/insert_anything/20250321_steps5000_pytorch_lora_weights.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Which OSInsert mode to run in this demo.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Torch device for the full pipeline, e.g. "cuda:0", "cuda:1", or "cpu".',
    )
    parser.add_argument(
        "--model_dir",
        default=OSINSERT_MODEL_DIR,
        help="Checkpoint root directory. Default: <repo_root>/{OSINSERT_MODEL_DIR}",
    )
    parser.add_argument(
        "--examples_dir",
        default="examples",
        help="Examples root directory. Default: <repo_root>/examples",
    )
    parser.add_argument(
        "--result_dir",
        default="result_dir",
        help="Result root directory. Default: <repo_root>/result_dir",
    )
    parser.add_argument(
        "--uniq_id",
        default="Bus_2",
        help="Example sample id (e.g., Bus_2).",
    )
    parser.add_argument(
        "--uniq_ids",
        default=None,
        help='Optional comma-separated list of uniq_ids to run as a batch. Example: "Bus_2,Demo_0". If set, overrides --uniq_id.',
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.5,
        help="Aggressive-mode mask schedule split ratio. First split_ratio of timesteps use SAM mask; remaining use bbox mask.",
    )
    parser.add_argument(
        "--split_ratios",
        default=None,
        help='Optional comma-separated list of split_ratio values to run as a sweep. Example: "0.33,0.5,0.67". If set, overrides --split_ratio.',
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, save intermediate artifacts into result_dir (debug mode).",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir

    def _resolve_optional_path(p: str | None) -> str | None:
        if p is None:
            return None
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str(repo_root / pp)

    objectstitch_ckpt_path = _resolve_optional_path(OBJECTSTITCH_CKPT_PATH)
    objectstitch_config_path = _resolve_optional_path(OBJECTSTITCH_CONFIG_PATH)
    objectstitch_clip_dir = _resolve_optional_path(OBJECTSTITCH_CLIP_DIR)
    sam_checkpoint = _resolve_optional_path(SAM_CHECKPOINT)
    flux_fill_path = _resolve_optional_path(FLUX_FILL_PATH)
    flux_redux_path = _resolve_optional_path(FLUX_REDUX_PATH)
    ia_lora_path = _resolve_optional_path(IA_LORA_PATH)

    examples_dir = Path(args.examples_dir)
    if not examples_dir.is_absolute():
        examples_dir = repo_root / examples_dir

    base_result_dir = Path(args.result_dir)
    if not base_result_dir.is_absolute():
        base_result_dir = repo_root / base_result_dir
    def _parse_csv_list(s: str | None) -> list[str] | None:
        if s is None:
            return None
        items = [x.strip() for x in s.split(",")]
        items = [x for x in items if x]
        return items if items else None

    def _parse_csv_floats(s: str | None) -> list[float] | None:
        items = _parse_csv_list(s)
        if items is None:
            return None
        return [float(x) for x in items]

    uniq_ids_arg = _parse_csv_list(args.uniq_ids)
    split_ratios_arg = _parse_csv_floats(args.split_ratios)
    uniq_ids = uniq_ids_arg or [args.uniq_id]
    split_ratios = split_ratios_arg or [float(args.split_ratio)]
    is_sweep = (uniq_ids_arg is not None) or (split_ratios_arg is not None)

    for r in split_ratios:
        if not (0.0 < float(r) < 1.0):
            raise ValueError(f"split_ratio must be in (0,1), got {r}")

    print("[INFO] repo_root =", repo_root)
    print("[INFO] model_dir =", model_dir)
    print("[INFO] device =", args.device)
    print("[INFO] examples =", examples_dir)
    print("[INFO] base_result_dir =", base_result_dir)
    print("[INFO] uniq_ids =", uniq_ids)
    print("[INFO] split_ratios =", split_ratios)

    osinsert = OSInsertModel(
        model_dir=model_dir,
        device=args.device,
        objectstitch_ckpt_path=objectstitch_ckpt_path,
        objectstitch_config_path=objectstitch_config_path,
        objectstitch_clip_dir=objectstitch_clip_dir,
        sam_checkpoint=sam_checkpoint,
        flux_fill_path=flux_fill_path,
        flux_redux_path=flux_redux_path,
        ia_lora_path=ia_lora_path,
    )
    print("[INFO] OSInsertModel created:", osinsert)

    for uniq in uniq_ids:
        bg = examples_dir / "background" / f"{uniq}.png"
        fg = examples_dir / "foreground" / f"{uniq}.png"
        fg_mask = examples_dir / "foreground_mask" / f"{uniq}.png"
        bbox_txt = examples_dir / "bbox" / f"{uniq}.txt"

        if args.verbose:
            print("[INFO] sample uniq =", uniq)
            print("[INFO] background =", bg)
            print("[INFO] foreground =", fg)
            print("[INFO] foreground_mask =", fg_mask)
            print("[INFO] bbox_txt =", bbox_txt)

        for split_ratio in split_ratios:
            if args.mode == "conservative":
                base_out = base_result_dir / "osinsert_demo"
                out_dir = (
                    base_out / f"{uniq}_split{split_ratio:g}_seed{args.seed}_strength{args.strength:g}"
                    if is_sweep
                    else base_out
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                osinsert(
                    background_path=bg,
                    foreground_path=fg,
                    foreground_mask_path=fg_mask,
                    bbox_txt_path=bbox_txt,
                    result_dir=out_dir,
                    mode="conservative",
                    verbose=args.verbose,
                    seed=args.seed,
                    strength=args.strength,
                    split_ratio=split_ratio,
                )
                print("[INFO] Conservative done ->", out_dir)

            if args.mode == "aggressive":
                base_out = base_result_dir / "osinsert_demo_aggressive"
                out_dir = (
                    base_out / f"{uniq}_split{split_ratio:g}_seed{args.seed}_strength{args.strength:g}"
                    if is_sweep
                    else base_out
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                osinsert(
                    background_path=bg,
                    foreground_path=fg,
                    foreground_mask_path=fg_mask,
                    bbox_txt_path=bbox_txt,
                    result_dir=out_dir,
                    mode="aggressive",
                    verbose=args.verbose,
                    seed=args.seed,
                    strength=args.strength,
                    split_ratio=split_ratio,
                )
                print("[INFO] Aggressive done ->", out_dir)


if __name__ == "__main__":
    main()
