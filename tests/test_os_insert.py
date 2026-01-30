"""One-click demo for OSInsertModel (conservative + aggressive modes).

This script assumes a single demo sample with the following layout:

    tests/source/background/Demo_0.png
    tests/source/foreground/Demo_0.png
    tests/source/foreground_mask/Demo_0.png
    tests/source/bbox/Demo_0.txt

It will run the conservative InsertAnything-only pipeline and write results to

    tests/result_dir/osinsert_demo
"""

import argparse
from pathlib import Path

from libcom.os_insert import OSInsertModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Which OSInsert mode to run in this demo.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    model_dir = repo_root / "model_dir"
    tests_source = repo_root / "tests" / "source"
    base_result_dir = repo_root / "tests" / "result_dir"
    result_dir_conservative = base_result_dir / "osinsert_demo"
    result_dir_aggressive = base_result_dir / "osinsert_demo_aggressive"

    bg = tests_source / "background" / "Demo_0.png"
    fg = tests_source / "foreground" / "Demo_0.png"
    fg_mask = tests_source / "foreground_mask" / "Demo_0.png"
    bbox_txt = tests_source / "bbox" / "Demo_0.txt"

    print("[INFO] repo_root =", repo_root)
    print("[INFO] model_dir =", model_dir)
    print("[INFO] tests/source =", tests_source)
    print("[INFO] result_dir_conservative =", result_dir_conservative)
    print("[INFO] result_dir_aggressive   =", result_dir_aggressive)
    print("[INFO] background =", bg)
    print("[INFO] foreground =", fg)
    print("[INFO] foreground_mask =", fg_mask)
    print("[INFO] bbox_txt =", bbox_txt)

    osinsert = OSInsertModel(model_dir=model_dir, device="cuda:0")
    print("[INFO] OSInsertModel created:", osinsert)

    if args.mode == "conservative":
        result_dir_conservative.mkdir(parents=True, exist_ok=True)
        osinsert(
            background_path=bg,
            foreground_path=fg,
            foreground_mask_path=fg_mask,
            bbox_txt_path=bbox_txt,
            result_dir=result_dir_conservative,
            mode="conservative",
            cleanup_intermediate=True,
            seed=123,
            strength=1.0,
        )
        print("[INFO] Conservative mode done. Results written to:", result_dir_conservative)

    if args.mode == "aggressive":
        result_dir_aggressive.mkdir(parents=True, exist_ok=True)
        osinsert(
            background_path=bg,
            foreground_path=fg,
            foreground_mask_path=fg_mask,
            bbox_txt_path=bbox_txt,
            result_dir=result_dir_aggressive,
            mode="aggressive",
            cleanup_intermediate=False,
            seed=123,
            strength=1.0,
        )
        print("[INFO] Aggressive mode done. Results written to:", result_dir_aggressive)


if __name__ == "__main__":
    main()
