import argparse
import os
import subprocess
from pathlib import Path


# ========= 用户配置（常用默认值集中在这里修改） =========
# 你可以直接修改下面这些默认值，然后只在命令行指定 --mode / --gpu 即可。
DEFAULT_GPU = 1
DEFAULT_OBJ_REPO = "/data/wangjingyuan/projects/ObjectStitch-Image-Composition"
DEFAULT_IA_REPO = "/data/wangjingyuan/projects/insert-anything"
DEFAULT_SAM_CKPT = "/data/wangjingyuan/models/sam/sam_vit_h_4b8939.pth"

# InsertAnything / FLUX 模型与 LoRA 路径（与原 insert-anything 仓库保持一致）
DEFAULT_FLUX_FILL_PATH = "/data/models/FLUX.1-Fill-dev"
DEFAULT_FLUX_REDUX_PATH = "/data/models/FLUX.1-Redux-dev"
DEFAULT_IA_LORA_PATH = "/data/models/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors"

# 处理图片与实验目录相关的默认路径（可以按需整体改到 /data/...）
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_ROOT = REPO_ROOT
DEFAULT_LIST_FILE = REPO_ROOT / "examples" / "samples_demo.tsv"
DEFAULT_OS_TEST_ROOT = REPO_ROOT / "os_test_demo"
DEFAULT_OS_OUT = REPO_ROOT / "objectstitch_out_demo"
DEFAULT_SAM_MASK_ROOT = REPO_ROOT / "sam_masks_demo"
DEFAULT_OSINSERT_OUT = REPO_ROOT / "osinsert_outputs_demo"
DEFAULT_TMP_ROOT = REPO_ROOT / "osinsert_tmp_demo"


def run(cmd, cwd=None, env=None):
    print("[RUN]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _safe_rmtree(path: Path):
    """递归删除目录，仅用于清理本脚本产生的中间结果。"""

    if not path.exists():
        return

    if path.is_file():
        path.unlink()
        return

    # 递归删除子内容
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            _safe_rmtree(child)

    # 最后删除空目录本身
    try:
        path.rmdir()
    except OSError:
        # 若目录非空（例如有并发写入），则保留，避免误删
        pass


def main():
    parser = argparse.ArgumentParser(
        description="One-click OSInsert pipeline: ObjectStitch + SAM + InsertAnything",
    )

    # 基本配置：GPU / 外部仓库路径 / 模式
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU, help="GPU id")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["aggressive", "conservative"],
        default="aggressive",
        help=(
            "OSInsert 运行模式：\n"
            "- aggressive：两阶段 OSInsert（ObjectStitch + SAM + InsertAnything）；\n"
            "- conservative：跳过 ObjectStitch & SAM，直接在 run_osinsert_pipeline.py 中走原始 InsertAnything 行为。"
        ),
    )
    parser.add_argument(
        "--obj_repo",
        type=str,
        default=DEFAULT_OBJ_REPO,
        help="Path to local ObjectStitch-Image-Composition repository",
    )
    parser.add_argument(
        "--ia_repo",
        type=str,
        default=DEFAULT_IA_REPO,
        help="Path to local insert-anything repository",
    )
    parser.add_argument(
        "--sam_ckpt",
        type=str,
        default=DEFAULT_SAM_CKPT,
        help="Path to SAM checkpoint (sam_vit_h_4b8939.pth)",
    )

    # demo 路径（多数情况下不用改，直接用仓库内 demo）
    parser.add_argument(
        "--exp_root",
        type=str,
        default=str(DEFAULT_EXP_ROOT),
        help="Experiment root directory (default: repository root)",
    )
    parser.add_argument(
        "--list_file",
        type=str,
        default=str(DEFAULT_LIST_FILE),
        help="TSV list file (uniq, bg, fg, fg_mask)",
    )
    parser.add_argument(
        "--os_test_root",
        type=str,
        default=str(DEFAULT_OS_TEST_ROOT),
        help="Root directory of os_test-style data (background/foreground/foreground_mask/bbox)",
    )
    parser.add_argument(
        "--os_out",
        type=str,
        default=str(DEFAULT_OS_OUT),
        help="Output directory for ObjectStitch results",
    )
    parser.add_argument(
        "--sam_mask_root",
        type=str,
        default=str(DEFAULT_SAM_MASK_ROOT),
        help="Output directory for SAM masks",
    )
    parser.add_argument(
        "--osinsert_out",
        type=str,
        default=str(DEFAULT_OSINSERT_OUT),
        help="Output directory for final OSInsert / InsertAnything results",
    )
    parser.add_argument(
        "--tmp_root",
        type=str,
        default=str(DEFAULT_TMP_ROOT),
        help="Temporary directory for intermediate visualizations",
    )

    # 是否在流程结束后清理中间结果
    parser.add_argument(
        "--cleanup_intermediate",
        action="store_true",
        help=(
            "若指定，则在流程结束后删除 ObjectStitch 输出目录、SAM mask 目录和 "
            "tmp_root 中的中间可视化结果。"
        ),
    )

    # ObjectStitch / InsertAnything 参数
    parser.add_argument("--os_seed", type=int, default=123)
    parser.add_argument("--os_steps", type=int, default=10)
    parser.add_argument("--ia_strength", type=float, default=1.0)
    parser.add_argument("--ia_seed", type=int, default=123)

    args = parser.parse_args()

    gpu_id = args.gpu
    exp_root = Path(args.exp_root)
    os_test_root = Path(args.os_test_root)
    list_file = Path(args.list_file)
    os_out = Path(args.os_out)
    sam_mask_root = Path(args.sam_mask_root)
    osinsert_out = Path(args.osinsert_out)
    tmp_root = Path(args.tmp_root)

    # ===== 阶段 1 & 2：仅在 aggressive 模式下运行 ObjectStitch + SAM =====
    if args.mode == "aggressive":
        print("[1/3] 在当前环境中运行 ObjectStitch 推理")
        run(
            [
                "python",
                "scripts/inference.py",
                "--outdir",
                str(os_out),
                "--testdir",
                str(os_test_root),
                "--num_samples",
                "1",
                "--sample_steps",
                str(args.os_steps),
                "--gpu",
                str(gpu_id),
                "--seed",
                str(args.os_seed),
                "--fixed_code",
                "True",
            ],
            cwd=args.obj_repo,
        )
        print(f"[1/3] ObjectStitch 完成，结果在 {os_out}")

        print("[2/3] 在当前环境中运行 SAM 提取前景 mask")
        if sam_mask_root.exists():
            # 清空旧目录
            for p in sam_mask_root.glob("*"):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    # 避免误删深层结构，这里只清空一层
                    for q in p.glob("*"):
                        if q.is_file():
                            q.unlink()
            # 不删除根目录本身
        sam_mask_root.mkdir(parents=True, exist_ok=True)

        run(
            [
                "python",
                "osinsert/run_sam_on_objectstitch.py",
                "--list",
                str(list_file),
                "--objectstitch_dir",
                str(os_out),
                "--bbox_dir",
                str(os_test_root / "bbox"),
                "--outdir",
                str(sam_mask_root),
                "--sam_checkpoint",
                args.sam_ckpt,
                "--model_type",
                "vit_h",
                "--device",
                f"cuda:{gpu_id}",
            ],
            cwd=str(exp_root),
        )

        mask_count = len(list(sam_mask_root.glob("*")))
        print(f"[2/3] SAM 完成，生成 mask 数量: {mask_count}")
    else:
        print("[1/3][2/3] conservative 模式：跳过 ObjectStitch 和 SAM，只在背景上做 InsertAnything")

    # ===== 阶段 3：OSInsert 第二阶段（InsertAnything） =====
    print("[3/3] 在当前环境中运行 InsertAnything (OSInsert 第二阶段)")

    env = os.environ.copy()

    # 让 osinsert/inference.py 能从环境变量读取 FLUX / LoRA 的路径
    env.setdefault("FLUX_FILL_PATH", DEFAULT_FLUX_FILL_PATH)
    env.setdefault("FLUX_REDUX_PATH", DEFAULT_FLUX_REDUX_PATH)
    env.setdefault("IA_LORA_PATH", DEFAULT_IA_LORA_PATH)

    ia_repo = args.ia_repo
    if ia_repo:
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ia_repo + (os.pathsep + prev if prev else "")

    run(
        [
            "python",
            "osinsert/run_osinsert_pipeline.py",
            "--mode",
            args.mode,
            "--list",
            str(list_file),
            "--bg_root",
            str(os_test_root / "background"),
            "--fg_root",
            str(os_test_root / "foreground"),
            "--fg_mask_root",
            str(os_test_root / "foreground_mask"),
            "--bbox_root",
            str(os_test_root / "bbox"),
            "--os_dir",
            str(os_out),
            "--sam_mask_root",
            str(sam_mask_root),
            "--outdir",
            str(osinsert_out),
            "--tmp_root",
            str(tmp_root),
            "--seed",
            str(args.ia_seed),
            "--strength",
            str(args.ia_strength),
            "--gpu",
            str(gpu_id),
        ],
        cwd=str(exp_root),
        env=env,
    )

    print(f"[3/3] OSInsert 完成，最终结果在 {osinsert_out}")
    if args.cleanup_intermediate:
        print("[CLEANUP] 根据 --cleanup_intermediate，开始删除中间目录 ...")
        for d in [os_out, sam_mask_root, tmp_root]:
            if d.exists():
                print(f"[CLEANUP] 删除 {d}")
                _safe_rmtree(d)
        print("[CLEANUP] 中间结果清理完成")
    print("=== 全流程结束：ObjectStitch + SAM + InsertAnything ===")


if __name__ == "__main__":
    main()
