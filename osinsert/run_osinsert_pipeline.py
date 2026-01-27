import os
import argparse

import cv2
import numpy as np

from tqdm import tqdm


def make_source_and_mask(bg_path, os_path, sam_mask_path):
    """构造第二阶段 InsertAnything 的 source image 和 mask。

    为了避免"原始背景中已有物体"与 ObjectStitch 结果在同一区域产生冲突，
    这里直接将完整的 ObjectStitch 图像作为 source，只用 SAM mask 限定可编辑区域。

    - source = resize(objectstitch, 与背景同分辨率)
    - mask   = resize(SAM mask, 与 source 同分辨率，二值 0/255)
    """
    bg = cv2.imread(bg_path)
    if bg is None:
        raise FileNotFoundError(bg_path)

    os_img = cv2.imread(os_path)
    if os_img is None:
        raise FileNotFoundError(os_path)

    mask = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(sam_mask_path)

    # 统一到与背景相同的分辨率，便于和 COCOEE/评测对齐
    h, w = bg.shape[:2]
    os_img = cv2.resize(os_img, (w, h), interpolation=cv2.INTER_AREA)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 直接使用 ObjectStitch 整图作为 source，避免与原始背景在插入区域发生混合冲突
    src = os_img

    return src, mask


def ensure_sam_mask(uniq, sam_mask_root):
    """返回已经预先计算好的 SAM mask 路径。

    这里假设第一阶段的 run_sam_on_objectstitch.py 已经在 sam_mask_root 下
    生成了 {uniq}.png；如果没有，就跳过该样本。
    """
    os.makedirs(sam_mask_root, exist_ok=True)
    out_path = os.path.join(sam_mask_root, f"{uniq}.png")
    if os.path.exists(out_path):
        return out_path
    # 若不存在，返回 None，由上层决定是否跳过
    return None


def process_one(uniq, args):
    """根据 mode 运行 OSInsert：

    - aggressive：完整两阶段流程（ObjectStitch + SAM + combined source + IA）。
    - conservative：跳过 ObjectStitch/SAM，仅使用 bg/fg/fg_mask/bbox 构造 mask，
      按原始 InsertAnything 方式在背景上编辑。
    """

    bg_path = os.path.join(args.bg_root, f"{uniq}.png")
    fg_path = os.path.join(args.fg_root, f"{uniq}.png")
    fg_mask_path = os.path.join(args.fg_mask_root, f"{uniq}.png")
    bbox_txt_path = os.path.join(args.bbox_root, f"{uniq}.txt")

    if not (os.path.exists(bg_path) and os.path.exists(fg_path)
            and os.path.exists(fg_mask_path) and os.path.exists(bbox_txt_path)):
        print("skip", uniq, "(missing inputs)")
        return

    # ===== aggressive：两阶段 OSInsert =====
    if args.mode == "aggressive":
        os_path = os.path.join(args.os_dir, f"{uniq}.jpg")
        if not os.path.exists(os_path):
            print("skip", uniq, "(missing ObjectStitch output)")
            return

        sam_mask_path = ensure_sam_mask(uniq, args.sam_mask_root)
        if sam_mask_path is None:
            print("skip", uniq, "(no SAM mask in", args.sam_mask_root, ")")
            return

        src_bgr, mask = make_source_and_mask(bg_path, os_path, sam_mask_path)

    # ===== conservative：仅使用原始 bg + bbox 作为编辑区域 =====
    else:
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            print("skip", uniq, "(failed to read background)")
            return

        h, w = bg_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(bbox_txt_path, "r") as f:
            line = f.readline().strip()
        try:
            x1, y1, x2, y2 = map(int, line.split())
        except Exception:
            print("skip", uniq, "(invalid bbox)")
            return

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            print("skip", uniq, "(degenerate bbox)")
            return

        mask[y1:y2, x1:x2] = 255
        src_bgr = bg_img

    # 保存中间结果（两种模式共用）
    os.makedirs(args.tmp_root, exist_ok=True)
    src_suffix = "aggressive" if args.mode == "aggressive" else "conservative"
    src_path = os.path.join(args.tmp_root, f"{uniq}_source_{src_suffix}.png")
    mask_path_out = os.path.join(args.tmp_root, f"{uniq}_mask_{src_suffix}.png")
    cv2.imwrite(src_path, src_bgr)
    cv2.imwrite(mask_path_out, mask)

    # 调用 InsertAnything diptych 流程
    out_dir = os.path.join(args.outdir, uniq)
    os.makedirs(out_dir, exist_ok=True)

    run_insertanything(
        source_image_path=src_path,
        mask_image_path=mask_path_out,
        ref_image_path=fg_path,
        ref_mask_path=fg_mask_path,
        seeds=[args.seed],
        strength=float(args.strength),
        save_path=out_dir,
    )
    print("done", uniq, "[mode=", args.mode, "]")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OSInsert pipeline: "
            "- aggressive: ObjectStitch + SAM -> combined source -> InsertAnything; "
            "- conservative: bg + bbox -> mask -> InsertAnything."
        )
    )
    parser.add_argument("--list", required=True, help="tsv like samples.tsv, first column is uniq id")
    parser.add_argument("--bg_root", default="os_test/background")
    parser.add_argument("--fg_root", default="os_test/foreground")
    parser.add_argument("--fg_mask_root", default="os_test/foreground_mask")
    parser.add_argument("--bbox_root", default="os_test/bbox")
    parser.add_argument("--os_dir", default="objectstitch_out")
    parser.add_argument("--sam_mask_root", default="sam_masks")
    parser.add_argument("--outdir", default="osinsert_outputs")
    parser.add_argument("--tmp_root", default="osinsert_tmp")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["aggressive", "conservative"],
        default="aggressive",
        help="OSInsert 模式：aggressive（两阶段）或 conservative（仅 bg+bbox）",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (mapped via CUDA_VISIBLE_DEVICES)")

    args = parser.parse_args()

    # 在导入 InsertAnything 之前指定要使用的 GPU，
    # 这样 inference.py 里 torch.device("cuda") 会绑定到这块卡。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 作为全局符号导入，供 process_one 调用
    global run_insertanything
    from inference import run_insertanything  # 第二阶段：InsertAnything diptych 推理

    with open(args.list) as f:
        lines = [line for line in f if line.strip()]

    for line in tqdm(lines, desc="OSInsert", ncols=80):
        parts = line.strip().split("\t")
        if not parts:
            continue
        uniq = parts[0]
        process_one(uniq, args)


if __name__ == "__main__":
    main()
