import os
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm


def load_bboxes(bbox_dir):
    """Load bboxes from txt files in bbox_dir."""
    mp = {}
    for fname in os.listdir(bbox_dir):
        if not fname.endswith(".txt"):
            continue
        uniq = fname[:-4]
        p = os.path.join(bbox_dir, fname)
        try:
            with open(p) as f:
                x1, y1, x2, y2 = map(int, f.readline().strip().split())
            mp[uniq] = (x1, y1, x2, y2)
        except Exception:
            continue
    return mp




def detect_yellow_bbox_from_grid(grid_image_bgr):
    """从 grid 图中检测黄色 bbox 的坐标。
    
    grid 图的格式：[背景+黄色bbox | 前景图 | 合成结果]
    黄色 bbox 在左边的背景图上，标记了前景图在合成结果中的位置。
    
    返回：(x1, y1, x2, y2) 相对于整个 grid 图的坐标，或 None 如果检测失败
    """
    # 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(grid_image_bgr, cv2.COLOR_BGR2HSV)
    
    # 黄色的 HSV 范围（OpenCV 中 H 是 0-180）
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    
    # 创建黄色掩码
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 找轮廓
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 找最大的轮廓（应该是黄色 bbox）
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return (x, y, x + w, y + h)


def extract_bbox_from_mask(mask_image):
    """从 mask 图中提取前景物体的 bbox。
    
    mask 图是二值图，白色（255）表示前景物体，黑色（0）表示背景。
    
    返回：(x1, y1, x2, y2) 或 None 如果没有前景像素
    """
    # 确保是灰度图
    if len(mask_image.shape) == 3:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    
    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 找最大的轮廓（前景物体）
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return (x, y, x + w, y + h)


def run_sam_on_image(predictor, image_bgr, box_xyxy):
    """Run SAM with a box prompt and return a clean binary mask.

    Strategy:
    - use SAM with box prompt to constrain the segmentation region
    - choose the mask with the highest confidence score (not just largest area)
    - apply a very relaxed area threshold to accept most masks
    """
    # SAM 期望 RGB
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    # 用 box prompt：框住整个 bbox 区域
    box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=True)
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM returned no masks")

    # 选择 SAM 置信度最高的 mask（而不是面积最大的）
    # scores 是 SAM 对每个 mask 的置信度评分
    best_idx = int(np.argmax(scores))
    m = masks[best_idx].astype(np.uint8)

    # 非常宽松的面积阈值：只要有一点点像素就接受（>=1% bbox 面积）
    x1, y1, x2, y2 = box_xyxy
    box_w = max(int(x2 - x1), 1)
    box_h = max(int(y2 - y1), 1)
    box_area = float(box_w * box_h)
    area = float(int(m.sum()))
    if area / box_area < 0.01:
        raise RuntimeError("SAM mask area too small")

    m = m.astype(np.uint8) * 255
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="tsv file like samples_50.tsv, first column is uniq id")
    ap.add_argument("--objectstitch_dir", required=True, help="directory containing ObjectStitch results, e.g. objectstitch_out")
    ap.add_argument("--bbox_dir", default="os_test/bbox", help="bbox txt dir (uniq.txt with x1 y1 x2 y2)")
    ap.add_argument("--outdir", default="sam_masks", help="where to save SAM masks as uniq.png")
    ap.add_argument("--sam_checkpoint", required=True, help="path to SAM checkpoint .pth")
    ap.add_argument("--model_type", default="vit_h", help="SAM model type, e.g. vit_h / vit_l / vit_b")
    ap.add_argument("--device", default="cuda", help="torch device, e.g. cuda:0 or cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 延迟导入 SAM，避免在没装包时影响别的脚本
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as e:
        raise ImportError("segment_anything package is required. Install from https://github.com/facebookresearch/segment-anything") from e

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # 从 bbox_dir 读取在背景图坐标系下的 bbox（x1, y1, x2, y2）
    bbox_map = load_bboxes(args.bbox_dir)

    with open(args.list) as f:
        lines = [line for line in f if line.strip()]

    for line in tqdm(lines, desc="SAM on ObjectStitch", ncols=80):
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        uniq = parts[0]
        bg_path = parts[1]  # 样本列表第二列是背景图路径

        # 读取背景图，得到原始尺寸（bbox 是在这个尺寸下定义的）
        bg_image_bgr = cv2.imread(bg_path)
        if bg_image_bgr is None:
            print("skip read fail bg image", uniq, bg_path)
            continue
        bg_h, bg_w = bg_image_bgr.shape[:2]

        # 读取 ObjectStitch 的合成结果（SAM 的输入）
        os_img_path = os.path.join(args.objectstitch_dir, uniq + ".jpg")
        if not os.path.exists(os_img_path):
            print("skip no os image", uniq)
            continue

        os_image_bgr = cv2.imread(os_img_path)
        if os_image_bgr is None:
            print("skip read fail os image", uniq)
            continue

        os_h, os_w = os_image_bgr.shape[:2]

        # 必须在 bbox_map 里有对应的 bbox
        if uniq not in bbox_map:
            print("skip no bbox", uniq)
            continue

        bx1, by1, bx2, by2 = bbox_map[uniq]

        # 按背景图尺寸缩放到 ObjectStitch 合成图的尺寸
        scale_x = os_w / float(bg_w) if bg_w > 0 else 1.0
        scale_y = os_h / float(bg_h) if bg_h > 0 else 1.0

        x1 = int(bx1 * scale_x)
        y1 = int(by1 * scale_y)
        x2 = int(bx2 * scale_x)
        y2 = int(by2 * scale_y)

        # 确保 bbox 在合成结果范围内
        x1 = max(0, min(x1, os_w - 1))
        y1 = max(0, min(y1, os_h - 1))
        x2 = max(x1 + 1, min(x2, os_w))
        y2 = max(y1 + 1, min(y2, os_h))

        try:
            os_mask = run_sam_on_image(predictor, os_image_bgr, (x1, y1, x2, y2))
        except Exception as e:
            print("sam failed", uniq, e)
            continue

        out_path = os.path.join(args.outdir, f"{uniq}.png")
        cv2.imwrite(out_path, os_mask)
        print("saved SAM mask", out_path)


if __name__ == "__main__":
    main()
