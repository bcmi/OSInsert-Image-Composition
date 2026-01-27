import sys,os, argparse, numpy as np, cv2, torch
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from utils.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask

proj_dir = "/data/wangjingyuan/projects/insert-anything"
sys.path.insert(0, proj_dir)
os.chdir(proj_dir)
dtype = torch.bfloat16
size = (768, 768)

def load_pipes(fill_dir, redux_dir, lora_path, device="cuda"):
    pipe = FluxFillPipeline.from_pretrained(fill_dir, torch_dtype=dtype)
    pipe.load_lora_weights(lora_path)
    pipe.to(device)
    redux = FluxPriorReduxPipeline.from_pretrained(redux_dir).to(dtype=dtype).to(device)
    return pipe, redux

def make_masks(bg_path, fg_mask_path, bbox, sam_mask_path=None):
    bg = cv2.imread(bg_path)
    if bg is None:
        raise FileNotFoundError(bg_path)
    # convert BGR (cv2) -> RGB for downstream PIL / diffusion
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    if fg_mask_path and os.path.exists(fg_mask_path):
        fm = cv2.imread(fg_mask_path, cv2.IMREAD_GRAYSCALE)
        _, fm = cv2.threshold(fm, 127, 255, cv2.THRESH_BINARY)
    else:
        fm = np.ones(bg.shape[:2], dtype=np.uint8)*255
    bg_h,bg_w = bg.shape[:2]

    # target mask for InsertAnything: prefer SAM mask if provided, otherwise use bbox
    if sam_mask_path is not None and os.path.exists(sam_mask_path):
        tar = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
        if tar is not None:
            # 将 SAM mask 重采样到和背景图一致的大小
            if tar.shape[0] != bg_h or tar.shape[1] != bg_w:
                tar = cv2.resize(tar, (bg_w, bg_h), interpolation=cv2.INTER_NEAREST)
            _, tar = cv2.threshold(tar, 127, 255, cv2.THRESH_BINARY)
            mask = (tar > 0).astype(np.uint8)
        else:
            # SAM mask 读取失败，回退到 bbox
            mask = np.zeros((bg_h,bg_w), dtype=np.uint8)
            x1,y1,x2,y2 = bbox
            mask[y1:y2, x1:x2] = 1
    else:
        # 没有 SAM mask，用 bbox
        mask = np.zeros((bg_h,bg_w), dtype=np.uint8)
        x1,y1,x2,y2 = bbox
        mask[y1:y2, x1:x2] = 1
    return bg, fm, mask

def process_one(pipe, redux, bg_path, fg_path, fg_mask_path, bbox, seed, out_path, sam_mask_path=None):
    bg, fg_mask, tar_mask = make_masks(bg_path, fg_mask_path, bbox, sam_mask_path=sam_mask_path)
    fg = cv2.imread(fg_path)
    if fg is None:
        raise FileNotFoundError(fg_path)
    # convert BGR (cv2) -> RGB to be consistent with bg / PIL
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    fg_mask_3 = np.stack([fg_mask,fg_mask,fg_mask], -1)
    ref_image = fg * (fg_mask_3>0) + np.ones_like(fg)*255*(fg_mask_3==0)
    ref_mask = fg_mask

    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    masked_ref_image = ref_image[ref_box_yyxx[0]:ref_box_yyxx[1], ref_box_yyxx[2]:ref_box_yyxx[3], :]
    ref_mask_crop   = ref_mask[ref_box_yyxx[0]:ref_box_yyxx[1], ref_box_yyxx[2]:ref_box_yyxx[3]]
    masked_ref_image, ref_mask_crop = expand_image_mask(masked_ref_image, ref_mask_crop, ratio=1.3)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)

    kernel = np.ones((7,7), np.uint8)
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=2)

    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
    tar_box_yyxx_crop =  expand_bbox(bg, tar_box_yyxx, ratio=2)
    tar_box_yyxx_crop = box2squre(bg, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx_crop

    old_tar_image = bg.copy()
    tar_image = bg[y1:y2,x1:x2,:]
    tar_mask_crop = tar_mask[y1:y2,x1:x2]

    H1, W1 = tar_image.shape[:2]
    tar_mask_crop = pad_to_square(tar_mask_crop, pad_value=0)
    tar_mask_crop = cv2.resize(tar_mask_crop, size)

    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    pipe_prior_output = redux(Image.fromarray(masked_ref_image))

    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2 = tar_image.shape[:2]
    tar_image = cv2.resize(tar_image, size)
    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)

    tar_mask_crop = np.stack([tar_mask_crop,tar_mask_crop,tar_mask_crop],-1)
    mask_black = np.ones_like(tar_image)*0
    mask_diptych = np.concatenate([mask_black, tar_mask_crop], axis=1)

    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych==1]=255
    mask_diptych = Image.fromarray(mask_diptych)

    generator = torch.Generator("cuda").manual_seed(seed)
    edited_image = pipe(
        image=diptych_ref_tar,
        mask_image=mask_diptych,
        height=mask_diptych.size[1],
        width=mask_diptych.size[0],
        max_sequence_length=512,
        generator=generator,
        **pipe_prior_output,
    ).images[0]

    width, height = edited_image.size
    left = width//2
    edited_image = edited_image.crop((left,0,width,height))
    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop))
    edited_image = Image.fromarray(edited_image)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    edited_image.save(out_path)
    print("saved", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[123,456])
    ap.add_argument("--fill_dir", default="/data/models/FLUX.1-Fill-dev")
    ap.add_argument("--redux_dir", default="/data/models/FLUX.1-Redux-dev")
    ap.add_argument("--lora", default="/data/models/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index, e.g. 0 or 1")
    ap.add_argument("--sam_mask_dir", default=None, help="optional dir containing SAM masks named as uniq.png")
    ap.add_argument("--bbox_root", default="/data/wangjingyuan/exp/os_test/bbox", help="directory containing {uniq}.txt bbox files")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    pipe, redux = load_pipes(args.fill_dir, args.redux_dir, args.lora, device=device)
    with open(args.list) as f:
        for line in f:
            parts=line.strip().split("\t")
            if len(parts)<4: continue
            uniq,bg,fg,mask = parts[:4]
            bbox_txt = os.path.join(args.bbox_root, f"{uniq}.txt")
            if not os.path.exists(bbox_txt):
                print("skip no bbox", uniq); continue
            with open(bbox_txt) as fb:
                x1,y1,x2,y2 = map(int, fb.readline().strip().split())
            for seed in args.seeds:
                out_path = os.path.join(args.outdir, uniq, f"seed{seed}.png")
                sam_mask_path = None
                if args.sam_mask_dir is not None:
                    sam_mask_path = os.path.join(args.sam_mask_dir, f"{uniq}.png")
                process_one(pipe, redux, bg, fg, mask, (x1,y1,x2,y2), seed, out_path, sam_mask_path=sam_mask_path)

if __name__ == "__main__":
    main()
