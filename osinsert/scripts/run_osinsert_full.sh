#!/usr/bin/env bash

# ==========================================================
# OSInsert 两阶段一键推理脚本（COCOEE / 任意 TSV 列表）
# 第一步：用 ObjectStitch 生成合成结果
# 第二步：对合成结果跑 SAM，得到前景 mask
# 第三步：组合 原始背景 + ObjectStitch 结果 + SAM mask，
#         构造 source image & mask，送入 InsertAnything 得到最终结果
#
# 说明：本脚本假设你已经在终端中手动 `conda activate osinsert` 或其它统一环境，
# 不再在脚本内部切换 conda 环境。
# ==========================================================

set -e

########## 用户可配置区 ##########

# GPU id
GPU_ID=1

# 仓库路径
# 当前 OSInsert 仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OSINSERT_DIR="${REPO_ROOT}/osinsert"   # run_sam_on_objectstitch.py / run_osinsert_pipeline.py 所在目录

# 外部仓库：根据你当前服务器实际路径填写
OBJ_REPO="/data/wangjingyuan/projects/ObjectStitch-Image-Composition"
IA_REPO="/data/wangjingyuan/projects/insert-anything"

# 实验根目录：为方便复现，默认使用当前仓库内的 demo 目录
# 如需在本地跑大规模实验，可将下方路径改为 /data/...，但不建议将这些绝对路径提交到 Git 仓库。
EXP_ROOT="${REPO_ROOT}"

# 列表与 os_test 结构（与 README 中 demo 说明保持一致）
LIST_FILE="${REPO_ROOT}/examples/samples_demo.tsv"   # TSV: uniq \t bg \t fg \t fgmask
OS_TEST_ROOT="${REPO_ROOT}/os_test_demo"             # demo: 预先提供/构建的 os_test 结构
OS_OUT="${REPO_ROOT}/objectstitch_out_demo"          # ObjectStitch 输出（demo）
SAM_MASK_ROOT="${REPO_ROOT}/sam_masks_demo"          # SAM 输出 mask 目录（demo）
OSINSERT_OUT="${REPO_ROOT}/osinsert_outputs_demo"    # OSInsert/InsertAnything 最终输出（demo）
TMP_ROOT="${REPO_ROOT}/osinsert_tmp_demo"            # 中间 source/mask 可视化目录（demo）

# ObjectStitch 采样参数
OS_SEED=123
OS_STEPS=10

# SAM 模型配置（已按你当前服务器路径填写）
SAM_CKPT="/data/wangjingyuan/models/sam/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE="vit_h"
SAM_DEVICE="cuda:${GPU_ID}"

# InsertAnything / OSInsert 参数
IA_STRENGTH=1.0
IA_SEED=123

########## 阶段 1：ObjectStitch 合成 ##########

echo "[1/3] 在当前环境中运行 ObjectStitch 推理"
cd "${OBJ_REPO}"

python scripts/inference.py \
  --outdir "${OS_OUT}" \
  --testdir "${OS_TEST_ROOT}" \
  --num_samples 1 --sample_steps ${OS_STEPS} --gpu ${GPU_ID} --seed ${OS_SEED} --fixed_code True

echo "[1/3] ObjectStitch 完成，结果在 ${OS_OUT}"

########## 阶段 2：SAM 提取前景 mask ##########

echo "[2/3] 在当前环境中运行 SAM 提取前景 mask"
cd "${EXP_ROOT}"

rm -rf "${SAM_MASK_ROOT}"
mkdir -p "${SAM_MASK_ROOT}"

python run_sam_on_objectstitch.py \
  --list "${LIST_FILE}" \
  --objectstitch_dir "${OS_OUT}" \
  --bbox_dir "${OS_TEST_ROOT}/bbox" \
  --outdir "${SAM_MASK_ROOT}" \
  --sam_checkpoint "${SAM_CKPT}" \
  --model_type "${SAM_MODEL_TYPE}" \
  --device "${SAM_DEVICE}"

MASK_COUNT=$(ls "${SAM_MASK_ROOT}" | wc -l)
echo "[2/3] SAM 完成，生成 mask 数量: ${MASK_COUNT}"

########## 阶段 3：OSInsert 第二阶段（InsertAnything） ##########

echo "[3/3] 在当前环境中运行 InsertAnything (OSInsert 第二阶段)"
cd "${EXP_ROOT}"

PYTHONPATH="${IA_REPO}" \
python run_osinsert_pipeline.py \
  --list "${LIST_FILE}" \
  --bg_root "${OS_TEST_ROOT}/background" \
  --fg_root "${OS_TEST_ROOT}/foreground" \
  --fg_mask_root "${OS_TEST_ROOT}/foreground_mask" \
  --bbox_root "${OS_TEST_ROOT}/bbox" \
  --os_dir "${OS_OUT}" \
  --sam_mask_root "${SAM_MASK_ROOT}" \
  --outdir "${OSINSERT_OUT}" \
  --tmp_root "${TMP_ROOT}" \
  --seed ${IA_SEED} \
  --strength ${IA_STRENGTH} \
  --gpu ${GPU_ID}
echo "[3/3] OSInsert 完成，最终结果在 ${OSINSERT_OUT}"
echo "=== 全流程结束：ObjectStitch + SAM + InsertAnything ==="
