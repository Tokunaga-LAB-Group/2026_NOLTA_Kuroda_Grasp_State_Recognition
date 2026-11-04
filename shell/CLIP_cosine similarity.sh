#!/bin/bash

# ======================================
# CLIP cosine similarity 推論用 shell
# ======================================

# ==== パス設定 ====

# このスクリプトの場所を基準にリポジトリのルートを求める
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${REPO_ROOT}/src/CLIP_cosine similarity.py"

# 入力データ（0/1 のサブディレクトリを想定）
INPUT_PATH="${REPO_ROOT}/dataset2/test/test_sampling"

# 実験名（ログ・保存先に使う）
EXP_NAME="CLIP_cosine_similarity_test_sampling"

# 保存先ルート
SAVE_ROOT="${REPO_ROOT}/results/CLIP_result"
SAVE_PATH="${SAVE_ROOT}/${EXP_NAME}"

# ログ保存先
LOG_ROOT="${REPO_ROOT}/logs"
LOG_PATH="${LOG_ROOT}/${EXP_NAME}"

# 使用GPU
GPU_ID=3
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ==== CLIP設定 ====
MODEL_NAME="ViT-L-14-quickgelu"
PRETRAINED="metaclip_fullcc"
CROP_SIZE=1080

# ==== テキスト設定 ====
TEXT_GRASP="Held object."
TEXT_NON_GRASP="Not held object"

# ==== ディレクトリ作成 ====
mkdir -p "${SAVE_PATH}"
mkdir -p "${LOG_PATH}"

# ==== 出力ファイル ====
OUT_CSV="${SAVE_PATH}/clip_results.csv"
OUT_LOG="${LOG_PATH}/clip.log"

# ==== 実行 ====
nohup python "${PYTHON_SCRIPT}" \
  --input_dir "${INPUT_PATH}" \
  --output_csv "${OUT_CSV}" \
  --gpu "${GPU_ID}" \
  --crop_size "${CROP_SIZE}" \
  --model "${MODEL_NAME}" \
  --pretrained "${PRETRAINED}" \
  --text_grasp "${TEXT_GRASP}" \
  --text_non_grasp "${TEXT_NON_GRASP}" \
  --strict \
  > "${OUT_LOG}" 2>&1 &

echo "======================================"
echo " CLIP inference started"
echo "  Input : ${INPUT_PATH}"
echo "  Output: ${OUT_CSV}"
echo "  GPU   : ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  Model : ${MODEL_NAME} (${PRETRAINED})"
echo "  TextG : ${TEXT_GRASP}"
echo "  TextN : ${TEXT_NON_GRASP}"
echo "  Log   : ${OUT_LOG}"
echo "======================================"
