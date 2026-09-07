#!/bin/bash

# ======================================
# CLIP cosine similarity 推論用 shell
# ======================================

# ==== パス設定 ====

# 入力データ（0/1 のサブディレクトリを想定）
INPUT_PATH="/data/Users/kuroda/2025_research/2025-12-28_kiban/test_sampling"

# 実験名（ログ・保存先に使う）
EXP_NAME="2026-1-5_kiban_test_demo_CLIP_results_dfn5b_debug"

# 保存先ルート
SAVE_ROOT="/data/Users/kuroda/2025_research/2025_result/2025-12-28_kiban/test/CLIP"
SAVE_PATH="${SAVE_ROOT}/${EXP_NAME}"

# ログ保存先
LOG_ROOT="/data/Users/kuroda/2025_research/2025_result/2025-12-28_kiban/log"
LOG_PATH="${LOG_ROOT}/${EXP_NAME}"

# 使用GPU
GPU_ID=3
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ==== CLIP設定 ====
MODEL_NAME="ViT-H-14-378-quickgelu" #デフォルト：ViT-H-14-378-quickgelu
PRETRAINED="dfn5b" #デフォルト：dfn5b
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
nohup python /data/Users/kuroda/2026_NOLTA_Kuroda_Grasp_State_Recognition/src/CLIP_cosine similarity.py \
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
