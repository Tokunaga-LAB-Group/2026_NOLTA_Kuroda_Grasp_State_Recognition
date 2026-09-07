#!/bin/bash
# ================================
# ResNet + SVM 推論スクリプト
# ================================

# -------- 推論対象データ --------
TEST_DATA_PATH="/data/Users/kuroda/2025_research/2025-12-09_test_sampling400/demo"
# 例:
# test/
#   ├─ 0/  (把持)
#   └─ 1/  (非把持)

# -------- 学習済み SVM --------
SVM_MODEL_PATH="/data/Users/kuroda/2025_research/2025_result/2025-12-08_train/2025-12-09_ResNet50_SVM_trainTest3/models/svm_classifier.pkl"

# -------- 実験名 --------
SAVE_NAME="2025-12-09-ResNet50_SVM_testTest3_sampling400"

# -------- 保存先 --------
SAVE_PATH="./2025_result/2025-12-09_test/${SAVE_NAME}"
LOG_PATH="./logs/${SAVE_NAME}"

# -------- 推論設定 --------
GPU_ID=0
IMG_H=256
IMG_W=256

# -------------------------------
# ディレクトリ作成
# -------------------------------
mkdir -p "${SAVE_PATH}"
mkdir -p "${LOG_PATH}"

# -------------------------------
# 推論実行
# -------------------------------
nohup python ./src/test.py \
    --gpu "${GPU_ID}" \
    --test_data_path "${TEST_DATA_PATH}" \
    --svm_model_path "${SVM_MODEL_PATH}" \
    --save_path "${SAVE_PATH}" \
    --resize "${IMG_H}" "${IMG_W}" \
    > "${LOG_PATH}/test.log" 2>&1 &

echo "========================================"
echo "Test job launched."
echo "Log file : ${LOG_PATH}/test.log"
echo "Save dir : ${SAVE_PATH}"
echo "========================================"
