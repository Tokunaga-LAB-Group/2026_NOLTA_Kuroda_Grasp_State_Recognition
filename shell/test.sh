#!/bin/bash
# このスクリプトの場所を基準にリポジトリのルートを求める
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${REPO_ROOT}/src/test.py"

#　推論対象データ 
TEST_DATA_PATH="${REPO_ROOT}/dataset2/test/test_sampling"

# リポジトリに収録済みの sampling100+100 モデルを使用
SVM_MODEL_PATH="${REPO_ROOT}/results/Supervsed_Learning/train/2026-1-15_TRAIN_sampling100+100/models/svm_classifier.pkl"

SAVE_NAME="test_sampling_SVM_sampling100+100"

# 保存先
SAVE_PATH="${REPO_ROOT}/results/Supervsed_Learning/test/${SAVE_NAME}"
LOG_PATH="${REPO_ROOT}/logs/${SAVE_NAME}"

# 推論設定 
GPU_ID=0
IMG_H=256
IMG_W=256

mkdir -p "${SAVE_PATH}"
mkdir -p "${LOG_PATH}"

# 実行
nohup python "${PYTHON_SCRIPT}" \
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
