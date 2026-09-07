#!/bin/bash

# ==== パスの設定 ====

# 訓練データ（0/1 のサブディレクトリを想定）
TRAIN_PATH="/data/Users/kuroda/2025_research/2025-12-08_train_data/train"

# 保存先
SAVE_NAME="2025-12-08_ResNet50_SVM_C"
SAVE_ROOT="./2025_result/2025-12-08_train"
SAVE_PATH="${SAVE_ROOT}/${SAVE_NAME}"
LOG_PATH="./logs"

# 使用GPU
GPU_ID=0

# Logディレクトリの作成
mkdir -p "${LOG_PATH}/${SAVE_NAME}"

# 結果保存ディレクトリの作成
mkdir -p "${SAVE_PATH}"

# ==== 実行 ====
nohup python ./src/train.py \
    --gpu ${GPU_ID} \
    --train_data_path "${TRAIN_PATH}" \
    --save_path "${SAVE_PATH}" \
    --resize 256 256 \
    --batch_size 32 \
    --C 10 \
    > "${LOG_PATH}/${SAVE_NAME}/training_phase.log" 2>&1 &

# ==== Notes ====
# - SVM kernel : rbf 固定
# - gamma     : scale（デフォルト）
# - C         : 固定値
