#!/bin/bash

# ==== パスの設定 ====

# このスクリプトの場所を基準にリポジトリのルートを求める
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${REPO_ROOT}/src/train.py"

# 訓練データ（0/1 のサブディレクトリを想定）
TRAIN_PATH="${REPO_ROOT}/dataset2/train/sampling100+100"

# 保存先
SAVE_NAME="train_sampling"
SAVE_ROOT="${REPO_ROOT}/results/Supervsed_Learning/train"
SAVE_PATH="${SAVE_ROOT}/${SAVE_NAME}"
LOG_PATH="${REPO_ROOT}/logs"

# 使用GPU
GPU_ID=0

# Logディレクトリの作成
mkdir -p "${LOG_PATH}/${SAVE_NAME}"

# 結果保存ディレクトリの作成
mkdir -p "${SAVE_PATH}"

# ==== 実行 ====
nohup python "${PYTHON_SCRIPT}" \
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
