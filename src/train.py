import os
import sys
import warnings

# ログや警告の抑制
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"  # 環境によってはなくてもOK
sys.path.append(os.getcwd())
sys.path.append("..")
warnings.filterwarnings("ignore")

import argparse
import pprint
import pickle
import json
import re
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as kimage
from sklearn.svm import SVC


# 引数の設定
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")

    # データ・保存先
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to training images (directory).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save SVM model and results.",
    )

    # 画像とバッチ
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size. H:int W:int (for ResNet).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )

    # ★ C固定値（shellで変更できる）
    parser.add_argument(
        "--C",
        type=float,
        default=10.0,
        help="Fixed C value for SVM (rbf). Default=10.0",
    )

    # group抽出に使う正規表現（ログ用：ファイル名に videoX が入る想定）
    parser.add_argument(
        "--video_regex",
        type=str,
        default=r"(video\d+)",
        help="Regex to extract video id from filename (e.g., '(video\\d+)').",
    )

    return parser.parse_args()


# ディレクトリ作成
def get_create_dir(*args):
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


# ResNetによる特徴量抽出
def build_resnet_feature_extractor(H_img, W_img):
    """
    ResNet を特徴量抽出器として構築する。
    画像を入力すると 1D ベクトル (グローバルプーリング後) を出力するモデル。
    """
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(H_img, W_img, 3),
        pooling="avg",
    )
    base_model.trainable = False
    return base_model


def list_image_files_with_labels(root_dir: str) -> List[Tuple[str, int]]:
    """
    root_dir/
      0/*.png ...
      1/*.png ...
    から (filepath, label) を集める
    """
    items: List[Tuple[str, int]] = []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    for label in [0, 1]:
        class_dir = os.path.join(root_dir, str(label))
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class dir not found: {class_dir}")

        for name in os.listdir(class_dir):
            if name.lower().endswith(exts):
                items.append((os.path.join(class_dir, name), label))

    # 再現性のために固定順
    items.sort(key=lambda x: x[0])
    return items


def extract_video_id(filename: str, pattern: str) -> str:
    """
    例: kiban_video4_0000.png -> video4
    """
    m = re.search(pattern, filename)
    if m:
        return m.group(1)
    return "unknown"


def extract_features_from_files(model, file_label_list, image_size, batch_size):
    """
    ファイルリストから直接読み込み → preprocess → ResNet特徴抽出
    """
    paths = [p for p, _ in file_label_list]
    labels = np.array([lab for _, lab in file_label_list], dtype=np.int64)

    feats = []
    n = len(paths)

    for i in range(0, n, batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_imgs = []

        for p in batch_paths:
            img = kimage.load_img(p, target_size=image_size)
            arr = kimage.img_to_array(img)
            batch_imgs.append(arr)

        batch = np.stack(batch_imgs, axis=0)
        batch = keras.applications.resnet50.preprocess_input(batch)

        f = model(batch, training=False).numpy()
        feats.append(f)

    X = np.concatenate(feats, axis=0)
    y = labels
    return X, y


def train_svm_fixedC(X, y, args):
    """
    C を固定して SVM(rbf) を学習する（gamma=scale）。
    """
    svm = SVC(
        kernel="rbf",
        gamma="scale",
        C=float(args.C),
        probability=False,
        class_weight="balanced",
        random_state=None,  # SVCは内部的に確率推定等しない限り基本決定的
    )
    svm.fit(X, y)
    return svm


def training_phase(args):
    SAVE_DIR = get_create_dir(args.save_path)
    MODEL_DIR = get_create_dir(SAVE_DIR, "models")

    H_img, W_img = args.resize

    # ファイル一覧（順序固定）
    items = list_image_files_with_labels(args.train_data_path)
    paths = [p for p, _ in items]

    # groups = videoID（ログ用）
    groups = np.array(
        [extract_video_id(os.path.basename(p), args.video_regex) for p in paths],
        dtype=object,
    )
    uniq, cnt = np.unique(groups, return_counts=True)
    print("[Group summary] video_id counts:")
    for u, c in zip(uniq, cnt):
        print(f"  {u}: {c}")

    # 特徴抽出
    feature_extractor = build_resnet_feature_extractor(H_img, W_img)
    print("Extracting features for training data...")
    X_train, y_train = extract_features_from_files(
        feature_extractor,
        items,
        image_size=(H_img, W_img),
        batch_size=args.batch_size,
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # C固定で学習
    print(f"Training SVM with fixed C={args.C} (rbf, gamma=scale, class_weight=balanced)...")
    svm = train_svm_fixedC(X_train, y_train, args)

    # モデル保存
    svm_path = os.path.join(MODEL_DIR, "svm_classifier.pkl")
    with open(svm_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"SVM model saved to: {svm_path}")

    # 学習設定保存（再現性用）
    train_info_path = os.path.join(SAVE_DIR, "train_info.json")
    with open(train_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "svm": {
                    "kernel": "rbf",
                    "C": float(args.C),
                    "gamma": "scale",
                    "class_weight": "balanced",
                },
                "resize": [int(H_img), int(W_img)],
                "batch_size": int(args.batch_size),
                "video_regex": args.video_regex,
                "group_counts": {str(k): int(v) for k, v in zip(uniq, cnt)},
                "train_data_path": args.train_data_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Train info saved to: {train_info_path}")

    # ラベル情報
    label_info_path = os.path.join(SAVE_DIR, "label_info.txt")
    with open(label_info_path, "w", encoding="utf-8") as f:
        f.write("0: grasp (物体を把持)\n")
        f.write("1: non_grasp (非把持)\n")
    print(f"Label info saved to: {label_info_path}")


if __name__ == "__main__":
    print(">>> Train Start", flush=True)
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    pprint.pprint(args.__dict__)
    training_phase(args)
    print(">>> Train Finished", flush=True)
