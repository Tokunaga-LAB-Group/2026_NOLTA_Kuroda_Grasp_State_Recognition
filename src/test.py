import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
sys.path.append(os.getcwd())
sys.path.append("..")
warnings.filterwarnings("ignore")

import argparse
import pprint
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import ( #scikit-learnの評価指標sklearn.metricsモジュール
    accuracy_score, #正解率
    precision_score, #適合率
    recall_score, #再現率
    f1_score, #F1スコア
    confusion_matrix, #混同行列
    classification_report, #分類レポート
)

def get_args(): #引数設定
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="0", help="GPU ID.")

    parser.add_argument( 
        "--test_data_path",
        type=str,
        required=True, #required=True:必須
        help="Path to test images directory. (contains subdirs '0' and '1')",
    )
    parser.add_argument(
        "--svm_model_path", #SVMモデルパス(pkl形式)
        type=str,
        required=True, 
        help="Path to trained SVM model (.pkl).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save results (metrics, csv).",
    )

    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size H W for ResNet.",
    )

    return parser.parse_args()

def build_resnet_feature_extractor(H_img, W_img): #訓練と同じ：画像→ResNetで特徴抽出→SVM推論
    """
    ResNet50 を特徴抽出器として構築する。
    画像 -> グローバル平均プーリング後の 1D ベクトル を出力。
    train.py と同じ設定にしておくことが重要。
    """
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(H_img, W_img, 3),
        pooling="avg",  # グローバル平均プーリング → ベクトル
    )
    base_model.trainable = False  # 特徴抽出だけに使う（学習しない）
    return base_model

def load_and_preprocess_image(path, image_size):
    """
    1枚の画像を読み込み → リサイズ → ResNet用前処理を行う。
    """
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3) #デコード：復号化　→　TensorFlowのテンソルに変換
    img.set_shape((None, None, 3)) #set_shape:テンソルの形状を設定
    img = tf.image.resize(img, image_size)
    img = keras.applications.resnet50.preprocess_input(img) #ResNetに合わせた前処理
    return img  # (H, W, 3), float32

def collect_image_paths_and_labels(root_dir): #画像パスとラベル収集
    """
    ディレクトリ構造:
      root_dir/
        0/  把持
        1/  非把持
    という前提で、画像パスと整数ラベルを集める。
    """
    image_paths = []
    labels = []

    # サブディレクトリ "0", "1" を固定で扱う
    for label_str in ["0", "1"]:
        class_dir = os.path.join(root_dir, label_str)
        if not os.path.isdir(class_dir): 
            print(f"[WARNING] Class directory not found: {class_dir}")
            continue

        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # 画像拡張子だけを対象にする
            lower = fname.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                image_paths.append(fpath)
                labels.append(int(label_str))

    return image_paths, np.array(labels, dtype=np.int32)

def main(args):
    # GPU 設定
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 保存先ディレクトリ作成
    SAVE_DIR = args.save_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    H_img, W_img = args.resize

    # 画像パスとラベル（0:把持, 1:非把持）を収集
    image_paths, y_true = collect_image_paths_and_labels(args.test_data_path) #image_paths:画像パス, y_true:正解ラベル
    if len(image_paths) == 0:
        print("No images found in test_data_path.")
        return

    print(f"Number of test images: {len(image_paths)}")

    # ResNet 特徴抽出器の構築
    print("Building ResNet50 feature extractor...")
    feature_extractor = build_resnet_feature_extractor(H_img, W_img) 

    # 画像ごとに特徴量を抽出
    # enumerate():インデックス番号（カウント、順番）と要素を同時に取得
    features_list = []
    for idx, path in enumerate(image_paths):
        img = load_and_preprocess_image(path, (H_img, W_img))
        img = tf.expand_dims(img, axis=0)  # (1, H, W, 3) expand_dims()：バッチ次元追加
        feat = feature_extractor(img, training=False).numpy()  # (1, D)
        features_list.append(feat[0])  # (D,)

        print(f"[{idx + 1:04}/{len(image_paths):04}] Tested", flush=True)

    X = np.stack(features_list, axis=0)  # (N, D)：Nは画像数、Dは特徴量次元数

    # 学習済み SVM モデルの読み込み
    print(f"Loading SVM model from: {args.svm_model_path}")
    with open(args.svm_model_path, "rb") as f:
        svm_clf = pickle.load(f)

    # 予測
    print("Predicting with SVM...")
    
    # 決定関数 f(x)（境界からの符号付き距離）
    decision = svm_clf.decision_function(X)
    
    y_pred = svm_clf.predict(X)

    # 必要なら確率も計算（probability=True のときのみ）
    y_score = None
    if hasattr(svm_clf, "predict_proba"):
        proba = svm_clf.predict_proba(X)  # (N, 2)
        # クラス1（非把持）側の確率をスコアとして保存しておく
        y_score = proba[:, 1]

    # --- 評価指標の計算（把持=0を正例として公式通り） ---
    # confusion_matrix: rows=true [0,1], cols=pred [0,1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # unpack (positive = 0: grasp)
    TP = int(cm[0, 0])  # true grasp, pred grasp
    FN = int(cm[0, 1])  # true grasp, pred non_grasp
    FP = int(cm[1, 0])  # true non_grasp, pred grasp
    TN = int(cm[1, 1])  # true non_grasp, pred non_grasp

    total = TP + TN + FP + FN
    eps = 1e-12  # 0除算防止（分母が0のとき0扱いでも可）

    accuracy  = (TP + TN) / (total + eps)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = (2 * precision * recall) / (precision + recall + eps)

    # （任意）他も欲しければ
    # specificity = TN / (TN + FP + eps)  # True Negative Rate
    # fpr = FP / (FP + TN + eps)

    # --- 画面出力（分かりやすい形式） ---
    print("Confusion Matrix (rows=true [grasp=0, non_grasp=1], cols=pred [0,1]):")
    print(cm)
    print()

    print("Confusion Matrix terms (positive = grasp=0):")
    print(f"TP (grasp->grasp)      : {TP}")
    print(f"FN (grasp->non_grasp)  : {FN}")
    print(f"FP (non_grasp->grasp)  : {FP}")
    print(f"TN (non_grasp->non)    : {TN}")
    print()

    print("=== Metrics (positive class = grasp=0) ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print()

    # --- metrics.txt に保存 ---
    metrics_path = os.path.join(SAVE_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Confusion Matrix (rows=true [grasp=0, non_grasp=1], cols=pred [0,1]):\n")
        f.write(str(cm) + "\n\n")

        f.write("Confusion Matrix terms (positive = grasp=0):\n")
        f.write(f"TP (grasp->grasp)      : {TP}\n")
        f.write(f"FN (grasp->non_grasp)  : {FN}\n")
        f.write(f"FP (non_grasp->grasp)  : {FP}\n")
        f.write(f"TN (non_grasp->non)    : {TN}\n\n")

        f.write("=== Metrics (positive class = grasp=0) ===\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n\n")

    print(f"Saved metrics to: {metrics_path}")



    # 各画像の結果を CSV に保存
    rel_paths = [os.path.relpath(p, args.test_data_path) for p in image_paths]
    df_dict = {
    "file_path": rel_paths,
    "true_label": y_true,
    "pred_label": y_pred,
    "decision_function": decision,         
    "margin_abs": np.abs(decision),          
    "correct": (y_true == y_pred).astype(int),
}
    if y_score is not None:
        df_dict["score_class1"] = y_score  # 非把持(1)である確率

    df = pd.DataFrame(df_dict)
    csv_path = os.path.join(SAVE_DIR, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved prediction CSV to: {csv_path}")

    # ラベルの意味もメモとして残しておく
    label_info_path = os.path.join(SAVE_DIR, "label_info.txt")
    with open(label_info_path, "w") as f:
        f.write("0: grasp (物体を把持)\n")
        f.write("1: non_grasp (非把持)\n")
    print(f"Saved label info to: {label_info_path}")
    
if __name__ == "__main__":
    print(">>> Test Start", flush=True)
    args = get_args()
    pprint.pprint(args.__dict__)
    main(args)
    print(">>> Test Finished", flush=True)
    
    
""" 
推論フロー（test.py）
 画像
  ↓
 ResNet（固定）
  ↓
 特徴ベクトル x (D次元)  
　↓
 決定関数f(x)の正負を調べる：SVM.predict(X)
  ↓
 0 or 1
 
 ※決定関数f(x)：decision = svm_clf.decision_function(X)を出力 predict(X)はf(x)の計算をしてさらにそのf(x)の正負から０、１クラスを判断する。
 　decisionはその離れている具合をcsvファイルで保存したいがために入れている
"""

"""
pklファイルに含まれているもの：
    ・サポートベクトル：決定境界を定義するデータポイント
    ・係数：サポートベクトルに関連付けられた重み
    ・バイアス項（切片）：決定境界の位置を調整する定数項
    ・ハイパーパラメータ：SVMの動作を制御するパラメータ（例：C、カーネルタイプ、ガンマなど）
    ・クラスラベル：分類タスクにおける各クラスのラベル情報
    ・カーネル関数の情報：使用されるカーネル関数の種類とそのパラメータ
"""