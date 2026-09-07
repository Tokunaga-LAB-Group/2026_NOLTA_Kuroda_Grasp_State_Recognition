import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
sys.path.append(os.getcwd())
sys.path.append("..")
warnings.filterwarnings("ignore")

from typing import Literal
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import keras
import tensorflow as tf
import onnx
import onnxruntime

import dnn
import src


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--save_path", type=str, required=True, help="Save path.")
    parser.add_argument("--video_path", type=str, default=None, help="Load test video path.")
    parser.add_argument("--resizes", type=int, nargs=2, default=(256, 256), help="Resize image shape (H, W).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deeplab_v3_plus",
        choices=["deeplab_v3_plus", "segformer"],
        help="Model name",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--top_n_percent", type=int, default=1, help="Top n percent pixels for anomaly score.")
    parser.add_argument("--trained_weights_file", type=str, default=None, help="Load trained weights.")
    parser.add_argument("--onnx_file", type=str, default=None, help="Load ONNX file.")
    return parser.parse_args()


def plot_score(scores, save_file=None, start_index=1, title="Anomaly Scores"):
    scores = np.asarray(scores).astype(np.int32)

    plt.figure(figsize=(7, 3), facecolor="whitesmoke")
    plt.title(title, fontsize=20)

    # x 軸
    plt.xticks(fontsize=15)
    plt.xlabel("Frame Index", fontsize=18)

    # y 軸
    plt.ylim(0, 260)
    major_ticks = sorted(list(range(0, 249, 50)) + [255])
    plt.yticks(major_ticks, fontsize=15)
    # --- マイナー目盛：10刻み ---
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    # --- グリッド線 ---
    plt.grid(True, which="major", linestyle="-", alpha=0.7)
    plt.grid(True, which="minor", linestyle="--", alpha=0.3)

    # プロット
    x = np.arange(start_index, start_index + len(scores))
    plt.plot(
        x,
        scores,
        marker="o",
        markersize=4,
        linestyle="-",
        markeredgewidth=1.0,
        markeredgecolor="k",
        c="cornflowerblue",
        label="score",
    )

    plt.grid(True, alpha=0.6)

    # 保存 or 表示
    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, bbox_inches="tight", pad_inches=0.05, dpi=200)
    else:
        plt.show()

    plt.clf()
    plt.close()


def predict(model, imgs, batch_size, session_type: Literal["onnx", "tf"] = "tf"):
    anomaly_maps = np.empty((imgs.shape[:-1]), dtype=np.float32)  # (N, H, W)

    if session_type == "onnx":
        for i, idx in enumerate(range(0, imgs.shape[0], batch_size)):
            print(f">>> {i+1:03d}/{int(np.ceil(imgs.shape[0]/batch_size)):03d}", flush=True)
            batch_imgs = imgs[idx : idx + batch_size]  # (B, H, W, C)

            input_feed = {model.get_inputs()[0].name: batch_imgs}

            anomaps = model.run(None, input_feed)  # [(B, H, W, K)]

            anomaps = anomaps[0]  # (B, H, W, K)
            anomaps = tf.nn.softmax(anomaps).numpy()  # (B, H, W, K)
            anomaps = anomaps[..., -1]  # (B, H, W)

            anomaly_maps[idx : idx + batch_size] = anomaps  # (B, H, W)
    elif session_type == "tf":
        anomaly_maps = model.predict(imgs, batch_size=batch_size, verbose=2)  # (N, H, W, K)
        anomaly_maps = tf.nn.softmax(anomaly_maps).numpy()  # (N, H, W, K)
        anomaly_maps = anomaly_maps[..., -1]  # (N, H, W)

    return anomaly_maps


def main(args):
    # Setting
    ALPHA = 0.6

    # 保存先のディレクトリ作成
    SAVE_PATH = src.utils.create_file_dir(args.save_path, "inference_phase", include_file=False)

    # テストデータの読み込み
    H, W = args.resizes
    C = 3
    K = 2
    video_files = src.data.path_to_files(args.video_path)
    print(video_files)

    pivot = max(1, (args.top_n_percent * H * W) // 100)

    if args.onnx_file is not None:
        # ONNXモデルの読み込み
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        model = onnxruntime.InferenceSession(args.onnx_file, providers=providers)

        for i, video_file in enumerate(video_files):
            # テンプレートパスの作成
            rel = os.path.relpath(video_file, args.video_path)
            base_name = os.path.splitext(os.path.basename(rel))[0]
            template_file = os.path.join(SAVE_PATH, "{dir_name}", base_name + ".{ext}")

            imgs = src.data.video_to_frames(video_file, resizes=(H, W))  # (N, H, W, C)

            print(f"[{i + 1:02}/{len(video_files):02}] Video File: {video_file} ({len(imgs):,} [フレーム])", flush=True)

            anomaly_maps = predict(model, imgs, args.batch_size, session_type="onnx")  # (N, H, W)

            # オーバーレイ画像の作成と保存
            lay_imgs = np.zeros_like(imgs)
            lay_imgs[..., 0] = anomaly_maps
            overlay_imgs = (1 - ALPHA) * imgs + ALPHA * lay_imgs  # (N, H, W, C)
            src.utils.save_video(
                overlay_imgs,
                video_file=video_file,
                save_file=src.utils.create_file_dir(template_file.format(dir_name="overlay_videos", ext="mp4")),
            )

            # 異常度スコアの計算と保存
            anomaly_scores = src.utils.cal_scores(anomaly_maps * 255, pivot)  # (N,)
            src.utils.save_csv(
                ([f"{str(i+1).zfill(4)}" for i in range(len(imgs))], anomaly_scores),
                src.utils.create_file_dir(template_file.format(dir_name="anomaly_scores", ext="csv")),
            )

            # 異常度スコアのプロット保存
            plot_score(
                anomaly_scores,
                src.utils.create_file_dir(template_file.format(dir_name="score_curves", ext="png")),
            )

    elif args.model_name is not None and args.trained_weights_file is not None:
        model = dnn.build_model(
            model_name=args.model_name,
            input_shape=(H, W, C),
            num_classes=K,
            name=args.model_name,
        )
        model.load_weights(args.trained_weights_file)

        for i, video_file in enumerate(video_files):
            # テンプレートパスの作成
            rel = os.path.relpath(video_file, args.video_path)
            base_name = os.path.splitext(os.path.basename(rel))[0]
            template_file = os.path.join(SAVE_PATH, "{dir_name}", base_name + ".{ext}")

            imgs = src.data.video_to_frames(video_file, resizes=(H, W))  # (N, H, W, C)

            print(f"[{i + 1:02}/{len(video_files):02}] Video File: {video_file} ({len(imgs):,} [フレーム])", flush=True)

            anomaly_maps = predict(model, imgs, args.batch_size, session_type="tf")  # (N, H, W)

            # オーバーレイ画像の作成と保存
            lay_imgs = np.zeros_like(imgs)
            lay_imgs[..., 0] = anomaly_maps
            overlay_imgs = (1 - ALPHA) * imgs + ALPHA * lay_imgs  # (N, H, W, C)
            src.utils.save_video(
                overlay_imgs,
                video_file=video_file,
                save_file=src.utils.create_file_dir(template_file.format(dir_name="overlay_videos", ext="mp4")),
            )

            # 異常度スコアの計算と保存
            anomaly_scores = src.utils.cal_scores(anomaly_maps, pivot)  # (N,)
            src.utils.save_csv(
                ([f"{str(i+1).zfill(4)}" for i in range(len(imgs))], anomaly_scores),
                src.utils.create_file_dir(template_file.format(dir_name="anomaly_scores", ext="csv")),
            )
            # 異常度スコアのプロット保存
            plot_score(
                anomaly_scores,
                src.utils.create_file_dir(template_file.format(dir_name="score_curves", ext="png")),
            )
    else:
        raise ValueError("Please specify either --onnx_file or --model_name with --trained_weights_file.")


if __name__ == "__main__":
    print(">>> Inference Start", flush=True)
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    pprint.pprint(args.__dict__)

    main(args)
    print(">>> Inference Finished", flush=True)
