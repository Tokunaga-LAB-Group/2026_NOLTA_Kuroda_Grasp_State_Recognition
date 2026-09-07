import os
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio

plt.rcParams["font.family"] = "DejaVu Serif"
# 軸の設定
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True
# 凡例の設定
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.handlelength"] = 1.0
plt.rcParams["legend.labelspacing"] = 0.5
plt.rcParams["legend.handletextpad"] = 1.0
plt.rcParams["legend.markerscale"] = 1.0


def create_file_dir(*args, include_file=True):
    path = os.path.join(*args)

    if include_file:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

    return path


def save_images(imgs, save_path, img_names=None, cmap="gray", vmin=0.0, vmax=1.0):
    for i, img in enumerate(imgs):
        img_name = img_names[i] if img_names is not None else str(i).zfill(len(str(len(imgs))) + 1) + ".jpg"

        plt.imsave(os.path.join(save_path, img_name), img, cmap=cmap, vmin=vmin, vmax=vmax)


def save_anomaly_map(file: str, test_path: str, save_path: str, anomaly_map: np.ndarray):
    save_cw_file = create_file_dir(
        file.replace(
            test_path,
            os.path.join(save_path, "anomaly_maps", "coolwarm"),
        )
    )
    save_gray_file = create_file_dir(
        file.replace(
            test_path,
            os.path.join(save_path, "anomaly_maps", "gray"),
        )
    )
    plt.imsave(save_cw_file, anomaly_map, cmap="coolwarm", vmin=0.0, vmax=1.0)
    plt.imsave(save_gray_file, anomaly_map, cmap="gray", vmin=0.0, vmax=1.0)


def save_video(imgs, video_file, save_file):
    # 元動画の fps を取得
    reader = imageio.get_reader(video_file)
    fps = reader.get_meta_data()["fps"]
    reader.close()

    # 出力設定
    writer = imageio.get_writer(
        save_file,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", "23"],
    )

    for img in imgs:
        # float32 → uint8
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        writer.append_data(img)

    writer.close()
    print("Saved:", save_file, flush=True)


def save_csv(data, save_file):
    df = pd.DataFrame({"file": data[0], "anomaly_score": data[1]})
    df.to_csv(save_file, index=False)


def cal_scores(anomaly_maps, pivot):
    scores = [np.mean(np.sort(anomap.flatten())[::-1][:pivot], dtype=np.float32) for anomap in anomaly_maps]
    scores = np.array(np.round(scores, 0), np.int32)  # [0, 255]

    return scores


def plot_LearningCurve(history, save_path=None, start_epoch=1, with_val=False):
    for key in history.history.keys():
        if with_val is False:
            hist_list = [history.history[key][start_epoch - 1 :]]
        elif with_val is True and ("val" in key):
            continue
        else:
            hist_list = [history.history[key][start_epoch - 1 :], history.history["val_" + key][start_epoch - 1 :]]

        N_hist = len(hist_list[0]) + start_epoch
        step = np.where(N_hist - start_epoch < 10, 1, (N_hist - start_epoch) // 10)

        # 学習曲線の描画
        plt.figure(figsize=(7, 3), facecolor="whitesmoke")
        plt.title(key, fontsize=20)
        # x軸
        plt.xlim(start_epoch - step * 0.2, N_hist + step * 0.2)
        xticks = np.arange(start_epoch, N_hist + 1, step)
        xticks[1:] = xticks[1:] - 1
        plt.xticks(xticks, fontsize=15)
        plt.xlabel("epochs", fontsize=18)
        # y軸
        hist_max = np.array([np.max(hist) for hist in hist_list]).max()
        hist_min = np.array([np.min(hist) for hist in hist_list]).min()
        y_max = np.where(hist_max > 0, hist_max * 1.1, hist_max * 0.9)
        y_min = np.where(hist_min > 0, 0, hist_min * 1.1)
        plt.ylim(y_min, y_max)
        plt.yticks(fontsize=15)

        for idx, hist in enumerate(hist_list):
            plt.plot(
                np.arange(start_epoch, N_hist),
                hist,
                marker="o",
                markeredgewidth=1.0,
                markeredgecolor="k",
                label=key,
                c="midnightblue" if (idx == 0) and ("val" not in key) else "coral",
            )
        if with_val is True:
            plt.legend(["train", "validation"], fontsize=15, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6)
        if save_path is not None:
            plt.savefig(
                os.path.join(save_path, "plot_{}.jpg".format(key)), bbox_inches="tight", pad_inches=0.05, dpi=300
            )
        else:
            plt.show()
        plt.clf()
        plt.close()


# 比較画像の取得
def get_compare_img(
    f_name,
    title_list,
    img_list,
    cmap_list,
    suptitle=None,
    pixel=(800, 2100),
    dpi=300,
):

    if suptitle is None:
        suptitle = os.path.splitext(f_name)[0]

    N_sub = len(title_list)

    fig_w = pixel[1] / dpi
    fig_h = pixel[0] / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", constrained_layout=True)

    fig.suptitle(suptitle, fontsize=20)

    for idx, data in enumerate(zip(title_list, img_list, cmap_list)):
        title, img, cmap = data
        ax = fig.add_subplot(1, N_sub, idx + 1)
        ax.set_title(title, fontsize=18)
        ax.axis("off")
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)

    # 描画
    canv = FigureCanvas(fig)
    fig.set_dpi(dpi)
    canv.draw()

    # 合成
    buf = np.asarray(canv.renderer.buffer_rgba())
    rgb = buf[..., :3].astype(np.float32)
    alpha = buf[..., 3:] / 255.0
    bg_arr = np.array([255, 255, 255], dtype=np.float32).reshape(1, 1, 3)
    comp = (rgb * alpha + bg_arr * (1.0 - alpha)).clip(0, 255).astype(np.uint8)

    plt.close(fig)

    return comp


def imgs_to_pdf(imgs, save_file, quality=90, subsampling="4:4:4"):
    c = canvas.Canvas(save_file)

    for img in imgs:
        img = Image.fromarray(img, mode="RGB")
        w, h = img.size

        # 1ページの大きさを画像サイズに合わせる
        c.setPageSize((w, h))

        # JPEG圧縮して貼り付け
        bio = io.BytesIO()
        img.save(bio, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=subsampling)
        bio.seek(0)
        c.drawImage(ImageReader(bio), 0, 0, width=w, height=h)

        c.showPage()
    c.save()
