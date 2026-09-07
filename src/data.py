import os
import glob
import json
import numpy as np
from natsort import natsorted
from PIL import Image
import tensorflow as tf
import cv2

AUTOTUNE = tf.data.AUTOTUNE
NORMAL = 0


# ============================================================
# データ読み込み
# ============================================================
def path_to_files(img_path: str) -> list[str]:
    """フォルダ以下の全ファイルを自然順に取得"""
    return natsorted(glob.glob(os.path.join(img_path, "**", "*.*"), recursive=True))


def files_to_images(files: list[str], resizes: tuple[int, int] = (256, 256)) -> np.ndarray:
    H, W = resizes

    imgs = [np.array(Image.open(f).convert("RGB").resize((W, H), Image.BILINEAR)) for f in files]
    imgs = np.stack(imgs, axis=0).astype(np.float32) / 255.0  # (N, H, W, C)

    return imgs


def video_to_frames(video_file: str, resizes: tuple[int, int]) -> list[np.ndarray]:
    H, W = resizes
    cap = cv2.VideoCapture(video_file)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize (W, H)
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

        # float32 正規化
        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

    cap.release()

    frames = np.stack(frames, axis=0)  # (num_frames, H, W, C)

    return frames


def tf_load_image(resizes: tuple[int, int] = (256, 256)):
    H, W = resizes
    C = 3

    def _load(path: tf.Tensor) -> tf.Tensor:
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=C, expand_animations=False)
        img = tf.image.resize(img, (H, W), method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape((H, W, C))
        return img

    return _load


def tf_load_annotation(resizes: tuple[int, int], num_classes: int = 2) -> tf.Tensor:
    def _load(json_file: tf.Tensor) -> tf.Tensor:
        mask, label = tf.py_function(
            func=lambda f: _load_anno_func(f, resizes=resizes),
            inp=[json_file],
            Tout=[tf.float32, tf.int32],
        )
        mask.set_shape((*resizes, num_classes))
        label.set_shape(())

        return mask, label

    return _load


def _load_anno_func(json_file, resizes: tuple[int, int]) -> np.ndarray:
    Hr, Wr = resizes

    # tf.Tensor → numpy
    if hasattr(json_file, "numpy"):
        json_file = json_file.numpy()

    # numpy→Python値
    if isinstance(json_file, np.ndarray):
        json_file = json_file.item()

    # bytes → str
    if isinstance(json_file, (bytes, np.bytes_)):
        json_file = json_file.decode("utf-8")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    H, W = data["y_size"], data["x_size"]
    K = data["num_classes"]
    points = data["points"]
    img_label = data["label_id"]

    mask = np.zeros((Hr, Wr, K), dtype=np.float32)

    if img_label > NORMAL:
        for p in points:
            h = p["y"]
            w = p["x"]
            k = p["label_id"]

            h_r = int(round(h * (Hr - 1) / (H - 1)))
            w_r = int(round(w * (Wr - 1) / (W - 1)))

            if not (0 <= h_r < Hr and 0 <= w_r < Wr and 0 <= k < K):
                raise ValueError(
                    f"境界外のポイントがあります: "
                    f"orig=({h}, {w}, {k}), "
                    f"scaled=({h_r}, {w_r}, {k}), "
                    f"mask_size=({Hr}, {Wr}, {K}), "
                    f"file={json_file}"
                )

            mask[h_r, w_r, k] = 1.0

    return mask, img_label


def tf_random_assignment(num_points: int):

    def process(mask: tf.Tensor, label: tf.Tensor):
        # mask: (H, W, K)
        H = tf.shape(mask)[0]
        W = tf.shape(mask)[1]

        def add_random_points():
            h_indices = tf.random.uniform(
                shape=(num_points,),
                minval=0,
                maxval=H,
                dtype=tf.int32,
            )  # (N,)
            w_indices = tf.random.uniform(
                shape=(num_points,),
                minval=0,
                maxval=W,
                dtype=tf.int32,
            )  # (N,)
            c_indices = tf.zeros((num_points,), dtype=tf.int32)  # (N,)

            indices = tf.stack([h_indices, w_indices, c_indices], axis=1)  # (N, 3)
            updates = tf.ones((num_points,), dtype=tf.float32)  # (N,)

            # 該当する座標に[1, 0, ...]を割り当てる
            sampled_mask = tf.tensor_scatter_nd_update(mask, indices, updates)

            return sampled_mask

        # 正常画像のときだけ座標をランダムに割り当て, それ以外はそのまま返す
        return tf.cond(tf.equal(label, NORMAL), add_random_points, lambda: mask)

    return process


def _file_check(img_files: list[str], anno_files: list[str]):
    if len(img_files) == 0:
        raise ValueError("画像ファイルが見つかりません。")
    if len(anno_files) == 0:
        raise ValueError("アノテーションファイルが見つかりません。")
    if len(img_files) != len(anno_files):
        raise ValueError("画像ファイルとアノテーションファイルの数が一致しません。")

    for img_file, anno_file in zip(img_files, anno_files):
        img_file_name = os.path.splitext(os.path.basename(img_file))[0]
        anno_file_name = os.path.splitext(os.path.basename(anno_file))[0]
        if img_file_name != anno_file_name:
            raise ValueError(f"ファイル名が一致しません: {img_file_name} vs {anno_file_name}")

        img_sizes = Image.open(img_file).size  # (width, height)
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            anno_sizes = (data["x_size"], data["y_size"])  # (width, height)
        if img_sizes != anno_sizes:
            raise ValueError(
                f"画像サイズが一致しません: {img_file_name} ({img_sizes}) vs {anno_file_name} ({anno_sizes})"
            )


# ============================================================
# Dataset Constructor
# ============================================================
def create_train_dataset(
    img_files: list[str],
    anno_files: list[str],
    resizes: tuple[int, int] = (256, 256),
    num_random_points: int = 6,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
    use_cache: bool = True,
):
    _file_check(img_files, anno_files)

    N = len(img_files)

    img_loader = tf_load_image(resizes)
    anno_loader = tf_load_annotation(resizes)
    random_assignment = tf_random_assignment(num_random_points)

    ds = tf.data.Dataset.from_tensor_slices((img_files, anno_files))

    if use_cache:
        ds = ds.map(
            lambda x, y: (img_loader(x), anno_loader(y)),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=min(2**13, N), seed=seed, reshuffle_each_iteration=True)
    else:
        if shuffle:
            ds = ds.shuffle(buffer_size=min(2**13, N), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x, y: (img_loader(x), anno_loader(y)),
            num_parallel_calls=AUTOTUNE,
            deterministic=not shuffle,
        )
    ds = ds.map(
        lambda x, y: (x, random_assignment(mask=y[0], label=y[1])),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    return ds
