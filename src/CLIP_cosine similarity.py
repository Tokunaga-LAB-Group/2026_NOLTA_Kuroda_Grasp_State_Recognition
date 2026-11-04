import argparse
import csv
import os
import re
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from clop import crop

# ファイル名から動画番号とフレーム番号を抽出する
_video_re = re.compile(r"(?:^|_)video(\d+)(?:_|\.|\b)", re.IGNORECASE)
_frame_re = re.compile(r"_(\d+)(?=\.[^.]+$)")


def filename_natural_key(filename: str):
    name = filename.strip()
    vm = _video_re.search(name)
    fm = _frame_re.search(name)

    if vm and fm:
        return (int(vm.group(1)), int(fm.group(1)))

    return (10**9, 10**9)


def build_ordered_image_list(root_dir: str, strict: bool = True):
    root = Path(root_dir)

    # 数字名のクラスディレクトリを番号順に並べる（0, 1, 2, ...）
    class_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    class_dirs.sort(key=lambda p: int(p.name))

    # 読み込む画像の拡張子
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    ordered = []
    for cd in class_dirs:
        files = [f for f in cd.iterdir() if f.is_file() and f.suffix.lower() in exts]

        if strict:
            bad = [
                f.name for f in files if filename_natural_key(f.name) == (10**9, 10**9)
            ]
            if bad:
                raise ValueError(
                    f"[ERROR] 命名規則にマッチしないファイル名が {cd} に {len(bad)} 個あります。\n"
                    f"例: {bad[:5]}"
                )

        # クラス内の画像を動画番号・フレーム番号の順に並べる
        files.sort(key=lambda f: filename_natural_key(f.name))
        ordered.extend(files)

    return [str(p) for p in ordered]


def get_args():
    p = argparse.ArgumentParser(
        description="CLIP inference: cosine similarity + margin output (ordered)."
    )

    # 入出力
    p.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input root directory that contains subdirs 0/1/... with images.",
    )
    p.add_argument("--output_csv", type=str, required=True, help="Output CSV path.")

    # GPU とクロップサイズ
    p.add_argument(
        "--gpu",
        type=int,
        default=3,
        help="Physical GPU id to use (CUDA_VISIBLE_DEVICES). Default: 3",
    )
    p.add_argument(
        "--crop_size", type=int, default=1080, help="Crop size (n x n). Default: 1080"
    )

    # CLIP モデルと事前学習済み重み
    p.add_argument(
        "--model",
        type=str,
        default="ViT-L-14-quickgelu",
        help="OpenCLIP model name. Default: ViT-L-14-quickgelu",
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default="metaclip_fullcc",
        help="Pretrained tag. Default: metaclip_fullcc",
    )

    # 把持・非把持のテキストプロンプト
    p.add_argument(
        "--text_grasp",
        type=str,
        default="Held object.",
        help='Text prompt for grasp. Default: "Held object."',
    )
    p.add_argument(
        "--text_non_grasp",
        type=str,
        default="Not held object",
        help='Text prompt for non-grasp. Default: "Not held object"',
    )

    # ファイル名の検証
    p.add_argument(
        "--strict",
        action="store_true",
        help="If set, stop when filenames don't match expected pattern for ordering.",
    )

    return p.parse_args()


def process_images_in_folder(
    folder_path: str,
    output_csv: str,
    crop_size: int,
    gpu_id: int,
    model_name: str,
    pretrained: str,
    text_grasp: str,
    text_non_grasp: str,
    strict: bool,
):
    """
    出力CSV列:
      - grasp_cos      : cos_sim(image, text_grasp)
      - non_grasp_cos  : cos_sim(image, text_non_grasp)
      - margin         : grasp_cos - non_grasp_cos（正に大きいほど把持寄り）

    ※softmaxは使わない（相対化しない）
    """
    # 使用する物理 GPU を指定し、初期化する
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDAが利用できません。GPU必須なので終了します。（指定GPU={gpu_id}）"
        )

    device = torch.device("cuda:0")

    try:
        torch.zeros(1, device=device)
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"GPU初期化に失敗しました。GPU必須なので終了します: {e}")

    print(f"Using GPU (physical): {gpu_id}")

    # CLIP モデルと前処理を読み込む
    print(f"[MODEL] {model_name} / pretrained={pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    # 把持・非把持のテキストをトークン化する
    tokenizer = open_clip.get_tokenizer(model_name)
    texts = [text_grasp, text_non_grasp]
    text_tokens = tokenizer(texts).to(device)

    # 全画像で共通のテキスト特徴を計算し、正規化する
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)  # (2, D)
        text_features = F.normalize(text_features, dim=-1)

    # 画像の処理順を確定する
    image_paths = build_ordered_image_list(folder_path, strict=strict)

    print(f"Total images found: {len(image_paths)}")
    if image_paths:
        print("[ORDER CHECK] first:", image_paths[0])
        print("[ORDER CHECK] last :", image_paths[-1])

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 画像ごとの類似度とマージンを CSV に保存する
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "grasp_cos", "non_grasp_cos", "margin"])

        for file_path in image_paths:
            try:
                # 画像を RGB に変換し、クロップ（n x n）と前処理を行う
                img = Image.open(file_path).convert("RGB")
                cropped_img = crop(img, crop_size)
                imgs = preprocess(cropped_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(imgs)  # (1, D)
                    image_features = F.normalize(image_features, dim=-1)

                    cos_sim = image_features @ text_features.T  # (1, 2)

                    grasp_cos = float(cos_sim[0, 0].item())
                    non_grasp_cos = float(cos_sim[0, 1].item())
                    margin = grasp_cos - non_grasp_cos

                writer.writerow([file_path, grasp_cos, non_grasp_cos, margin])

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Saved CSV: {output_csv}")


def main():
    args = get_args()

    process_images_in_folder(
        folder_path=args.input_dir,
        output_csv=args.output_csv,
        crop_size=args.crop_size,
        gpu_id=args.gpu,
        model_name=args.model,
        pretrained=args.pretrained,
        text_grasp=args.text_grasp,
        text_non_grasp=args.text_non_grasp,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
