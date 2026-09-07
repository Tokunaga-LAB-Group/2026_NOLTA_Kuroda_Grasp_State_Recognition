import os
import csv
import re
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip

from hol import src


# =========================
# 並び順（このプロジェクト用）
# 例: kiban_〇〇〇〇_video5_0000.png
#     kiban_video5_0000.png（〇〇〇〇が空でもOK）
# =========================
_video_re = re.compile(r"(?:^|_)video(\d+)(?:_|\.|\b)", re.IGNORECASE)
_frame_re = re.compile(r"_(\d+)(?=\.[^.]+$)")  # 拡張子直前の _数字 をフレーム扱い


def filename_natural_key(filename: str):
    """
    対応例:
      kiban_video5_0000.png
      kiban_demo_video5_0000.png
      kiban_xxx_yyy_video12_0089.jpg

    key = (video_id, frame_id)
    """
    name = filename.strip()

    vm = _video_re.search(name)
    fm = _frame_re.search(name)

    if vm and fm:
        return (int(vm.group(1)), int(fm.group(1)))

    # 命名規則に合わないものは最後へ
    return (10**9, 10**9)


def build_ordered_image_list(root_dir: str, strict: bool = True):
    root = Path(root_dir)

    # 直下のディレクトリのうち、数字名だけを対象にする（例: "0","1"）
    class_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    class_dirs.sort(key=lambda p: int(p.name))  # 0→1→2... を保証

    # 対象拡張子（必要なら追加）
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    ordered = []
    for cd in class_dirs:
        files = [f for f in cd.iterdir() if f.is_file() and f.suffix.lower() in exts]

        if strict:
            bad = [f.name for f in files if filename_natural_key(f.name) == (10**9, 10**9)]
            if bad:
                raise ValueError(
                    f"[ERROR] 命名規則にマッチしないファイル名が {cd} に {len(bad)} 個あります。\n"
                    f"例: {bad[:5]}"
                )

        # “そのディレクトリ内で” video/frame の自然順を保証
        files.sort(key=lambda f: filename_natural_key(f.name))
        ordered.extend(files)

    return [str(p) for p in ordered]


def get_args():
    p = argparse.ArgumentParser(description="CLIP inference: cosine similarity + margin output (ordered).")

    p.add_argument("--input_dir", type=str, required=True,
                   help="Input root directory that contains subdirs 0/1/... with images.")
    p.add_argument("--output_csv", type=str, required=True,
                   help="Output CSV path.")
    p.add_argument("--gpu", type=int, default=3,
                   help="Physical GPU id to use (CUDA_VISIBLE_DEVICES). Default: 3")
    p.add_argument("--crop_size", type=int, default=1080,
                   help="Crop size (n x n). Default: 1080")

    # 実験しやすいように model/pretrained も引数化（デフォルトはあなたの今の設定）
    p.add_argument("--model", type=str, default="ViT-L-14-quickgelu",
                   help="OpenCLIP model name. Default: ViT-L-14-quickgelu")
    p.add_argument("--pretrained", type=str, default="metaclip_fullcc",
                   help="Pretrained tag. Default: metaclip_fullcc")

    # テキストは固定で良いと言ってたのでデフォルト固定、必要なら変更も可能にしておく
    p.add_argument("--text_grasp", type=str, default="Held object.",
                   help='Text prompt for grasp. Default: "Held object."')
    p.add_argument("--text_non_grasp", type=str, default="Not held object",
                   help='Text prompt for non-grasp. Default: "Not held object"')

    p.add_argument("--strict", action="store_true",
                   help="If set, stop when filenames don't match expected pattern for ordering.")
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

    # ===== GPU固定（物理GPU指定） =====
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDAが利用できません。GPU必須なので終了します。（指定GPU={gpu_id}）")

    device = torch.device("cuda:0")

    try:
        torch.zeros(1, device=device)
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"GPU初期化に失敗しました。GPU必須なので終了します: {e}")

    print(f"Using GPU (physical): {gpu_id}")

    # ===== CLIPロード =====
    print(f"[MODEL] {model_name} / pretrained={pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    # ===== テキスト（2文） =====
    tokenizer = open_clip.get_tokenizer(model_name)
    texts = [text_grasp, text_non_grasp]
    text_tokens = tokenizer(texts).to(device)

    # ===== テキスト特徴（固定） =====
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)       # (2, D)
        text_features = F.normalize(text_features, dim=-1)   # (2, D)

    # ===== “評価順リスト” を確定 =====
    image_paths = build_ordered_image_list(folder_path, strict=strict)

    print(f"Total images found: {len(image_paths)}")
    if image_paths:
        print("[ORDER CHECK] first:", image_paths[0])
        print("[ORDER CHECK] last :", image_paths[-1])

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ===== CSV出力 =====
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "grasp_cos", "non_grasp_cos", "margin"])

        for file_path in image_paths:
            try:
                img = Image.open(file_path).convert("RGB")

                # クロップ（n x n）
                cropped_img = src.clop.crop(img, crop_size)

                imgs = preprocess(cropped_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(imgs)           # (1, D)
                    image_features = F.normalize(image_features, dim=-1)

                    cos_sim = image_features @ text_features.T         # (1, 2)

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
