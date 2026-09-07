import os
import cv2
import csv
from typing import List, Tuple

# 自然順ソート（video10 が video2 の前に来る問題を防ぐ）
try:
    from natsort import natsorted
except ImportError:
    natsorted = None


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def get_folder_path() -> str | None:
    folder = input("アノテーション対象のフォルダパスを入力してください：").strip()
    if not os.path.isdir(folder):
        print("指定されたフォルダが存在しません。")
        return None
    return folder


def list_images_sorted(folder_path: str) -> List[str]:
    """フォルダ内画像を安定した順序（自然順優先）で返す（ファイル名のみ）。"""
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(IMG_EXTS)]
    if not files:
        return []

    # natsort が使えるなら自然順、なければ通常のsorted
    if natsorted is not None:
        return natsorted(files)
    return sorted(files)


def manual_annotation(folder_path: str) -> None:
    image_files = list_images_sorted(folder_path)
    if not image_files:
        print("画像が見つかりません。")
        return

    print(f"\nFound {len(image_files)} images in: {folder_path}")
    print("操作: キー '0' (把持) / '1' (非把持) を押してください。終了は 'q'。\n")

    annotations: List[Tuple[str, int]] = []
    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"画像を読み込めません: {image_file}")
            continue

        # 画面にファイル名を表示（視認性UP）
        disp = image.copy()
        cv2.putText(
            disp,
            f"{idx}/{len(image_files)}  {image_file}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Annotate Image (Press 0 or 1, q to quit)", disp)

        while True:
            key = cv2.waitKey(0)
            if key == ord("0"):
                annotations.append((image_path, 0))
                print(f"{image_file}: 0")
                break
            elif key == ord("1"):
                annotations.append((image_path, 1))
                print(f"{image_file}: 1")
                break
            elif key == ord("q"):
                print("途中終了します。ここまでのアノテーションを保存します。")
                cv2.destroyAllWindows()
                save_annotations(folder_path, annotations)
                return
            else:
                print("無効な入力です。0 または 1（終了は q）を押してください。")

        cv2.destroyAllWindows()

    save_annotations(folder_path, annotations)


def auto_annotation(folder_path: str, value: int) -> None:
    image_files = list_images_sorted(folder_path)
    if not image_files:
        print("画像が見つかりません。")
        return

    annotations = [(os.path.join(folder_path, image_file), value) for image_file in image_files]

    # 必要なら表示（うるさければ消してOK）
    for image_path, _ in annotations[:10]:
        print(f"{image_path}: {value}")
    if len(annotations) > 10:
        print(f"... ({len(annotations)} images total)")

    save_annotations(folder_path, annotations)


def save_annotations(folder_path: str, annotations: List[Tuple[str, int]]) -> None:
    csv_path = os.path.join(folder_path, "annotations.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Annotation"])
        writer.writerows(annotations)

    print(f"\nアノテーションが完了しました。結果は {csv_path} に保存されました。")
    print("※ 画像順は安定化（自然順優先）されています。")


if __name__ == "__main__":
    folder_path = get_folder_path()
    if folder_path:
        print("\n1: 手動アノテーション")
        print("2: 自動アノテーション（すべて1に設定）")
        print("3: 自動アノテーション（すべて0に設定）")
        mode = input("モードを選択してください（1, 2, または 3）：").strip()

        if mode == "1":
            manual_annotation(folder_path)
        elif mode == "2":
            auto_annotation(folder_path, value=1)
        elif mode == "3":
            auto_annotation(folder_path, value=0)
        else:
            print("無効な選択です。プログラムを終了します。")
    else:
        print("フォルダが選択されませんでした。")
