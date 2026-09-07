import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


# ==============================
# CLIP結果CSVの読み込み（形式ゆれ対応・握りつぶさない）
# ==============================
def load_clip_csv_flexible(path: str) -> pd.DataFrame:
    """
    対応するCSV例:
    [新] Image, grasp_cos, non_grasp_cos, margin
    [新] Image, grasp_cos, non_grasp_cos   (margin無しでもOK)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Image" not in df.columns:
        raise ValueError(f"'Image' column not found in result CSV. columns={list(df.columns)}")

    # 列名ゆれを吸収
    grasp_col = None
    non_col = None

    # 新形式優先
    if "grasp_cos" in df.columns and "non_grasp_cos" in df.columns:
        grasp_col, non_col = "grasp_cos", "non_grasp_cos"
    else:
        # 旧形式
        for g in ["grasp_score", "held_score", "grasp", "held"]:
            if g in df.columns:
                grasp_col = g
                break
        for n in ["non_grasp_score", "not_held_score", "non_grasp", "not_held"]:
            if n in df.columns:
                non_col = n
                break

    if grasp_col is None or non_col is None:
        raise ValueError(
            "Could not detect score columns in result CSV.\n"
            f"columns={list(df.columns)}\n"
            "Expected one of:\n"
            "  (grasp_cos, non_grasp_cos) or (grasp_score/held_score, non_grasp_score/not_held_score)"
        )

    out = df[["Image", grasp_col, non_col]].copy()
    out = out.rename(columns={grasp_col: "grasp", non_col: "non_grasp"})

    out["Image"] = out["Image"].astype(str).str.strip()
    out["grasp"] = pd.to_numeric(out["grasp"], errors="raise")
    out["non_grasp"] = pd.to_numeric(out["non_grasp"], errors="raise")

    # margin列があるなら使う（ただし型変換は厳しめ）
    if "margin" in df.columns:
        out["margin"] = pd.to_numeric(df["margin"], errors="raise")
    else:
        out["margin"] = out["grasp"] - out["non_grasp"]

    # NaNがあるならここで落とす（静かに落とすのではなく、数を表示して気づけるように）
    n_before = len(out)
    out = out.dropna(subset=["Image", "grasp", "non_grasp", "margin"])
    n_after = len(out)
    if n_after != n_before:
        print(f"[WARN] Dropped rows with NaN: {n_before - n_after}")

    return out


# ==============================
# 評価（marginで判定・ROCもmargin）
# positive = grasp(0)
# pred: margin > thresh_hold => grasp(0) else non_grasp(1)
# ==============================
def result_check(
    result_path: str,
    annotation_path: str,
    output_csv: str,
    output_roc_grasp: str,
    thresh_hold: float = 0.0,  # ★margin閾値。2文なら0.0が自然（argmaxと一致）
    eps: float = 1e-12,
):
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation CSV not found: {annotation_path}")

    # ===== 読み込み（annotation）=====
    ann = pd.read_csv(annotation_path)
    ann.columns = [c.strip() for c in ann.columns]

    if "Image" not in ann.columns or "Annotation" not in ann.columns:
        raise ValueError(f"Annotation CSV must have columns 'Image' and 'Annotation'. columns={list(ann.columns)}")

    ann["Image"] = ann["Image"].astype(str).str.strip()
    ann["Annotation"] = pd.to_numeric(ann["Annotation"], errors="raise").astype(int)

    # ===== 読み込み（result）=====
    res = load_clip_csv_flexible(result_path)

    # ===== マージ =====
    df = pd.merge(ann, res, on="Image", how="inner")
    print("merged rows:", len(df))
    if len(df) == 0:
        raise RuntimeError("マージ結果が0行。Imageが一致していない可能性。")

    # ===== 予測（margin閾値）=====
    # margin > thresh_hold => grasp(0)
    # else => non_grasp(1)
    df["pred"] = (df["margin"].astype(float) <= float(thresh_hold)).astype(int)

    y_true = df["Annotation"].astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()

    # ===== 混同行列 =====
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion Matrix (rows=true [0,1], cols=pred [0,1])\n", cm)

    TP = int(cm[0, 0])
    FN = int(cm[0, 1])
    FP = int(cm[1, 0])
    TN = int(cm[1, 1])
    total = TP + TN + FP + FN

    accuracy  = (TP + TN) / (total + eps)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = (2 * precision * recall) / (precision + recall + eps)

    # ===== ROC（把持を正例・スコアはmargin）=====
    os.makedirs(os.path.dirname(output_roc_grasp), exist_ok=True)
    y_true_grasp = (y_true == 0)  # Trueが把持
    y_score = df["margin"].astype(float).to_numpy()

    fpr, tpr, _ = roc_curve(y_true_grasp, y_score)
    auc_grasp = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC(grasp)= {auc_grasp:.3f}")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (Positive=grasp(0), score=margin, thr={thresh_hold})")
    plt.legend()
    plt.grid()
    plt.savefig(output_roc_grasp, dpi=200)
    plt.close()

    # ===== 表示 =====
    print("\nConfusion Matrix terms (positive = grasp=0):")
    print(f"TP (grasp->grasp)      : {TP}")
    print(f"FN (grasp->non_grasp)  : {FN}")
    print(f"FP (non_grasp->grasp)  : {FP}")
    print(f"TN (non_grasp->non)    : {TN}")

    print("\n=== Metrics (positive class = grasp=0) ===")
    print(f"Threshold(margin): {thresh_hold}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc_grasp:.4f}")

    # ===== レポートCSV保存 =====
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    cm_row0 = f"[{cm[0,0]} {cm[0,1]}]"
    cm_row1 = f"[{cm[1,0]} {cm[1,1]}]"

    report_rows = [
        ("Result file", result_path),
        ("Annotation file", annotation_path),
        ("Count", str(total)),
        ("Threshold(margin)", str(thresh_hold)),
        ("", ""),

        ("Confusion Matrix (rows=true [grasp=0, non_grasp=1], cols=pred [0,1])", ""),
        ("CM row0", cm_row0),
        ("CM row1", cm_row1),
        ("", ""),

        ("Confusion Matrix terms (positive = grasp=0)", ""),
        ("TP (grasp->grasp)", str(TP)),
        ("FN (grasp->non_grasp)", str(FN)),
        ("FP (non_grasp->grasp)", str(FP)),
        ("TN (non_grasp->non)", str(TN)),
        ("", ""),

        ("Metrics (positive class = grasp=0)", ""),
        ("Accuracy", f"{accuracy:.4f}"),
        ("Precision", f"{precision:.4f}"),
        ("Recall", f"{recall:.4f}"),
        ("F1-score", f"{f1:.4f}"),
        ("AUC(grasp0)", f"{auc_grasp:.4f}"),
        ("ROC image", output_roc_grasp),
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Item", "Value"])
        w.writerows(report_rows)

    print(f"\nSaved report CSV: {output_csv}")
    print(f"Saved ROC image : {output_roc_grasp}")


if __name__ == "__main__":
    result_path = "/data/Users/kuroda/2025_research/2025_result/2025-12-28_kiban/test/CLIP/2026-1-5_kiban_test2_demo_CLIP_results_metaclip_fullcc/clip_cosine_results.csv" #CLIP結果csv
    output_csv = "/data/Users/kuroda/2025_research/2025_result/2025-12-28_kiban/CLIP_result/test2/2026-1-7_test2_CLIP_results_metaclip_fullcc_debug.csv" #出力
    output_roc_grasp = "/data/Users/kuroda/2025_research/2025_result/2025-12-28_kiban/CLIP_result/test2/roc_metaclip_fullcc_debug.png" #出力roc画像
    annotation_path = "/data/Users/kuroda/2025_research/2025-12-28_kiban/CLIP_csv/marge/merged_annotations_2026_1_5_kiban_test2.csv" #アノテーションcsv

    #閾値：margin
    thresh_hold = 0.0

    result_check(
        result_path=result_path,
        annotation_path=annotation_path,
        output_csv=output_csv,
        output_roc_grasp=output_roc_grasp,
        thresh_hold=thresh_hold,
        eps=1e-4,
    )
