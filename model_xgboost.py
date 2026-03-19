# -*- coding: utf-8 -*-
"""
model_XGBoost.ipynb

Refactor from colab_2.py:
- 讀取 train_df.csv / test_df.csv
- 特徵清理（drop user_id + 時間欄位 + 非數值欄位）
- Train/Valid split
- 訓練 XGBoost
- 模型評估 + Feature Importance + Threshold tuning + SHAP
- 輸出 submission.csv
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def find_best_threshold(y_true, y_prob, th_min=0.01, th_max=0.6, step=0.005):
    thresholds = np.arange(th_min, th_max, step)
    best_f1, best_th = 0.0, 0.5
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)
    return best_th, best_f1


def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    y = train_df["status"]
    X = train_df.drop(columns=["status"]).copy()
    test_X = test_df.copy()

    datetime_cols = X.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    time_related_cols = [
        "confirmed_at", "level1_finished_at", "level2_finished_at",
        "twd_first_time", "twd_last_time",
        "crypto_first_time", "crypto_last_time",
        "trade_first_time", "trade_last_time",
        "swap_first_time", "swap_last_time",
        "overall_first_time", "overall_last_time",
    ]

    drop_cols = datetime_cols + time_related_cols + ["user_id"]
    X = X.drop(columns=drop_cols, errors="ignore")
    test_X = test_X.drop(columns=drop_cols, errors="ignore")

    non_numeric_cols = X.select_dtypes(
        exclude=["int", "float", "bool"]).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)
        test_X = test_X.drop(columns=non_numeric_cols, errors="ignore")

    X = X.fillna(-1)
    test_X = test_X.fillna(-1)
    return X, y, test_X


def plot_top20_feature_importance(model, X, title="Top 20 Feature Importance"):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("[WARN] model has no feature_importances_. Skip feature importance.")
        return

    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()


def plot_threshold_curve(y_true, y_prob, best_th):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    thresholds_plot = thresholds
    precisions_plot = precisions[:-1]
    recalls_plot = recalls[:-1]
    f1_scores = 2 * (precisions_plot * recalls_plot) / \
        (precisions_plot + recalls_plot + 1e-10)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds_plot, precisions_plot, label="Precision")
    plt.plot(thresholds_plot, recalls_plot, label="Recall")
    plt.plot(thresholds_plot, f1_scores, label="F1")
    plt.axvline(x=best_th, color="red", linestyle="--",
                label=f"Best th={best_th:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_shap_summary(model, X_valid, sample_size=500):
    if not HAS_SHAP:
        print("[WARN] shap 未安裝，跳過 SHAP。可安裝: pip install shap")
        return

    X_sample = X_valid.iloc[:sample_size, :]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_to_plot, X_sample, plot_type="dot")


def main(
    train_path="train_df.csv",
    test_path="test_df.csv",
    out_path="submission.csv",
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X, y, test_X = prepare_xy(train_df, test_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=42,
        verbosity=0,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    valid_prob = model.predict_proba(X_valid)[:, 1]
    best_th, best_f1 = find_best_threshold(y_valid, valid_prob)

    valid_pred = (valid_prob >= best_th).astype(int)
    print("=== XGBoost Validation ===")
    print("best_th:", best_th, "best_f1:", best_f1)
    print("Confusion Matrix:\n", confusion_matrix(y_valid, valid_pred))
    print("Classification Report:\n", classification_report(y_valid, valid_pred))
    print(f"F1: {f1_score(y_valid, valid_pred, zero_division=0):.4f}")
    print("AUC:", roc_auc_score(y_valid, valid_prob))
    print("Precision:", precision_score(y_valid, valid_pred, zero_division=0))
    print("Recall:", recall_score(y_valid, valid_pred, zero_division=0))

    plot_top20_feature_importance(
        model, X, title="XGBoost - Top 20 Feature Importance")
    plot_threshold_curve(y_valid, valid_prob, best_th=best_th)
    plot_shap_summary(model, X_valid, sample_size=500)

    test_prob = model.predict_proba(test_X)[:, 1]
    test_pred = (test_prob >= best_th).astype(int)

    submission = pd.DataFrame({
        "user_id": test_df["user_id"],
        "status": test_pred,
    })
    submission.to_csv(out_path, index=False)
    print(f"{out_path} saved")

    plt.figure(figsize=(8, 5))
    sns.histplot(test_prob, bins=50, kde=True)
    plt.axvline(best_th, color="red", linestyle="--",
                label=f"Threshold: {best_th:.2f}")
    plt.title("Distribution of Risk Scores (Probability)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
