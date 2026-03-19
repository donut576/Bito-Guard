# -*- coding: utf-8 -*-
"""
model_LightGBM.ipynb

Refactor from colab_2.py:
- 讀取 train_df.csv / test_df.csv
- 特徵清理（drop user_id + 時間欄位 + 非數值欄位）
- Train/Valid split
- 訓練 LightGBM
-（可選）Optuna 調參
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

import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ----------------------------
# Utils
# ----------------------------
def find_best_threshold(y_true, y_prob, th_min=0.01, th_max=0.6, step=0.005):
    """掃 threshold 找最佳 F1"""
    thresholds = np.arange(th_min, th_max, step)
    best_f1, best_th = 0.0, 0.5
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)
    return best_th, best_f1


def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    依照 colab_2.py 的做法：
    - y = status
    - drop user_id + datetime 欄位 + time_related_cols
    - 移除非數值欄位
    """
    y = train_df["status"]
    X = train_df.drop(columns=["status"]).copy()
    test_X = test_df.copy()

    # datetime 欄位（若有）
    datetime_cols = X.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    # 既知時間欄位（colab_2.py）
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

    # 移除非數值欄位（保留 int/float/bool）
    non_numeric_cols = X.select_dtypes(
        exclude=["int", "float", "bool"]).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)
        test_X = test_X.drop(columns=non_numeric_cols, errors="ignore")

    # 建議填補缺失值，避免部分模型/SHAP 出現問題（你若不想填可刪掉）
    X = X.fillna(-1)
    test_X = test_X.fillna(-1)

    return X, y, test_X


def train_lgbm_with_optional_optuna(X_train, y_train, X_valid, y_valid, pos_weight, use_optuna=True):
    """
    把 colab_2.py 的 Optuna/手動參數邏輯包起來。
    回傳：best_params dict
    """
    best_params = None

    HAS_OPTUNA = False
    optuna = None
    if use_optuna:
        try:
            import optuna as _optuna
            _optuna.logging.set_verbosity(_optuna.logging.WARNING)
            optuna = _optuna
            HAS_OPTUNA = True
        except ImportError:
            HAS_OPTUNA = False

    if HAS_OPTUNA:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "scale_pos_weight": pos_weight,
                "random_state": 42,
                "verbose": -1,
            }
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(
                    50, verbose=False), lgb.log_evaluation(-1)],
            )
            prob = m.predict_proba(X_valid)[:, 1]
            _, best_f1 = find_best_threshold(y_valid, prob)
            return best_f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        best_params = dict(study.best_params)
        best_params["scale_pos_weight"] = pos_weight
        best_params["random_state"] = 42
        best_params["verbose"] = -1
        print(f"[Optuna] best_f1={study.best_value:.4f}")
        print(f"[Optuna] best_params={best_params}")
    else:
        # colab_2.py 的手動參數
        best_params = {
            "n_estimators": 500,
            "learning_rate": 0.02,
            "num_leaves": 127,
            "max_depth": 7,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": pos_weight,
            "random_state": 42,
            "verbose": -1,
        }
        print("[Info] Optuna not available, using manual params.")

    return best_params


def plot_top20_feature_importance(model, X, title="Top 20 Feature Importance"):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("[WARN] model has no feature_importances_. Skip feature importance.")
        return

    indices = np.argsort(importances)[-20:]  # Top 20
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()


def plot_threshold_curve(y_true, y_prob, best_th):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve 的 thresholds 少 1，需要對齊
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

    # 二元分類可能回 list，取 class=1
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_to_plot, X_sample, plot_type="dot")


# ----------------------------
# Main
# ----------------------------
def main(
    train_path="train_df.csv",
    test_path="test_df.csv",
    out_path="submission.csv",
    use_optuna=True,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X, y, test_X = prepare_xy(train_df, test_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"pos_weight: {pos_weight:.2f}")

    best_params = train_lgbm_with_optional_optuna(
        X_train, y_train, X_valid, y_valid, pos_weight, use_optuna=use_optuna
    )

    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(
            100, verbose=False), lgb.log_evaluation(-1)],
    )

    valid_prob = model.predict_proba(X_valid)[:, 1]
    best_th, best_f1 = find_best_threshold(y_valid, valid_prob)

    valid_pred = (valid_prob >= best_th).astype(int)
    print("=== LightGBM Validation ===")
    print("best_th:", best_th, "best_f1:", best_f1)
    print("Confusion Matrix:\n", confusion_matrix(y_valid, valid_pred))
    print("Classification Report:\n", classification_report(y_valid, valid_pred))
    print(f"F1: {f1_score(y_valid, valid_pred, zero_division=0):.4f}")
    print("AUC:", roc_auc_score(y_valid, valid_prob))
    print("Precision:", precision_score(y_valid, valid_pred, zero_division=0))
    print("Recall:", recall_score(y_valid, valid_pred, zero_division=0))

    # ===== 額外輸出：Feature Importance / Threshold curve / SHAP =====
    plot_top20_feature_importance(
        model, X, title="LightGBM - Top 20 Feature Importance")
    plot_threshold_curve(y_valid, valid_prob, best_th=best_th)
    plot_shap_summary(model, X_valid, sample_size=500)

    # ===== Test / submission =====
    test_prob = model.predict_proba(test_X)[:, 1]
    test_pred = (test_prob >= best_th).astype(int)

    submission = pd.DataFrame({
        "user_id": test_df["user_id"],
        "status": test_pred,
    })
    submission.to_csv(out_path, index=False)
    print(f"{out_path} saved")

    # 分佈圖
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
