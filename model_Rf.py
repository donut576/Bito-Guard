# -*- coding: utf-8 -*-
"""
model_random_forest.py

目的：
1. 讀取 train_feature.csv / test_feature.csv
2. 提供多種特徵版本做 ablation study（消融實驗），評估 data leakage 影響：
   - full    : 全部可用數值欄位（上限分數，但可能含 leakage）
   - no_leak : 移除高風險可疑欄位（較保守）
   - safe    : 移除高風險可疑欄位 + 人口學欄位（最保守，最接近真實部署情境）
3. 優先使用 time-based split（模擬真實時序預測場景）；
   若無可用時間欄位，退回 random stratified split
4. 可選用 Optuna 自動調參（直接優化 F1），或使用預設手動參數
5. 輸出：
   - validation metrics（含 PR-AUC）
   - confusion matrix / classification report
   - feature importance CSV + 圖
   - threshold 分析 CSV + 曲線圖
   - SHAP summary plot（TreeExplainer）
   - submission.csv
   - compare_modes.csv
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# =========================================================
# 1. 全域設定區
# =========================================================

TARGET_COL = "status"
ID_COL     = "user_id"

HIGH_LEAKAGE_RISK_KEYWORDS = [
    "last_time",
    "active_span",
    "overall",
    "network",
    "wallet",
    "shared_ip",
    "total_unique_ips",
    "total_ip_usage_count",
    "swap_total",
    "swap_max",
    "swap_avg",
    "crypto_out_total",
    "iforest_score",
    "anomaly",
]

DEMOGRAPHIC_COLS_CANDIDATE = [
    "sex", "age", "career", "income_source", "user_source", "birthday",
]

PREFERRED_TIME_COLS = [
    "overall_first_time",
    "confirmed_at",
    "twd_first_time",
    "crypto_first_time",
    "trade_first_time",
    "swap_first_time",
]

TIME_RELATED_RAW_COLS = [
    "confirmed_at", "level1_finished_at", "level2_finished_at",
    "twd_first_time", "twd_last_time",
    "crypto_first_time", "crypto_last_time",
    "trade_first_time", "trade_last_time",
    "swap_first_time", "swap_last_time",
    "overall_first_time", "overall_last_time",
]


# =========================================================
# 2. Threshold 搜尋
# =========================================================
def find_best_threshold(y_true, y_prob, th_min=0.01, th_max=0.95, step=0.005):
    thresholds = np.arange(th_min, th_max, step)
    rows = []
    best_f1, best_th = 0.0, 0.5

    for th in thresholds:
        pred      = (y_prob >= th).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall    = recall_score(y_true, pred, zero_division=0)
        f1        = f1_score(y_true, pred, zero_division=0)

        rows.append({
            "threshold": th,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)

    threshold_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    return best_th, best_f1, threshold_df


# =========================================================
# 3. Random Forest 調參
# =========================================================
def tune_rf_with_optuna(X_train, y_train, X_valid, y_valid, n_trials=50):
    """
    用 Optuna 搜尋 Random Forest 參數，直接優化 validation F1。
    """
    if not HAS_OPTUNA:
        print("[INFO] Optuna 未安裝，使用手動參數。可安裝: pip install optuna")
        return _default_rf_params()

    def objective(trial):
        params = {
            "n_estimators"      : trial.suggest_int("n_estimators", 200, 1200),
            "max_depth"         : trial.suggest_int("max_depth", 3, 20),
            "min_samples_split" : trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf"  : trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features"      : trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight"      : trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
            "bootstrap"         : True,
            "random_state"      : 42,
            "n_jobs"            : -1,
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_valid)[:, 1]
        _, best_f1, _ = find_best_threshold(y_valid, prob)
        return best_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params.update({
        "bootstrap"    : True,
        "random_state" : 42,
        "n_jobs"       : -1,
    })

    print(f"[Optuna] best_f1={study.best_value:.4f}")
    print(f"[Optuna] best_params={best_params}")
    return best_params


def _default_rf_params():
    """
    Optuna 不可用時的預設 Random Forest 參數。
    """
    return {
        "n_estimators"      : 500,
        "max_depth"         : 10,
        "min_samples_split" : 10,
        "min_samples_leaf"  : 5,
        "max_features"      : "sqrt",
        "class_weight"      : "balanced_subsample",
        "bootstrap"         : True,
        "random_state"      : 42,
        "n_jobs"            : -1,
    }


# =========================================================
# 4. 畫圖工具
# =========================================================
def plot_top20_feature_importance(model, X, title="Top 20 Feature Importance"):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("[WARN] model has no feature_importances_. Skip.")
        return

    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_threshold_curve(threshold_df, best_th, title="Precision / Recall / F1 vs Threshold"):
    plot_df = threshold_df.sort_values("threshold")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["threshold"], plot_df["precision"], label="Precision")
    plt.plot(plot_df["threshold"], plot_df["recall"],    label="Recall")
    plt.plot(plot_df["threshold"], plot_df["f1"],        label="F1")
    plt.axvline(x=best_th, color="red", linestyle="--", label=f"Best th={best_th:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_shap_summary(model, X_valid, sample_size=300):
    """
    Random Forest 跑 SHAP 會比 boosting 模型更慢，所以 sample_size 設小一點。
    """
    if not HAS_SHAP:
        print("[WARN] shap 未安裝，跳過 SHAP。可安裝: pip install shap")
        return
    if len(X_valid) == 0:
        print("[WARN] X_valid 為空，跳過 SHAP。")
        return

    X_sample = X_valid.iloc[:sample_size, :]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_to_plot, X_sample, plot_type="dot")


# =========================================================
# 5. 欄位工具
# =========================================================
def cleanup_xy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns.tolist()
    to_drop = [
        c for c in cols
        if (c.endswith("_x") or c.endswith("_y")) and c[:-2] in cols
    ]
    return df.drop(columns=to_drop, errors="ignore")


def parse_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_like_cols = [
        c for c in df.columns
        if ("time" in c.lower()) or ("date" in c.lower()) or ("_at" in c.lower())
    ]
    for col in time_like_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.5:
                df[col] = parsed
        except Exception:
            pass
    return df


def get_demographic_cols(df: pd.DataFrame):
    return [c for c in DEMOGRAPHIC_COLS_CANDIDATE if c in df.columns]


def get_high_leakage_risk_cols(df: pd.DataFrame):
    return [
        c for c in df.columns
        if any(k in c.lower() for k in HIGH_LEAKAGE_RISK_KEYWORDS)
    ]


def get_datetime_cols(df: pd.DataFrame):
    return df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()


def choose_split_time_col(df: pd.DataFrame):
    for c in PREFERRED_TIME_COLS:
        if c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    return c
            except Exception:
                continue
    return None


# =========================================================
# 6. 資料準備
# =========================================================
def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, mode="full"):
    train_df = train_df.copy()
    test_df  = test_df.copy()

    y      = train_df[TARGET_COL].copy()
    X      = train_df.drop(columns=[TARGET_COL]).copy()
    test_X = test_df.copy()

    demographic_cols       = get_demographic_cols(train_df)
    high_leakage_risk_cols = get_high_leakage_risk_cols(train_df)
    datetime_cols          = get_datetime_cols(train_df)

    drop_cols = [ID_COL] + TIME_RELATED_RAW_COLS + datetime_cols

    if mode == "no_leak":
        drop_cols += high_leakage_risk_cols
    elif mode == "safe":
        drop_cols += high_leakage_risk_cols + demographic_cols
    elif mode != "full":
        raise ValueError(f"未知 mode: {mode}，請使用 'full' / 'no_leak' / 'safe'")

    X      = X.drop(columns=drop_cols, errors="ignore")
    test_X = test_X.drop(columns=drop_cols, errors="ignore")

    non_numeric_cols = X.select_dtypes(exclude=["int", "float", "bool"]).columns.tolist()
    if non_numeric_cols:
        X      = X.drop(columns=non_numeric_cols, errors="ignore")
        test_X = test_X.drop(columns=non_numeric_cols, errors="ignore")

    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    for col in test_X.select_dtypes(include="bool").columns:
        test_X[col] = test_X[col].astype(int)

    X      = X.replace([np.inf, -np.inf], np.nan)
    test_X = test_X.replace([np.inf, -np.inf], np.nan)

    missing_in_test = sorted(set(X.columns) - set(test_X.columns))
    extra_in_test   = sorted(set(test_X.columns) - set(X.columns))

    for col in missing_in_test:
        test_X[col] = 0

    test_X = test_X.drop(columns=extra_in_test, errors="ignore")
    test_X = test_X[X.columns]

    X      = X.fillna(0)
    test_X = test_X.fillna(0)

    constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if constant_cols:
        X      = X.drop(columns=constant_cols, errors="ignore")
        test_X = test_X.drop(columns=constant_cols, errors="ignore")

    info = {
        "mode"                    : mode,
        "demographic_cols"        : demographic_cols,
        "high_leakage_risk_cols"  : high_leakage_risk_cols,
        "datetime_cols"           : datetime_cols,
        "non_numeric_cols_removed": non_numeric_cols,
        "constant_cols_removed"   : constant_cols,
        "missing_in_test"         : missing_in_test,
        "extra_in_test"           : extra_in_test,
        "final_feature_count"     : X.shape[1],
    }
    return X, y, test_X, info


# =========================================================
# 7. 切分函式
# =========================================================
def split_data(X, y, raw_train_df, split_time_col=None, test_size=0.2):
    if split_time_col and split_time_col in raw_train_df.columns:
        time_series = pd.to_datetime(raw_train_df[split_time_col], errors="coerce")
        usable_idx  = raw_train_df.loc[time_series.notna()].sort_values(split_time_col).index.tolist()

        if len(usable_idx) >= 100:
            split_point = int(len(usable_idx) * (1 - test_size))
            train_idx, valid_idx = usable_idx[:split_point], usable_idx[split_point:]
            return (
                X.loc[train_idx], X.loc[valid_idx],
                y.loc[train_idx], y.loc[valid_idx],
                f"time_based({split_time_col})"
            )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_valid, y_train, y_valid, "random_stratified"


# =========================================================
# 8. 單一實驗函式
# =========================================================
def run_experiment(train_df, test_df, mode="full", out_dir="output_rf", use_optuna=True):
    print("=" * 100)
    print(f"[INFO] Running mode = {mode}  |  use_optuna = {use_optuna}")

    X, y, test_X, prep_info = prepare_xy(train_df, test_df, mode=mode)

    split_time_col = choose_split_time_col(train_df)
    X_train, X_valid, y_train, y_valid, split_method = split_data(
        X, y, train_df, split_time_col=split_time_col, test_size=0.2
    )

    print(f"[INFO] split_method    : {split_method}")
    print(f"[INFO] train shape     : {X_train.shape}, valid shape: {X_valid.shape}")
    print(f"[INFO] train pos rate  : {y_train.mean():.4f}, valid pos rate: {y_valid.mean():.4f}")

    best_params = tune_rf_with_optuna(
        X_train, y_train, X_valid, y_valid, n_trials=50
    ) if use_optuna else _default_rf_params()

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    best_th, best_f1, threshold_df = find_best_threshold(y_valid, valid_prob)
    valid_pred = (valid_prob >= best_th).astype(int)

    metrics = {
        "mode"           : mode,
        "split_method"   : split_method,
        "n_features"     : X.shape[1],
        "best_threshold" : best_th,
        "accuracy"       : accuracy_score(y_valid, valid_pred),
        "f1"             : f1_score(y_valid, valid_pred, zero_division=0),
        "auc"            : roc_auc_score(y_valid, valid_prob),
        "pr_auc"         : average_precision_score(y_valid, valid_prob),
        "precision"      : precision_score(y_valid, valid_pred, zero_division=0),
        "recall"         : recall_score(y_valid, valid_pred, zero_division=0),
    }

    print("\n=== Validation Result ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_valid, valid_pred))
    print("\nClassification Report:")
    print(classification_report(y_valid, valid_pred, digits=4))

    mode_out_dir = os.path.join(out_dir, mode)
    os.makedirs(mode_out_dir, exist_ok=True)

    feature_importance_df = pd.DataFrame({
        "feature"   : X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    feature_importance_df["rank"] = range(1, len(feature_importance_df) + 1)
    feature_importance_df.to_csv(os.path.join(mode_out_dir, "feature_importance.csv"), index=False)

    threshold_df.to_csv(os.path.join(mode_out_dir, "threshold_analysis.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(mode_out_dir, "metrics.csv"), index=False)

    pd.DataFrame({
        "parameter": list(model.get_params().keys()),
        "value"    : list(model.get_params().values())
    }).to_csv(os.path.join(mode_out_dir, "best_params.csv"), index=False)

    pd.DataFrame({
        "user_id"   : train_df.loc[X_valid.index, ID_COL].values,
        "true_label": y_valid.values,
        "pred_prob" : valid_prob,
        "pred_label": valid_pred
    }).to_csv(os.path.join(mode_out_dir, "valid_detail.csv"), index=False)

    plot_top20_feature_importance(model, X, title=f"Random Forest - Top 20 Feature Importance ({mode})")
    plot_threshold_curve(threshold_df, best_th, title=f"Threshold Curve ({mode})")
    plot_shap_summary(model, X_valid, sample_size=300)

    # --- 測試集預測 & submission ---
    test_prob = model.predict_proba(test_X)[:, 1]
    test_pred = (test_prob >= best_th).astype(int)

    # 比賽提交檔（0/1）
    pd.DataFrame({
        "user_id": test_df[ID_COL],
        "status" : test_pred,
    }).to_csv(os.path.join(mode_out_dir, "submission.csv"), index=False)

    # 給 ensemble / 投票用的測試集完整分數
    pd.DataFrame({
        "user_id"  : test_df[ID_COL],
        "pred_prob": test_prob,
        "pred_label": test_pred,
        "best_threshold": best_th,
        "mode": mode,
    }).to_csv(os.path.join(mode_out_dir, "test_scores.csv"), index=False)
    # 比賽提交檔（0/1）
    pd.DataFrame({
        "user_id": test_df[ID_COL],
        "status" : test_pred,
    }).to_csv(os.path.join(mode_out_dir, "submission.csv"), index=False)

    # 給 ensemble / 投票用的測試集完整分數
    pd.DataFrame({
        "user_id"  : test_df[ID_COL],
        "pred_prob": test_prob,
        "pred_label": test_pred,
        "best_threshold": best_th,
        "mode": mode,
    }).to_csv(os.path.join(mode_out_dir, "test_scores.csv"), index=False)
    pd.DataFrame({
        "user_id": test_df[ID_COL],
        "status" : test_pred,
    }).to_csv(os.path.join(mode_out_dir, "submission.csv"), index=False)

    pd.DataFrame({
        "user_id"  : test_df[ID_COL],
        "pred_prob": test_prob,
        "status"   : test_pred,
    }).to_csv(os.path.join(mode_out_dir, "test_scores.csv"), index=False)

    plt.figure(figsize=(8, 5))
    sns.histplot(test_prob, bins=50, kde=True)
    plt.axvline(best_th, color="red", linestyle="--", label=f"Threshold: {best_th:.3f}")
    plt.title(f"Distribution of Risk Scores ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[INFO] mode={mode} done. submission saved to {mode_out_dir}/submission.csv")

    return {
        "mode"                 : mode,
        "model"                : model,
        "metrics"              : metrics,
        "prep_info"            : prep_info,
        "feature_importance_df": feature_importance_df,
        "threshold_df"         : threshold_df,
    }


# =========================================================
# 9. 主程式
# =========================================================
def main(
    train_path="train_feature.csv",
    test_path="test_feature.csv",
    out_dir="output_rf",
    use_optuna=True,
):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_df = cleanup_xy_columns(train_df)
    test_df  = cleanup_xy_columns(test_df)
    train_df = parse_time_columns(train_df)
    test_df  = parse_time_columns(test_df)

    print(f"[INFO] train_df shape: {train_df.shape}")
    print(f"[INFO] test_df shape : {test_df.shape}")
    print(f"[INFO] 正樣本比例    : {train_df[TARGET_COL].mean():.4f}")

    modes   = ["full", "no_leak", "safe"]
    results = []

    for mode in modes:
        res = run_experiment(
            train_df=train_df,
            test_df=test_df,
            mode=mode,
            out_dir=out_dir,
            use_optuna=use_optuna,
        )
        results.append(res["metrics"])

    compare_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    compare_df.to_csv(os.path.join(out_dir, "compare_modes.csv"), index=False)

    print("\n" + "=" * 100)
    print("=== Compare Modes ===")
    print(compare_df.to_string(index=False))

    best_mode = compare_df.iloc[0]["mode"]
    print(f"\n[RECOMMEND] F1 最高的版本是 '{best_mode}'。")
    print("[RECOMMEND] 若 full 與 safe 分數差距 > 0.05，建議優先提交 safe 版本，避免 leakage 造成線上線下分數落差。")


if __name__ == "__main__":
    main()
