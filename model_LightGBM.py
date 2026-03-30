# -*- coding: utf-8 -*-
"""
model_lightgbm.py

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
   - validation metrics（含 PR-AUC，對不平衡資料更有參考價值）
   - confusion matrix / classification report
   - feature importance CSV + 圖
   - threshold 分析 CSV + 曲線圖
   - SHAP summary plot
   - submission.csv
   - 各版本比較表 compare_modes.csv
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

import lightgbm as lgb
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

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imbalanced-learn 未安裝，SMOTE 將被跳過。可安裝: pip install imbalanced-learn")


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
# 2. Focal Loss（自訂 LightGBM objective）
# =========================================================
def focal_loss_lgb(alpha=0.25, gamma=2.0):
    """
    Focal Loss 是針對不平衡分類問題設計的 loss function。
    
    標準 cross-entropy 的問題：
    - 對「容易分類的負樣本（大量正常用戶）」給予和「難分類的正樣本（少數黑名單）」
      相同的 loss 權重，導致模型被大量負樣本主導，忽略少數正樣本。
    
    Focal Loss 的改進：
    - 引入 (1 - pt)^gamma 調製因子，讓「容易分類的樣本」的 loss 自動縮小
    - 模型越確定某個樣本是負類（pt 接近 1），那個樣本的 loss 就越小
    - 讓模型把更多注意力放在「難分類的正樣本」上
    
    參數說明：
    - alpha : 正樣本的權重（0~1），通常設 0.25，讓正樣本得到更多關注
    - gamma : 調製強度（>= 0），gamma=0 退化成標準 cross-entropy
              gamma=2 是最常用的值，讓容易樣本的 loss 大幅縮小
    
    回傳：
        custom_obj : LightGBM 自訂 objective 函式，輸入 (y_pred, dataset)，
                     輸出 (gradient, hessian)
    """
    def custom_obj(labels, preds):
        """
        LGBMClassifier 的自訂 objective 透過 sklearn API 呼叫時，
        傳入的是 (labels, preds) 兩個 numpy array，
        不是 LightGBM 原生 API 的 (y_pred, Dataset)。
        """
        y_true = labels
        # sigmoid 轉換：把 raw score 轉成機率
        p = 1.0 / (1.0 + np.exp(-preds))
        # pt：預測正確的機率（正樣本看 p，負樣本看 1-p）
        pt = np.where(y_true == 1, p, 1 - p)
        # alpha_t：正樣本用 alpha，負樣本用 1-alpha
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
        # 調製因子：(1-pt)^gamma，越容易分類的樣本這個值越小
        modulating = (1 - pt) ** gamma
        # gradient（一階導數）
        grad = alpha_t * modulating * (p - y_true) - \
               alpha_t * gamma * modulating * np.log(pt + 1e-8) * pt * (1 - pt) * np.sign(y_true - p)
        # hessian（二階導數，用近似值讓訓練更穩定）
        hess = alpha_t * modulating * p * (1 - p)
        return grad, hess
    return custom_obj


def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    """
    SMOTE（Synthetic Minority Over-sampling Technique）過採樣
    只對訓練集做，避免 data leakage
    """
    if not HAS_SMOTE:
        print("[WARN] SMOTE 未安裝，跳過過採樣，使用原始訓練集")
        return X_train, y_train

    print(f"[INFO] SMOTE 前：正樣本 {y_train.sum()} / 總樣本 {len(y_train)} ({y_train.mean():.4f})")

    # 新版 imbalanced-learn 的 SMOTE 已不支援 n_jobs 參數
    sm = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    X_res, y_res = sm.fit_resample(X_train, y_train)
    y_res = pd.Series(y_res, name=y_train.name)

    print(f"[INFO] SMOTE 後：正樣本 {y_res.sum()} / 總樣本 {len(y_res)} ({y_res.mean():.4f})")
    return X_res, y_res
# =========================================================
# 3. Threshold 搜尋
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
            "f1": f1
        })

        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)

    threshold_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    return best_th, best_f1, threshold_df


# =========================================================
# 4. LightGBM 調參
# =========================================================
def tune_lgb_with_optuna(X_train, y_train, X_valid, y_valid, scale_pos_weight, n_trials=100):
    """
    用 Optuna 搜尋 LightGBM 超參數，直接優化 validation F1。
    
    改進項目（相較於基本版）：
    1. n_trials 從 50 提升到 100，搜尋更充分
    2. 加入 boosting_type 選擇（gbdt vs dart）
       - gbdt：標準梯度提升，支援 early stopping，速度快
       - dart：用 dropout 概念訓練，對不平衡資料有時更穩定，但不支援 early stopping
    3. 加入 Focal Loss 的 alpha/gamma 搜尋
       - 讓 Optuna 自動找最適合這份資料的 Focal Loss 強度
    4. scale_pos_weight 只在 gbdt 模式下使用
       - dart + focal loss 已經處理不平衡問題，不需要再疊加 scale_pos_weight
    """
    if not HAS_OPTUNA:
        print("[INFO] Optuna 未安裝，使用手動參數。可安裝: pip install optuna")
        return _default_lgb_params(scale_pos_weight)

    def objective(trial):
        # 選擇 booster 類型
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

        params = {
            "boosting_type"     : boosting_type,
            "n_estimators"      : trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate"     : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves"        : trial.suggest_int("num_leaves", 15, 255),
            "max_depth"         : trial.suggest_int("max_depth", 3, 12),
            "min_child_samples" : trial.suggest_int("min_child_samples", 10, 100),
            "subsample"         : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain"    : trial.suggest_float("min_split_gain", 0.0, 2.0),
            "random_state"      : 42,
            "n_jobs"            : -1,
            "verbosity"         : -1,
        }

        # dart 專屬參數
        if boosting_type == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)
            # dart 不支援 early stopping，限制 n_estimators 避免 trial 跑太久
            params["n_estimators"] = min(params["n_estimators"], 500)

        # 選擇是否使用 Focal Loss
        use_focal = trial.suggest_categorical("use_focal_loss", [True, False])

        if use_focal:
            # Focal Loss 模式：用自訂 objective，不用 scale_pos_weight
            focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.5)
            focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
            params["objective"] = focal_loss_lgb(alpha=focal_alpha, gamma=focal_gamma)
            params["metric"]    = "binary_logloss"  # Focal Loss 用 logloss 監控
        else:
            # 標準模式：用 scale_pos_weight 處理不平衡
            params["objective"]        = "binary"
            params["metric"]           = "average_precision"
            params["scale_pos_weight"] = trial.suggest_float(
                "scale_pos_weight",
                max(1.0, scale_pos_weight * 0.5),
                scale_pos_weight * 2.0,
            )

        model = lgb.LGBMClassifier(**params)

        if boosting_type == "dart":
            # dart 不支援 early stopping，直接用固定 n_estimators 訓練
            model.fit(X_train, y_train)
        else:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )

        # 自訂 objective 時 predict_proba 回傳 raw score（1D），需手動 sigmoid
        # 標準 objective 時 predict_proba 回傳機率矩陣（2D），取第二欄
        if use_focal:
            raw = model.predict(X_valid)
            prob = 1.0 / (1.0 + np.exp(-raw))  # sigmoid
        else:
            prob = model.predict_proba(X_valid)[:, 1]
        _, best_f1, _ = find_best_threshold(y_valid, prob)
        return best_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_params)

    # 把 Focal Loss 的 alpha/gamma 從 best_params 裡取出來重建 objective
    use_focal   = best_params.pop("use_focal_loss", False)
    focal_alpha = best_params.pop("focal_alpha", 0.25)
    focal_gamma = best_params.pop("focal_gamma", 2.0)

    if use_focal:
        best_params["objective"] = focal_loss_lgb(alpha=focal_alpha, gamma=focal_gamma)
        best_params["metric"]    = "binary_logloss"
        best_params["focal_alpha_used"] = focal_alpha  # 記錄用了哪組值，方便 debug
        best_params["focal_gamma_used"] = focal_gamma
        print(f"[Optuna] 使用 Focal Loss: alpha={focal_alpha:.3f}, gamma={focal_gamma:.3f}")
    else:
        best_params["objective"] = "binary"
        best_params["metric"]    = "average_precision"

    best_params.update({"random_state": 42, "n_jobs": -1, "verbosity": -1})

    print(f"[Optuna] best_f1={study.best_value:.4f}")
    print(f"[Optuna] boosting_type={best_params.get('boosting_type')}, use_focal={use_focal}")
    return best_params


def _default_lgb_params(scale_pos_weight):
    """
    Optuna 不可用時的預設 LightGBM 參數。
    """
    return {
        "n_estimators"      : 500,
        "learning_rate"     : 0.02,
        "num_leaves"        : 63,
        "max_depth"         : 6,
        "min_child_samples" : 30,
        "subsample"         : 0.8,
        "colsample_bytree"  : 0.8,
        "reg_alpha"         : 0.1,
        "reg_lambda"        : 1.0,
        "min_split_gain"    : 0.1,
        "scale_pos_weight"  : scale_pos_weight,

        "objective"         : "binary",
        "metric"            : "average_precision",
        "random_state"      : 42,
        "n_jobs"            : -1,
        "verbosity"         : -1,
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


def plot_shap_summary(model, X_valid, sample_size=500):
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
def run_experiment(train_df, test_df, mode="full", out_dir="output_lgb", use_optuna=True):
    print("=" * 100)
    print(f"[INFO] Running mode = {mode}  |  use_optuna = {use_optuna}")

    X, y, test_X, prep_info = prepare_xy(train_df, test_df, mode=mode)

    split_time_col = choose_split_time_col(train_df)
    X_train, X_valid, y_train, y_valid, split_method = split_data(
        X, y, train_df, split_time_col=split_time_col, test_size=0.2
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print(f"[INFO] split_method    : {split_method}")
    print(f"[INFO] scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"[INFO] train shape     : {X_train.shape}, valid shape: {X_valid.shape}")
    print(f"[INFO] train pos rate  : {y_train.mean():.4f}, valid pos rate: {y_valid.mean():.4f}")

    # ---------------------------------------------------------
    # 特徵篩選：先用預設參數快速訓練一次，移除 importance=0 的特徵
    # 再用篩選後的特徵跑 Optuna，讓調參更聚焦在有用的特徵上
    # ---------------------------------------------------------
    print("[INFO] 第一輪：快速訓練找出 importance=0 的特徵...")
    _screening_model = lgb.LGBMClassifier(**_default_lgb_params(scale_pos_weight))
    _screening_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    _importance_series = pd.Series(
        _screening_model.feature_importances_, index=X_train.columns
    )
    zero_importance_cols = _importance_series[_importance_series == 0].index.tolist()

    if zero_importance_cols:
        print(f"[INFO] 移除 {len(zero_importance_cols)} 個 importance=0 的特徵: {zero_importance_cols[:10]}{'...' if len(zero_importance_cols) > 10 else ''}")
        X       = X.drop(columns=zero_importance_cols, errors="ignore")
        test_X  = test_X.drop(columns=zero_importance_cols, errors="ignore")
        X_train = X_train.drop(columns=zero_importance_cols, errors="ignore")
        X_valid = X_valid.drop(columns=zero_importance_cols, errors="ignore")
        print(f"[INFO] 篩選後特徵數: {X.shape[1]}")
    else:
        print("[INFO] 沒有 importance=0 的特徵，跳過篩選")

    best_params = tune_lgb_with_optuna(
        X_train, y_train, X_valid, y_valid,
        scale_pos_weight=scale_pos_weight,
        n_trials=100
    ) if use_optuna else _default_lgb_params(scale_pos_weight)

    # ---------------------------------------------------------
    # SMOTE 過採樣（只對訓練集做，驗證集和測試集絕對不動）
    # 把正樣本比例從 ~3.2% 提升到 10%，讓模型看到更多正樣本的特徵變化
    # 注意：SMOTE 要在 Optuna 之後做，因為 Optuna 內部已經用原始分布調參
    #       這裡的 SMOTE 只影響最終模型的訓練，不影響 Optuna 的搜尋結果
    # ---------------------------------------------------------
    X_train_final, y_train_final = apply_smote(
        X_train, y_train,
        sampling_strategy=0.1,  # 正樣本比例目標：10%
        random_state=42,
    )

    # 取出 Focal Loss 相關資訊（如果 Optuna 選了 Focal Loss）
    focal_alpha_used = best_params.pop("focal_alpha_used", None)
    focal_gamma_used = best_params.pop("focal_gamma_used", None)
    boosting_type    = best_params.get("boosting_type", "gbdt")

    model = lgb.LGBMClassifier(**best_params)

    if boosting_type == "dart":
        # dart 不支援 early stopping，用 SMOTE 後的完整訓練集訓練
        print("[INFO] boosting_type=dart，使用固定 n_estimators 訓練（不支援 early stopping）")
        model.fit(X_train_final, y_train_final)
    else:
        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

    # 自訂 objective（Focal Loss）時 predict_proba 回傳 raw score，需手動 sigmoid
    use_focal_final = focal_alpha_used is not None
    if use_focal_final:
        raw = model.predict(X_valid)
        valid_prob = 1.0 / (1.0 + np.exp(-raw))
    else:
        valid_prob = model.predict_proba(X_valid)[:, 1]
    best_th, best_f1, threshold_df = find_best_threshold(y_valid, valid_prob)
    valid_pred = (valid_prob >= best_th).astype(int)

    # ---------------------------------------------------------
    # Pseudo-labeling：用第一輪模型對 test 預測，
    # 把高信心的預測加回訓練集，再重新訓練一次最終模型。
    #
    # 為什麼有效？
    # - 你的 test 有 12753 筆，其中可能有幾百筆高信心正樣本
    # - 把這些「幾乎確定是黑名單」的樣本加進訓練集，
    #   讓模型看到更多正樣本的特徵分布，提升 recall
    # - 同時加入高信心負樣本，避免模型偏向正類
    #
    # 注意事項：
    # - 只用「高信心」的預測（正樣本 > pos_threshold，負樣本 < neg_threshold）
    # - 信心門檻設太低會引入噪音，反而讓模型變差
    # - 負樣本數量設為正樣本的 neg_ratio 倍，維持合理的正負比例
    # ---------------------------------------------------------
    if use_focal_final:
        raw_test_round1 = model.predict(test_X)
        test_prob_round1 = 1.0 / (1.0 + np.exp(-raw_test_round1))
    else:
        test_prob_round1 = model.predict_proba(test_X)[:, 1]

    pos_threshold = 0.80   # 機率 > 80% 才當正樣本偽標籤（保守，避免噪音）
    neg_threshold = 0.03   # 機率 < 3% 才當負樣本偽標籤
    neg_ratio     = 5      # 負樣本偽標籤數量 = 正樣本偽標籤數量 × neg_ratio

    pseudo_pos_mask = test_prob_round1 > pos_threshold
    pseudo_neg_mask = test_prob_round1 < neg_threshold

    n_pseudo_pos = pseudo_pos_mask.sum()
    n_pseudo_neg = min(pseudo_neg_mask.sum(), n_pseudo_pos * neg_ratio)

    print(f"\n[Pseudo-labeling] 高信心正樣本: {n_pseudo_pos}, 高信心負樣本候選: {pseudo_neg_mask.sum()}, 實際使用負樣本: {n_pseudo_neg}")

    if n_pseudo_pos >= 10:
        # 取出 test 的原始特徵（已篩選過 importance=0 的欄位）
        pseudo_pos_X = test_X[pseudo_pos_mask].copy()
        pseudo_pos_y = pd.Series(
            np.ones(n_pseudo_pos, dtype=int),
            index=pseudo_pos_X.index,
            name=TARGET_COL,
        )

        # 負樣本隨機取樣，避免全部用上造成不平衡
        neg_indices = np.where(pseudo_neg_mask)[0]
        np.random.seed(42)
        sampled_neg_indices = np.random.choice(neg_indices, size=n_pseudo_neg, replace=False)
        pseudo_neg_X = test_X.iloc[sampled_neg_indices].copy()
        pseudo_neg_y = pd.Series(
            np.zeros(n_pseudo_neg, dtype=int),
            index=pseudo_neg_X.index,
            name=TARGET_COL,
        )

        # 合併：原始訓練集 + 偽標籤正樣本 + 偽標籤負樣本
        X_aug = pd.concat([X_train_final, pseudo_pos_X, pseudo_neg_X], ignore_index=True)
        y_aug = pd.concat([y_train_final, pseudo_pos_y, pseudo_neg_y], ignore_index=True)

        print(f"[Pseudo-labeling] 擴充後訓練集: {len(X_aug)} 筆，正樣本比例: {y_aug.mean():.4f}")

        # 用擴充後的訓練集重新訓練（使用相同的 best_params）
        model_pl = lgb.LGBMClassifier(**best_params)
        if boosting_type == "dart":
            model_pl.fit(X_aug, y_aug)
        else:
            model_pl.fit(
                X_aug, y_aug,
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )

        # 用 pseudo-labeling 模型重新評估驗證集
        if use_focal_final:
            raw_pl = model_pl.predict(X_valid)
            valid_prob_pl = 1.0 / (1.0 + np.exp(-raw_pl))
        else:
            valid_prob_pl = model_pl.predict_proba(X_valid)[:, 1]

        _, best_f1_pl, _ = find_best_threshold(y_valid, valid_prob_pl)
        print(f"[Pseudo-labeling] Round 1 F1={best_f1:.4f} → Round 2 F1={best_f1_pl:.4f}")

        # 如果 pseudo-labeling 有改善，就用新模型
        if best_f1_pl >= best_f1:
            print("[Pseudo-labeling] 採用 pseudo-labeling 模型（有改善）")
            model      = model_pl
            valid_prob = valid_prob_pl
            best_th, best_f1, threshold_df = find_best_threshold(y_valid, valid_prob)
            valid_pred = (valid_prob >= best_th).astype(int)
        else:
            print("[Pseudo-labeling] 保留原始模型（pseudo-labeling 未改善）")
    else:
        print(f"[Pseudo-labeling] 高信心正樣本不足（{n_pseudo_pos} < 10），跳過")

    metrics = {
        "mode"            : mode,
        "split_method"    : split_method,
        "n_features"      : X.shape[1],
        "best_threshold"  : best_th,
        "accuracy"        : accuracy_score(y_valid, valid_pred),
        "f1"              : f1_score(y_valid, valid_pred, zero_division=0),
        "auc"             : roc_auc_score(y_valid, valid_prob),
        "pr_auc"          : average_precision_score(y_valid, valid_prob),
        "precision"       : precision_score(y_valid, valid_pred, zero_division=0),
        "recall"          : recall_score(y_valid, valid_pred, zero_division=0),
        "boosting_type"   : boosting_type,
        "used_focal_loss" : focal_alpha_used is not None,
        "focal_alpha"     : focal_alpha_used,
        "focal_gamma"     : focal_gamma_used,
        "used_smote"      : HAS_SMOTE,
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

    plot_top20_feature_importance(model, X, title=f"LightGBM - Top 20 Feature Importance ({mode})")
    plot_threshold_curve(threshold_df, best_th, title=f"Threshold Curve ({mode})")
    plot_shap_summary(model, X_valid, sample_size=500)

    if use_focal_final:
        raw_test = model.predict(test_X)
        test_prob = 1.0 / (1.0 + np.exp(-raw_test))
    else:
        test_prob = model.predict_proba(test_X)[:, 1]
    test_pred = (test_prob >= best_th).astype(int)

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
    out_dir="output_lgb",
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

    modes   = ["full"] # "no_leak", "safe"
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
