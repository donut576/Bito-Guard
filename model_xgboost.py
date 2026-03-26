# -*- coding: utf-8 -*-
"""
model_xgboost.py

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

import xgboost as xgb
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
#    集中管理所有「欄位名稱規則」，方便日後維護
# =========================================================

TARGET_COL = "status"   # 預測目標：1=人頭戶/黑名單，0=正常
ID_COL     = "user_id"  # 唯一識別碼，不能當特徵

# ----------------------------------------------------------
# 高風險 leakage 欄位關鍵字
# 這些欄位「可能」在標籤產生後才被記錄，或與標籤高度相關，
# 放進模型會讓驗證分數虛高，但實際部署時根本拿不到。
# 做 ablation 時用 no_leak / safe mode 移除，觀察分數差距。
# ----------------------------------------------------------
HIGH_LEAKAGE_RISK_KEYWORDS = [
    "last_time",          # 最後活動時間：標籤確定後才有意義
    "active_span",        # 活躍時間跨度：同上
    "overall",            # 整體統計：可能包含標籤期間的行為
    "network",            # 網路特徵：可能是事後分析結果
    "wallet",             # 錢包特徵：同上
    "shared_ip",          # 共用 IP：可能是事後調查結果
    "total_unique_ips",   # IP 數量：同上
    "total_ip_usage_count",
    "swap_total",         # swap 總量：若 swap 行為本身就是標籤依據
    "swap_max",
    "swap_avg",
    "crypto_out_total",   # 出金總量：可能直接關聯到標籤
    "iforest_score",      # Isolation Forest 分數：若是用全量資料算的就是 leakage
    "anomaly",            # 異常分數：同上
]

# ----------------------------------------------------------
# 人口學 / 靜態屬性欄位
# 這些欄位本身不是 leakage，但在 safe mode 中移除，
# 是為了測試「純行為特徵」的預測能力，
# 也避免模型學到人口學偏見。
# ----------------------------------------------------------
DEMOGRAPHIC_COLS_CANDIDATE = [
    "sex", "age", "career", "income_source", "user_source", "birthday",
]

# ----------------------------------------------------------
# 優先用來做 time-based split 的欄位（按優先順序排列）
# time-based split 的邏輯：用早期資料訓練，晚期資料驗證，
# 模擬真實部署時「用歷史預測未來」的場景，比 random split 更可信。
# ----------------------------------------------------------
PREFERRED_TIME_COLS = [
    "overall_first_time",
    "confirmed_at",
    "twd_first_time",
    "crypto_first_time",
    "trade_first_time",
    "swap_first_time",
]

# 原始時間欄位：不直接進模型（datetime 格式無法被樹模型處理）
# 若需要時間特徵，應在特徵工程階段轉成數值（如距今天數、時間差秒數等）
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
#    XGBoost 預設用 0.5 切分，但對不平衡資料通常不是最佳點。
#    這裡掃描整個 [th_min, th_max] 區間，找讓 F1 最高的切點。
# =========================================================
def find_best_threshold(y_true, y_prob, th_min=0.01, th_max=0.95, step=0.005):
    """
    掃描 threshold，找出 F1 最佳的切點。

    為什麼不用 precision_recall_curve 直接算？
    因為我們想同時輸出每個 threshold 的詳細數據（存成 CSV），
    方便事後分析 precision/recall 的 trade-off。

    回傳：
        best_th       : 最佳 threshold
        best_f1       : 該 threshold 下的 F1
        threshold_df  : 每個 threshold 的 precision / recall / f1 明細
    """
    thresholds = np.arange(th_min, th_max, step)
    rows = []
    best_f1, best_th = 0.0, 0.5

    for th in thresholds:
        pred      = (y_prob >= th).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall    = recall_score(y_true, pred, zero_division=0)
        f1        = f1_score(y_true, pred, zero_division=0)
        rows.append({"threshold": th, "precision": precision, "recall": recall, "f1": f1})

        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)

    threshold_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    return best_th, best_f1, threshold_df


# =========================================================
# 3. Optuna 超參數調優
#    直接以「驗證集 F1」為優化目標，比用 AUC 更直接。
#    若未安裝 optuna，自動退回手動參數。
# =========================================================
def tune_xgb_with_optuna(X_train, y_train, X_valid, y_valid, scale_pos_weight, n_trials=50):
    """
    用 Optuna 搜尋 XGBoost 超參數，優化目標為驗證集 F1。

    搜尋空間說明：
    - max_depth        : 樹的深度，越深越容易 overfit
    - learning_rate    : 學習率，配合 early stopping 通常設小一點
    - subsample        : 每棵樹用多少比例的樣本，降低 overfit
    - colsample_bytree : 每棵樹用多少比例的特徵，降低 overfit
    - min_child_weight : 葉節點最小樣本權重，越大越保守
    - gamma            : 分裂所需最小 loss 減少量，越大越保守
    - reg_alpha        : L1 正則化，讓特徵權重稀疏
    - reg_lambda       : L2 正則化，讓特徵權重平滑
    - scale_pos_weight : 正負樣本權重比，處理不平衡問題的關鍵參數，
                         這裡讓 Optuna 在 [基礎值*0.5, 基礎值*2] 範圍內微調

    回傳：best_params dict
    """
    if not HAS_OPTUNA:
        print("[INFO] Optuna 未安裝，使用手動參數。可安裝: pip install optuna")
        return _default_xgb_params(scale_pos_weight)

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 9),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 50),
            "gamma"            : trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # scale_pos_weight 在基礎值附近微調，讓模型自己找最佳不平衡補償
            "scale_pos_weight" : trial.suggest_float(
                "scale_pos_weight",
                max(1.0, scale_pos_weight * 0.5),
                scale_pos_weight * 2.0
            ),
            "eval_metric"          : "aucpr",  # PR-AUC 對不平衡資料的 early stopping 更敏感
            "early_stopping_rounds": 50,
            "random_state"         : 42,
            "verbosity"            : 0,
            "tree_method"          : "hist",
            "n_jobs"               : -1,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        prob = m.predict_proba(X_valid)[:, 1]
        # 直接優化 F1，而不是 AUC，更符合最終評估目標
        _, best_f1, _ = find_best_threshold(y_valid, prob)
        return best_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    # 補上 Optuna 不搜尋的固定參數
    best_params.update({
        "eval_metric"          : "aucpr",
        "early_stopping_rounds": 100,
        "random_state"         : 42,
        "verbosity"            : 0,
        "tree_method"          : "hist",
        "n_jobs"               : -1,
    })
    print(f"[Optuna] best_f1={study.best_value:.4f}")
    print(f"[Optuna] best_params={best_params}")
    return best_params


def _default_xgb_params(scale_pos_weight):
    """
    Optuna 不可用時的手動參數。
    這組參數是根據常見不平衡分類任務的經驗值設定的：
    - learning_rate=0.02 配合 early_stopping 讓模型慢慢收斂
    - max_depth=6 是 XGBoost 的常用預設值
    - min_child_weight=5 比預設值(1)更保守，避免在少數類別上 overfit
    - gamma=0.1 要求分裂有一定的 loss 改善才執行
    - reg_alpha/lambda 加入輕微正則化
    """
    return {
        "n_estimators"         : 500,
        "learning_rate"        : 0.02,
        "max_depth"            : 6,
        "subsample"            : 0.8,
        "colsample_bytree"     : 0.8,
        "min_child_weight"     : 5,
        "gamma"                : 0.1,
        "reg_alpha"            : 0.1,
        "reg_lambda"           : 1.0,
        "scale_pos_weight"     : scale_pos_weight,
        "eval_metric"          : "aucpr",
        "early_stopping_rounds": 100,
        "random_state"         : 42,
        "verbosity"            : 0,
        "tree_method"          : "hist",
        "n_jobs"               : -1,
    }


# =========================================================
# 4. 畫圖工具函式
# =========================================================
def plot_top20_feature_importance(model, X, title="Top 20 Feature Importance"):
    """
    繪製前 20 名特徵重要度（XGBoost 預設用 gain，即分裂帶來的 loss 減少量）。
    注意：feature importance 只反映訓練集的分裂情況，
    若想了解對預測的實際貢獻，建議搭配 SHAP 一起看。
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("[WARN] model has no feature_importances_. Skip.")
        return

    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()


def plot_threshold_curve(threshold_df, best_th, title="Precision / Recall / F1 vs Threshold"):
    """
    繪製 threshold 與 precision / recall / F1 的關係圖。
    這張圖幫助你理解：
    - 降低 threshold → recall 上升（抓到更多黑名單），但 precision 下降（誤判增加）
    - 提高 threshold → precision 上升，但 recall 下降（漏掉更多黑名單）
    - 紅線標示目前選定的最佳 F1 切點
    """
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
    """
    繪製 SHAP summary plot（蜂群圖）。
    每個點代表一筆樣本的某個特徵的 SHAP 值：
    - 橫軸：SHAP 值（正值 → 推高預測機率，負值 → 拉低預測機率）
    - 顏色：特徵值大小（紅=高，藍=低）
    這張圖能回答「哪些特徵對模型預測影響最大，方向是什麼」。
    """
    if not HAS_SHAP:
        print("[WARN] shap 未安裝，跳過 SHAP。可安裝: pip install shap")
        return
    if len(X_valid) == 0:
        print("[WARN] X_valid 為空，跳過 SHAP。")
        return

    X_sample  = X_valid.iloc[:sample_size, :]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 二元分類時 shap_values 可能是 list，取 class=1（正類）的 SHAP 值
    shap_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_to_plot, X_sample, plot_type="dot")


# =========================================================
# 5. 欄位整理工具函式
# =========================================================
def cleanup_xy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理 DataFrame merge 後常見的 _x / _y 重複欄位。
    例如：若 df 同時有 'age' 和 'age_x'，則刪除 'age_x'（保留主體欄位）。
    """
    df   = df.copy()
    cols = df.columns.tolist()
    to_drop = [
        c for c in cols
        if (c.endswith("_x") or c.endswith("_y")) and c[:-2] in cols
    ]
    return df.drop(columns=to_drop, errors="ignore")


def parse_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    嘗試把欄位名看起來像時間的欄位轉成 datetime 型別。
    目的：讓後續 choose_split_time_col() 能正確識別時間欄位做 time-based split。
    只有當超過 50% 的值能成功解析時才轉換，避免誤轉純數字欄位。
    """
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
    """回傳實際存在於 df 中的人口學欄位"""
    return [c for c in DEMOGRAPHIC_COLS_CANDIDATE if c in df.columns]


def get_high_leakage_risk_cols(df: pd.DataFrame):
    """
    根據欄位名稱關鍵字找出高風險可疑欄位。
    用關鍵字比對而非完整欄位名，是因為特徵工程後欄位名可能有前後綴。
    """
    return [
        c for c in df.columns
        if any(k in c.lower() for k in HIGH_LEAKAGE_RISK_KEYWORDS)
    ]


def get_datetime_cols(df: pd.DataFrame):
    """回傳 df 中所有 datetime 型別的欄位名稱"""
    return df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()


def choose_split_time_col(df: pd.DataFrame):
    """
    從 PREFERRED_TIME_COLS 中挑第一個「有效」的時間欄位，
    用來做 time-based split。
    「有效」定義：欄位存在，且超過 50% 的值能被解析為 datetime。
    """
    for c in PREFERRED_TIME_COLS:
        if c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    return c
            except Exception:
                continue
    return None  # 找不到就回傳 None，讓 split_data() 退回 random split


# =========================================================
# 6. 資料準備函式
# =========================================================
def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, mode="full"):
    """
    根據 mode 建立 X / y / test_X，並回傳處理過程的 info dict。

    mode 說明：
    - full    : 全部可用數值欄位（分數最高，但可能含 leakage，僅供參考上限）
    - no_leak : 移除高風險可疑欄位（較保守，建議作為主要參考版本）
    - safe    : 移除高風險可疑欄位 + 人口學欄位（最保守，最接近真實部署情境）

    處理流程：
    1. 分離 target / ID
    2. 根據 mode 決定要刪哪些欄位
    3. 移除非數值欄位（樹模型不能直接處理 string/object）
    4. bool → int（XGBoost 對 bool 型別有時會有問題）
    5. inf → nan（避免 XGBoost 在 inf 值上產生奇怪的分裂）
    6. 對齊 train / test 欄位（test 可能缺某些欄位，補 0）
    7. 缺值填補（用 0，比 -1 更不容易讓模型把「缺值」切成強規則）
    8. 刪除常數欄位（nunique=1 的欄位對模型沒有任何資訊量）
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    y      = train_df[TARGET_COL].copy()
    X      = train_df.drop(columns=[TARGET_COL]).copy()
    test_X = test_df.copy()

    demographic_cols       = get_demographic_cols(train_df)
    high_leakage_risk_cols = get_high_leakage_risk_cols(train_df)
    datetime_cols          = get_datetime_cols(train_df)

    # 基礎必刪欄位：ID + 原始時間欄位 + datetime 型別欄位
    drop_cols = [ID_COL] + TIME_RELATED_RAW_COLS + datetime_cols

    # 根據 mode 額外刪欄
    if mode == "no_leak":
        drop_cols += high_leakage_risk_cols
    elif mode == "safe":
        drop_cols += high_leakage_risk_cols + demographic_cols
    elif mode != "full":
        raise ValueError(f"未知 mode: {mode}，請使用 'full' / 'no_leak' / 'safe'")

    X      = X.drop(columns=drop_cols, errors="ignore")
    test_X = test_X.drop(columns=drop_cols, errors="ignore")

    # 移除非數值欄位
    non_numeric_cols = X.select_dtypes(exclude=["int", "float", "bool"]).columns.tolist()
    if non_numeric_cols:
        X      = X.drop(columns=non_numeric_cols, errors="ignore")
        test_X = test_X.drop(columns=non_numeric_cols, errors="ignore")

    # bool → int（True/False → 1/0）
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    for col in test_X.select_dtypes(include="bool").columns:
        test_X[col] = test_X[col].astype(int)

    # inf → nan，避免 XGBoost 在無窮大值上產生不合理的分裂
    X      = X.replace([np.inf, -np.inf], np.nan)
    test_X = test_X.replace([np.inf, -np.inf], np.nan)

    # 對齊 train / test 欄位
    # test 缺少的欄位補 0（代表該行為未發生）
    # test 多出的欄位直接刪除（train 沒見過，模型不認識）
    missing_in_test = sorted(set(X.columns) - set(test_X.columns))
    extra_in_test   = sorted(set(test_X.columns) - set(X.columns))
    for col in missing_in_test:
        test_X[col] = 0
    test_X = test_X.drop(columns=extra_in_test, errors="ignore")
    test_X = test_X[X.columns]  # 確保欄位順序與 train 一致

    # 缺值填補：用 0 而非 -1
    # 用 -1 的問題：樹模型可能把「缺值(-1)」切成一個強規則，
    # 但 -1 在原始特徵空間中可能是個有意義的值（如負數金額）
    X      = X.fillna(0)
    test_X = test_X.fillna(0)

    # 刪除常數欄位（nunique <= 1 代表所有值都一樣，對模型沒有資訊量）
    constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if constant_cols:
        X      = X.drop(columns=constant_cols, errors="ignore")
        test_X = test_X.drop(columns=constant_cols, errors="ignore")

    info = {
        "mode"                   : mode,
        "demographic_cols"       : demographic_cols,
        "high_leakage_risk_cols" : high_leakage_risk_cols,
        "datetime_cols"          : datetime_cols,
        "non_numeric_cols_removed": non_numeric_cols,
        "constant_cols_removed"  : constant_cols,
        "missing_in_test"        : missing_in_test,
        "extra_in_test"          : extra_in_test,
        "final_feature_count"    : X.shape[1],
    }
    return X, y, test_X, info


# =========================================================
# 7. 切分函式
# =========================================================
def split_data(X, y, raw_train_df, split_time_col=None, test_size=0.2):
    """
    優先使用 time-based split，找不到合適時間欄位才退回 random stratified split。

    time-based split 的邏輯：
    - 按時間排序，前 80% 當訓練集，後 20% 當驗證集
    - 這樣驗證集的時間點都在訓練集之後，模擬真實「用歷史預測未來」的場景
    - 比 random split 更能反映模型在實際部署時的表現

    random stratified split 的問題：
    - 驗證集可能包含比訓練集更早的資料，造成「未來資訊洩漏到訓練集」的假象
    - 但若資料沒有時序性，random split 仍是合理選擇

    回傳：X_train, X_valid, y_train, y_valid, split_method（字串，記錄用了哪種切法）
    """
    if split_time_col and split_time_col in raw_train_df.columns:
        time_series  = pd.to_datetime(raw_train_df[split_time_col], errors="coerce")
        usable_idx   = raw_train_df.loc[time_series.notna()].sort_values(split_time_col).index.tolist()

        # 至少要有 100 筆才值得做 time-based split，否則驗證集太小
        if len(usable_idx) >= 100:
            split_point = int(len(usable_idx) * (1 - test_size))
            train_idx, valid_idx = usable_idx[:split_point], usable_idx[split_point:]
            return (
                X.loc[train_idx], X.loc[valid_idx],
                y.loc[train_idx], y.loc[valid_idx],
                f"time_based({split_time_col})"
            )

    # fallback：random stratified split（保持正負樣本比例）
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_valid, y_train, y_valid, "random_stratified"


# =========================================================
# 8. 單一實驗函式
# =========================================================
def run_experiment(train_df, test_df, mode="full", out_dir="output", use_optuna=True):
    """
    執行單一 mode 的完整實驗流程：
    資料準備 → 切分 → 調參（可選）→ 訓練 → 評估 → 輸出

    參數：
        use_optuna : True 時使用 Optuna 調參（需安裝 optuna），False 使用手動參數
    """
    print("=" * 100)
    print(f"[INFO] Running mode = {mode}  |  use_optuna = {use_optuna}")

    # --- 資料準備 ---
    X, y, test_X, prep_info = prepare_xy(train_df, test_df, mode=mode)

    # --- 切分資料 ---
    split_time_col = choose_split_time_col(train_df)
    X_train, X_valid, y_train, y_valid, split_method = split_data(
        X, y, train_df, split_time_col=split_time_col, test_size=0.2
    )

    # --- 計算正負樣本比例 ---
    # scale_pos_weight = 負樣本數 / 正樣本數
    # 告訴模型正樣本（黑名單）比較稀少，要給予更高的懲罰權重
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print(f"[INFO] split_method    : {split_method}")
    print(f"[INFO] scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"[INFO] train shape     : {X_train.shape}, valid shape: {X_valid.shape}")
    print(f"[INFO] train pos rate  : {y_train.mean():.4f}, valid pos rate: {y_valid.mean():.4f}")

    # --- 超參數調優或使用手動參數 ---
    best_params = tune_xgb_with_optuna(
        X_train, y_train, X_valid, y_valid,
        scale_pos_weight=scale_pos_weight,
        n_trials=50
    ) if use_optuna else _default_xgb_params(scale_pos_weight)

    # --- 訓練最終模型 ---
    # 用調好的參數重新訓練（Optuna 內部訓練的模型只用來搜尋參數，不保留）
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    # --- 驗證集評估 ---
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
        # PR-AUC（Average Precision）：在不平衡資料上比 ROC-AUC 更有鑑別力，
        # 因為 ROC-AUC 在負樣本很多時容易虛高
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

    # --- 輸出 CSV ---
    mode_out_dir = os.path.join(out_dir, mode)
    os.makedirs(mode_out_dir, exist_ok=True)

    # 特徵重要性（按 importance 降序排列，rank=1 最重要）
    feature_importance_df = pd.DataFrame({
        "feature"   : X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    feature_importance_df["rank"] = range(1, len(feature_importance_df) + 1)
    feature_importance_df.to_csv(os.path.join(mode_out_dir, "feature_importance.csv"), index=False)

    # threshold 分析（每個 threshold 的 precision/recall/f1，方便事後調整）
    threshold_df.to_csv(os.path.join(mode_out_dir, "threshold_analysis.csv"), index=False)

    # 評估指標
    pd.DataFrame([metrics]).to_csv(os.path.join(mode_out_dir, "metrics.csv"), index=False)

    # 模型參數（方便復現）
    pd.DataFrame({
        "parameter": list(model.get_params().keys()),
        "value"    : list(model.get_params().values())
    }).to_csv(os.path.join(mode_out_dir, "best_params.csv"), index=False)

    # 驗證集預測明細（方便分析哪些樣本被誤判）
    pd.DataFrame({
        "user_id"   : train_df.loc[X_valid.index, ID_COL].values,
        "true_label": y_valid.values,
        "pred_prob" : valid_prob,
        "pred_label": valid_pred
    }).to_csv(os.path.join(mode_out_dir, "valid_detail.csv"), index=False)

    # --- 畫圖 ---
    plot_top20_feature_importance(model, X, title=f"XGBoost - Top 20 Feature Importance ({mode})")
    plot_threshold_curve(threshold_df, best_th, title=f"Threshold Curve ({mode})")
    plot_shap_summary(model, X_valid, sample_size=500)

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
    pd.DataFrame({
        "user_id": test_df[ID_COL],
        "status" : test_pred,
    }).to_csv(os.path.join(mode_out_dir, "submission.csv"), index=False)

    # 測試集分數分布圖（觀察模型是否有足夠的分辨力）
    plt.figure(figsize=(8, 5))
    sns.histplot(test_prob, bins=50, kde=True)
    plt.axvline(best_th, color="red", linestyle="--", label=f"Threshold: {best_th:.3f}")
    plt.title(f"Distribution of Risk Scores ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[INFO] mode={mode} done. submission saved to {mode_out_dir}/submission.csv")

    return {
        "mode"                : mode,
        "model"               : model,
        "metrics"             : metrics,
        "prep_info"           : prep_info,
        "feature_importance_df": feature_importance_df,
        "threshold_df"        : threshold_df,
    }


# =========================================================
# 9. 主程式
# =========================================================
def main(
    train_path="train_feature.csv",
    test_path="test_feature.csv",
    out_dir="output_xgb",
    use_optuna=True,   # 設為 False 可跳過 Optuna，直接用手動參數（速度快很多）
):
    # --- 讀資料 ---
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # 清理 merge 產生的 _x/_y 重複欄位，再嘗試解析時間欄位
    train_df = cleanup_xy_columns(train_df)
    test_df  = cleanup_xy_columns(test_df)
    train_df = parse_time_columns(train_df)
    test_df  = parse_time_columns(test_df)

    print(f"[INFO] train_df shape: {train_df.shape}")
    print(f"[INFO] test_df shape : {test_df.shape}")
    print(f"[INFO] 正樣本比例    : {train_df[TARGET_COL].mean():.4f}")

    # --- 跑三種 mode 做 ablation ---
    # 建議先看 full vs no_leak 的分數差距：
    # - 差距大 → 有明顯 leakage，no_leak 的分數更可信
    # - 差距小 → leakage 影響不大，full 的特徵可以放心用
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

    # --- 匯總比較表 ---
    compare_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    compare_df.to_csv(os.path.join(out_dir, "compare_modes.csv"), index=False)

    print("\n" + "=" * 100)
    print("=== Compare Modes ===")
    print(compare_df.to_string(index=False))

    # 建議：
    # - 若 full >> safe，代表有 leakage，提交時用 safe 版本
    # - 若 full ≈ safe，代表特徵品質好，可以用 full 版本
    best_mode = compare_df.iloc[0]["mode"]
    print(f"\n[RECOMMEND] F1 最高的版本是 '{best_mode}'。")
    print("[RECOMMEND] 若 full 與 safe 分數差距 > 0.05，建議優先提交 safe 版本，避免 leakage 造成線上線下分數落差。")


if __name__ == "__main__":
    main()
