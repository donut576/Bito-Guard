# -*- coding: utf-8 -*-
"""
model_stack.py

改進項目：
1. 讀取單模 Optuna best_params.csv，直接用最強參數當 base model
2. 加入 CatBoost 增加模型多樣性
3. TTA（Test-Time Augmentation）降低 test 預測 variance
4. build_meta_features 加入 rank features + 原始 top-N 特徵
5. meta-model 用 Optuna 調參
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, average_precision_score,
)
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# =========================================================
# 1. 全域設定
# =========================================================
TARGET_COL   = "status"
ID_COL       = "user_id"
RANDOM_STATE = 42
N_SPLITS     = 5
TTA_ROUNDS   = 3       # Test-Time Augmentation 次數
TTA_NOISE    = 0.005   # TTA 加的 noise 標準差（很小，只是擾動）

HIGH_LEAKAGE_RISK_KEYWORDS = [
    "last_time", "active_span", "overall", "network", "wallet",
    "shared_ip", "total_unique_ips", "total_ip_usage_count",
    "swap_total", "swap_max", "swap_avg", "crypto_out_total",
    "iforest_score", "anomaly",
]

DEMOGRAPHIC_COLS_CANDIDATE = [
    "sex", "age", "career", "income_source", "user_source", "birthday",
]

PREFERRED_TIME_COLS = [
    "overall_first_time", "confirmed_at", "twd_first_time",
    "crypto_first_time", "trade_first_time", "swap_first_time",
]

TIME_RELATED_RAW_COLS = [
    "confirmed_at", "level1_finished_at", "level2_finished_at",
    "twd_first_time", "twd_last_time", "crypto_first_time", "crypto_last_time",
    "trade_first_time", "trade_last_time", "swap_first_time", "swap_last_time",
    "overall_first_time", "overall_last_time",
]


# =========================================================
# 2. 基本工具
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def cleanup_xy_columns(df):
    df = df.copy()
    cols = df.columns.tolist()
    to_drop = [c for c in cols if (c.endswith("_x") or c.endswith("_y")) and c[:-2] in cols]
    return df.drop(columns=to_drop, errors="ignore")

def parse_time_columns(df):
    df = df.copy()
    time_like_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "_at"])]
    for col in time_like_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.5:
                df[col] = parsed
        except Exception:
            pass
    return df

def get_demographic_cols(df):
    return [c for c in DEMOGRAPHIC_COLS_CANDIDATE if c in df.columns]

def get_high_leakage_risk_cols(df):
    return [c for c in df.columns if any(k in c.lower() for k in HIGH_LEAKAGE_RISK_KEYWORDS)]

def get_datetime_cols(df):
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

def find_best_threshold(y_true, y_prob, th_min=0.01, th_max=0.95, step=0.005):
    thresholds = np.arange(th_min, th_max, step)
    rows, best_f1, best_th = [], 0.0, 0.5
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        rows.append({"threshold": th, "precision": p, "recall": r, "f1": f})
        if f > best_f1:
            best_f1, best_th = float(f), float(th)
    return best_th, best_f1, pd.DataFrame(rows).sort_values("f1", ascending=False)

def evaluate_result(y_true, y_prob, threshold, model_name="model"):
    pred = (y_prob >= threshold).astype(int)
    metrics = {
        "model": model_name, "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }
    return metrics, confusion_matrix(y_true, pred), classification_report(y_true, pred, digits=4, zero_division=0), pred


# =========================================================
# 3. 讀取單模 Optuna best params
# =========================================================
def load_best_params_from_csv(csv_path):
    """
    讀取單模輸出的 best_params.csv，自動轉型。
    找不到檔案時回傳 None，讓 build_base_models 退回手動參數。
    """
    if not os.path.exists(csv_path):
        print(f"[INFO] best_params not found: {csv_path}, using default params")
        return None
    df = pd.read_csv(csv_path)
    params = {}
    for _, row in df.iterrows():
        k, v = str(row["parameter"]), str(row["value"])
        if v in ("True",):
            params[k] = True
        elif v in ("False",):
            params[k] = False
        elif v in ("None", "nan", "NaN", ""):
            # missing=nan 是 XGBoost 預設值，直接跳過讓它用預設
            continue
        else:
            try:
                params[k] = int(v) if ("." not in v and "e" not in v.lower()) else float(v)
            except ValueError:
                params[k] = v
    print(f"[INFO] Loaded best_params from {csv_path}")
    return params


# =========================================================
# 4. 資料準備
# =========================================================
def prepare_xy(train_df, test_df, mode="full"):
    train_df, test_df = train_df.copy(), test_df.copy()
    y      = train_df[TARGET_COL].copy()
    X      = train_df.drop(columns=[TARGET_COL]).copy()
    test_X = test_df.copy()

    demographic_cols       = get_demographic_cols(train_df)
    high_leakage_risk_cols = get_high_leakage_risk_cols(train_df)
    datetime_cols          = get_datetime_cols(train_df)
    drop_cols              = [ID_COL] + TIME_RELATED_RAW_COLS + datetime_cols

    if mode == "no_leak":
        drop_cols += high_leakage_risk_cols
    elif mode == "safe":
        drop_cols += high_leakage_risk_cols + demographic_cols
    elif mode != "full":
        raise ValueError(f"未知 mode: {mode}")

    X      = X.drop(columns=drop_cols, errors="ignore")
    test_X = test_X.drop(columns=drop_cols, errors="ignore")

    non_num = X.select_dtypes(exclude=["int", "float", "bool"]).columns.tolist()
    X      = X.drop(columns=non_num, errors="ignore")
    test_X = test_X.drop(columns=non_num, errors="ignore")

    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    for col in test_X.select_dtypes(include="bool").columns:
        test_X[col] = test_X[col].astype(int)

    X      = X.replace([np.inf, -np.inf], np.nan)
    test_X = test_X.replace([np.inf, -np.inf], np.nan)

    for col in sorted(set(X.columns) - set(test_X.columns)):
        test_X[col] = 0
    test_X = test_X.drop(columns=sorted(set(test_X.columns) - set(X.columns)), errors="ignore")
    test_X = test_X[X.columns]

    X      = X.fillna(0)
    test_X = test_X.fillna(0)

    constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
    X      = X.drop(columns=constant_cols, errors="ignore")
    test_X = test_X.drop(columns=constant_cols, errors="ignore")

    return X, y, test_X, {
        "mode": mode, "final_feature_count": X.shape[1],
        "high_leakage_risk_cols": high_leakage_risk_cols,
        "demographic_cols": demographic_cols,
        "constant_cols_removed": constant_cols,
    }


# =========================================================
# 5. 建立 base models（優先讀 Optuna best params）
# =========================================================
def build_base_models(scale_pos_weight,
                      xgb_params_path="output_xgb/full/best_params.csv",
                      lgb_params_path="output_lgb/full/best_params.csv"):
    models = {}

    # --- XGBoost ---
    if HAS_XGB:
        xgb_params = load_best_params_from_csv(xgb_params_path)
        if xgb_params:
            # 強制用當前資料的 scale_pos_weight，不用單模 split 算出來的值
            xgb_params["scale_pos_weight"]     = scale_pos_weight
            xgb_params["eval_metric"]          = "aucpr"
            xgb_params["early_stopping_rounds"]= 100
            xgb_params["random_state"]         = RANDOM_STATE
            xgb_params["verbosity"]            = 0
            xgb_params["tree_method"]          = "hist"
            xgb_params["n_jobs"]               = -1
            models["xgb"] = xgb.XGBClassifier(**xgb_params)
        else:
            models["xgb"] = xgb.XGBClassifier(
                n_estimators=700, learning_rate=0.02, max_depth=6,
                subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                early_stopping_rounds=100, random_state=RANDOM_STATE,
                verbosity=0, tree_method="hist", n_jobs=-1,
            )

    # --- LightGBM ---
    if HAS_LGB:
        lgb_params = load_best_params_from_csv(lgb_params_path)
        if lgb_params:
            # 只借用架構參數，正則化參數重設為保守值避免在 OOF 小訓練集上過擬合
            models["lgb"] = lgb.LGBMClassifier(
                n_estimators      = lgb_params.get("n_estimators", 700),
                learning_rate     = lgb_params.get("learning_rate", 0.02),
                num_leaves        = lgb_params.get("num_leaves", 63),
                max_depth         = lgb_params.get("max_depth", 6),
                min_child_samples = lgb_params.get("min_child_samples", 30),
                subsample         = lgb_params.get("subsample", 0.8),
                colsample_bytree  = lgb_params.get("colsample_bytree", 0.8),
                # 正則化重設，避免單模調出來的極端值在 OOF 上過保守
                reg_alpha         = min(lgb_params.get("reg_alpha", 0.1), 1.0),
                reg_lambda        = lgb_params.get("reg_lambda", 1.0),
                min_split_gain    = min(lgb_params.get("min_split_gain", 0.1), 0.5),
                scale_pos_weight=scale_pos_weight, objective="binary",
                metric="average_precision", random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=-1,
            )

    # --- CatBoost ---
    if HAS_CB:
        models["cb"] = cb.CatBoostClassifier(
            iterations=700, learning_rate=0.02, depth=6,
            l2_leaf_reg=3.0, scale_pos_weight=scale_pos_weight,
            eval_metric="F1", random_state=RANDOM_STATE,
            verbose=0, thread_count=-1,
        )

    return models


# =========================================================
# 6. Meta features（含 rank features + 原始 top-N）
# =========================================================
def build_meta_features(pred_df, X_original=None, top_n_features=20):
    """
    1. base predictions + 衍生統計
    2. rank features（把機率轉成百分位，更穩定）
    3. 原始特徵 top-N（用 LGB 快速選）
    """
    df = pred_df.copy()
    base_pred_cols = [c for c in df.columns if c.endswith("_pred")]
    out = df[base_pred_cols].copy()

    if len(base_pred_cols) >= 2:
        out["pred_mean"] = out[base_pred_cols].mean(axis=1)
        out["pred_std"]  = out[base_pred_cols].std(axis=1).fillna(0)
        out["pred_gap"]  = out[base_pred_cols].max(axis=1) - out[base_pred_cols].min(axis=1)
        out["pred_max"]  = out[base_pred_cols].max(axis=1)
        out["pred_min"]  = out[base_pred_cols].min(axis=1)

        # rank features：把機率轉成 0~1 百分位
        for col in base_pred_cols:
            out[f"{col}_rank"] = out[col].rank(pct=True)

        if len(base_pred_cols) == 2:
            a, b = base_pred_cols[0], base_pred_cols[1]
            out["both_high"] = ((out[a] > 0.3) & (out[b] > 0.3)).astype(int)
            out["both_low"]  = ((out[a] < 0.1) & (out[b] < 0.1)).astype(int)
            out["disagree"]  = (np.abs(out[a] - out[b]) > 0.2).astype(int)

    # 原始特徵 top-N
    if X_original is not None and top_n_features > 0 and HAS_LGB:
        _selector = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        )
        _proxy_y = (out["pred_mean"] > out["pred_mean"].median()).astype(int)
        _selector.fit(X_original.reset_index(drop=True), _proxy_y)
        top_cols = pd.Series(
            _selector.feature_importances_, index=X_original.columns
        ).nlargest(top_n_features).index.tolist()
        out = pd.concat(
            [out.reset_index(drop=True), X_original[top_cols].reset_index(drop=True)],
            axis=1,
        )

    return out


# =========================================================
# 7. Meta-model Optuna 調參
# =========================================================
def tune_meta_model(meta_train_X, y, n_trials=30):
    """用 Optuna 在 OOF 上搜尋 meta LGB 的最佳參數"""
    if not HAS_OPTUNA or not HAS_LGB:
        return _default_meta_params()

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 7, 63),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 50),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight": "balanced",
            "objective": "binary",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, va_idx in skf.split(meta_train_X, y):
            model.fit(meta_train_X.iloc[tr_idx], y.iloc[tr_idx])
            prob = model.predict_proba(meta_train_X.iloc[va_idx])[:, 1]
            _, f1, _ = find_best_threshold(y.iloc[va_idx], prob)
            scores.append(f1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"class_weight": "balanced", "objective": "binary",
                 "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
    print(f"[Meta Optuna] best CV F1: {study.best_value:.4f}, params: {best}")
    return best


def _default_meta_params():
    return {
        "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
        "max_depth": 3, "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "class_weight": "balanced", "objective": "binary",
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1,
    }


# =========================================================
# 8. 主訓練流程
# =========================================================
def run_stacking(train_df, test_df, mode="full", out_dir="output_stacking",
                 use_meta_optuna=True, top_n_features=20):
    ensure_dir(out_dir)
    mode_out_dir = os.path.join(out_dir, mode)
    ensure_dir(mode_out_dir)

    print("=" * 100)
    print(f"[INFO] Running stacking mode={mode}, TTA={TTA_ROUNDS}, top_n_features={top_n_features}")

    X, y, test_X, prep_info = prepare_xy(train_df, test_df, mode=mode)
    print(f"[INFO] feature count: {X.shape[1]}")

    y = y.astype(int)
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"[INFO] scale_pos_weight: {scale_pos_weight:.4f}")

    models     = build_base_models(scale_pos_weight)
    model_names = list(models.keys())
    print(f"[INFO] base models: {model_names}")

    skf       = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds  = {name: np.zeros(len(X))      for name in model_names}
    test_preds = {name: np.zeros(len(test_X)) for name in model_names}
    fold_records = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n[Fold {fold}/{N_SPLITS}]")
        X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

        for name, model in models.items():
            print(f"  -> training {name}")

            if name == "lgb":
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                          eval_metric="average_precision",
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            elif name == "xgb":
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            elif name == "cb":
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                          early_stopping_rounds=100, verbose=False)
            else:
                model.fit(X_train, y_train)

            valid_prob = model.predict_proba(X_valid)[:, 1]
            oof_preds[name][va_idx] = valid_prob

            # TTA：對 test 加微小 noise 預測多次再平均，降低 variance
            test_prob_acc = np.zeros(len(test_X))
            for _ in range(TTA_ROUNDS):
                noise = np.random.normal(0, TTA_NOISE, test_X.shape)
                test_X_noisy = test_X + noise
                test_prob_acc += model.predict_proba(test_X_noisy)[:, 1]
            test_preds[name] += (test_prob_acc / TTA_ROUNDS) / N_SPLITS

            best_th, best_f1, _ = find_best_threshold(y_valid, valid_prob)
            fold_records.append({
                "fold": fold, "model": name,
                "best_threshold": best_th, "best_f1": best_f1,
                "pr_auc": average_precision_score(y_valid, valid_prob),
                "auc": roc_auc_score(y_valid, valid_prob),
            })

    pd.DataFrame(fold_records).to_csv(os.path.join(mode_out_dir, "fold_metrics.csv"), index=False)

    # --- Base model OOF evaluation ---
    base_metrics = []
    oof_base_df  = pd.DataFrame({"user_id": train_df[ID_COL], "status": y})
    test_base_df = pd.DataFrame({"user_id": test_df[ID_COL]})

    for name in model_names:
        oof_base_df[f"{name}_pred"]  = oof_preds[name]
        test_base_df[f"{name}_pred"] = test_preds[name]

        best_th, best_f1, th_df = find_best_threshold(y, oof_preds[name])
        metrics, cm, report, _ = evaluate_result(y, oof_preds[name], best_th, model_name=name)
        metrics["best_f1_from_oof"] = best_f1
        base_metrics.append(metrics)

        print("\n" + "-" * 80)
        print(f"[Base Model: {name}]  OOF F1={best_f1:.4f}  threshold={best_th:.3f}")
        print(cm)
        print(report)
        th_df.to_csv(os.path.join(mode_out_dir, f"{name}_threshold_analysis.csv"), index=False)

    oof_base_df.to_csv(os.path.join(mode_out_dir, "oof_base_predictions.csv"), index=False)
    test_base_df.to_csv(os.path.join(mode_out_dir, "test_base_predictions.csv"), index=False)

    # --- Weighted Blend（搜尋最佳權重）---
    if len(model_names) >= 2:
        best_blend_f1, best_weights = 0.0, None
        names = model_names
        # grid search over equal-step weights for 2~3 models
        if len(names) == 2:
            for w in np.arange(0.1, 1.0, 0.05):
                prob = w * oof_preds[names[0]] + (1 - w) * oof_preds[names[1]]
                _, f1, _ = find_best_threshold(y, prob)
                if f1 > best_blend_f1:
                    best_blend_f1 = f1
                    best_weights  = [w, 1 - w]
        else:
            # 3 models: enumerate combinations summing to 1
            for w0 in np.arange(0.1, 0.8, 0.1):
                for w1 in np.arange(0.1, 0.8, 0.1):
                    w2 = 1.0 - w0 - w1
                    if w2 < 0.05:
                        continue
                    prob = w0*oof_preds[names[0]] + w1*oof_preds[names[1]] + w2*oof_preds[names[2]]
                    _, f1, _ = find_best_threshold(y, prob)
                    if f1 > best_blend_f1:
                        best_blend_f1 = f1
                        best_weights  = [w0, w1, w2]

        print(f"\n[Blend Search] best weights: {dict(zip(names, [f'{w:.2f}' for w in best_weights]))}, OOF F1={best_blend_f1:.4f}")
        blend_oof_prob  = sum(w * oof_preds[n]  for w, n in zip(best_weights, names))
        blend_test_prob = sum(w * test_preds[n] for w, n in zip(best_weights, names))
    else:
        blend_oof_prob  = oof_preds[model_names[0]]
        blend_test_prob = test_preds[model_names[0]]

    blend_best_th, blend_best_f1, blend_th_df = find_best_threshold(y, blend_oof_prob)
    blend_metrics, blend_cm, blend_report, blend_oof_pred = evaluate_result(
        y, blend_oof_prob, blend_best_th, model_name="blend")
    blend_metrics["best_f1_from_oof"] = blend_best_f1

    print("\n" + "=" * 100)
    print(f"[BLEND RESULT]  OOF F1={blend_best_f1:.4f}  threshold={blend_best_th:.3f}")
    print(blend_cm)
    print(blend_report)
    blend_th_df.to_csv(os.path.join(mode_out_dir, "blend_threshold_analysis.csv"), index=False)

    # --- Stacking with meta-model ---
    meta_train_X = build_meta_features(
        oof_base_df[[f"{n}_pred" for n in model_names]],
        X_original=X, top_n_features=top_n_features,
    )
    meta_test_X = build_meta_features(
        test_base_df[[f"{n}_pred" for n in model_names]],
        X_original=test_X, top_n_features=top_n_features,
    )

    meta_params = tune_meta_model(meta_train_X, y, n_trials=30) if use_meta_optuna else _default_meta_params()
    meta_model  = lgb.LGBMClassifier(**meta_params)
    meta_model.fit(meta_train_X, y)

    stack_oof_prob  = meta_model.predict_proba(meta_train_X)[:, 1]
    stack_test_prob = meta_model.predict_proba(meta_test_X)[:, 1]

    stack_best_th, stack_best_f1, stack_th_df = find_best_threshold(y, stack_oof_prob)
    stack_metrics, stack_cm, stack_report, stack_oof_pred = evaluate_result(
        y, stack_oof_prob, stack_best_th, model_name="stacking")
    stack_metrics["best_f1_from_oof"] = stack_best_f1

    print("\n" + "=" * 100)
    print(f"[STACKING RESULT]  OOF F1={stack_best_f1:.4f}  threshold={stack_best_th:.3f}")
    print(stack_cm)
    print(stack_report)

    # --- Save all outputs ---
    metrics_df = pd.DataFrame(base_metrics + [blend_metrics, stack_metrics])
    metrics_df.to_csv(os.path.join(mode_out_dir, "metrics_summary.csv"), index=False)
    stack_th_df.to_csv(os.path.join(mode_out_dir, "stacking_threshold_analysis.csv"), index=False)

    test_pred       = (stack_test_prob >= stack_best_th).astype(int)
    blend_test_pred = (blend_test_prob  >= blend_best_th).astype(int)

    pd.DataFrame({
        "user_id": train_df[ID_COL], "status": y,
        **{f"{n}_oof_pred": oof_preds[n] for n in model_names},
        "blend_oof_prob": blend_oof_prob, "blend_oof_pred": blend_oof_pred,
        "stack_oof_prob": stack_oof_prob, "stack_oof_pred": stack_oof_pred,
    }).to_csv(os.path.join(mode_out_dir, "train_prediction_detail.csv"), index=False)

    pd.DataFrame({
        "user_id": test_df[ID_COL],
        **{f"{n}_test_pred": test_preds[n] for n in model_names},
        "blend_test_prob": blend_test_prob, "blend_test_pred": blend_test_pred,
        "stack_test_prob": stack_test_prob, "stack_test_pred": test_pred,
    }).to_csv(os.path.join(mode_out_dir, "test_prediction_detail.csv"), index=False)

    pd.DataFrame({"user_id": test_df[ID_COL], "status": test_pred}
                 ).to_csv(os.path.join(mode_out_dir, "submission_stacking.csv"), index=False)
    pd.DataFrame({"user_id": test_df[ID_COL], "status": blend_test_pred}
                 ).to_csv(os.path.join(mode_out_dir, "submission_blend.csv"), index=False)

    pd.DataFrame({"user_id": train_df[ID_COL], "status": y}
                 ).to_csv(os.path.join(mode_out_dir, "train_id_status.csv"), index=False)
    pd.DataFrame({"user_id": test_df[ID_COL], "status": test_pred}
                 ).to_csv(os.path.join(mode_out_dir, "test_id_status.csv"), index=False)
    pd.concat([
        pd.DataFrame({"user_id": train_df[ID_COL], "status": y}),
        pd.DataFrame({"user_id": test_df[ID_COL],  "status": test_pred}),
    ], ignore_index=True).to_csv(os.path.join(mode_out_dir, "all_id_status.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(stack_test_prob, bins=50, kde=True)
    plt.axvline(stack_best_th, color="red", linestyle="--", label=f"Threshold: {stack_best_th:.3f}")
    plt.title(f"Stacking Score Distribution ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mode_out_dir, "stacking_score_distribution.png"))
    plt.close()

    return {"mode": mode, "prep_info": prep_info, "metrics_df": metrics_df}


# =========================================================
# 9. 主程式
# =========================================================
def main(
    train_path="train_feature.csv",
    test_path="test_feature.csv",
    out_dir="output_stacking",
    use_meta_optuna=True,
    top_n_features=20,
):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    train_df = cleanup_xy_columns(parse_time_columns(train_df))
    test_df  = cleanup_xy_columns(parse_time_columns(test_df))

    print(f"[INFO] train: {train_df.shape}, test: {test_df.shape}")
    print(f"[INFO] 正樣本比例: {train_df[TARGET_COL].mean():.4f}")

    all_results = []
    for mode in ["full"]:   # 可改成 ["full", "no_leak", "safe"]
        res = run_stacking(train_df, test_df, mode=mode, out_dir=out_dir,
                           use_meta_optuna=use_meta_optuna, top_n_features=top_n_features)
        tmp = res["metrics_df"].copy()
        tmp["mode"] = mode
        all_results.append(tmp)

    compare_df = pd.concat(all_results, ignore_index=True)
    ensure_dir(out_dir)
    compare_df.to_csv(os.path.join(out_dir, "compare_all_models.csv"), index=False)
    print("\n" + "=" * 100)
    print("All done.")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
