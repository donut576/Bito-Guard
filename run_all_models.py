"""
run_all_models.py

一鍵跑完三個模型（XGBoost、LightGBM、Random Forest），
並輸出統一的 dashboard 結果到 output_results/。

使用方式：
  python run_all_models.py
  python run_all_models.py --no-optuna          # 跳過 Optuna 調參，用預設參數（快）
  python run_all_models.py --mode safe          # 只跑 safe mode
  python run_all_models.py --s3-bucket my-bucket  # 訓練完自動上傳結果到 S3
  python run_all_models.py --s3-download        # 先從 S3 下載 feature CSV 再訓練

AWS 整合說明：
  - 設定環境變數 AML_S3_BUCKET 或傳入 --s3-bucket 參數
  - feature CSV 預期放在 s3://<bucket>/data/train_feature.csv 等路徑
  - 訓練結果會上傳到 s3://<bucket>/output/<model>/<mode>/
  - summary.csv 上傳到 s3://<bucket>/output/summary.csv
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

# ── 確認 CSV 存在 ──────────────────────────────────────────
TRAIN_PATH = "train_feature.csv"
TEST_PATH  = "test_feature.csv"


def _s3_client(region: str = "ap-northeast-1"):
    import boto3
    return boto3.client("s3", region_name=region)


def download_features_from_s3(bucket: str, region: str = "ap-northeast-1") -> None:
    """從 S3 下載 feature CSV 到本地（若本地已存在則跳過）。"""
    s3 = _s3_client(region)
    for fname in [TRAIN_PATH, TEST_PATH]:
        if os.path.exists(fname):
            print(f"[S3] {fname} 已存在，跳過下載")
            continue
        s3_key = f"data/{fname}"
        print(f"[S3] 下載 s3://{bucket}/{s3_key} → {fname}")
        s3.download_file(bucket, s3_key, fname)
        print(f"[S3] ✓ {fname}")


def upload_results_to_s3(out_base: str, bucket: str, region: str = "ap-northeast-1") -> None:
    """將 output_results/ 下所有檔案上傳到 s3://<bucket>/output/。"""
    s3 = _s3_client(region)
    root = pathlib.Path(out_base)
    uploaded = 0
    for local_file in root.rglob("*"):
        if not local_file.is_file():
            continue
        s3_key = "output/" + str(local_file.relative_to(root)).replace("\\", "/")
        s3.upload_file(str(local_file), bucket, s3_key)
        uploaded += 1
    print(f"[S3] ✓ 上傳 {uploaded} 個檔案 → s3://{bucket}/output/")


def check_csvs():
    missing = [p for p in [TRAIN_PATH, TEST_PATH] if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] 找不到以下檔案：{missing}")
        print("[ERROR] 請先執行：python feature_engineering.py")
        print("[ERROR] 或加上 --s3-download 從 S3 下載")
        sys.exit(1)

# ── SHAP 計算並輸出 JSON ───────────────────────────────────
def compute_shap_json(model, X_valid, feature_cols, out_path, sample_size=300):
    try:
        import shap
        X_sample = X_valid.iloc[:sample_size]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)
        rows = []
        for i, feat in enumerate(feature_cols):
            rows.append({
                "feature": feat,
                "shap_value": round(float(mean_signed[i]), 4),
                "abs_shap": round(float(mean_abs[i]), 4),
                "direction": "positive" if mean_signed[i] >= 0 else "negative",
            })
        rows.sort(key=lambda x: x["abs_shap"], reverse=True)
        with open(out_path, "w") as f:
            json.dump(rows[:20], f, indent=2)
        print(f"  [SHAP] saved → {out_path}")
    except Exception as e:
        print(f"  [SHAP] skipped: {e}")


# ── 通用 prepare_xy (shared across models) ────────────────
HIGH_LEAKAGE_RISK_KEYWORDS = [
    "last_time", "active_span", "overall", "network", "wallet",
    "shared_ip", "total_unique_ips", "total_ip_usage_count",
    "swap_total", "swap_max", "swap_avg", "crypto_out_total",
    "iforest_score", "anomaly",
]
DEMOGRAPHIC_COLS = ["sex", "age", "career", "income_source", "user_source", "birthday"]
TIME_RELATED_RAW_COLS = [
    "confirmed_at", "level1_finished_at", "level2_finished_at",
    "twd_first_time", "twd_last_time", "crypto_first_time", "crypto_last_time",
    "trade_first_time", "trade_last_time", "swap_first_time", "swap_last_time",
    "overall_first_time", "overall_last_time",
]
TARGET_COL = "status"
ID_COL = "user_id"


def prepare_xy(train_df, test_df, mode="safe"):
    y = train_df[TARGET_COL].copy()
    X = train_df.drop(columns=[TARGET_COL]).copy()
    test_X = test_df.copy()

    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    leakage_cols = [c for c in X.columns if any(k in c.lower() for k in HIGH_LEAKAGE_RISK_KEYWORDS)]
    demo_cols = [c for c in DEMOGRAPHIC_COLS if c in X.columns]

    drop = [ID_COL] + TIME_RELATED_RAW_COLS + datetime_cols
    if mode in ("no_leak", "safe"):
        drop += leakage_cols
    if mode == "safe":
        drop += demo_cols

    X = X.drop(columns=drop, errors="ignore")
    test_X = test_X.drop(columns=drop, errors="ignore")

    non_num = X.select_dtypes(exclude=["int", "float", "bool"]).columns.tolist()
    X = X.drop(columns=non_num, errors="ignore")
    test_X = test_X.drop(columns=non_num, errors="ignore")

    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    for col in test_X.select_dtypes(include="bool").columns:
        test_X[col] = test_X[col].astype(int)

    for col in set(X.columns) - set(test_X.columns):
        test_X[col] = 0
    test_X = test_X[[c for c in X.columns if c in test_X.columns]]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    test_X = test_X.replace([np.inf, -np.inf], np.nan).fillna(0)

    const = X.columns[X.nunique() <= 1].tolist()
    X = X.drop(columns=const, errors="ignore")
    test_X = test_X.drop(columns=const, errors="ignore")

    return X, y, test_X


def split_data(X, y, train_df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    preferred = ["overall_first_time", "confirmed_at", "twd_first_time", "crypto_first_time"]
    for col in preferred:
        if col in train_df.columns:
            ts = pd.to_datetime(train_df[col], errors="coerce")
            idx = train_df.loc[ts.notna()].sort_values(col).index.tolist()
            if len(idx) >= 100:
                sp = int(len(idx) * (1 - test_size))
                return X.loc[idx[:sp]], X.loc[idx[sp:]], y.loc[idx[:sp]], y.loc[idx[sp:]]
    X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    return X_tr, X_v, y_tr, y_v


def find_best_threshold(y_true, y_prob):
    from sklearn.metrics import f1_score, precision_score, recall_score
    rows, best_f1, best_th = [], 0.0, 0.5
    for t in np.arange(0.01, 0.95, 0.005):
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        rows.append({"threshold": round(t, 3), "precision": round(precision_score(y_true, pred, zero_division=0), 4),
                     "recall": round(recall_score(y_true, pred, zero_division=0), 4), "f1": round(f, 4)})
        if f > best_f1:
            best_f1, best_th = f, t
    return best_th, best_f1, pd.DataFrame(rows)


# ── 各模型訓練函式 ─────────────────────────────────────────

def run_xgboost(X_tr, y_tr, X_v, y_v, use_optuna):
    import xgboost as xgb
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    if use_optuna:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            def obj(trial):
                m = xgb.XGBClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 200, 1000),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    scale_pos_weight=spw, tree_method="hist", random_state=42, verbosity=0,
                )
                m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
                _, f1, _ = find_best_threshold(y_v, m.predict_proba(X_v)[:, 1])
                return f1
            study = optuna.create_study(direction="maximize")
            study.optimize(obj, n_trials=30, show_progress_bar=True)
            p = study.best_params
            model = xgb.XGBClassifier(**p, scale_pos_weight=spw, tree_method="hist", random_state=42, verbosity=0)
        except Exception:
            model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                       tree_method="hist", random_state=42, verbosity=0)
    else:
        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                   tree_method="hist", random_state=42, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    return model


def run_lightgbm(X_tr, y_tr, X_v, y_v, use_optuna):
    import lightgbm as lgb
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = {
        "n_estimators": 500, "learning_rate": 0.02, "num_leaves": 63,
        "max_depth": 6, "min_child_samples": 30, "subsample": 0.8,
        "colsample_bytree": 0.8, "scale_pos_weight": spw,
        "objective": "binary", "random_state": 42, "n_jobs": -1, "verbosity": -1,
    }
    if use_optuna:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            def obj(trial):
                p = {**params,
                     "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                     "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                     "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                     "max_depth": trial.suggest_int("max_depth", 3, 10),
                }
                m = lgb.LGBMClassifier(**p)
                m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                _, f1, _ = find_best_threshold(y_v, m.predict_proba(X_v)[:, 1])
                return f1
            study = optuna.create_study(direction="maximize")
            study.optimize(obj, n_trials=30, show_progress_bar=True)
            params.update(study.best_params)
        except Exception:
            pass
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return model


def run_rf(X_tr, y_tr, use_optuna):
    from sklearn.ensemble import RandomForestClassifier
    params = {"n_estimators": 500, "max_depth": 10, "min_samples_split": 10,
              "min_samples_leaf": 5, "max_features": "sqrt",
              "class_weight": "balanced_subsample", "random_state": 42, "n_jobs": -1}
    if use_optuna:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            def obj(trial):
                p = {**params,
                     "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                     "max_depth": trial.suggest_int("max_depth", 3, 15),
                     "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
                }
                m = RandomForestClassifier(**p)
                m.fit(X_tr, y_tr)
                return 0.0  # RF doesn't have eval_set, skip optuna for speed
            # RF optuna is slow, skip
        except Exception:
            pass
    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)
    return model


# ── 主流程 ─────────────────────────────────────────────────

def run_model(name, train_fn, train_df, test_df, mode, use_optuna, out_base):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, average_precision_score

    print(f"\n{'='*60}")
    print(f"  {name.upper()} | mode={mode}")
    print(f"{'='*60}")

    X, y, test_X = prepare_xy(train_df, test_df, mode)
    X_tr, X_v, y_tr, y_v = split_data(X, y, train_df)

    print(f"  train: {X_tr.shape}, valid: {X_v.shape}, features: {X.shape[1]}")
    print(f"  pos_rate train={y_tr.mean():.3f}, valid={y_v.mean():.3f}")

    model = train_fn(X_tr, y_tr, X_v, y_v, use_optuna) if name != "rf" else train_fn(X_tr, y_tr, use_optuna)

    y_prob = model.predict_proba(X_v)[:, 1]
    best_th, _, threshold_df = find_best_threshold(y_v, y_prob)
    y_pred = (y_prob >= best_th).astype(int)

    metrics = {
        "model": name, "mode": mode,
        "f1": round(f1_score(y_v, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_v, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_v, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_v, y_prob), 4),
        "pr_auc": round(average_precision_score(y_v, y_prob), 4),
        "accuracy": round(accuracy_score(y_v, y_pred), 4),
        "threshold": round(best_th, 4),
        "n_features": X.shape[1],
    }
    print(f"  F1={metrics['f1']}  AUC={metrics['auc']}  Precision={metrics['precision']}  Recall={metrics['recall']}")

    out_dir = os.path.join(out_base, name, mode)
    os.makedirs(out_dir, exist_ok=True)

    # feature importance
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp["rank"] = range(1, len(feat_imp) + 1)
    feat_imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    # threshold analysis
    threshold_df.sort_values("threshold").to_csv(os.path.join(out_dir, "threshold_analysis.csv"), index=False)

    # metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # valid detail
    pd.DataFrame({
        "user_id": train_df.loc[X_v.index, ID_COL].values,
        "true_label": y_v.values,
        "pred_prob": y_prob,
        "pred_label": y_pred,
    }).to_csv(os.path.join(out_dir, "valid_detail.csv"), index=False)

    # SHAP JSON
    compute_shap_json(model, X_v, list(X.columns), os.path.join(out_dir, "shap.json"))

    # test predictions
    test_prob = model.predict_proba(test_X)[:, 1]
    test_pred = (test_prob >= best_th).astype(int)
    pd.DataFrame({"user_id": test_df[ID_COL], "pred_prob": test_prob, "status": test_pred}
                 ).to_csv(os.path.join(out_dir, "test_scores.csv"), index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna, use default params")
    parser.add_argument("--mode", default="all", choices=["all", "safe", "no_leak", "full"])
    parser.add_argument("--s3-bucket", default=os.environ.get("AML_S3_BUCKET", ""), help="S3 bucket 名稱")
    parser.add_argument("--s3-download", action="store_true", help="從 S3 下載 feature CSV 再訓練")
    parser.add_argument("--s3-upload", action="store_true", help="訓練完後上傳結果到 S3")
    parser.add_argument("--aws-region", default=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-1"))
    args = parser.parse_args()

    s3_bucket = args.s3_bucket

    # ── 從 S3 下載 feature CSV（若指定）──────────────────────
    if args.s3_download:
        if not s3_bucket:
            print("[ERROR] --s3-download 需要指定 --s3-bucket 或設定 AML_S3_BUCKET 環境變數")
            sys.exit(1)
        download_features_from_s3(s3_bucket, args.aws_region)

    check_csvs()
    use_optuna = not args.no_optuna
    modes = ["full", "no_leak", "safe"] if args.mode == "all" else [args.mode]
    out_base = "output_results"

    print(f"\n[INFO] Loading CSVs...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # parse time cols
    for df in [train_df, test_df]:
        for col in df.columns:
            if any(k in col.lower() for k in ["time", "date", "_at"]):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() > 0.5:
                        df[col] = parsed
                except Exception:
                    pass

    print(f"[INFO] train={train_df.shape}, test={test_df.shape}, pos_rate={train_df[TARGET_COL].mean():.4f}")
    print(f"[INFO] use_optuna={use_optuna}, modes={modes}")

    all_metrics = []
    model_fns = [
        ("xgb", run_xgboost),
        ("lgb", run_lightgbm),
        ("rf",  run_rf),
    ]

    for mode in modes:
        for name, fn in model_fns:
            try:
                m = run_model(name, fn, train_df, test_df, mode, use_optuna, out_base)
                all_metrics.append(m)
            except Exception as e:
                print(f"  [ERROR] {name}/{mode}: {e}")

    # summary
    if all_metrics:
        summary = pd.DataFrame(all_metrics).sort_values(["mode", "f1"], ascending=[True, False])
        summary.to_csv(os.path.join(out_base, "summary.csv"), index=False)
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(summary.to_string(index=False))
        print(f"\n[DONE] Results saved to {out_base}/")

    # ── 上傳結果到 S3（若指定）────────────────────────────────
    do_upload = args.s3_upload or bool(s3_bucket and not args.s3_download)
    if do_upload and s3_bucket:
        try:
            upload_results_to_s3(out_base, s3_bucket, args.aws_region)
        except Exception as e:
            print(f"[WARN] S3 上傳失敗（不影響本地結果）: {e}")
    elif do_upload and not s3_bucket:
        print("[WARN] 未指定 S3 bucket，跳過上傳。請設定 AML_S3_BUCKET 或傳入 --s3-bucket")


if __name__ == "__main__":
    main()
