import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ===============================
# 1. 讀取三個模型的 validation 結果
# ===============================

xgb = pd.read_csv("output_xgb/full/valid_detail.csv")
lgb = pd.read_csv("output_lgb/full/valid_detail.csv")
rf  = pd.read_csv("output_rf/valid_detail.csv")

# 先確認 user_id 對齊
df = xgb[["user_id", "true_label"]].copy()
df["xgb_prob"] = xgb["pred_prob"]
df["lgb_prob"] = lgb["pred_prob"]
df["rf_prob"]  = rf["pred_prob"]

# ===============================
# 2. 各模型 threshold
#    先用 0.5，之後可以改成各模型最佳 threshold
# ===============================

xgb_th = 0.5
lgb_th = 0.5
rf_th  = 0.5

df["xgb_pred"] = (df["xgb_prob"] >= xgb_th).astype(int)
df["lgb_pred"] = (df["lgb_prob"] >= lgb_th).astype(int)
df["rf_pred"]  = (df["rf_prob"]  >= rf_th ).astype(int)

# ===============================
# 3. baseline（先用 LGB 當主模型）
# ===============================

df["final_pred"] = df["lgb_pred"].copy()

# ===============================
# 4. 找 decision boundary（不確定區）
# ===============================

LOW_TH  = 0.3
HIGH_TH = 0.7

uncertain_mask = (df["lgb_prob"] >= LOW_TH) & (df["lgb_prob"] <= HIGH_TH)

# ===============================
# 5. 多數投票
# ===============================

df["votes"] = df["xgb_pred"] + df["lgb_pred"] + df["rf_pred"]

df.loc[uncertain_mask, "final_pred"] = (
    df.loc[uncertain_mask, "votes"] >= 2
).astype(int)

# ===============================
# 6. 評分
# ===============================

y_true = df["true_label"]
y_pred = df["final_pred"]

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)

print("\n=== Ensemble Evaluation on Validation ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# ===============================
# 7. 存結果
# ===============================

df.to_csv("ensemble_valid_result.csv", index=False)
print("\nensemble_valid_result.csv 已輸出")
