import requests
import pandas as pd
import numpy as np
import time
from IPython.display import display

# ============================================================
# 實驗二
# 選用特徵：
#   - 交易頻率激增：5分鐘內 > 5筆 (created_at)
#   - created_at, ori_samount, twd_srate
#   - 小額多筆後接大額（試水溫）
#   - 金額相對個人歷史均值的倍數 (max/mean)
#   - 金額標準差 / CV
#   - 相鄰交易金額比率
#   - 夜間/凌晨/週末夜間交易 (23:00-05:00)
#   - 多筆交易間隔極短
# ============================================================

BASE_URL = "https://aws-event-api.bitopro.com"

def fetch_table_paginated(name, batch_size=50000):
    """分頁抓取指定資料表，回傳完整 DataFrame"""
    all_dfs = []
    offset = 0
    while True:
        url = f"{BASE_URL}/{name}?limit={batch_size}&offset={offset}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        if len(data) == 0:
            print(f"{name}: 抓取完成，共 {offset} 筆以上。")
            break
        df_batch = pd.DataFrame(data)
        all_dfs.append(df_batch)
        print(f"{name}: 已抓取 {len(df_batch)} 筆, offset={offset}")
        offset += batch_size
        time.sleep(0.1)
    return pd.concat(all_dfs, ignore_index=True)

# ============================================================
# Step 1: 抓取資料
# ============================================================
twd_transfer = fetch_table_paginated("twd_transfer")
crypto_transfer = fetch_table_paginated("crypto_transfer")
train_label = fetch_table_paginated("train_label")
predict_label = fetch_table_paginated("predict_label")

# ============================================================
# Step 2: 時間與金額欄位處理
# ============================================================
# 法幣交易：金額直接除以 1e8 還原
twd_transfer["created_at"] = pd.to_datetime(twd_transfer["created_at"], errors="coerce")
twd_transfer["amount_twd"] = twd_transfer["ori_samount"] * 1e-8

# 虛擬幣交易：需乘上 twd_srate 換算成台幣
# twd_srate 也是整數格式，需除以 1e8
crypto_transfer["created_at"] = pd.to_datetime(crypto_transfer["created_at"], errors="coerce")
if "twd_srate" in crypto_transfer.columns:
    crypto_transfer["amount_twd"] = (
        crypto_transfer["ori_samount"] * 1e-8 * (crypto_transfer["twd_srate"] * 1e-8)
    )
else:
    # 若無匯率欄位，先以原始金額代替（單位不同，僅作為相對比較用）
    print("警告：crypto_transfer 無 twd_srate 欄位，amount_twd 使用 ori_samount * 1e-8")
    crypto_transfer["amount_twd"] = crypto_transfer["ori_samount"] * 1e-8

# ============================================================
# Step 3: 合併法幣與虛擬幣為統一交易流
# ============================================================
all_tx = pd.concat([
    twd_transfer[["user_id", "created_at", "amount_twd"]],
    crypto_transfer[["user_id", "created_at", "amount_twd"]]
]).sort_values(["user_id", "created_at"]).reset_index(drop=True)

# ============================================================
# Step 4: 衍生欄位計算（row-level，聚合前）
# ============================================================

# 4-1. 交易間隔（秒）：同一 user 相鄰兩筆的時間差
all_tx["gap_sec"] = all_tx.groupby("user_id")["created_at"].diff().dt.total_seconds()

# 4-2. 相鄰交易金額比率：當筆 / 前一筆（偵測突然放大的交易）
all_tx["prev_amount"] = all_tx.groupby("user_id")["amount_twd"].shift(1)
all_tx["amount_ratio"] = all_tx["amount_twd"] / (all_tx["prev_amount"].abs() + 1e-9)

# 4-3. 夜間交易標記（23:00 ~ 05:00）
all_tx["is_night"] = all_tx["created_at"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5]).astype(int)

# ============================================================
# Step 5: 交易頻率激增 — 5分鐘內 > 5筆（rolling window）
# ============================================================
all_tx_indexed = all_tx.set_index("created_at")

surge_flags = []
for uid, group in all_tx_indexed.groupby("user_id"):
    # 對每筆交易計算前 5 分鐘內（含自身）的交易筆數
    rolling_count = group["amount_twd"].rolling("5min").count()
    # 只要有任一時間點超過 5 筆，就標記為有頻率激增
    has_surge = int((rolling_count > 5).any())
    surge_flags.append({"user_id": uid, "tx_surge_flag": has_surge})

surge_df = pd.DataFrame(surge_flags)

# ============================================================
# Step 6: 小額多筆後接大額（試水溫偵測）
# 定義：前 3 筆均 < 個人均值的 20%，且下一筆 > 個人均值的 3 倍
# ============================================================
user_mean = all_tx.groupby("user_id")["amount_twd"].mean().rename("user_mean_amount")
all_tx = all_tx.merge(user_mean, on="user_id", how="left")

# 標記小額（< 個人均值 20%）與大額（> 個人均值 300%）
all_tx["is_small"] = (all_tx["amount_twd"] < all_tx["user_mean_amount"] * 0.2).astype(int)
all_tx["is_large"] = (all_tx["amount_twd"] > all_tx["user_mean_amount"] * 3.0).astype(int)

# 計算連續小額筆數（rolling 3 筆都是小額）
all_tx["small_streak"] = all_tx.groupby("user_id")["is_small"].transform(
    lambda x: x.rolling(3, min_periods=3).sum()
)
# 前 3 筆都是小額，且當筆是大額 → 試水溫
all_tx["probe_pattern"] = ((all_tx["small_streak"] == 3) & (all_tx["is_large"] == 1)).astype(int)

probe_df = (
    all_tx.groupby("user_id")["probe_pattern"]
    .max()
    .reset_index()
    .rename(columns={"probe_pattern": "probe_trade_flag"})
)

# ============================================================
# Step 7: User-level 聚合特徵
# ============================================================
feature_df = all_tx.groupby("user_id").agg(
    # 交易間隔統計
    avg_gap_sec=("gap_sec", "mean"),       # 平均交易間隔
    min_gap_sec=("gap_sec", "min"),        # 最短交易間隔（極端值）

    # 金額統計
    total_amount_twd=("amount_twd", "sum"),
    mean_amount_twd=("amount_twd", "mean"),
    max_amount_twd=("amount_twd", "max"),
    std_amount_twd=("amount_twd", "std"),

    # 相鄰金額比率
    avg_amount_ratio=("amount_ratio", "mean"),   # 平均相鄰比率
    max_amount_ratio=("amount_ratio", "max"),    # 最大相鄰比率（突增偵測）

    # 夜間交易比例
    night_tx_ratio=("is_night", "mean"),
).reset_index()

# 4. 金額相對個人歷史均值的倍數：最大單筆 / 均值（正確版本）
feature_df["amount_to_mean_multiplier"] = (
    feature_df["max_amount_twd"] / (feature_df["mean_amount_twd"] + 1e-9)
)

# 5. 金額變異係數 CV = std / mean（衡量金額分散程度）
feature_df["amount_cv"] = (
    feature_df["std_amount_twd"] / (feature_df["mean_amount_twd"] + 1e-9)
)

# ============================================================
# Step 8: 合併頻率激增與試水溫特徵
# ============================================================
feature_df = feature_df.merge(surge_df, on="user_id", how="left")
feature_df = feature_df.merge(probe_df, on="user_id", how="left")
feature_df = feature_df.fillna(0)

display(feature_df.head())
print(f"特徵欄位：{feature_df.columns.tolist()}")

# ============================================================
# Step 9: 合併 Label，產出訓練集與測試集
# ============================================================
train_df = train_label.merge(feature_df, on="user_id", how="left").fillna(0)
test_df = predict_label.merge(feature_df, on="user_id", how="left").fillna(0)

print(f"訓練集形狀: {train_df.shape}")
print(f"測試集形狀: {test_df.shape}")

# ============================================================
# Step 10: 儲存結果
# ============================================================
train_df.to_csv("train_df_exp2.csv", index=False)
test_df.to_csv("test_df_exp2.csv", index=False)
feature_df.to_csv("feature_df_exp2.csv", index=False)

print("實驗二特徵處理完成，請前往 Colab-2 進行模型訓練。")
