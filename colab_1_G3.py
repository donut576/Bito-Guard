import requests
import pandas as pd
import numpy as np
import time
from IPython.display import display

# ============================================================
# 實驗三
# 選用特徵：
#   - from_wallet_hash, to_wallet_hash, relation_user_id
#   - 地址重複率（from/to wallet 重複使用比例）
#   - 共同來源錢包數（同一 from_wallet 對應多少 user）
#   - 共同去向錢包數（同一 to_wallet 對應多少 user）
#   - 同一母錢包衍生的子地址數
#   - A→B→C→A 類循環轉移
#   - 錢包入度 / 出度
#   - 短期內新地址都流向同一收款地址
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
crypto_transfer = fetch_table_paginated("crypto_transfer")
train_label = fetch_table_paginated("train_label")
predict_label = fetch_table_paginated("predict_label")

# ============================================================
# Step 2: 基礎清洗與時間排序
# ============================================================
crypto_df = crypto_transfer.copy()
crypto_df["created_at"] = pd.to_datetime(crypto_df["created_at"], errors="coerce")
crypto_df = crypto_df.sort_values(["user_id", "created_at"]).reset_index(drop=True)

print("crypto_transfer 欄位：", crypto_df.columns.tolist())

# ============================================================
# Step 3: 地址重複率
# 定義：每個 user 的 from/to wallet 中，重複出現的比例
# 計算方式：1 - (唯一錢包數 / 總錢包出現次數)
# ============================================================
wallet_stats = crypto_df.groupby("user_id").agg(
    unique_from_wallets=("from_wallet_hash", "nunique"),
    unique_to_wallets=("to_wallet_hash", "nunique"),
    total_tx=("from_wallet_hash", "count")  # 總交易筆數
).reset_index()

# 唯一錢包數 / 總交易數 越小 → 重複率越高
wallet_stats["wallet_reuse_rate"] = 1 - (
    (wallet_stats["unique_from_wallets"] + wallet_stats["unique_to_wallets"])
    / (wallet_stats["total_tx"] * 2).clip(lower=1)  # 避免除以 0
)

# ============================================================
# Step 4: 共同來源 / 去向錢包數
# 計算每個 from_wallet / to_wallet 被多少不同 user 使用
# 值越大 → 該錢包是多人共用的「集中地址」，異常風險高
# ============================================================
from_wallet_users = crypto_df.groupby("from_wallet_hash")["user_id"].nunique().to_dict()
to_wallet_users = crypto_df.groupby("to_wallet_hash")["user_id"].nunique().to_dict()

crypto_df["common_source_count"] = crypto_df["from_wallet_hash"].map(from_wallet_users)
crypto_df["common_target_count"] = crypto_df["to_wallet_hash"].map(to_wallet_users)

# ============================================================
# Step 5: 錢包入度（In-degree）與出度（Out-degree）
# 入度：某錢包作為收款方（to_wallet）的次數
# 出度：某錢包作為發款方（from_wallet）的次數
# 高入度 → 集中收款地址；高出度 → 大量發送地址
# ============================================================
in_degree = crypto_df.groupby("to_wallet_hash").size().to_dict()
out_degree = crypto_df.groupby("from_wallet_hash").size().to_dict()

crypto_df["wallet_in_degree"] = crypto_df["to_wallet_hash"].map(in_degree)
crypto_df["wallet_out_degree"] = crypto_df["from_wallet_hash"].map(out_degree)

# ============================================================
# Step 6: 同一母錢包衍生的子地址數
# 定義：同一個 from_wallet 對應到多少不同的 to_wallet
# 值越大 → 該錢包像是「分發器」，可能是洗錢中繼站
# ============================================================
parent_child = crypto_df.groupby("from_wallet_hash")["to_wallet_hash"].nunique().to_dict()
crypto_df["child_wallet_count"] = crypto_df["from_wallet_hash"].map(parent_child)

# ============================================================
# Step 7: A→B→C→A 循環轉移偵測
# 簡化邏輯：檢查同一 user 的 from_wallet 與 to_wallet 是否有交集
# 有交集 → 資金在自己控制的地址間循環
# ============================================================
user_flow_overlap = crypto_df.groupby("user_id").apply(
    lambda x: len(set(x["from_wallet_hash"]) & set(x["to_wallet_hash"]))
).reset_index(name="circular_transfer_count")

# ============================================================
# Step 8: 短期內新地址都流向同一收款地址
# 定義：1 小時內，同一 to_wallet 收到來自多少不同 from_wallet
# 值越大 → 短時間內大量不同地址匯入同一地址，異常集中
# 修正：需先 set_index(created_at) 才能用 pd.Grouper
# ============================================================
crypto_df_indexed = crypto_df.set_index("created_at")

short_term_flow = (
    crypto_df_indexed
    .groupby(["to_wallet_hash", pd.Grouper(freq="1h")])["from_wallet_hash"]
    .nunique()
    .reset_index()
    .rename(columns={"from_wallet_hash": "hourly_unique_senders"})
)

# 取每個 to_wallet 的最大值，再 map 回 user
to_wallet_max_flow = (
    short_term_flow.groupby("to_wallet_hash")["hourly_unique_senders"].max().to_dict()
)
crypto_df["short_term_target_flow"] = crypto_df["to_wallet_hash"].map(to_wallet_max_flow)

# ============================================================
# Step 9: User-level 聚合特徵
# ============================================================
agg_cols = {
    "common_source_count": "mean",   # 平均共同來源錢包數
    "common_target_count": "mean",   # 平均共同去向錢包數
    "wallet_in_degree": "max",       # 最大入度
    "wallet_out_degree": "max",      # 最大出度
    "child_wallet_count": "max",     # 最大子地址數（母錢包衍生）
    "short_term_target_flow": "max", # 短期最大集中流入數
}

feature_df_exp3 = crypto_df.groupby("user_id").agg(
    avg_common_source=("common_source_count", "mean"),
    avg_common_target=("common_target_count", "mean"),
    max_wallet_in_degree=("wallet_in_degree", "max"),
    max_wallet_out_degree=("wallet_out_degree", "max"),
    max_child_wallet_count=("child_wallet_count", "max"),
    max_short_term_flow=("short_term_target_flow", "max"),
).reset_index()

# 加入 relation_user_id 特徵（若欄位存在）
if "relation_user_id" in crypto_df.columns:
    relation_feat = (
        crypto_df.groupby("user_id")["relation_user_id"]
        .nunique()
        .reset_index()
        .rename(columns={"relation_user_id": "relation_user_nunique"})
    )
    feature_df_exp3 = feature_df_exp3.merge(relation_feat, on="user_id", how="left")
else:
    print("警告：crypto_transfer 無 relation_user_id 欄位，跳過此特徵")

# 合併地址重複率與循環轉移特徵
feature_df_exp3 = feature_df_exp3.merge(
    wallet_stats[["user_id", "wallet_reuse_rate"]], on="user_id", how="left"
)
feature_df_exp3 = feature_df_exp3.merge(user_flow_overlap, on="user_id", how="left")

feature_df_exp3 = feature_df_exp3.fillna(0)

display(feature_df_exp3.head())
print(f"特徵欄位：{feature_df_exp3.columns.tolist()}")

# ============================================================
# Step 10: 合併 Label，產出訓練集與測試集
# ============================================================
train_df = train_label.merge(feature_df_exp3, on="user_id", how="left").fillna(0)
test_df = predict_label.merge(feature_df_exp3, on="user_id", how="left").fillna(0)

print(f"訓練集形狀: {train_df.shape}")
print(f"測試集形狀: {test_df.shape}")

# ============================================================
# Step 11: 儲存結果
# ============================================================
train_df.to_csv("train_df_exp3.csv", index=False)
test_df.to_csv("test_df_exp3.csv", index=False)
feature_df_exp3.to_csv("feature_df_exp3.csv", index=False)

print("實驗三特徵處理完成，請前往 Colab-2 進行模型訓練。")
