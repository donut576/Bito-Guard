import requests
import pandas as pd
import numpy as np
import time
from IPython.display import display

# ============================================================
# 實驗五
# 選用特徵：
#   - currency（幣種）
#   - kind / sub_kind（交易類型）
#   - protocol（協議）
#   - 高風險幣種標記
#   - TRC20 / BSC 風險標記
#   - protocol × 交易頻率（per user per protocol）
#   - protocol × 小額高頻（小額交易次數 per protocol）
#   - protocol × 自轉移（from_wallet = to_wallet）
#   - protocol × 地址重複率
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
usdt_twd_trading = fetch_table_paginated("usdt_twd_trading")
usdt_swap = fetch_table_paginated("usdt_swap")
train_label = fetch_table_paginated("train_label")
predict_label = fetch_table_paginated("predict_label")

print("crypto_transfer 欄位：", crypto_transfer.columns.tolist())

# ============================================================
# Step 2: 基礎清洗
# ============================================================
crypto_df = crypto_transfer.copy()
crypto_df["created_at"] = pd.to_datetime(crypto_df["created_at"], errors="coerce")
crypto_df["ori_samount"] = pd.to_numeric(crypto_df["ori_samount"], errors="coerce").fillna(0)

# ============================================================
# Step 3: 高風險幣種標記
# 定義高風險幣種（隱私幣、常見洗錢幣種）
# 可依實際資料調整幣種名稱
# ============================================================
HIGH_RISK_CURRENCIES = ["XMR", "ZEC", "DASH", "USDT"]  # 依實際資料調整
if "currency" in crypto_df.columns:
    crypto_df["is_high_risk_currency"] = crypto_df["currency"].isin(HIGH_RISK_CURRENCIES).astype(int)
else:
    print("警告：無 currency 欄位，補 0")
    crypto_df["is_high_risk_currency"] = 0

# ============================================================
# Step 4: TRC20 / BSC 風險標記
# protocol 值需依實際 API 資料確認，以下為常見對應：
#   TRC20 = 4, BSC (BEP20) = 5（請執行後確認實際值）
# ============================================================
if "protocol" in crypto_df.columns:
    print("protocol 唯一值：", crypto_df["protocol"].unique())
    TRC20_BSC_PROTOCOLS = [4, 5]  # 請依實際資料確認
    crypto_df["is_trc20_bsc"] = crypto_df["protocol"].isin(TRC20_BSC_PROTOCOLS).astype(int)
else:
    print("警告：無 protocol 欄位，補 0")
    crypto_df["protocol"] = "unknown"
    crypto_df["is_trc20_bsc"] = 0

# ============================================================
# Step 5: kind / sub_kind 特徵
# 計算每個 user 各 kind 的交易次數（one-hot 風格）
# ============================================================
kind_features = pd.DataFrame({"user_id": crypto_df["user_id"].unique()})

if "kind" in crypto_df.columns:
    kind_counts = (
        crypto_df.groupby(["user_id", "kind"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # 欄位加前綴避免衝突
    kind_counts.columns = ["user_id"] + [f"kind_{c}" for c in kind_counts.columns[1:]]
    kind_features = kind_features.merge(kind_counts, on="user_id", how="left")
else:
    print("警告：無 kind 欄位，跳過")

if "sub_kind" in crypto_df.columns:
    subkind_counts = (
        crypto_df.groupby(["user_id", "sub_kind"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    subkind_counts.columns = ["user_id"] + [f"subkind_{c}" for c in subkind_counts.columns[1:]]
    kind_features = kind_features.merge(subkind_counts, on="user_id", how="left")
else:
    print("警告：無 sub_kind 欄位，跳過")

# ============================================================
# Step 6: protocol × 交易頻率
# 每個 user 在各 protocol 下的交易次數，取最大值作為 user-level 特徵
# 修正：不在 agg lambda 裡 filter 外部 DataFrame（效能極差），改用 pivot 再 merge
# ============================================================
protocol_freq = (
    crypto_df.groupby(["user_id", "protocol"])
    .size()
    .reset_index(name="protocol_tx_count")
)
# 取每個 user 在所有 protocol 中的最大交易次數
max_protocol_freq = (
    protocol_freq.groupby("user_id")["protocol_tx_count"]
    .max()
    .reset_index()
    .rename(columns={"protocol_tx_count": "max_tx_per_protocol"})
)

# ============================================================
# Step 7: protocol × 小額高頻
# 小額定義：ori_samount * 1e-8 < 100（即實際金額 < 100 USDT/TWD）
# ori_samount 是整數格式，100 USDT 對應 100 * 1e8 = 1e10
# ============================================================
SMALL_AMOUNT_THRESHOLD = 100 * 1e8  # 100 USDT（整數格式）
crypto_df["is_small_amount"] = (crypto_df["ori_samount"] < SMALL_AMOUNT_THRESHOLD).astype(int)

protocol_small_freq = (
    crypto_df.groupby(["user_id", "protocol"])["is_small_amount"]
    .sum()
    .reset_index(name="small_tx_count")
)
# 取每個 user 在所有 protocol 中的最大小額交易次數
max_small_freq = (
    protocol_small_freq.groupby("user_id")["small_tx_count"]
    .max()
    .reset_index()
    .rename(columns={"small_tx_count": "max_small_tx_per_protocol"})
)

# ============================================================
# Step 8: protocol × 自轉移
# 自轉移：from_wallet_hash == to_wallet_hash（資金在自己地址間循環）
# ============================================================
crypto_df["is_self_transfer"] = (
    crypto_df["from_wallet_hash"] == crypto_df["to_wallet_hash"]
).astype(int)

protocol_self = (
    crypto_df.groupby(["user_id", "protocol"])["is_self_transfer"]
    .sum()
    .reset_index(name="self_transfer_count")
)
max_self_transfer = (
    protocol_self.groupby("user_id")["self_transfer_count"]
    .max()
    .reset_index()
    .rename(columns={"self_transfer_count": "max_self_transfer_per_protocol"})
)

# ============================================================
# Step 9: protocol × 地址重複率
# 重複率 = 1 - (唯一地址數 / 總交易數*2)
# ============================================================
protocol_addr = crypto_df.groupby(["user_id", "protocol"]).agg(
    unique_from=("from_wallet_hash", "nunique"),
    unique_to=("to_wallet_hash", "nunique"),
    total_tx=("from_wallet_hash", "count"),
).reset_index()
protocol_addr["addr_reuse_rate"] = 1 - (
    (protocol_addr["unique_from"] + protocol_addr["unique_to"])
    / (protocol_addr["total_tx"] * 2).clip(lower=1)
)
avg_reuse = (
    protocol_addr.groupby("user_id")["addr_reuse_rate"]
    .mean()
    .reset_index()
    .rename(columns={"addr_reuse_rate": "avg_addr_reuse_rate"})
)

# ============================================================
# Step 10: User-level 基礎聚合特徵
# ============================================================
feature_df_exp5 = crypto_df.groupby("user_id").agg(
    trc20_bsc_tx_count=("is_trc20_bsc", "sum"),       # TRC20/BSC 交易總次數
    high_risk_currency_count=("is_high_risk_currency", "sum"),  # 高風險幣種交易次數
    total_self_transfers=("is_self_transfer", "sum"),  # 自轉移總次數
    total_small_tx=("is_small_amount", "sum"),         # 小額交易總次數
    unique_protocol_count=("protocol", "nunique"),     # 使用過的協議種類數
).reset_index()

# ============================================================
# Step 11: 合併所有 protocol 衍生特徵
# ============================================================
feature_df_exp5 = feature_df_exp5.merge(max_protocol_freq, on="user_id", how="left")
feature_df_exp5 = feature_df_exp5.merge(max_small_freq, on="user_id", how="left")
feature_df_exp5 = feature_df_exp5.merge(max_self_transfer, on="user_id", how="left")
feature_df_exp5 = feature_df_exp5.merge(avg_reuse, on="user_id", how="left")
feature_df_exp5 = feature_df_exp5.merge(kind_features, on="user_id", how="left")

feature_df_exp5 = feature_df_exp5.fillna(0)

display(feature_df_exp5.head())
print(f"特徵欄位：{feature_df_exp5.columns.tolist()}")

# ============================================================
# Step 12: 合併 Label，產出訓練集與測試集
# ============================================================
train_df = train_label.merge(feature_df_exp5, on="user_id", how="left").fillna(0)
test_df = predict_label.merge(feature_df_exp5, on="user_id", how="left").fillna(0)

print(f"訓練集形狀: {train_df.shape}")
print(f"測試集形狀: {test_df.shape}")

# ============================================================
# Step 13: 儲存結果
# ============================================================
train_df.to_csv("train_df_exp5.csv", index=False)
test_df.to_csv("test_df_exp5.csv", index=False)
feature_df_exp5.to_csv("feature_df_exp5.csv", index=False)

print("實驗五特徵處理完成，請前往 Colab-2 進行模型訓練。")
