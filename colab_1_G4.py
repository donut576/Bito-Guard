import requests
import pandas as pd
import numpy as np
import time
from IPython.display import display

# ============================================================
# 實驗四
# 選用特徵：
#   - source_ip_hash
#   - IP 共用人數（同一 IP 對應幾個 user）
#   - 同 IP 關聯帳號數
#   - 帳號近期 IP 切換次數
#   - 首次登入 IP 與交易 IP 是否差異大
#   - 深夜從新 IP 發生高額交易
#   - IP × user_source 交叉特徵
#   - IP × 註冊時間密集度
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
        # 欄位名稱空格統一換成底線（防止 API 回傳帶空格的欄位名）
        df_batch.columns = [c.replace(" ", "_") for c in df_batch.columns]
        all_dfs.append(df_batch)
        print(f"{name}: 已抓取 {len(df_batch)} 筆, offset={offset}")
        offset += batch_size
        time.sleep(0.1)
    return pd.concat(all_dfs, ignore_index=True)

def ensure_column(df, col_name):
    """若欄位不存在則補 NaN，避免後續 KeyError"""
    if col_name not in df.columns:
        print(f"警告：{col_name} 欄位不存在，補 NaN")
        df[col_name] = np.nan
    return df

# ============================================================
# Step 1: 抓取資料
# ============================================================
user_info = fetch_table_paginated("user_info")
twd_transfer = fetch_table_paginated("twd_transfer")
crypto_transfer = fetch_table_paginated("crypto_transfer")
usdt_twd_trading = fetch_table_paginated("usdt_twd_trading")
train_label = fetch_table_paginated("train_label")
predict_label = fetch_table_paginated("predict_label")

# 確認 IP 欄位名稱（可能是 source_ip 或 source_ip_hash）
print("twd_transfer 欄位：", twd_transfer.columns.tolist())
print("crypto_transfer 欄位：", crypto_transfer.columns.tolist())

# ============================================================
# Step 2: 統一 IP 欄位名稱，補齊缺失欄位
# ============================================================
# 若 API 回傳的是 source_ip_hash，統一改名為 source_ip 方便後續處理
for df in [twd_transfer, crypto_transfer, usdt_twd_trading]:
    if "source_ip_hash" in df.columns and "source_ip" not in df.columns:
        df.rename(columns={"source_ip_hash": "source_ip"}, inplace=True)

twd_transfer = ensure_column(twd_transfer, "source_ip")
crypto_transfer = ensure_column(crypto_transfer, "source_ip")
usdt_twd_trading = ensure_column(usdt_twd_trading, "source_ip")

# ============================================================
# Step 3: 彙整所有交易的 IP 紀錄
# ============================================================
all_ip_records = pd.concat([
    twd_transfer[["user_id", "source_ip", "created_at", "ori_samount"]].rename(
        columns={"created_at": "ts"}),
    crypto_transfer[["user_id", "source_ip", "created_at", "ori_samount"]].rename(
        columns={"created_at": "ts"}),
    usdt_twd_trading[["user_id", "source_ip", "updated_at", "trade_samount"]].rename(
        columns={"updated_at": "ts", "trade_samount": "ori_samount"}),
]).dropna(subset=["source_ip"])

all_ip_records["ts"] = pd.to_datetime(all_ip_records["ts"], errors="coerce")
all_ip_records["ori_samount"] = pd.to_numeric(all_ip_records["ori_samount"], errors="coerce").fillna(0)
all_ip_records = all_ip_records.sort_values(["user_id", "ts"]).reset_index(drop=True)

# ============================================================
# Step 4: IP 共用人數 / 同 IP 關聯帳號數
# 同一 IP 被多少不同 user 使用 → 值越大越可疑
# ============================================================
ip_share_map = all_ip_records.groupby("source_ip")["user_id"].nunique().to_dict()
all_ip_records["ip_shared_users"] = all_ip_records["source_ip"].map(ip_share_map)

# ============================================================
# Step 5: 帳號近期 IP 切換次數
# 相鄰兩筆交易的 IP 不同 → 計為一次切換
# ============================================================
all_ip_records["ip_changed"] = (
    all_ip_records.groupby("user_id")["source_ip"]
    .transform(lambda x: (x != x.shift()).astype(int))
)
# 第一筆不算切換（shift 後為 NaN，會被標為 1，需修正）
first_tx_mask = all_ip_records.groupby("user_id").cumcount() == 0
all_ip_records.loc[first_tx_mask, "ip_changed"] = 0

# ============================================================
# Step 6: 深夜從新 IP 發生高額交易
# 條件：凌晨 0~5 點 + 金額 > 50,000 TWD + 該 IP 是首次出現
# ============================================================
all_ip_records["is_late_night"] = all_ip_records["ts"].dt.hour.between(0, 5).astype(int)
all_ip_records["amount_twd"] = all_ip_records["ori_samount"] * 1e-8

# 標記每個 user 的首次出現 IP（按時間排序後，第一次見到該 IP）
all_ip_records["is_new_ip"] = ~all_ip_records.duplicated(subset=["user_id", "source_ip"], keep="first")
all_ip_records["late_night_new_ip_high"] = (
    (all_ip_records["is_late_night"] == 1) &
    (all_ip_records["is_new_ip"]) &
    (all_ip_records["amount_twd"] > 50000)
).astype(int)

# ============================================================
# Step 7: 首次登入 IP 與交易 IP 是否差異大
# 定義：user 的第一筆交易 IP 與後續交易中不同 IP 的比例
# ============================================================
first_ip = (
    all_ip_records.groupby("user_id")["source_ip"]
    .first()
    .reset_index()
    .rename(columns={"source_ip": "first_ip"})
)
all_ip_records = all_ip_records.merge(first_ip, on="user_id", how="left")
# 與首次 IP 不同的交易比例
all_ip_records["diff_from_first_ip"] = (
    all_ip_records["source_ip"] != all_ip_records["first_ip"]
).astype(int)

# ============================================================
# Step 8: User-level 聚合特徵
# ============================================================
feature_df_exp4 = all_ip_records.groupby("user_id").agg(
    avg_ip_shared_users=("ip_shared_users", "mean"),    # 平均 IP 共用人數
    max_ip_shared_users=("ip_shared_users", "max"),     # 最大 IP 共用人數
    total_ip_changes=("ip_changed", "sum"),             # 總 IP 切換次數
    unique_ip_count=("source_ip", "nunique"),           # 使用過的不同 IP 數
    late_night_new_ip_high_tx=("late_night_new_ip_high", "sum"),  # 深夜新IP高額交易次數
    diff_ip_ratio=("diff_from_first_ip", "mean"),       # 與首次 IP 不同的比例
).reset_index()

# ============================================================
# Step 9: IP × user_source 交叉特徵
# 計算每個 IP 對應的 user_source 種類數（多種來源共用同一 IP → 可疑）
# ============================================================
user_info["confirmed_at"] = pd.to_datetime(user_info["confirmed_at"], errors="coerce")

# 取每個 user 最常用的 IP 作為代表 IP
user_main_ip = (
    all_ip_records.groupby("user_id")["source_ip"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"source_ip": "main_ip"})
)
user_reg_with_ip = user_info.merge(user_main_ip, on="user_id", how="left")

# IP × user_source：同一 IP 有幾種不同的 user_source
if "user_source" in user_info.columns:
    ip_source_diversity = (
        user_reg_with_ip.dropna(subset=["main_ip"])
        .groupby("main_ip")["user_source"]
        .nunique()
        .reset_index()
        .rename(columns={"user_source": "ip_source_diversity"})
    )
    user_main_ip = user_main_ip.merge(ip_source_diversity, on="main_ip", how="left")
    feature_df_exp4 = feature_df_exp4.merge(
        user_main_ip[["user_id", "ip_source_diversity"]], on="user_id", how="left"
    )
else:
    print("警告：user_info 無 user_source 欄位，跳過 IP×usersource 特徵")

# ============================================================
# Step 10: IP × 註冊時間密集度
# 計算同一 IP 在 1 小時內有幾個帳號完成註冊
# 修正：先 groupby 算出密集度，再 map 回 user（不用 merge on 時間）
# ============================================================
reg_df = user_reg_with_ip.dropna(subset=["main_ip", "confirmed_at"]).copy()
reg_df = reg_df.set_index("confirmed_at")

ip_reg_density_map = {}
for ip, group in reg_df.groupby("main_ip"):
    # 每個時間點前 1 小時內的註冊人數
    rolling_count = group["user_id"].resample("1h").count()
    ip_reg_density_map[ip] = rolling_count.max()  # 取最高峰值

user_main_ip["ip_reg_density"] = user_main_ip["main_ip"].map(ip_reg_density_map)
feature_df_exp4 = feature_df_exp4.merge(
    user_main_ip[["user_id", "ip_reg_density"]], on="user_id", how="left"
)

# ============================================================
# Step 11: 合併 user_source
# ============================================================
feature_df_exp4 = feature_df_exp4.merge(
    user_info[["user_id", "user_source"]], on="user_id", how="left"
)

feature_df_exp4 = feature_df_exp4.fillna(0)
display(feature_df_exp4.head())
print(f"特徵欄位：{feature_df_exp4.columns.tolist()}")

# ============================================================
# Step 12: 合併 Label，產出訓練集與測試集
# ============================================================
train_df = train_label.merge(feature_df_exp4, on="user_id", how="left").fillna(0)
test_df = predict_label.merge(feature_df_exp4, on="user_id", how="left").fillna(0)

print(f"訓練集形狀: {train_df.shape}")
print(f"測試集形狀: {test_df.shape}")

# ============================================================
# Step 13: 儲存結果
# ============================================================
train_df.to_csv("train_df_exp4.csv", index=False)
test_df.to_csv("test_df_exp4.csv", index=False)
feature_df_exp4.to_csv("feature_df_exp4.csv", index=False)

print("實驗四特徵處理完成，請前往 Colab-2 進行模型訓練。")
