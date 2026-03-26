# -*- coding: utf-8 -*-
"""
feature_engineering.py

輸出：
  - train_feature.csv
  - test_feature.csv
  - feature_full.csv
"""

import time
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 200)


# ============================================================
# 1. 通用函式
# ============================================================
def fetch_table_paginated(name: str, batch_size: int = 50000, sleep_sec: float = 0.15) -> pd.DataFrame:
    """
    分頁抓取 API 資料，直到回傳空資料為止。

    📌 資料來源：
    Swagger UI（查看欄位與結構）：
    https://aws-event-docs.bitopro.com/

    API Endpoint（實際抓資料）：
    https://aws-event-api.bitopro.com/

    📌 說明：
    - Swagger UI 用來查詢有哪些 table / 欄位
    - API endpoint 才是實際拿資料的地方

    📌 為什麼要分頁？
    API 單次回傳有筆數上限，若資料量大（如 crypto_transfer 可達數十萬筆以上），
    一次抓取容易 timeout 或記憶體爆掉，因此採用 offset + limit 分頁拉取。

    📌 參數：
    - name: table 名稱（如 user_info, twd_transfer）
    - batch_size: 每次抓取筆數
    - sleep_sec: 每次請求間隔，避免打爆 API
    """

    all_dfs = []
    offset = 0

    while True:
        url = f"https://aws-event-api.bitopro.com/{name}?limit={batch_size}&offset={offset}"

        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()

        if len(data) == 0:
            print(f"{name}: stop at offset={offset}")
            break

        df_batch = pd.DataFrame(data)
        all_dfs.append(df_batch)

        print(f"{name}: fetched {len(df_batch)} rows, offset={offset}")

        offset += batch_size
        time.sleep(sleep_sec)

    final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    print(f"{name}: final shape = {final_df.shape}")

    return final_df

def get_existing_col(df: pd.DataFrame, candidates: list) -> object:
    """
    從多個候選欄位名稱中，找出第一個實際存在於 df 的欄位。
    用途：相容脫敏版 API（source_ip 或 source_ip_hash），自動適配不同版本資料。
    回傳：找到的欄位名稱，若都不存在則回傳 None。
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_divide(a, b):
    """
    安全除法，分母加上極小值 1e-9 避免除以零。
    對 pandas Series 和純量都適用。
    """
    return a / (b + 1e-9)


def add_time_cols(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    將指定時間欄位轉為 datetime，並萃取常用時間維度特徵。

    萃取的欄位：
    - hour        : 小時（0-23），用於偵測深夜交易
    - day_of_week : 星期幾（0=週一, 6=週日），用於偵測週末異常
    - date_only   : 日期（不含時間），用於計算活躍天數
    - is_night    : 凌晨 0-5 點為 1，人頭戶常在非正常時段操作
    - is_weekend  : 週六/週日為 1
    """
    df = df.copy()
    df[time_col]      = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"]        = df[time_col].dt.hour
    df["day_of_week"] = df[time_col].dt.dayofweek
    df["date_only"]   = df[time_col].dt.date
    df["is_night"]    = df["hour"].between(0, 5).astype(int)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    return df


def calc_hour_entropy(series: pd.Series) -> float:
    """
    計算時間分布的資訊熵（Shannon Entropy）。

    熵的意義：
    - 熵高 → 交易時間分散，行為較隨機（正常用戶通常分散）
    - 熵低 → 交易時間高度集中，行為規律（可能是程式自動化操作）

    公式：H = -Σ p(x) * log(p(x))
    加 1e-9 避免 log(0) 的數值問題。
    """
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    p = s.value_counts(normalize=True)
    return float(-(p * np.log(p + 1e-9)).sum())


def calc_gap_features(df: pd.DataFrame, user_col: str, time_col: str, prefix: str) -> pd.DataFrame:
    """
    計算同一使用者相鄰兩筆交易的時間間隔（gap）統計特徵。

    為什麼 gap 特徵重要？
    人頭戶常有「短時間內大量操作」的特徵（burstiness 高），
    而正常用戶的交易間隔通常較長且分散。

    Burstiness index 來自 Goh & Barabási (2008)：
    (std - mean) / (std + mean)，值域 [-1, 1]，越高代表越突發性。
    """
    tmp = df[[user_col, time_col]].dropna().sort_values([user_col, time_col]).copy()
    tmp["gap_sec"] = tmp.groupby(user_col)[time_col].diff().dt.total_seconds()

    feat = tmp.groupby(user_col).agg(
        **{
            f"{prefix}_gap_mean_sec"     : ("gap_sec", "mean"),
            f"{prefix}_gap_std_sec"      : ("gap_sec", "std"),
            f"{prefix}_gap_min_sec"      : ("gap_sec", "min"),
            f"{prefix}_gap_max_sec"      : ("gap_sec", "max"),
            f"{prefix}_gap_lt_5min_ratio": ("gap_sec", lambda x: (x < 300).mean()),
            f"{prefix}_gap_lt_1hr_ratio" : ("gap_sec", lambda x: (x < 3600).mean()),
        }
    ).reset_index()

    feat[f"{prefix}_burstiness"] = (
        (feat[f"{prefix}_gap_std_sec"].fillna(0) - feat[f"{prefix}_gap_mean_sec"].fillna(0))
        / (feat[f"{prefix}_gap_std_sec"].fillna(0) + feat[f"{prefix}_gap_mean_sec"].fillna(0) + 1e-9)
    )
    return feat


def calc_active_day_features(df: pd.DataFrame, user_col: str, date_col: str, prefix: str) -> pd.DataFrame:
    """
    計算使用者的活躍天數（有交易紀錄的不重複日期數）。

    活躍天數用途：
    - 搭配交易筆數可算出「每天平均交易頻率」
    - 活躍天數極少但交易量大，是人頭戶的常見特徵
    """
    feat = df.groupby(user_col).agg(
        **{f"{prefix}_active_days": (date_col, "nunique")}
    ).reset_index()
    return feat


# ============================================================
# 2. 資料清洗
# ============================================================

def prepare_user_info(user_info: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 user_info 並建立 KYC 時間差特徵。

    KYC 流程說明（依 schema）：
    - confirmed_at       : 完成 Email 驗證
    - level1_finished_at : 完成手機驗證（KYC Level 1）
    - level2_finished_at : 完成身份驗證（KYC Level 2），完成後才能法幣出入金

    為什麼 KYC 時間差是重要特徵？
    人頭戶通常是被人操控快速完成 KYC，
    lvl1_minus_confirm_sec 極短（幾秒內完成）是異常訊號。
    """
    user_info = user_info.copy()
    for col in ["confirmed_at", "level1_finished_at", "level2_finished_at"]:
        if col in user_info.columns:
            user_info[col] = pd.to_datetime(user_info[col], errors="coerce")

    user_info["has_confirmed"] = user_info["confirmed_at"].notna().astype(int)
    user_info["has_level1"]    = user_info["level1_finished_at"].notna().astype(int)
    user_info["has_level2"]    = user_info["level2_finished_at"].notna().astype(int)

    # KYC 各階段時間差（秒）；負值代表時間順序異常，本身也是異常旗標
    user_info["lvl1_minus_confirm_sec"] = (
        user_info["level1_finished_at"] - user_info["confirmed_at"]
    ).dt.total_seconds()
    user_info["lvl2_minus_lvl1_sec"] = (
        user_info["level2_finished_at"] - user_info["level1_finished_at"]
    ).dt.total_seconds()
    user_info["lvl2_minus_confirm_sec"] = (
        user_info["level2_finished_at"] - user_info["confirmed_at"]
    ).dt.total_seconds()
    return user_info


def prepare_twd_transfer(twd_transfer: pd.DataFrame, twd_ip_col) -> pd.DataFrame:
    """
    清洗法幣（台幣）出入金資料。

    金額縮放：ori_samount * 1e-8 = 實際台幣金額。
    kind 說明：0=加值（入金），1=提領（出金）。

    注意：twd_transfer schema 無 id 欄位，
    後續 count 改用 kind（必定存在）。
    """
    twd_transfer = twd_transfer.copy()
    twd_transfer = add_time_cols(twd_transfer, "created_at")
    twd_transfer["amount"]     = twd_transfer["ori_samount"] * 1e-8
    twd_transfer["amount_log"] = np.log1p(twd_transfer["amount"])
    twd_transfer["is_deposit"]  = (twd_transfer["kind"] == 0).astype(int)
    twd_transfer["is_withdraw"] = (twd_transfer["kind"] == 1).astype(int)
    # 統一 IP 欄位名稱（若無 IP 欄位則填 NaN）
    twd_transfer["ip_for_feat"] = twd_transfer[twd_ip_col] if twd_ip_col else np.nan
    return twd_transfer


def prepare_crypto_transfer(
    crypto_transfer: pd.DataFrame,
    crypto_ip_col,
    from_wallet_col,
    to_wallet_col,
) -> pd.DataFrame:
    """
    清洗虛擬貨幣轉帳資料，並補上鏈別風險特徵所需欄位。

    金額計算：
    - crypto_amount = ori_samount * 1e-8
    - twd_rate      = twd_srate * 1e-8
    - twd_value     = crypto_amount * twd_rate（換算台幣，跨幣種比較用）

    sub_kind：0=外部（鏈上），1=內部（站內轉帳）
    protocol：0=Self, 1=ERC20, 2=OMNI, 3=BNB, 4=TRC20, 5=BSC, 6=Polygon

    鏈別切換（is_protocol_switch）：
    同一用戶相鄰兩筆使用不同協議，頻繁切換可能代表刻意規避追蹤。

    小額旗標（is_small_twd_value）：
    台幣價值 <= 1000 元，人頭戶常先做小額測試確認地址可用。
    """
    crypto_transfer = crypto_transfer.copy()
    crypto_transfer = add_time_cols(crypto_transfer, "created_at")

    crypto_transfer["crypto_amount"]     = crypto_transfer["ori_samount"] * 1e-8
    crypto_transfer["twd_rate"]          = crypto_transfer["twd_srate"] * 1e-8
    crypto_transfer["twd_value"]         = crypto_transfer["crypto_amount"] * crypto_transfer["twd_rate"]
    crypto_transfer["crypto_amount_log"] = np.log1p(crypto_transfer["crypto_amount"])
    crypto_transfer["twd_value_log"]     = np.log1p(crypto_transfer["twd_value"])

    crypto_transfer["is_deposit"]        = (crypto_transfer["kind"] == 0).astype(int)
    crypto_transfer["is_withdraw"]       = (crypto_transfer["kind"] == 1).astype(int)
    crypto_transfer["is_external"]       = (crypto_transfer["sub_kind"] == 0).astype(int)
    crypto_transfer["is_internal"]       = (crypto_transfer["sub_kind"] == 1).astype(int)
    crypto_transfer["has_relation_user"] = crypto_transfer["relation_user_id"].notna().astype(int)

    crypto_transfer["ip_for_feat"]      = crypto_transfer[crypto_ip_col]   if crypto_ip_col   else np.nan
    crypto_transfer["from_wallet_feat"] = crypto_transfer[from_wallet_col] if from_wallet_col else np.nan
    crypto_transfer["to_wallet_feat"]   = crypto_transfer[to_wallet_col]   if to_wallet_col   else np.nan

    crypto_transfer["has_from_wallet"] = crypto_transfer["from_wallet_feat"].notna().astype(int)
    crypto_transfer["has_to_wallet"]   = crypto_transfer["to_wallet_feat"].notna().astype(int)

    # 鏈別旗標（TRC20=4, BSC=5）
    crypto_transfer["is_trc20"] = (crypto_transfer["protocol"] == 4).astype(int)
    crypto_transfer["is_bsc"]   = (crypto_transfer["protocol"] == 5).astype(int)

    # 鏈別切換旗標：同一用戶相鄰兩筆使用不同協議
    crypto_transfer = crypto_transfer.sort_values(["user_id", "created_at"]).copy()
    crypto_transfer["prev_protocol"]      = crypto_transfer.groupby("user_id")["protocol"].shift(1)
    crypto_transfer["is_protocol_switch"] = (
        crypto_transfer["protocol"].ne(crypto_transfer["prev_protocol"])
        & crypto_transfer["prev_protocol"].notna()
    ).astype(int)

    # 小額交易旗標（台幣價值 <= 1000 元）
    crypto_transfer["is_small_twd_value"] = (crypto_transfer["twd_value"] <= 1000).astype(int)

    # 自轉移旗標：from_wallet == to_wallet（主要對外部交易有意義）
    crypto_transfer["is_self_wallet_transfer"] = (
        crypto_transfer["from_wallet_feat"].notna()
        & crypto_transfer["to_wallet_feat"].notna()
        & crypto_transfer["from_wallet_feat"].eq(crypto_transfer["to_wallet_feat"])
    ).astype(int)

    return crypto_transfer


def prepare_trade(usdt_twd_trading: pd.DataFrame, trade_ip_col) -> pd.DataFrame:
    """
    清洗撮合交易（掛單簿）資料。

    金額縮放：trade_samount * 1e-8 = 實際 USDT 數量；twd_srate * 1e-8 = 實際匯率。
    時間欄位：用 updated_at（訂單完成時間），不是 created_at。
    is_buy：1=買單（TWD→USDT），0=賣單（USDT→TWD）。
    source：下單來源（0=WEB, 1=APP, 2=API）。

    注意：usdt_twd_trading schema 無 id 欄位，count 改用 is_buy。
    """
    usdt_twd_trading = usdt_twd_trading.copy()
    usdt_twd_trading = add_time_cols(usdt_twd_trading, "updated_at")
    usdt_twd_trading["amount"]     = usdt_twd_trading["trade_samount"] * 1e-8
    usdt_twd_trading["price"]      = usdt_twd_trading["twd_srate"] * 1e-8
    usdt_twd_trading["amount_log"] = np.log1p(usdt_twd_trading["amount"])
    usdt_twd_trading["price_log"]  = np.log1p(usdt_twd_trading["price"])
    usdt_twd_trading["is_sell"]    = (usdt_twd_trading["is_buy"] == 0).astype(int)
    usdt_twd_trading["ip_for_feat"] = usdt_twd_trading[trade_ip_col] if trade_ip_col else np.nan
    return usdt_twd_trading


def prepare_swap(usdt_swap: pd.DataFrame) -> pd.DataFrame:
    """
    清洗一鍵買賣（swap）資料。

    金額縮放：twd_samount * 1e-8 = 實際台幣；currency_samount * 1e-8 = 實際虛幣。
    kind：0=買幣（TWD→USDT），1=賣幣（USDT→TWD）。

    注意：usdt_swap schema 無 id 欄位，count 改用 kind。
    """
    usdt_swap = usdt_swap.copy()
    usdt_swap = add_time_cols(usdt_swap, "created_at")
    usdt_swap["twd_amount"]        = usdt_swap["twd_samount"] * 1e-8
    usdt_swap["crypto_amount"]     = usdt_swap["currency_samount"] * 1e-8
    usdt_swap["twd_amount_log"]    = np.log1p(usdt_swap["twd_amount"])
    usdt_swap["crypto_amount_log"] = np.log1p(usdt_swap["crypto_amount"])
    usdt_swap["is_buy_coin"]  = (usdt_swap["kind"] == 0).astype(int)
    usdt_swap["is_sell_coin"] = (usdt_swap["kind"] == 1).astype(int)
    return usdt_swap


# ============================================================
# 3. 基礎聚合特徵
# ============================================================

def build_twd_features(twd_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    建立台幣出入金的使用者層級聚合特徵。

    Bug 修正：原 feature_test_1.py 使用 ("id", "count")，
    但 twd_transfer schema 無 id 欄位，改用 ("kind", "count")。

    特徵設計邏輯：
    - 金額統計（sum/mean/max/std/median）：捕捉資金規模與分布
    - 入金/出金分開統計：計算出入金比例，人頭戶通常出金 >> 入金
    - 夜間/週末比例：異常時段操作
    - IP 多樣性：同一用戶使用多個 IP 是異常訊號
    - 時間熵：交易時間是否集中（低熵=高度規律=可能是程式操作）
    - active_span_sec：從第一筆到最後一筆的總時間跨度
    """
    # 修正：twd_transfer 無 id 欄位，用 kind 做 count
    twd_feat = twd_transfer.groupby("user_id").agg(
        twd_txn_count     =("kind",        "count"),
        twd_total_amount  =("amount",      "sum"),
        twd_avg_amount    =("amount",      "mean"),
        twd_max_amount    =("amount",      "max"),
        twd_min_amount    =("amount",      "min"),
        twd_std_amount    =("amount",      "std"),
        twd_median_amount =("amount",      "median"),
        twd_deposit_count =("is_deposit",  "sum"),
        twd_withdraw_count=("is_withdraw", "sum"),
        twd_night_ratio   =("is_night",    "mean"),
        twd_weekend_ratio =("is_weekend",  "mean"),
        twd_unique_ip     =("ip_for_feat", "nunique"),
        twd_hour_entropy  =("hour",        calc_hour_entropy),
        twd_first_time    =("created_at",  "min"),
        twd_last_time     =("created_at",  "max"),
    ).reset_index()

    twd_in = twd_transfer[twd_transfer["is_deposit"] == 1].groupby("user_id").agg(
        twd_in_total_amount=("amount", "sum"),
        twd_in_avg_amount  =("amount", "mean"),
        twd_in_max_amount  =("amount", "max"),
    ).reset_index()

    twd_out = twd_transfer[twd_transfer["is_withdraw"] == 1].groupby("user_id").agg(
        twd_out_total_amount=("amount", "sum"),
        twd_out_avg_amount  =("amount", "mean"),
        twd_out_max_amount  =("amount", "max"),
    ).reset_index()

    twd_active_days = calc_active_day_features(twd_transfer, "user_id", "date_only", "twd")
    twd_gap_feat    = calc_gap_features(twd_transfer, "user_id", "created_at", "twd")

    for feat in [twd_in, twd_out, twd_active_days, twd_gap_feat]:
        twd_feat = twd_feat.merge(feat, on="user_id", how="left")

    twd_feat["twd_deposit_ratio"]          = safe_divide(twd_feat["twd_deposit_count"],    twd_feat["twd_txn_count"])
    twd_feat["twd_withdraw_ratio"]         = safe_divide(twd_feat["twd_withdraw_count"],   twd_feat["twd_txn_count"])
    # 出金次數 / 入金次數：> 1 代表出金比入金更頻繁
    twd_feat["twd_withdraw_deposit_ratio"] = safe_divide(twd_feat["twd_withdraw_count"],   twd_feat["twd_deposit_count"])
    # 出金金額 / 入金金額：接近 1 代表幾乎全部入金都被提走（快進快出）
    twd_feat["twd_out_in_amount_ratio"]    = safe_divide(twd_feat["twd_out_total_amount"], twd_feat["twd_in_total_amount"])
    twd_feat["twd_txn_per_active_day"]     = safe_divide(twd_feat["twd_txn_count"],        twd_feat["twd_active_days"])
    twd_feat["twd_active_span_sec"]        = (
        twd_feat["twd_last_time"] - twd_feat["twd_first_time"]
    ).dt.total_seconds()

    return twd_feat


def _build_protocol_detail(g: pd.DataFrame) -> pd.Series:
    """
    針對單一用戶，計算 TRC20 和 BSC 的細粒度風險特徵。

    為什麼要分鏈計算？
    TRC20 和 BSC 是常見洗錢鏈（手續費低、速度快、較難追蹤），
    若用戶大量使用這兩條鏈，且集中在夜間、地址重用率高，
    複合風險訊號會比單一特徵更強。

    地址重用率（reuse_rate）：
    1 - (唯一地址數 / 總交易筆數)，越接近 1 代表一直用同一個地址。

    流向不平衡（in_out_imbalance）：
    |入金筆數 - 出金筆數| / 總筆數，越高代表資金流向高度單向。
    """
    total = len(g)
    trc   = g[g["is_trc20"] == 1]
    bsc   = g[g["is_bsc"]   == 1]

    def reuse_rate(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return 0.0
        return float(1 - s.nunique() / (len(s) + 1e-9))

    return pd.Series({
        "trc20_ratio"               : safe_divide(len(trc), total),
        "bsc_ratio"                 : safe_divide(len(bsc), total),
        "trc20_night_ratio"         : trc["is_night"].mean()   if len(trc) > 0 else 0.0,
        "bsc_night_ratio"           : bsc["is_night"].mean()   if len(bsc)  > 0 else 0.0,
        "trc20_weekend_ratio"       : trc["is_weekend"].mean() if len(trc) > 0 else 0.0,
        "bsc_weekend_ratio"         : bsc["is_weekend"].mean() if len(bsc)  > 0 else 0.0,
        "trc20_unique_to_wallet"    : trc["to_wallet_feat"].nunique(dropna=True),
        "bsc_unique_to_wallet"      : bsc["to_wallet_feat"].nunique(dropna=True),
        "trc20_addr_reuse_rate"     : reuse_rate(trc["to_wallet_feat"]),
        "bsc_addr_reuse_rate"       : reuse_rate(bsc["to_wallet_feat"]),
        "trc20_from_addr_reuse_rate": reuse_rate(trc["from_wallet_feat"]),
        "bsc_from_addr_reuse_rate"  : reuse_rate(bsc["from_wallet_feat"]),
        "trc20_inflow_ratio"        : trc["is_deposit"].mean()  if len(trc) > 0 else 0.0,
        "trc20_outflow_ratio"       : trc["is_withdraw"].mean() if len(trc) > 0 else 0.0,
        "bsc_inflow_ratio"          : bsc["is_deposit"].mean()  if len(bsc)  > 0 else 0.0,
        "bsc_outflow_ratio"         : bsc["is_withdraw"].mean() if len(bsc)  > 0 else 0.0,
        # 流向不平衡：|入金筆數 - 出金筆數| / 總筆數
        "trc20_in_out_imbalance"    : abs(trc["is_deposit"].sum() - trc["is_withdraw"].sum()) / (len(trc) + 1e-9),
        "bsc_in_out_imbalance"      : abs(bsc["is_deposit"].sum() - bsc["is_withdraw"].sum()) / (len(bsc)  + 1e-9),
    })


def build_crypto_features(crypto_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    建立虛擬貨幣轉帳的使用者層級聚合特徵，並整合鏈別風險欄位。

    特徵設計邏輯：
    - 金額統計：同時統計虛幣數量和台幣換算價值（跨幣種比較用 twd_value）
    - 外部/內部交易分開統計：外部（鏈上）比例高代表資金流出平台
    - 關聯用戶：內轉對象的多樣性（network 特徵的前置）
    - 幣種/協議多樣性：使用越多種幣種/鏈，行為越複雜
    - 鏈別風險特徵（TRC20/BSC）：由 _build_protocol_detail 計算
    - 協議切換率：頻繁切換鏈可能代表刻意規避追蹤
    """
    crypto_feat = crypto_transfer.groupby("user_id").agg(
        crypto_txn_count           =("kind",              "count"),
        crypto_total_amount        =("crypto_amount",     "sum"),
        crypto_avg_amount          =("crypto_amount",     "mean"),
        crypto_max_amount          =("crypto_amount",     "max"),
        crypto_std_amount          =("crypto_amount",     "std"),
        crypto_total_twd_value     =("twd_value",         "sum"),
        crypto_avg_twd_value       =("twd_value",         "mean"),
        crypto_max_twd_value       =("twd_value",         "max"),
        crypto_std_twd_value       =("twd_value",         "std"),
        crypto_deposit_count       =("is_deposit",        "sum"),
        crypto_withdraw_count      =("is_withdraw",       "sum"),
        crypto_external_count      =("is_external",       "sum"),
        crypto_internal_count      =("is_internal",       "sum"),
        crypto_relation_count      =("has_relation_user", "sum"),
        crypto_unique_currency     =("currency",          "nunique"),
        crypto_unique_protocol     =("protocol",          "nunique"),
        crypto_unique_relation_user=("relation_user_id",  "nunique"),
        crypto_unique_ip           =("ip_for_feat",       "nunique"),
        crypto_unique_from_wallet  =("from_wallet_feat",  "nunique"),
        crypto_unique_to_wallet    =("to_wallet_feat",    "nunique"),
        crypto_night_ratio         =("is_night",          "mean"),
        crypto_weekend_ratio       =("is_weekend",        "mean"),
        crypto_hour_entropy        =("hour",              calc_hour_entropy),
        crypto_first_time          =("created_at",        "min"),
        crypto_last_time           =("created_at",        "max"),
    ).reset_index()

    crypto_in = crypto_transfer[crypto_transfer["is_deposit"] == 1].groupby("user_id").agg(
        crypto_in_total_twd_value=("twd_value", "sum"),
        crypto_in_avg_twd_value  =("twd_value", "mean"),
    ).reset_index()

    crypto_out = crypto_transfer[crypto_transfer["is_withdraw"] == 1].groupby("user_id").agg(
        crypto_out_total_twd_value=("twd_value", "sum"),
        crypto_out_avg_twd_value  =("twd_value", "mean"),
    ).reset_index()

    crypto_active_days = calc_active_day_features(crypto_transfer, "user_id", "date_only", "crypto")
    crypto_gap_feat    = calc_gap_features(crypto_transfer, "user_id", "created_at", "crypto")

    # 協議層級彙總特徵（不含 trc20_night_ratio / bsc_night_ratio，
    # 這兩個由 protocol_detail_feat 計算，避免 merge 後產生 _x/_y 重複欄位）
    protocol_feat = crypto_transfer.groupby("user_id").agg(
        trc20_tx_count                    =("is_trc20",                 "sum"),
        bsc_tx_count                      =("is_bsc",                   "sum"),
        protocol_switch_count             =("is_protocol_switch",       "sum"),
        protocol_switch_rate              =("is_protocol_switch",       "mean"),
        crypto_small_twd_ratio            =("is_small_twd_value",       "mean"),
        crypto_self_wallet_transfer_count =("is_self_wallet_transfer",  "sum"),
        crypto_self_wallet_transfer_ratio =("is_self_wallet_transfer",  "mean"),
    ).reset_index()

    # 每個用戶的 TRC20/BSC 細粒度特徵
    protocol_detail_feat = (
        crypto_transfer.groupby("user_id")
        .apply(_build_protocol_detail)
        .reset_index()
    )

    for feat in [crypto_in, crypto_out, crypto_active_days, crypto_gap_feat,
                 protocol_feat, protocol_detail_feat]:
        crypto_feat = crypto_feat.merge(feat, on="user_id", how="left")

    crypto_feat["crypto_withdraw_ratio"]      = safe_divide(crypto_feat["crypto_withdraw_count"],      crypto_feat["crypto_txn_count"])
    crypto_feat["crypto_external_ratio"]      = safe_divide(crypto_feat["crypto_external_count"],      crypto_feat["crypto_txn_count"])
    crypto_feat["crypto_internal_ratio"]      = safe_divide(crypto_feat["crypto_internal_count"],      crypto_feat["crypto_txn_count"])
    crypto_feat["crypto_relation_ratio"]      = safe_divide(crypto_feat["crypto_relation_count"],      crypto_feat["crypto_txn_count"])
    # 出金台幣價值 / 入金台幣價值：接近 1 代表快進快出
    crypto_feat["crypto_out_in_amount_ratio"] = safe_divide(crypto_feat["crypto_out_total_twd_value"], crypto_feat["crypto_in_total_twd_value"])
    crypto_feat["crypto_txn_per_active_day"]  = safe_divide(crypto_feat["crypto_txn_count"],           crypto_feat["crypto_active_days"])
    # 每個目標地址的平均交易筆數：越高代表資金集中流向少數地址
    crypto_feat["crypto_txn_per_wallet"]      = safe_divide(crypto_feat["crypto_txn_count"],           crypto_feat["crypto_unique_to_wallet"])
    crypto_feat["crypto_active_span_sec"]     = (
        crypto_feat["crypto_last_time"] - crypto_feat["crypto_first_time"]
    ).dt.total_seconds()

    return crypto_feat


def build_trade_features(usdt_twd_trading: pd.DataFrame) -> pd.DataFrame:
    """
    建立撮合交易（掛單簿）的使用者層級聚合特徵。

    Bug 修正：原 feature_test_1.py 使用 ("id", "count")，
    但 usdt_twd_trading schema 無 id 欄位，改用 ("is_buy", "count")。

    特徵設計邏輯：
    - 買/賣比例：人頭戶通常只做單向操作（只買或只賣）
    - 市價單比例：市價單不在乎成交價，急於成交是異常訊號
    - source 多樣性：同時用 WEB/APP/API 下單，可能是程式化操作
    - 每 IP 的交易筆數：單一 IP 大量下單
    """
    # 修正：usdt_twd_trading 無 id 欄位，用 is_buy 做 count
    trade_feat = usdt_twd_trading.groupby("user_id").agg(
        trade_count         =("is_buy",      "count"),
        trade_total_amount  =("amount",      "sum"),
        trade_avg_amount    =("amount",      "mean"),
        trade_max_amount    =("amount",      "max"),
        trade_std_amount    =("amount",      "std"),
        trade_avg_price     =("price",       "mean"),
        trade_max_price     =("price",       "max"),
        trade_std_price     =("price",       "std"),
        trade_buy_count     =("is_buy",      "sum"),
        trade_sell_count    =("is_sell",     "sum"),
        trade_market_count  =("is_market",   "sum"),
        trade_unique_ip     =("ip_for_feat", "nunique"),
        trade_source_nunique=("source",      "nunique"),
        trade_night_ratio   =("is_night",    "mean"),
        trade_weekend_ratio =("is_weekend",  "mean"),
        trade_hour_entropy  =("hour",        calc_hour_entropy),
        trade_first_time    =("updated_at",  "min"),
        trade_last_time     =("updated_at",  "max"),
    ).reset_index()

    trade_active_days = calc_active_day_features(usdt_twd_trading, "user_id", "date_only", "trade")
    trade_gap_feat    = calc_gap_features(usdt_twd_trading, "user_id", "updated_at", "trade")

    for feat in [trade_active_days, trade_gap_feat]:
        trade_feat = trade_feat.merge(feat, on="user_id", how="left")

    trade_feat["trade_buy_ratio"]          = safe_divide(trade_feat["trade_buy_count"],    trade_feat["trade_count"])
    trade_feat["trade_sell_ratio"]         = safe_divide(trade_feat["trade_sell_count"],   trade_feat["trade_count"])
    trade_feat["trade_market_ratio"]       = safe_divide(trade_feat["trade_market_count"], trade_feat["trade_count"])
    trade_feat["trade_txn_per_active_day"] = safe_divide(trade_feat["trade_count"],        trade_feat["trade_active_days"])
    trade_feat["trade_txn_per_ip"]         = safe_divide(trade_feat["trade_count"],        trade_feat["trade_unique_ip"])
    trade_feat["trade_active_span_sec"]    = (
        trade_feat["trade_last_time"] - trade_feat["trade_first_time"]
    ).dt.total_seconds()

    return trade_feat


def build_swap_features(usdt_swap: pd.DataFrame) -> pd.DataFrame:
    """
    建立一鍵買賣（swap）的使用者層級聚合特徵。

    Bug 修正：原 feature_test_1.py 使用 ("id", "count")，
    但 usdt_swap schema 無 id 欄位，改用 ("kind", "count")。

    Swap 的風險意義：
    - Swap 不留掛單紀錄，比撮合交易更難追蹤
    - swap_sell_buy_twd_ratio 接近 1：買進後幾乎全部賣出（快進快出）
    """
    # 修正：usdt_swap 無 id 欄位，用 kind 做 count
    swap_feat = usdt_swap.groupby("user_id").agg(
        swap_count        =("kind",          "count"),
        swap_total_twd    =("twd_amount",    "sum"),
        swap_avg_twd      =("twd_amount",    "mean"),
        swap_max_twd      =("twd_amount",    "max"),
        swap_std_twd      =("twd_amount",    "std"),
        swap_total_crypto =("crypto_amount", "sum"),
        swap_avg_crypto   =("crypto_amount", "mean"),
        swap_max_crypto   =("crypto_amount", "max"),
        swap_std_crypto   =("crypto_amount", "std"),
        swap_buy_count    =("is_buy_coin",   "sum"),
        swap_sell_count   =("is_sell_coin",  "sum"),
        swap_night_ratio  =("is_night",      "mean"),
        swap_weekend_ratio=("is_weekend",    "mean"),
        swap_hour_entropy =("hour",          calc_hour_entropy),
        swap_first_time   =("created_at",    "min"),
        swap_last_time    =("created_at",    "max"),
    ).reset_index()

    swap_buy  = usdt_swap[usdt_swap["is_buy_coin"]  == 1].groupby("user_id").agg(
        swap_buy_total_twd=("twd_amount", "sum")
    ).reset_index()
    swap_sell = usdt_swap[usdt_swap["is_sell_coin"] == 1].groupby("user_id").agg(
        swap_sell_total_twd=("twd_amount", "sum")
    ).reset_index()

    swap_active_days = calc_active_day_features(usdt_swap, "user_id", "date_only", "swap")
    swap_gap_feat    = calc_gap_features(usdt_swap, "user_id", "created_at", "swap")

    for feat in [swap_buy, swap_sell, swap_active_days, swap_gap_feat]:
        swap_feat = swap_feat.merge(feat, on="user_id", how="left")

    swap_feat["swap_buy_ratio"]          = safe_divide(swap_feat["swap_buy_count"],      swap_feat["swap_count"])
    swap_feat["swap_sell_ratio"]         = safe_divide(swap_feat["swap_sell_count"],     swap_feat["swap_count"])
    swap_feat["swap_sell_buy_twd_ratio"] = safe_divide(swap_feat["swap_sell_total_twd"], swap_feat["swap_buy_total_twd"])
    swap_feat["swap_txn_per_active_day"] = safe_divide(swap_feat["swap_count"],          swap_feat["swap_active_days"])
    swap_feat["swap_active_span_sec"]    = (
        swap_feat["swap_last_time"] - swap_feat["swap_first_time"]
    ).dt.total_seconds()

    return swap_feat


# ============================================================
# 4. 進階特徵
# ============================================================

def build_network_features(crypto_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    利用 relation_user_id 建立使用者間的內轉網路特徵。

    Bug 修正：原 feature_test_1.py 使用 ("id", "count")，
    但 crypto_transfer schema 無 id 欄位，改用 ("user_id", "count")。

    網路特徵的意義：
    人頭戶通常是一個「資金中繼站」，會從多個來源接收資金（in_degree 高），
    再轉給少數幾個目標（out_degree 低），或反過來。
    這種不對稱的網路結構是重要的風險訊號。
    """
    tmp = crypto_transfer[crypto_transfer["has_relation_user"] == 1].copy()
    if len(tmp) == 0:
        return pd.DataFrame({"user_id": []})

    # 修正：用 user_id 做 count（代替不存在的 id）
    network_out = tmp.groupby("user_id").agg(
        network_out_degree     =("relation_user_id", "nunique"),
        network_out_txn_count  =("user_id",          "count"),
        network_out_total_value=("twd_value",         "sum"),
    ).reset_index()

    network_in = tmp.groupby("relation_user_id").agg(
        network_in_degree     =("user_id", "nunique"),
        network_in_txn_count  =("user_id", "count"),
        network_in_total_value=("twd_value", "sum"),
    ).reset_index().rename(columns={"relation_user_id": "user_id"})

    network_feat = network_out.merge(network_in, on="user_id", how="outer").fillna(0)

    network_feat["network_total_degree"]        = network_feat["network_out_degree"] + network_feat["network_in_degree"]
    network_feat["network_out_in_degree_ratio"] = safe_divide(network_feat["network_out_degree"],      network_feat["network_in_degree"])
    network_feat["network_out_in_value_ratio"]  = safe_divide(network_feat["network_out_total_value"], network_feat["network_in_total_value"])
    # 不對稱指數：越接近 1 代表網路結構越不對稱（只進或只出）
    network_feat["network_degree_imbalance"]    = safe_divide(
        np.abs(network_feat["network_out_degree"] - network_feat["network_in_degree"]),
        network_feat["network_total_degree"] + 1,
    )
    return network_feat


def analyze_wallet_risk(crypto_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    分析錢包地址的共用風險與地址重用率。

    Bug 修正：原 feature_test_1.py 使用 ("id", "count")，
    但 crypto_transfer schema 無 id 欄位，改用 ("user_id", "count")。

    高風險錢包定義：
    同一個 to_wallet 地址被 >= 5 個不同用戶使用，
    代表這個地址是「共用收款地址」，可能是洗錢集散地。

    地址重用率（addr_reuse_rate）：
    1 - (唯一地址數 / 總交易筆數)
    正常用戶每次提領通常用不同地址；
    人頭戶可能一直提到同一個地址（重用率高）。
    """
    tmp = crypto_transfer.copy()
    if tmp["to_wallet_feat"].notna().sum() == 0:
        return pd.DataFrame({"user_id": []})

    # 修正：用 user_id 做 count（代替不存在的 id）
    to_wallet_stats = tmp.groupby("to_wallet_feat").agg(
        wallet_user_count =("user_id", "nunique"),
        wallet_txn_count  =("user_id", "count"),
        wallet_total_value=("twd_value", "sum"),
    )
    from_wallet_stats = tmp.groupby("from_wallet_feat").agg(
        wallet_user_count=("user_id", "nunique"),
        wallet_txn_count =("user_id", "count"),
    )

    high_risk_to_wallets   = to_wallet_stats[to_wallet_stats["wallet_user_count"]   >= 5].index
    high_risk_from_wallets = from_wallet_stats[from_wallet_stats["wallet_user_count"] >= 5].index

    tmp["is_high_risk_to_wallet"]   = tmp["to_wallet_feat"].isin(high_risk_to_wallets).astype(int)
    tmp["is_high_risk_from_wallet"] = tmp["from_wallet_feat"].isin(high_risk_from_wallets).astype(int)

    wallet_risk_feat = tmp.groupby("user_id").agg(
        high_risk_to_wallet_count  =("is_high_risk_to_wallet",   "sum"),
        high_risk_from_wallet_count=("is_high_risk_from_wallet", "sum"),
        high_risk_to_wallet_ratio  =("is_high_risk_to_wallet",   "mean"),
        high_risk_from_wallet_ratio=("is_high_risk_from_wallet", "mean"),
        addr_reuse_rate      =("to_wallet_feat",   lambda x: 1 - x.nunique() / (len(x) + 1e-9)),
        from_addr_reuse_rate =("from_wallet_feat", lambda x: 1 - x.nunique() / (len(x) + 1e-9)),
    ).reset_index()

    return wallet_risk_feat


def analyze_ip_patterns(
    twd_transfer: pd.DataFrame,
    crypto_transfer: pd.DataFrame,
    usdt_twd_trading: pd.DataFrame,
) -> pd.DataFrame:
    """
    跨三個交易表彙整 IP 使用特徵。

    共用 IP 定義：同一個 IP 被 >= 3 個不同用戶使用，
    代表這個 IP 可能是「共用設備」或「代操帳號」的訊號。

    ip_jump_rate：唯一 IP 數 / 總 IP 使用次數
    越接近 1 代表每次都用不同 IP（可能在規避追蹤）。

    注意：usdt_swap 沒有 IP 欄位（schema 未提供），故不納入此分析。
    """
    all_ips = pd.concat([
        twd_transfer[["user_id", "ip_for_feat"]].assign(source="twd"),
        crypto_transfer[["user_id", "ip_for_feat"]].assign(source="crypto"),
        usdt_twd_trading[["user_id", "ip_for_feat"]].assign(source="trade"),
    ], ignore_index=True)

    all_ips = all_ips.dropna(subset=["ip_for_feat"])
    if len(all_ips) == 0:
        return pd.DataFrame({"user_id": []})

    ip_user_count = all_ips.groupby("ip_for_feat")["user_id"].nunique()
    shared_ips    = ip_user_count[ip_user_count >= 3].index
    all_ips["is_shared_ip"] = all_ips["ip_for_feat"].isin(shared_ips).astype(int)

    ip_feat = all_ips.groupby("user_id").agg(
        total_unique_ips    =("ip_for_feat", "nunique"),
        total_ip_usage_count=("ip_for_feat", "count"),
        shared_ip_count     =("is_shared_ip", "sum"),
        shared_ip_ratio     =("is_shared_ip", "mean"),
        ip_source_diversity =("source",       "nunique"),
    ).reset_index()

    ip_feat["ip_jump_rate"] = safe_divide(ip_feat["total_unique_ips"], ip_feat["total_ip_usage_count"])
    return ip_feat


def extract_temporal_anomalies(
    twd_transfer: pd.DataFrame,
    crypto_transfer: pd.DataFrame,
    usdt_twd_trading: pd.DataFrame,
    usdt_swap: pd.DataFrame,
) -> pd.DataFrame:
    """
    量化跨渠道的時間規律性與異常時段特徵。

    avg_time_entropy：所有渠道時間熵的平均值
    min_time_entropy：所有渠道中最低的時間熵（最規律的渠道）

    注意：欄位加 _2 後綴，避免與 build_*_features 中已有的同名欄位衝突。
    """
    temporal_feats = []
    for df, prefix in [
        (twd_transfer,    "twd"),
        (crypto_transfer, "crypto"),
        (usdt_twd_trading,"trade"),
        (usdt_swap,       "swap"),
    ]:
        feat = df.groupby("user_id").agg(
            **{
                f"{prefix}_hour_entropy_2" : ("hour",        calc_hour_entropy),
                f"{prefix}_day_entropy"    : ("day_of_week", calc_hour_entropy),
                f"{prefix}_midnight_ratio" : ("is_night",    "mean"),
                f"{prefix}_weekend_ratio_2": ("is_weekend",  "mean"),
            }
        ).reset_index()
        temporal_feats.append(feat)

    temporal_feat = reduce(
        lambda left, right: left.merge(right, on="user_id", how="outer"),
        temporal_feats
    ).fillna(0)

    entropy_cols = [c for c in temporal_feat.columns if "entropy" in c]
    temporal_feat["avg_time_entropy"] = temporal_feat[entropy_cols].mean(axis=1)
    temporal_feat["min_time_entropy"] = temporal_feat[entropy_cols].min(axis=1)
    return temporal_feat


def detect_amount_anomalies(
    twd_transfer: pd.DataFrame,
    crypto_transfer: pd.DataFrame,
    usdt_twd_trading: pd.DataFrame,
    usdt_swap: pd.DataFrame,
) -> pd.DataFrame:
    """
    建立跨渠道的金額規律性與異常特徵。

    整數金額比例（round_ratio）：
    人頭戶常使用整數金額（如 10000、50000），因為是被人指定金額操作。

    變異係數（CV = std / mean）：
    CV 低代表每次交易金額都差不多（規律性高），可能是程式化操作。

    偏態（skew）：
    正偏態高代表大多數交易金額小，但偶爾有極大金額，
    可能是「測試小額 + 一次大額轉出」的洗錢模式。
    """
    twd_amount_feat = twd_transfer.groupby("user_id").agg(
        twd_round_1000_ratio =("amount", lambda x: (x % 1000  == 0).mean()),
        twd_round_10000_ratio=("amount", lambda x: (x % 10000 == 0).mean()),
        twd_amount_cv        =("amount", lambda x: x.std() / (x.mean() + 1e-9)),
        twd_amount_skew      =("amount", lambda x: x.skew() if len(x) > 2 else 0),
    ).reset_index()

    crypto_amount_feat = crypto_transfer.groupby("user_id").agg(
        crypto_twd_value_cv  =("twd_value", lambda x: x.std() / (x.mean() + 1e-9)),
        crypto_twd_value_skew=("twd_value", lambda x: x.skew() if len(x) > 2 else 0),
    ).reset_index()

    trade_amount_feat = usdt_twd_trading.groupby("user_id").agg(
        trade_amount_cv=("amount", lambda x: x.std() / (x.mean() + 1e-9)),
        trade_price_cv =("price",  lambda x: x.std() / (x.mean() + 1e-9)),
    ).reset_index()

    swap_amount_feat = usdt_swap.groupby("user_id").agg(
        swap_twd_cv           =("twd_amount", lambda x: x.std() / (x.mean() + 1e-9)),
        swap_round_10000_ratio=("twd_amount", lambda x: (x % 10000 == 0).mean()),
    ).reset_index()

    amount_feat = twd_amount_feat.merge(crypto_amount_feat, on="user_id", how="outer")
    amount_feat = amount_feat.merge(trade_amount_feat, on="user_id", how="outer")
    amount_feat = amount_feat.merge(swap_amount_feat, on="user_id", how="outer")
    return amount_feat.fillna(0)


def calculate_fund_flow_patterns(
    twd_transfer: pd.DataFrame,
    crypto_transfer: pd.DataFrame,
    usdt_twd_trading: pd.DataFrame,
    usdt_swap: pd.DataFrame,
) -> pd.DataFrame:
    """
    建立跨產品的資金流動時序特徵，捕捉「快進快出」行為。

    快進快出的典型模式：
    1. 台幣入金（twd_transfer, kind=0）
    2. 在 24 小時內開始虛幣提領（crypto_transfer, kind=1）
    → is_fast_twd_to_crypto = 1

    overall_active_span_sec：
    從所有渠道的第一筆交易到最後一筆的總時間跨度。
    跨度極短但交易量大，是人頭戶的典型特徵。
    """
    twd_time   = twd_transfer.groupby("user_id")["created_at"].agg(["min", "max"]).reset_index()
    twd_time.columns = ["user_id", "twd_first", "twd_last"]

    crypto_time = crypto_transfer.groupby("user_id")["created_at"].agg(["min", "max"]).reset_index()
    crypto_time.columns = ["user_id", "crypto_first", "crypto_last"]

    # 注意：撮合交易用 updated_at（訂單完成時間），不是 created_at
    trade_time  = usdt_twd_trading.groupby("user_id")["updated_at"].agg(["min", "max"]).reset_index()
    trade_time.columns = ["user_id", "trade_first", "trade_last"]

    swap_time   = usdt_swap.groupby("user_id")["created_at"].agg(["min", "max"]).reset_index()
    swap_time.columns = ["user_id", "swap_first", "swap_last"]

    flow_feat = twd_time.merge(crypto_time, on="user_id", how="outer")
    flow_feat = flow_feat.merge(trade_time, on="user_id", how="outer")
    flow_feat = flow_feat.merge(swap_time,  on="user_id", how="outer")

    flow_feat["overall_first_time"]      = flow_feat[["twd_first", "crypto_first", "trade_first", "swap_first"]].min(axis=1)
    flow_feat["overall_last_time"]       = flow_feat[["twd_last",  "crypto_last",  "trade_last",  "swap_last"]].max(axis=1)
    flow_feat["overall_active_span_sec"] = (
        flow_feat["overall_last_time"] - flow_feat["overall_first_time"]
    ).dt.total_seconds()

    flow_feat["twd_to_crypto_gap_sec"] = (flow_feat["crypto_first"] - flow_feat["twd_first"]).dt.total_seconds()
    flow_feat["twd_to_trade_gap_sec"]  = (flow_feat["trade_first"]  - flow_feat["twd_first"]).dt.total_seconds()

    flow_feat["is_fast_twd_to_crypto"] = (
        (flow_feat["twd_to_crypto_gap_sec"] > 0) & (flow_feat["twd_to_crypto_gap_sec"] < 86400)
    ).astype(int)
    flow_feat["is_fast_twd_to_trade"]  = (
        (flow_feat["twd_to_trade_gap_sec"]  > 0) & (flow_feat["twd_to_trade_gap_sec"]  < 86400)
    ).astype(int)

    flow_feat["has_twd"]    = flow_feat["twd_first"].notna().astype(int)
    flow_feat["has_crypto"] = flow_feat["crypto_first"].notna().astype(int)
    flow_feat["has_trade"]  = flow_feat["trade_first"].notna().astype(int)
    flow_feat["has_swap"]   = flow_feat["swap_first"].notna().astype(int)
    flow_feat["txn_type_diversity"] = (
        flow_feat["has_twd"] + flow_feat["has_crypto"] + flow_feat["has_trade"] + flow_feat["has_swap"]
    )

    numeric_cols = [
        "user_id", "overall_active_span_sec",
        "twd_to_crypto_gap_sec", "twd_to_trade_gap_sec",
        "is_fast_twd_to_crypto", "is_fast_twd_to_trade",
        "has_twd", "has_crypto", "has_trade", "has_swap", "txn_type_diversity",
    ]
    return flow_feat[numeric_cols].fillna(0)


def extract_sequence_features(twd_transfer: pd.DataFrame, crypto_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    建立台幣/虛幣跨產品行為序列特徵，捕捉「操作順序」的異常模式。

    可疑序列定義：
    台幣入金（twd_in）→ 1 小時內 → 虛幣提領（crypto_out）
    這是最典型的洗錢路徑：台幣入金後快速換成虛幣提走。

    repeat_action_ratio：
    連續做同樣動作的比例（如連續多次 twd_in），
    可能代表分批入金（規避大額申報）。
    """
    twd_seq = twd_transfer[["user_id", "created_at", "kind"]].copy()
    twd_seq["action"] = twd_seq["kind"].map({0: "twd_in", 1: "twd_out"})
    twd_seq = twd_seq[["user_id", "created_at", "action"]]

    crypto_seq = crypto_transfer[["user_id", "created_at", "kind"]].copy()
    crypto_seq["action"] = crypto_seq["kind"].map({0: "crypto_in", 1: "crypto_out"})
    crypto_seq = crypto_seq[["user_id", "created_at", "action"]]

    all_seq = pd.concat([twd_seq, crypto_seq], ignore_index=True).sort_values(["user_id", "created_at"])

    all_seq["prev_action"] = all_seq.groupby("user_id")["action"].shift(1)
    all_seq["next_action"] = all_seq.groupby("user_id")["action"].shift(-1)
    all_seq["is_repeat"]   = (all_seq["action"] == all_seq["prev_action"]).astype(int)
    all_seq["time_to_next"] = all_seq.groupby("user_id")["created_at"].diff(-1).dt.total_seconds().abs()

    # 可疑序列：twd_in → 1 小時內 → crypto_out
    all_seq["is_suspicious_seq"] = (
        (all_seq["action"]       == "twd_in")
        & (all_seq["next_action"] == "crypto_out")
        & (all_seq["time_to_next"] < 3600)
    ).astype(int)

    seq_feat = all_seq.groupby("user_id").agg(
        total_actions        =("action",            "count"),
        repeat_action_ratio  =("is_repeat",         "mean"),
        unique_action_types  =("action",            "nunique"),
        suspicious_seq_count =("is_suspicious_seq", "sum"),
        suspicious_seq_ratio =("is_suspicious_seq", "mean"),
    ).reset_index()

    return seq_feat


# ============================================================
# 5. 交叉特徵
# ============================================================

def build_cross_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    建立跨模組的交叉特徵，補強複合風險訊號。

    為什麼需要交叉特徵？
    單一特徵的鑑別力有限，但多個特徵的組合往往能捕捉更複雜的風險模式。
    例如：「深夜交易」本身不一定是風險，但「深夜 + 大金額 + TRC20」的組合就很可疑。

    所有交叉特徵都先檢查欄位是否存在（issubset），
    避免在某些情況下欄位不存在時報錯。
    """
    df = feature_df.copy()

    # 深夜 × 金額：深夜大額交易的複合風險訊號
    if {"trade_night_ratio", "trade_total_amount"}.issubset(df.columns):
        df["trade_night_amount_cross"] = df["trade_night_ratio"] * df["trade_total_amount"]
    if {"crypto_night_ratio", "crypto_total_twd_value"}.issubset(df.columns):
        df["crypto_night_value_cross"] = df["crypto_night_ratio"] * df["crypto_total_twd_value"]

    # KYC 完成時間 × 行為密度
    # 邏輯：KYC 完成越快（lvl2_minus_confirm_sec 越小）+ 交易越多 → 風險越高
    # 用 log1p 壓縮，避免極端值主導
    if {"lvl2_minus_confirm_sec", "trade_count"}.issubset(df.columns):
        df["kyc_trade_cross"]  = np.log1p(df["lvl2_minus_confirm_sec"].clip(lower=0)) * np.log1p(df["trade_count"])
    if {"lvl2_minus_confirm_sec", "crypto_txn_count"}.issubset(df.columns):
        df["kyc_crypto_cross"] = np.log1p(df["lvl2_minus_confirm_sec"].clip(lower=0)) * np.log1p(df["crypto_txn_count"])

    # 法幣 / 虛幣資金不對稱
    # twd_crypto_value_gap 接近 0：台幣入金幾乎全部換成虛幣（快進快出）
    if {"twd_total_amount", "crypto_total_twd_value"}.issubset(df.columns):
        df["twd_crypto_value_gap"]   = df["twd_total_amount"] - df["crypto_total_twd_value"]
        df["twd_crypto_value_ratio"] = safe_divide(df["crypto_total_twd_value"], df["twd_total_amount"])

    # 行為多樣性綜合分數：使用的交易類型數 + 協議多樣性 + 下單來源多樣性
    diversity_cols = [c for c in ["txn_type_diversity", "crypto_unique_protocol", "trade_source_nunique"] if c in df.columns]
    if diversity_cols:
        df["behavior_diversity_score"] = df[diversity_cols].fillna(0).sum(axis=1)

    # TRC20/BSC 鏈別交叉特徵
    # trc20_night_cross：TRC20 比例 × 深夜比例，兩者都高才觸發
    if {"trc20_ratio", "crypto_night_ratio"}.issubset(df.columns):
        df["trc20_night_cross"] = df["trc20_ratio"] * df["crypto_night_ratio"]
    if {"bsc_ratio", "crypto_night_ratio"}.issubset(df.columns):
        df["bsc_night_cross"]   = df["bsc_ratio"]   * df["crypto_night_ratio"]

    # 協議切換 × 最短交易間隔：頻繁切換鏈 + 高頻操作的複合訊號
    # clip(lower=1) 避免 gap_min_sec=0 造成除以零
    if {"protocol_switch_rate", "crypto_gap_min_sec"}.issubset(df.columns):
        df["protocol_switch_fast_cross"] = df["protocol_switch_rate"] * (
            1 / (df["crypto_gap_min_sec"].clip(lower=1))
        )

    # 地址重用 × 共用 IP：地址重用 + 共用 IP 的複合風險
    # 可能是同一批人操控多帳號，且都提到同一個地址
    if {"trc20_addr_reuse_rate", "shared_ip_ratio"}.issubset(df.columns):
        df["trc20_addr_ip_cross"] = df["trc20_addr_reuse_rate"] * df["shared_ip_ratio"]
    if {"bsc_addr_reuse_rate", "shared_ip_ratio"}.issubset(df.columns):
        df["bsc_addr_ip_cross"]   = df["bsc_addr_reuse_rate"]   * df["shared_ip_ratio"]

    # 可疑序列 × 快進快出旗標：兩個快進快出訊號同時出現
    if {"suspicious_seq_ratio", "is_fast_twd_to_crypto"}.issubset(df.columns):
        df["suspicious_fast_cross"] = df["suspicious_seq_ratio"] * df["is_fast_twd_to_crypto"]

    # 出入金比 × 活躍時長：短時間內大量出金（快進快出的量化）
    if {"twd_out_in_amount_ratio", "twd_active_span_sec"}.issubset(df.columns):
        # 活躍時長越短 + 出入金比越高 → 風險越高
        # 用 1/(span+1) 讓短時間的值更大
        df["twd_fast_outflow_cross"] = df["twd_out_in_amount_ratio"] * (
            1 / (df["twd_active_span_sec"].clip(lower=1))
        )

    return df


# ============================================================
# 6. IsolationForest 異常分數
# ============================================================

def add_iforest_score(
    train_feature: pd.DataFrame,
    test_feature: pd.DataFrame,
    drop_cols: list,
) -> tuple:
    """
    使用 IsolationForest 計算非監督式異常分數。

    修正說明（相較於 feature_test_1.py）：
    原版只用 train_df 做 fit，再分別 transform train/test。
    但這樣 test 的分數分布可能與 train 不一致（因為 fit 時沒看到 test 的分布）。

    改進做法：
    1. 合併 train + test 的特徵矩陣做 fit（讓模型看到完整分布）
    2. 再分別 transform train 和 test
    3. 這樣 train/test 的 iforest_score 分布更一致，模型訓練更穩定

    注意：fit 時不能用 status 欄位（標籤洩漏），drop_cols 需包含 status。
    """
    # 取出數值欄位，排除 ID 和標籤
    num_cols = [
        c for c in train_feature.select_dtypes(include=[np.number]).columns
        if c not in drop_cols
    ]

    # 合併 train + test 做 fit（確保分布一致）
    X_train = train_feature[num_cols].replace([np.inf, -np.inf], 0).fillna(0)
    X_test  = test_feature[num_cols].replace([np.inf, -np.inf], 0).fillna(0)
    X_all   = pd.concat([X_train, X_test], ignore_index=True)

    iso = IsolationForest(
        n_estimators=200,    # 200 棵樹，增加穩定性
        contamination=0.03,  # 預期約 3% 的異常比例（人頭戶比例估計）
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_all)  # 用 train+test 合併後的資料 fit，確保分布一致

    # 負號轉換：score_samples 預設「越高越正常」，取負後「越高越異常」
    train_feature = train_feature.copy()
    test_feature  = test_feature.copy()
    train_feature["iforest_score"] = -iso.score_samples(X_train)
    test_feature["iforest_score"]  = -iso.score_samples(X_test)

    print("IsolationForest score added (fit on train+test combined)")
    return train_feature, test_feature


# ============================================================
# 7. 主流程
# ============================================================

def build_feature_dataset() -> tuple:
    """
    完整特徵工程主流程：抓資料 → 清洗 → 建特徵 → 合併 → IsolationForest → 輸出。

    流程說明（共 9 步）：
    1. 分頁抓取所有原始資料表
    2. 偵測欄位名稱（相容脫敏版 API）
    3. 清洗各資料表，建立中間欄位
    4. 建立各渠道的基礎聚合特徵
    5. 建立進階特徵（網路、錢包、IP、時間、金額、資金流、序列）
    6. 合併所有特徵到 user_info 為基底的寬表
    7. 補缺值（數值欄位填 0，datetime 欄位保留）
    8. 切出 train / test，加入 IsolationForest 異常分數
    9. 輸出 CSV

    回傳：(train_feature, test_feature, feature_df)
    """

    print("\n[1/9] 抓取資料...")
    user_info        = fetch_table_paginated("user_info")
    twd_transfer     = fetch_table_paginated("twd_transfer")
    crypto_transfer  = fetch_table_paginated("crypto_transfer")
    usdt_twd_trading = fetch_table_paginated("usdt_twd_trading")
    usdt_swap        = fetch_table_paginated("usdt_swap")
    train_label      = fetch_table_paginated("train_label")
    predict_label    = fetch_table_paginated("predict_label")

    print("\n[2/9] 欄位相容偵測...")
    twd_ip_col      = get_existing_col(twd_transfer,     ["source_ip_hash", "source_ip"])
    crypto_ip_col   = get_existing_col(crypto_transfer,  ["source_ip_hash", "source_ip"])
    trade_ip_col    = get_existing_col(usdt_twd_trading, ["source_ip_hash", "source_ip"])
    from_wallet_col = get_existing_col(crypto_transfer,  ["from_wallet_hash", "from_wallet"])
    to_wallet_col   = get_existing_col(crypto_transfer,  ["to_wallet_hash",   "to_wallet"])

    print(f"  twd_ip_col      : {twd_ip_col}")
    print(f"  crypto_ip_col   : {crypto_ip_col}")
    print(f"  trade_ip_col    : {trade_ip_col}")
    print(f"  from_wallet_col : {from_wallet_col}")
    print(f"  to_wallet_col   : {to_wallet_col}")

    print("\n[3/9] 資料清洗...")
    user_info        = prepare_user_info(user_info)
    twd_transfer     = prepare_twd_transfer(twd_transfer, twd_ip_col)
    crypto_transfer  = prepare_crypto_transfer(crypto_transfer, crypto_ip_col, from_wallet_col, to_wallet_col)
    usdt_twd_trading = prepare_trade(usdt_twd_trading, trade_ip_col)
    usdt_swap        = prepare_swap(usdt_swap)

    print("\n[4/9] 建立基礎聚合特徵...")
    twd_feat    = build_twd_features(twd_transfer)
    crypto_feat = build_crypto_features(crypto_transfer)
    trade_feat  = build_trade_features(usdt_twd_trading)
    swap_feat   = build_swap_features(usdt_swap)

    print("\n[5/9] 建立進階特徵...")
    network_feat     = build_network_features(crypto_transfer)
    wallet_risk_feat = analyze_wallet_risk(crypto_transfer)
    ip_feat          = analyze_ip_patterns(twd_transfer, crypto_transfer, usdt_twd_trading)
    temporal_feat    = extract_temporal_anomalies(twd_transfer, crypto_transfer, usdt_twd_trading, usdt_swap)
    amount_feat      = detect_amount_anomalies(twd_transfer, crypto_transfer, usdt_twd_trading, usdt_swap)
    flow_feat        = calculate_fund_flow_patterns(twd_transfer, crypto_transfer, usdt_twd_trading, usdt_swap)
    seq_feat         = extract_sequence_features(twd_transfer, crypto_transfer)

    print("\n[6/9] 合併所有特徵...")
    # 以 user_info 為基底（保留所有用戶），依序 left join 各特徵表
    feature_df = user_info.copy()

    for feat in [twd_feat, crypto_feat, trade_feat, swap_feat]:
        feature_df = feature_df.merge(feat, on="user_id", how="left")

    for feat in [network_feat, wallet_risk_feat, ip_feat, temporal_feat, amount_feat, flow_feat, seq_feat]:
        feature_df = feature_df.merge(feat, on="user_id", how="left")

    # 建立交叉特徵（在合併後的完整特徵表上計算）
    feature_df = build_cross_features(feature_df)

    print(f"  feature_df shape before fillna: {feature_df.shape}")

    print("\n[7/9] 補缺值...")
    # datetime 欄位保留 NaT（不填 0，避免時間計算出錯）
    # 其他欄位（數值）填 0（代表該行為未發生）
    datetime_cols = feature_df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    for col in feature_df.columns:
        if col not in datetime_cols:
            feature_df[col] = feature_df[col].fillna(0)

    print(f"  feature_df shape after fillna: {feature_df.shape}")

    print("\n[8/9] 切出 train / test，加入 IsolationForest 異常分數...")
    train_feature = train_label.merge(feature_df, on="user_id", how="left")
    test_feature  = predict_label.merge(feature_df, on="user_id", how="left")

    # 補缺值（merge 後可能有少數 NaN）
    for col in train_feature.columns:
        if col not in datetime_cols:
            train_feature[col] = train_feature[col].fillna(0)
    for col in test_feature.columns:
        if col not in datetime_cols:
            test_feature[col] = test_feature[col].fillna(0)

    # IsolationForest：用 train+test 合併後的特徵矩陣 fit，再分別 transform
    # drop_cols 排除 user_id（ID）和 status（標籤，避免洩漏）
    iforest_drop = ["user_id", "status"]
    train_feature, test_feature = add_iforest_score(train_feature, test_feature, iforest_drop)

    print("\n[9/9] 輸出 CSV...")
    # 同時把 iforest_score 回寫到 feature_df（供後續分析用）
    feature_df = feature_df.merge(
        train_feature[["user_id", "iforest_score"]],
        on="user_id", how="left"
    )

    train_feature.to_csv("train_feature.csv", index=False)
    test_feature.to_csv("test_feature.csv",   index=False)
    feature_df.to_csv("feature_full.csv",     index=False)

    print(f"  train_feature shape : {train_feature.shape}")
    print(f"  test_feature  shape : {test_feature.shape}")
    print(f"  feature_df    shape : {feature_df.shape}")
    print("  已輸出：train_feature_v2.csv / test_feature_v2.csv / feature_full_v2.csv")

    return train_feature, test_feature, feature_df


if __name__ == "__main__":
    build_feature_dataset()
