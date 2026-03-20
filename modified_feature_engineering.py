# -*- coding: utf-8 -*-
"""
modified_feature_engineering.py

用途：
1. 依照 feature_test_1 的完整架構建立使用者層級特徵矩陣
2. 保留原本 notebook 的核心設計：
   - user_info / twd_transfer / crypto_transfer / usdt_twd_trading / usdt_swap
   - 基礎聚合、gap、活躍天、network、wallet risk、IP、temporal、amount、flow、sequence、cross features
3. 額外補強你前面提到的 test5 鏈別風險特徵：
   - TRC20 / BSC 比例
   - protocol switch rate
   - 協議內夜間比例
   - 協議內地址重用率
   - 協議流向不平衡

輸出：
- train_feature_modified.csv
- test_feature_modified.csv
- feature_full_modified.csv

注意：
- 這支程式是「特徵工程」腳本，不含模型訓練。
- 若 API 欄位為脫敏版本（source_ip_hash / from_wallet_hash / to_wallet_hash），程式會自動適配。
- 金額縮放依照 notebook：原始 samount 與 srate 乘上 1e-8 轉換為實際數值。
"""

import time
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 200)


# =========================
# 1. 通用函式
# =========================

def fetch_table_paginated(name: str, batch_size: int = 50000, sleep_sec: float = 0.15) -> pd.DataFrame:
    """
    分頁抓取 API 資料，直到回傳空資料為止。

    為什麼要分頁？
    API 單次回傳有筆數上限，若資料量大（如 crypto_transfer 可能有數百萬筆），
    一次性抓取會 timeout 或 OOM，分頁可以穩定地把全量資料拉回來。

    參數：
        name       : API endpoint 名稱（如 "user_info", "twd_transfer"）
        batch_size : 每次抓取的筆數，預設 50000
        sleep_sec  : 每次請求後的等待秒數，避免打爆 API rate limit
    """
    all_dfs = []
    offset = 0

    while True:
        url = f"https://aws-event-api.bitopro.com/{name}?limit={batch_size}&offset={offset}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()  # 若 HTTP 狀態碼非 2xx，直接拋出例外
        data = r.json()

        # 回傳空陣列代表已抓完所有資料
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


def get_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    從多個候選欄位名稱中，找出第一個實際存在於 df 的欄位。

    用途：相容脫敏版 API（欄位名稱可能是 source_ip 或 source_ip_hash），
    讓程式不需要寫死欄位名稱，自動適配不同版本的資料。

    回傳：找到的欄位名稱，若都不存在則回傳 None。
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_divide(a, b):
    """
    安全除法，分母加上極小值 1e-9 避免除以零。

    為什麼不用 np.where？
    這個函式設計給 pandas Series 和純量都能用，
    加 1e-9 比 np.where 更簡潔，且對比率特徵影響可忽略不計。
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
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
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

    產出欄位（以 prefix 為前綴）：
    - gap_mean_sec      : 平均間隔秒數
    - gap_std_sec       : 間隔標準差（越小代表越規律）
    - gap_min_sec       : 最短間隔（偵測高頻操作）
    - gap_max_sec       : 最長間隔
    - gap_lt_5min_ratio : 間隔 < 5 分鐘的比例（高頻旗標）
    - gap_lt_1hr_ratio  : 間隔 < 1 小時的比例
    - burstiness        : (std - mean) / (std + mean)，越接近 1 代表越突發性
    """
    tmp = df[[user_col, time_col]].dropna().sort_values([user_col, time_col]).copy()
    # diff() 計算同一 user 相鄰兩筆的時間差，第一筆會是 NaT
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

    # Burstiness index：來自 Goh & Barabási (2008)
    # 值域 [-1, 1]，越高代表越突發（短時間爆量後長時間靜止）
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


# =========================
# 2. 清洗資料
# =========================

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
    反之，正常用戶可能隔幾天才完成 Level 2。
    """
    user_info = user_info.copy()

    # 將三個 KYC 時間欄位統一轉為 datetime，errors="coerce" 讓無效值變 NaT 而非報錯
    for col in ["confirmed_at", "level1_finished_at", "level2_finished_at"]:
        if col in user_info.columns:
            user_info[col] = pd.to_datetime(user_info[col], errors="coerce")

    # KYC 完成旗標：1=已完成，0=未完成（NaT）
    user_info["has_confirmed"] = user_info["confirmed_at"].notna().astype(int)
    user_info["has_level1"]    = user_info["level1_finished_at"].notna().astype(int)
    user_info["has_level2"]    = user_info["level2_finished_at"].notna().astype(int)

    # KYC 各階段時間差（秒）
    # 負值代表時間順序異常（如 level1 早於 confirmed_at），可作為異常旗標
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


def prepare_twd_transfer(twd_transfer: pd.DataFrame, twd_ip_col: str | None) -> pd.DataFrame:
    """
    清洗法幣（台幣）出入金資料。

    金額縮放說明：
    ori_samount 是整數格式，需乘以 1e-8 才是實際台幣金額。
    例如：ori_samount = 998500000000 → 實際金額 = 9985 TWD

    欄位說明：
    - kind == 0 : 加值（入金，台幣匯入）
    - kind == 1 : 提領（出金，台幣匯出）
    - ip_for_feat : 統一命名的 IP 欄位，相容脫敏版（source_ip_hash）
                    注意：外部匯入加值時 source_ip 為空值（schema 說明）
    """
    twd_transfer = twd_transfer.copy()
    twd_transfer = add_time_cols(twd_transfer, "created_at")

    # 金額轉換：整數 → 實際台幣金額
    twd_transfer["amount"]     = twd_transfer["ori_samount"] * 1e-8
    twd_transfer["amount_log"] = np.log1p(twd_transfer["amount"])  # log 壓縮，降低極端值影響

    # 交易方向旗標
    twd_transfer["is_deposit"]  = (twd_transfer["kind"] == 0).astype(int)
    twd_transfer["is_withdraw"] = (twd_transfer["kind"] == 1).astype(int)

    # 統一 IP 欄位名稱（若無 IP 欄位則填 NaN，後續特徵計算會自動忽略）
    twd_transfer["ip_for_feat"] = twd_transfer[twd_ip_col] if twd_ip_col else np.nan

    return twd_transfer


def prepare_crypto_transfer(
    crypto_transfer: pd.DataFrame,
    crypto_ip_col: str | None,
    from_wallet_col: str | None,
    to_wallet_col: str | None,
) -> pd.DataFrame:
    """
    清洗虛擬貨幣轉帳資料，並補上鏈別風險特徵所需欄位。

    金額計算說明：
    - crypto_amount = ori_samount * 1e-8  （虛幣數量）
    - twd_rate      = twd_srate * 1e-8    （對台幣匯率）
    - twd_value     = crypto_amount * twd_rate  （換算台幣價值，跨幣種比較用）

    sub_kind 說明：
    - 0 = 外部（鏈上交易，有 from/to wallet 地址）
    - 1 = 內部（交易所內轉，relation_user_id 有值，wallet 地址通常為空）

    protocol 說明（依 schema）：
    - 0: Self（內部）, 1: ERC20, 2: OMNI, 3: BNB, 4: TRC20, 5: BSC, 6: Polygon
    TRC20 和 BSC 是常見洗錢鏈，因手續費低、匿名性較高。

    鏈別切換（is_protocol_switch）：
    同一用戶在相鄰兩筆交易中使用不同協議，
    頻繁切換可能代表刻意規避追蹤。
    """
    crypto_transfer = crypto_transfer.copy()
    crypto_transfer = add_time_cols(crypto_transfer, "created_at")

    # 金額轉換
    crypto_transfer["crypto_amount"]     = crypto_transfer["ori_samount"] * 1e-8
    crypto_transfer["twd_rate"]          = crypto_transfer["twd_srate"] * 1e-8
    crypto_transfer["twd_value"]         = crypto_transfer["crypto_amount"] * crypto_transfer["twd_rate"]
    crypto_transfer["crypto_amount_log"] = np.log1p(crypto_transfer["crypto_amount"])
    crypto_transfer["twd_value_log"]     = np.log1p(crypto_transfer["twd_value"])

    # 交易方向與類型旗標
    crypto_transfer["is_deposit"]        = (crypto_transfer["kind"] == 0).astype(int)
    crypto_transfer["is_withdraw"]       = (crypto_transfer["kind"] == 1).astype(int)
    crypto_transfer["is_external"]       = (crypto_transfer["sub_kind"] == 0).astype(int)  # 鏈上交易
    crypto_transfer["is_internal"]       = (crypto_transfer["sub_kind"] == 1).astype(int)  # 站內轉帳
    crypto_transfer["has_relation_user"] = crypto_transfer["relation_user_id"].notna().astype(int)

    # 統一欄位名稱（相容脫敏版 API）
    crypto_transfer["ip_for_feat"]       = crypto_transfer[crypto_ip_col]    if crypto_ip_col    else np.nan
    crypto_transfer["from_wallet_feat"]  = crypto_transfer[from_wallet_col]  if from_wallet_col  else np.nan
    crypto_transfer["to_wallet_feat"]    = crypto_transfer[to_wallet_col]    if to_wallet_col    else np.nan

    # 錢包存在旗標（外部交易才有地址，內部轉帳通常為空）
    crypto_transfer["has_from_wallet"] = crypto_transfer["from_wallet_feat"].notna().astype(int)
    crypto_transfer["has_to_wallet"]   = crypto_transfer["to_wallet_feat"].notna().astype(int)

    # 鏈別旗標（TRC20=4, BSC=5，依 schema 定義）
    crypto_transfer["is_trc20"] = (crypto_transfer["protocol"] == 4).astype(int)
    crypto_transfer["is_bsc"]   = (crypto_transfer["protocol"] == 5).astype(int)

    # 鏈別切換旗標：同一用戶相鄰兩筆使用不同協議
    # 先按 user_id + created_at 排序，確保 shift(1) 取到的是時間上的前一筆
    crypto_transfer = crypto_transfer.sort_values(["user_id", "created_at"]).copy()
    crypto_transfer["prev_protocol"]      = crypto_transfer.groupby("user_id")["protocol"].shift(1)
    crypto_transfer["is_protocol_switch"] = (
        crypto_transfer["protocol"].ne(crypto_transfer["prev_protocol"])
        & crypto_transfer["prev_protocol"].notna()  # 第一筆沒有前一筆，排除
    ).astype(int)

    # 小額交易旗標：台幣價值 <= 1000 元視為小額
    # 人頭戶常先做小額測試轉帳，確認地址可用後再大額轉出
    crypto_transfer["is_small_twd_value"] = (crypto_transfer["twd_value"] <= 1000).astype(int)

    # 自轉移旗標：來源地址 == 目標地址
    # 注意：內部轉帳（sub_kind=1）的 wallet 欄位通常為空，此旗標主要對外部交易有意義
    crypto_transfer["is_self_wallet_transfer"] = (
        crypto_transfer["from_wallet_feat"].notna()
        & crypto_transfer["to_wallet_feat"].notna()
        & crypto_transfer["from_wallet_feat"].eq(crypto_transfer["to_wallet_feat"])
    ).astype(int)

    return crypto_transfer


def prepare_trade(usdt_twd_trading: pd.DataFrame, trade_ip_col: str | None) -> pd.DataFrame:
    """
    清洗撮合交易（掛單簿）資料。

    欄位說明：
    - trade_samount : 已成交數量（USDT），乘以 1e-8 得實際數量
    - twd_srate     : 成交匯率（USDT/TWD），乘以 1e-8 得實際匯率
    - is_buy == 1   : 買單（用 TWD 買 USDT）
    - is_buy == 0   : 賣單（把 USDT 賣成 TWD）
    - is_market     : 市價單（1）vs 限價單（0）
    - source        : 下單來源（0=WEB, 1=APP, 2=API）

    注意：撮合交易的時間欄位是 updated_at（訂單完成時間），
    不是 created_at，計算 gap 時要用 updated_at。
    """
    usdt_twd_trading = usdt_twd_trading.copy()
    usdt_twd_trading = add_time_cols(usdt_twd_trading, "updated_at")

    # 金額與價格轉換
    usdt_twd_trading["amount"]     = usdt_twd_trading["trade_samount"] * 1e-8
    usdt_twd_trading["price"]      = usdt_twd_trading["twd_srate"] * 1e-8
    usdt_twd_trading["amount_log"] = np.log1p(usdt_twd_trading["amount"])
    usdt_twd_trading["price_log"]  = np.log1p(usdt_twd_trading["price"])

    # 賣單旗標（is_buy 的反向）
    usdt_twd_trading["is_sell"]      = (usdt_twd_trading["is_buy"] == 0).astype(int)
    usdt_twd_trading["ip_for_feat"]  = usdt_twd_trading[trade_ip_col] if trade_ip_col else np.nan

    return usdt_twd_trading


def prepare_swap(usdt_swap: pd.DataFrame) -> pd.DataFrame:
    """
    清洗一鍵買賣（swap）資料。

    Swap vs 撮合交易的差異：
    - Swap 是「一鍵買賣」，由平台直接報價成交，不進掛單簿
    - 撮合交易是掛單等待對手方成交
    - 人頭戶可能偏好 swap（速度快、不留掛單紀錄）

    欄位說明：
    - kind == 0 : 用戶買幣（用 TWD 買 USDT）
    - kind == 1 : 用戶賣幣（把 USDT 賣成 TWD）
    - twd_samount      : 成交台幣數量，乘以 1e-8
    - currency_samount : 成交虛幣數量，乘以 1e-8
    """
    usdt_swap = usdt_swap.copy()
    usdt_swap = add_time_cols(usdt_swap, "created_at")

    usdt_swap["twd_amount"]        = usdt_swap["twd_samount"] * 1e-8
    usdt_swap["crypto_amount"]     = usdt_swap["currency_samount"] * 1e-8
    usdt_swap["twd_amount_log"]    = np.log1p(usdt_swap["twd_amount"])
    usdt_swap["crypto_amount_log"] = np.log1p(usdt_swap["crypto_amount"])

    # 買賣方向旗標
    usdt_swap["is_buy_coin"]  = (usdt_swap["kind"] == 0).astype(int)
    usdt_swap["is_sell_coin"] = (usdt_swap["kind"] == 1).astype(int)

    return usdt_swap


# =========================
# 3. 基礎聚合特徵
# =========================

def build_twd_features(twd_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    建立台幣出入金的使用者層級聚合特徵。

    特徵設計邏輯：
    - 金額統計（sum/mean/max/std/median）：捕捉資金規模與分布
    - 入金/出金分開統計：計算出入金比例，人頭戶通常出金 >> 入金
    - 夜間/週末比例：異常時段操作
    - IP 多樣性：同一用戶使用多個 IP 是異常訊號
    - 時間熵：交易時間是否集中（低熵=高度規律=可能是程式操作）
    - 活躍天數 + gap 特徵：搭配筆數算密度
    - active_span_sec：從第一筆到最後一筆的總時間跨度
    """
    # schema 中 twd_transfer 沒有 id 欄位，用 kind 做 count（必定存在）
    twd_feat = twd_transfer.groupby("user_id").agg(
        twd_txn_count    =("kind",         "count"),
        twd_total_amount =("amount",       "sum"),
        twd_avg_amount   =("amount",       "mean"),
        twd_max_amount   =("amount",       "max"),
        twd_min_amount   =("amount",       "min"),
        twd_std_amount   =("amount",       "std"),
        twd_median_amount=("amount",       "median"),
        twd_deposit_count=("is_deposit",   "sum"),
        twd_withdraw_count=("is_withdraw", "sum"),
        twd_night_ratio  =("is_night",     "mean"),
        twd_weekend_ratio=("is_weekend",   "mean"),
        twd_unique_ip    =("ip_for_feat",  "nunique"),
        twd_hour_entropy =("hour",         calc_hour_entropy),
        twd_first_time   =("created_at",   "min"),
        twd_last_time    =("created_at",   "max"),
    ).reset_index()

    # 入金子集統計（只看 kind==0 的筆）
    twd_in = twd_transfer[twd_transfer["is_deposit"] == 1].groupby("user_id").agg(
        twd_in_total_amount=("amount", "sum"),
        twd_in_avg_amount  =("amount", "mean"),
        twd_in_max_amount  =("amount", "max"),
    ).reset_index()

    # 出金子集統計（只看 kind==1 的筆）
    twd_out = twd_transfer[twd_transfer["is_withdraw"] == 1].groupby("user_id").agg(
        twd_out_total_amount=("amount", "sum"),
        twd_out_avg_amount  =("amount", "mean"),
        twd_out_max_amount  =("amount", "max"),
    ).reset_index()

    twd_active_days = calc_active_day_features(twd_transfer, "user_id", "date_only", "twd")
    twd_gap_feat    = calc_gap_features(twd_transfer, "user_id", "created_at", "twd")

    for feat in [twd_in, twd_out, twd_active_days, twd_gap_feat]:
        twd_feat = twd_feat.merge(feat, on="user_id", how="left")

    # 衍生比率特徵
    twd_feat["twd_deposit_ratio"]          = safe_divide(twd_feat["twd_deposit_count"],    twd_feat["twd_txn_count"])
    twd_feat["twd_withdraw_ratio"]         = safe_divide(twd_feat["twd_withdraw_count"],   twd_feat["twd_txn_count"])
    # 出金次數 / 入金次數：> 1 代表出金比入金更頻繁
    twd_feat["twd_withdraw_deposit_ratio"] = safe_divide(twd_feat["twd_withdraw_count"],   twd_feat["twd_deposit_count"])
    # 出金金額 / 入金金額：接近 1 代表幾乎全部入金都被提走（快進快出）
    twd_feat["twd_out_in_amount_ratio"]    = safe_divide(twd_feat["twd_out_total_amount"], twd_feat["twd_in_total_amount"])
    # 每活躍天的平均交易筆數：高頻旗標
    twd_feat["twd_txn_per_active_day"]     = safe_divide(twd_feat["twd_txn_count"],        twd_feat["twd_active_days"])
    # 活躍時間跨度（秒）：跨度短但交易量大是異常訊號
    twd_feat["twd_active_span_sec"] = (
        twd_feat["twd_last_time"] - twd_feat["twd_first_time"]
    ).dt.total_seconds()

    return twd_feat


def build_crypto_features(crypto_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    建立虛擬貨幣轉帳的使用者層級聚合特徵，並整合鏈別風險欄位。

    特徵設計邏輯：
    - 金額統計：同時統計虛幣數量和台幣換算價值（跨幣種比較用 twd_value）
    - 外部/內部交易分開統計：外部（鏈上）比例高代表資金流出平台
    - 關聯用戶：內轉對象的多樣性（network 特徵的前置）
    - 幣種/協議多樣性：使用越多種幣種/鏈，行為越複雜
    - 錢包多樣性：to_wallet 越多代表資金分散到多個地址
    - 鏈別風險特徵（TRC20/BSC）：見 build_protocol_detail 說明
    """
    # schema 中 crypto_transfer 有 user_id，用 kind 做 count
    crypto_feat = crypto_transfer.groupby("user_id").agg(
        crypto_txn_count          =("kind",              "count"),
        crypto_total_amount       =("crypto_amount",     "sum"),
        crypto_avg_amount         =("crypto_amount",     "mean"),
        crypto_max_amount         =("crypto_amount",     "max"),
        crypto_std_amount         =("crypto_amount",     "std"),
        crypto_total_twd_value    =("twd_value",         "sum"),
        crypto_avg_twd_value      =("twd_value",         "mean"),
        crypto_max_twd_value      =("twd_value",         "max"),
        crypto_std_twd_value      =("twd_value",         "std"),
        crypto_deposit_count      =("is_deposit",        "sum"),
        crypto_withdraw_count     =("is_withdraw",       "sum"),
        crypto_external_count     =("is_external",       "sum"),
        crypto_internal_count     =("is_internal",       "sum"),
        crypto_relation_count     =("has_relation_user", "sum"),
        crypto_unique_currency    =("currency",          "nunique"),
        crypto_unique_protocol    =("protocol",          "nunique"),
        crypto_unique_relation_user=("relation_user_id", "nunique"),
        crypto_unique_ip          =("ip_for_feat",       "nunique"),
        crypto_unique_from_wallet =("from_wallet_feat",  "nunique"),
        crypto_unique_to_wallet   =("to_wallet_feat",    "nunique"),
        crypto_night_ratio        =("is_night",          "mean"),
        crypto_weekend_ratio      =("is_weekend",        "mean"),
        crypto_hour_entropy       =("hour",              calc_hour_entropy),
        crypto_first_time         =("created_at",        "min"),
        crypto_last_time          =("created_at",        "max"),
    ).reset_index()

    # 入金/出金子集（台幣換算價值）
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

    # 協議層級彙總特徵
    # 注意：trc20_night_ratio / bsc_night_ratio 由下方 protocol_detail_feat 計算，
    # 這裡不占位，避免 merge 後產生 _x/_y 重複欄位
    protocol_feat = crypto_transfer.groupby("user_id").agg(
        trc20_tx_count                   =("is_trc20",                  "sum"),
        bsc_tx_count                     =("is_bsc",                    "sum"),
        protocol_switch_count            =("is_protocol_switch",        "sum"),
        protocol_switch_rate             =("is_protocol_switch",        "mean"),  # 切換頻率
        crypto_small_twd_ratio           =("is_small_twd_value",        "mean"),  # 小額交易比例
        crypto_self_wallet_transfer_count=("is_self_wallet_transfer",   "sum"),
        crypto_self_wallet_transfer_ratio=("is_self_wallet_transfer",   "mean"),
    ).reset_index()

    def build_protocol_detail(g: pd.DataFrame) -> pd.Series:
        """
        針對單一用戶，計算 TRC20 和 BSC 的細粒度風險特徵。

        為什麼要分鏈計算？
        TRC20 和 BSC 是常見洗錢鏈（手續費低、速度快、較難追蹤），
        若用戶大量使用這兩條鏈，且集中在夜間、地址重用率高，
        複合風險訊號會比單一特徵更強。

        地址重用率（reuse_rate）：
        1 - (唯一地址數 / 總交易筆數)
        越接近 1 代表一直用同一個地址，可能是固定的洗錢地址。

        流向不平衡（in_out_imbalance）：
        |入金筆數 - 出金筆數| / 總筆數
        越高代表該鏈的資金流向高度單向（只進或只出）。
        """
        total = len(g)
        trc   = g[g["is_trc20"] == 1]
        bsc   = g[g["is_bsc"]   == 1]

        def reuse_rate(s: pd.Series) -> float:
            """地址重用率：1 - 唯一地址數/總筆數，越高代表越常重用同一地址"""
            s = s.dropna()
            if len(s) == 0:
                return 0.0
            return float(1 - s.nunique() / (len(s) + 1e-9))

        return pd.Series({
            # 各鏈佔總交易的比例
            "trc20_ratio"            : safe_divide(len(trc), total),
            "bsc_ratio"              : safe_divide(len(bsc), total),
            # 各鏈的夜間/週末比例（深夜 TRC20 交易是高風險訊號）
            "trc20_night_ratio"      : trc["is_night"].mean()   if len(trc) > 0 else 0.0,
            "bsc_night_ratio"        : bsc["is_night"].mean()   if len(bsc)  > 0 else 0.0,
            "trc20_weekend_ratio"    : trc["is_weekend"].mean() if len(trc) > 0 else 0.0,
            "bsc_weekend_ratio"      : bsc["is_weekend"].mean() if len(bsc)  > 0 else 0.0,
            # 各鏈的唯一目標地址數（越少代表資金集中流向固定地址）
            "trc20_unique_to_wallet" : trc["to_wallet_feat"].nunique(dropna=True),
            "bsc_unique_to_wallet"   : bsc["to_wallet_feat"].nunique(dropna=True),
            # 目標地址重用率（to_wallet）
            "trc20_addr_reuse_rate"      : reuse_rate(trc["to_wallet_feat"]),
            "bsc_addr_reuse_rate"        : reuse_rate(bsc["to_wallet_feat"]),
            # 來源地址重用率（from_wallet）
            "trc20_from_addr_reuse_rate" : reuse_rate(trc["from_wallet_feat"]),
            "bsc_from_addr_reuse_rate"   : reuse_rate(bsc["from_wallet_feat"]),
            # 各鏈的入金/出金比例
            "trc20_inflow_ratio"     : trc["is_deposit"].mean()  if len(trc) > 0 else 0.0,
            "trc20_outflow_ratio"    : trc["is_withdraw"].mean() if len(trc) > 0 else 0.0,
            "bsc_inflow_ratio"       : bsc["is_deposit"].mean()  if len(bsc)  > 0 else 0.0,
            "bsc_outflow_ratio"      : bsc["is_withdraw"].mean() if len(bsc)  > 0 else 0.0,
            # 流向不平衡：|入金筆數 - 出金筆數| / 總筆數
            "trc20_in_out_imbalance" : abs(trc["is_deposit"].sum() - trc["is_withdraw"].sum()) / (len(trc) + 1e-9),
            "bsc_in_out_imbalance"   : abs(bsc["is_deposit"].sum() - bsc["is_withdraw"].sum()) / (len(bsc)  + 1e-9),
        })

    # groupby.apply 對每個 user_id 執行 build_protocol_detail，回傳 DataFrame
    protocol_detail_feat = crypto_transfer.groupby("user_id").apply(build_protocol_detail).reset_index()

    # 依序 merge 所有子特徵表（left join 保留所有用戶）
    for feat in [crypto_in, crypto_out, crypto_active_days, crypto_gap_feat, protocol_feat, protocol_detail_feat]:
        crypto_feat = crypto_feat.merge(feat, on="user_id", how="left")

    # 衍生比率特徵
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

    特徵設計邏輯：
    - 買/賣比例：人頭戶通常只做單向操作（只買或只賣）
    - 市價單比例：市價單不在乎成交價，急於成交是異常訊號
    - source 多樣性：同時用 WEB/APP/API 下單，可能是程式化操作
    - IP 多樣性：多 IP 下單是異常訊號
    - 每 IP 的交易筆數：單一 IP 大量下單
    """
    # schema 中 usdt_twd_trading 沒有 id 欄位，用 is_buy 做 count（必定存在）
    trade_feat = usdt_twd_trading.groupby("user_id").agg(
        trade_count         =("is_buy",       "count"),
        trade_total_amount  =("amount",       "sum"),
        trade_avg_amount    =("amount",       "mean"),
        trade_max_amount    =("amount",       "max"),
        trade_std_amount    =("amount",       "std"),
        trade_avg_price     =("price",        "mean"),
        trade_max_price     =("price",        "max"),
        trade_std_price     =("price",        "std"),
        trade_buy_count     =("is_buy",       "sum"),
        trade_sell_count    =("is_sell",      "sum"),
        trade_market_count  =("is_market",    "sum"),   # 市價單筆數
        trade_unique_ip     =("ip_for_feat",  "nunique"),
        trade_source_nunique=("source",       "nunique"),  # 下單來源多樣性（WEB/APP/API）
        trade_night_ratio   =("is_night",     "mean"),
        trade_weekend_ratio =("is_weekend",   "mean"),
        trade_hour_entropy  =("hour",         calc_hour_entropy),
        trade_first_time    =("updated_at",   "min"),
        trade_last_time     =("updated_at",   "max"),
    ).reset_index()

    trade_active_days = calc_active_day_features(usdt_twd_trading, "user_id", "date_only", "trade")
    trade_gap_feat    = calc_gap_features(usdt_twd_trading, "user_id", "updated_at", "trade")

    for feat in [trade_active_days, trade_gap_feat]:
        trade_feat = trade_feat.merge(feat, on="user_id", how="left")

    trade_feat["trade_buy_ratio"]          = safe_divide(trade_feat["trade_buy_count"],    trade_feat["trade_count"])
    trade_feat["trade_sell_ratio"]         = safe_divide(trade_feat["trade_sell_count"],   trade_feat["trade_count"])
    trade_feat["trade_market_ratio"]       = safe_divide(trade_feat["trade_market_count"], trade_feat["trade_count"])
    trade_feat["trade_txn_per_active_day"] = safe_divide(trade_feat["trade_count"],        trade_feat["trade_active_days"])
    # 每個 IP 的平均交易筆數：高代表單一 IP 大量下單
    trade_feat["trade_txn_per_ip"]         = safe_divide(trade_feat["trade_count"],        trade_feat["trade_unique_ip"])
    trade_feat["trade_active_span_sec"]    = (
        trade_feat["trade_last_time"] - trade_feat["trade_first_time"]
    ).dt.total_seconds()

    return trade_feat


def build_swap_features(usdt_swap: pd.DataFrame) -> pd.DataFrame:
    """
    建立一鍵買賣（swap）的使用者層級聚合特徵。

    Swap 的風險意義：
    - Swap 不留掛單紀錄，比撮合交易更難追蹤
    - swap_sell_buy_twd_ratio 接近 1：買進後幾乎全部賣出（快進快出）
    - swap 與 twd_transfer 搭配看：台幣入金 → swap 買幣 → 虛幣提領，是典型洗錢路徑
    """
    # schema 中 usdt_swap 沒有 id 欄位，用 kind 做 count（必定存在）
    swap_feat = usdt_swap.groupby("user_id").agg(
        swap_count       =("kind",          "count"),
        swap_total_twd   =("twd_amount",    "sum"),
        swap_avg_twd     =("twd_amount",    "mean"),
        swap_max_twd     =("twd_amount",    "max"),
        swap_std_twd     =("twd_amount",    "std"),
        swap_total_crypto=("crypto_amount", "sum"),
        swap_avg_crypto  =("crypto_amount", "mean"),
        swap_max_crypto  =("crypto_amount", "max"),
        swap_std_crypto  =("crypto_amount", "std"),
        swap_buy_count   =("is_buy_coin",   "sum"),
        swap_sell_count  =("is_sell_coin",  "sum"),
        swap_night_ratio =("is_night",      "mean"),
        swap_weekend_ratio=("is_weekend",   "mean"),
        swap_hour_entropy=("hour",          calc_hour_entropy),
        swap_first_time  =("created_at",    "min"),
        swap_last_time   =("created_at",    "max"),
    ).reset_index()

    # 買幣/賣幣子集的台幣金額統計
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
    # 賣幣台幣金額 / 買幣台幣金額：接近 1 代表買進後幾乎全部賣出
    swap_feat["swap_sell_buy_twd_ratio"] = safe_divide(swap_feat["swap_sell_total_twd"], swap_feat["swap_buy_total_twd"])
    swap_feat["swap_txn_per_active_day"] = safe_divide(swap_feat["swap_count"],          swap_feat["swap_active_days"])
    swap_feat["swap_active_span_sec"]    = (
        swap_feat["swap_last_time"] - swap_feat["swap_first_time"]
    ).dt.total_seconds()

    return swap_feat


# =========================
# 4. 進階特徵
# =========================

def build_network_features(crypto_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    利用 relation_user_id 建立使用者間的內轉網路特徵。

    網路特徵的意義：
    人頭戶通常是一個「資金中繼站」，會從多個來源接收資金（in_degree 高），
    再轉給少數幾個目標（out_degree 低），或反過來。
    這種不對稱的網路結構是重要的風險訊號。

    out_degree : 該用戶主動轉出給多少個不同用戶
    in_degree  : 有多少個不同用戶轉入給該用戶
    degree_imbalance : |out - in| / total，越高代表越不對稱
    """
    # 只取有 relation_user_id 的內轉紀錄
    tmp = crypto_transfer[crypto_transfer["has_relation_user"] == 1].copy()
    if len(tmp) == 0:
        return pd.DataFrame({"user_id": []})

    # 從「發起方」角度統計：我轉出給多少人
    network_out = tmp.groupby("user_id").agg(
        network_out_degree     =("relation_user_id", "nunique"),  # 轉出對象數
        network_out_txn_count  =("user_id",          "count"),    # 轉出筆數（用 user_id 代替不存在的 id）
        network_out_total_value=("twd_value",         "sum"),     # 轉出總台幣價值
    ).reset_index()

    # 從「接收方」角度統計：有多少人轉入給我
    network_in = tmp.groupby("relation_user_id").agg(
        network_in_degree     =("user_id",   "nunique"),  # 轉入來源數
        network_in_txn_count  =("user_id",   "count"),    # 轉入筆數
        network_in_total_value=("twd_value", "sum"),      # 轉入總台幣價值
    ).reset_index().rename(columns={"relation_user_id": "user_id"})

    # outer join：有些用戶只有轉出或只有轉入，都要保留
    network_feat = network_out.merge(network_in, on="user_id", how="outer").fillna(0)

    network_feat["network_total_degree"]        = network_feat["network_out_degree"] + network_feat["network_in_degree"]
    # 轉出/轉入對象比：> 1 代表轉出給更多人（資金分散），< 1 代表接收更多人的資金（資金匯聚）
    network_feat["network_out_in_degree_ratio"] = safe_divide(network_feat["network_out_degree"],       network_feat["network_in_degree"])
    network_feat["network_out_in_value_ratio"]  = safe_divide(network_feat["network_out_total_value"],  network_feat["network_in_total_value"])
    # 不對稱指數：越接近 1 代表網路結構越不對稱
    network_feat["network_degree_imbalance"]    = safe_divide(
        np.abs(network_feat["network_out_degree"] - network_feat["network_in_degree"]),
        network_feat["network_total_degree"] + 1,
    )
    return network_feat


def analyze_wallet_risk(crypto_transfer: pd.DataFrame) -> pd.DataFrame:
    """
    分析錢包地址的共用風險與地址重用率。

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

    # 統計每個目標地址被多少個不同用戶使用
    to_wallet_stats = tmp.groupby("to_wallet_feat").agg(
        wallet_user_count=("user_id", "nunique"),
        wallet_txn_count =("user_id", "count"),    # 用 user_id 代替不存在的 id
        wallet_total_value=("twd_value", "sum"),
    )
    from_wallet_stats = tmp.groupby("from_wallet_feat").agg(
        wallet_user_count=("user_id", "nunique"),
        wallet_txn_count =("user_id", "count"),    # 用 user_id 代替不存在的 id
    )

    # 高風險地址：被 >= 5 個不同用戶使用
    high_risk_to_wallets   = to_wallet_stats[to_wallet_stats["wallet_user_count"]   >= 5].index
    high_risk_from_wallets = from_wallet_stats[from_wallet_stats["wallet_user_count"] >= 5].index

    tmp["is_high_risk_to_wallet"]   = tmp["to_wallet_feat"].isin(high_risk_to_wallets).astype(int)
    tmp["is_high_risk_from_wallet"] = tmp["from_wallet_feat"].isin(high_risk_from_wallets).astype(int)

    wallet_risk_feat = tmp.groupby("user_id").agg(
        high_risk_to_wallet_count  =("is_high_risk_to_wallet",   "sum"),
        high_risk_from_wallet_count=("is_high_risk_from_wallet", "sum"),
        high_risk_to_wallet_ratio  =("is_high_risk_to_wallet",   "mean"),
        high_risk_from_wallet_ratio=("is_high_risk_from_wallet", "mean"),
        # 目標地址重用率：1 - 唯一地址數/總筆數
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

    為什麼要跨表合併 IP？
    同一個 IP 可能在不同交易類型中出現，
    跨表合併才能看到完整的 IP 使用行為。

    共用 IP 定義：同一個 IP 被 >= 3 個不同用戶使用，
    代表這個 IP 可能是「共用設備」或「代操帳號」的訊號。

    ip_jump_rate：唯一 IP 數 / 總 IP 使用次數
    越接近 1 代表每次都用不同 IP（可能在規避追蹤）；
    越接近 0 代表一直用同一個 IP（正常用戶的典型行為）。

    注意：usdt_swap 沒有 IP 欄位（schema 未提供），故不納入此分析。
    """
    all_ips = pd.concat([
        twd_transfer[["user_id", "ip_for_feat"]].assign(source="twd"),
        crypto_transfer[["user_id", "ip_for_feat"]].assign(source="crypto"),
        usdt_twd_trading[["user_id", "ip_for_feat"]].assign(source="trade"),
    ], ignore_index=True)

    # 移除 IP 為空的紀錄（外部匯入加值時 source_ip 為空，schema 有說明）
    all_ips = all_ips.dropna(subset=["ip_for_feat"])
    if len(all_ips) == 0:
        return pd.DataFrame({"user_id": []})

    # 找出共用 IP（被 >= 3 個不同用戶使用）
    ip_user_count = all_ips.groupby("ip_for_feat")["user_id"].nunique()
    shared_ips    = ip_user_count[ip_user_count >= 3].index
    all_ips["is_shared_ip"] = all_ips["ip_for_feat"].isin(shared_ips).astype(int)

    ip_feat = all_ips.groupby("user_id").agg(
        total_unique_ips    =("ip_for_feat", "nunique"),  # 使用過多少個不同 IP
        total_ip_usage_count=("ip_for_feat", "count"),    # 總 IP 使用次數
        shared_ip_count     =("is_shared_ip", "sum"),     # 使用共用 IP 的次數
        shared_ip_ratio     =("is_shared_ip", "mean"),    # 共用 IP 比例
        ip_source_diversity =("source",       "nunique"), # 在幾種交易類型中有 IP 紀錄
    ).reset_index()

    # IP 跳躍率：越高代表越常換 IP
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

    為什麼要跨渠道計算時間特徵？
    單一渠道的時間特徵可能不夠強，
    但若一個用戶在所有渠道都集中在深夜操作，
    複合訊號會更有說服力。

    avg_time_entropy：所有渠道時間熵的平均值
    min_time_entropy：所有渠道中最低的時間熵（最規律的渠道）

    注意：這裡的 _hour_entropy_2 和 _weekend_ratio_2 是為了避免
    與 build_*_features 中已有的同名欄位衝突而加後綴。
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
                f"{prefix}_hour_entropy_2" : ("hour",        calc_hour_entropy),  # 小時分布熵
                f"{prefix}_day_entropy"    : ("day_of_week", calc_hour_entropy),  # 星期分布熵
                f"{prefix}_midnight_ratio" : ("is_night",    "mean"),             # 深夜比例
                f"{prefix}_weekend_ratio_2": ("is_weekend",  "mean"),             # 週末比例
            }
        ).reset_index()
        temporal_feats.append(feat)

    # 用 reduce + outer merge 合併四個渠道的特徵，缺值填 0
    temporal_feat = reduce(
        lambda left, right: left.merge(right, on="user_id", how="outer"),
        temporal_feats
    ).fillna(0)

    # 跨渠道時間熵的彙總統計
    entropy_cols = [c for c in temporal_feat.columns if "entropy" in c]
    temporal_feat["avg_time_entropy"] = temporal_feat[entropy_cols].mean(axis=1)  # 平均熵
    temporal_feat["min_time_entropy"] = temporal_feat[entropy_cols].min(axis=1)   # 最低熵（最規律的渠道）
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
    人頭戶常使用整數金額（如 10000、50000），
    因為是被人指定金額操作，而非自然產生的交易行為。

    變異係數（CV = std / mean）：
    CV 低代表每次交易金額都差不多（規律性高），
    可能是程式化操作或被指定固定金額。

    偏態（skew）：
    正偏態高代表大多數交易金額小，但偶爾有極大金額，
    可能是「測試小額 + 一次大額轉出」的洗錢模式。
    """
    twd_amount_feat = twd_transfer.groupby("user_id").agg(
        twd_round_1000_ratio =("amount", lambda x: (x % 1000  == 0).mean()),  # 整千比例
        twd_round_10000_ratio=("amount", lambda x: (x % 10000 == 0).mean()),  # 整萬比例
        twd_amount_cv        =("amount", lambda x: x.std() / (x.mean() + 1e-9)),
        twd_amount_skew      =("amount", lambda x: x.skew() if len(x) > 2 else 0),
    ).reset_index()

    crypto_amount_feat = crypto_transfer.groupby("user_id").agg(
        crypto_twd_value_cv  =("twd_value", lambda x: x.std() / (x.mean() + 1e-9)),
        crypto_twd_value_skew=("twd_value", lambda x: x.skew() if len(x) > 2 else 0),
    ).reset_index()

    trade_amount_feat = usdt_twd_trading.groupby("user_id").agg(
        trade_amount_cv=("amount", lambda x: x.std() / (x.mean() + 1e-9)),
        trade_price_cv =("price",  lambda x: x.std() / (x.mean() + 1e-9)),  # 成交價格的穩定性
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

    txn_type_diversity：
    使用了幾種交易類型（twd/crypto/trade/swap），
    越多代表行為越複雜，但也可能是正常活躍用戶。
    """
    # 各渠道的最早/最晚交易時間
    twd_time   = twd_transfer.groupby("user_id")["created_at"].agg(["min", "max"]).reset_index()
    twd_time.columns = ["user_id", "twd_first", "twd_last"]

    crypto_time = crypto_transfer.groupby("user_id")["created_at"].agg(["min", "max"]).reset_index()
    crypto_time.columns = ["user_id", "crypto_first", "crypto_last"]

    # 注意：撮合交易用 updated_at（訂單完成時間），不是 created_at
    trade_time  = usdt_twd_trading.groupby("user_id")["updated_at"].agg(["min", "max"]).reset_index()
    trade_time.columns = ["user_id", "trade_first", "trade_last"]

    swap_time   = usdt_swap.groupby("user_id")["created_at"].agg(["min", "max"]).reset_index()
    swap_time.columns = ["user_id", "swap_first", "swap_last"]

    # outer join 合併四個渠道（有些用戶可能只有部分渠道的紀錄）
    flow_feat = twd_time.merge(crypto_time, on="user_id", how="outer")
    flow_feat = flow_feat.merge(trade_time, on="user_id", how="outer")
    flow_feat = flow_feat.merge(swap_time,  on="user_id", how="outer")

    # 跨渠道的整體活躍時間範圍
    flow_feat["overall_first_time"]     = flow_feat[["twd_first", "crypto_first", "trade_first", "swap_first"]].min(axis=1)
    flow_feat["overall_last_time"]      = flow_feat[["twd_last",  "crypto_last",  "trade_last",  "swap_last"]].max(axis=1)
    flow_feat["overall_active_span_sec"] = (
        flow_feat["overall_last_time"] - flow_feat["overall_first_time"]
    ).dt.total_seconds()

    # 台幣入金到虛幣/撮合交易的時間差（秒）
    # 正值代表先有台幣入金，再開始虛幣/撮合操作（符合洗錢流程）
    flow_feat["twd_to_crypto_gap_sec"] = (flow_feat["crypto_first"] - flow_feat["twd_first"]).dt.total_seconds()
    flow_feat["twd_to_trade_gap_sec"]  = (flow_feat["trade_first"]  - flow_feat["twd_first"]).dt.total_seconds()

    # 快進快出旗標：台幣入金後 24 小時內開始虛幣/撮合操作
    flow_feat["is_fast_twd_to_crypto"] = (
        (flow_feat["twd_to_crypto_gap_sec"] > 0) & (flow_feat["twd_to_crypto_gap_sec"] < 86400)
    ).astype(int)
    flow_feat["is_fast_twd_to_trade"]  = (
        (flow_feat["twd_to_trade_gap_sec"]  > 0) & (flow_feat["twd_to_trade_gap_sec"]  < 86400)
    ).astype(int)

    # 各渠道是否有交易紀錄的旗標
    flow_feat["has_twd"]    = flow_feat["twd_first"].notna().astype(int)
    flow_feat["has_crypto"] = flow_feat["crypto_first"].notna().astype(int)
    flow_feat["has_trade"]  = flow_feat["trade_first"].notna().astype(int)
    flow_feat["has_swap"]   = flow_feat["swap_first"].notna().astype(int)
    # 使用的交易類型數量（0-4）
    flow_feat["txn_type_diversity"] = (
        flow_feat["has_twd"] + flow_feat["has_crypto"] + flow_feat["has_trade"] + flow_feat["has_swap"]
    )

    # 只保留數值欄位輸出（時間欄位已完成計算，不需要進模型）
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

    unique_action_types：
    使用了幾種不同的操作類型（twd_in/twd_out/crypto_in/crypto_out），
    只用 1-2 種代表行為單一（可能是人頭戶的特徵）。
    """
    # 建立台幣操作序列
    twd_seq = twd_transfer[["user_id", "created_at", "kind"]].copy()
    twd_seq["action"] = twd_seq["kind"].map({0: "twd_in", 1: "twd_out"})
    twd_seq = twd_seq[["user_id", "created_at", "action"]]

    # 建立虛幣操作序列
    crypto_seq = crypto_transfer[["user_id", "created_at", "kind"]].copy()
    crypto_seq["action"] = crypto_seq["kind"].map({0: "crypto_in", 1: "crypto_out"})
    crypto_seq = crypto_seq[["user_id", "created_at", "action"]]

    # 合併並按時間排序，建立完整的操作時間序列
    all_seq = pd.concat([twd_seq, crypto_seq], ignore_index=True).sort_values(["user_id", "created_at"])

    # 前一個動作（用於偵測重複操作）
    all_seq["prev_action"] = all_seq.groupby("user_id")["action"].shift(1)
    # 下一個動作（用於偵測可疑序列）
    all_seq["next_action"] = all_seq.groupby("user_id")["action"].shift(-1)
    # 是否與前一個動作相同（重複操作旗標）
    all_seq["is_repeat"]   = (all_seq["action"] == all_seq["prev_action"]).astype(int)
    # 到下一個動作的時間差（秒）
    all_seq["time_to_next"] = all_seq.groupby("user_id")["created_at"].diff(-1).dt.total_seconds().abs()

    # 可疑序列旗標：twd_in → 1 小時內 → crypto_out
    all_seq["is_suspicious_seq"] = (
        (all_seq["action"]      == "twd_in")
        & (all_seq["next_action"] == "crypto_out")
        & (all_seq["time_to_next"] < 3600)  # 1 小時 = 3600 秒
    ).astype(int)

    seq_feat = all_seq.groupby("user_id").agg(
        total_actions        =("action",           "count"),
        repeat_action_ratio  =("is_repeat",        "mean"),   # 重複操作比例
        unique_action_types  =("action",           "nunique"),# 操作類型多樣性
        suspicious_seq_count =("is_suspicious_seq","sum"),    # 可疑序列次數
        suspicious_seq_ratio =("is_suspicious_seq","mean"),   # 可疑序列比例
    ).reset_index()

    return seq_feat


# =========================
# 5. 交叉特徵
# =========================

def build_cross_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    建立跨模組的交叉特徵，補強複合風險訊號。

    為什麼需要交叉特徵？
    單一特徵的鑑別力有限，但多個特徵的組合往往能捕捉更複雜的風險模式。
    例如：「深夜交易」本身不一定是風險，但「深夜 + 大金額 + TRC20」的組合就很可疑。

    所有交叉特徵都先檢查欄位是否存在（issubset），
    避免在某些 mode 下欄位被移除後報錯。
    """
    df = feature_df.copy()

    # 深夜 × 金額：深夜大額交易的複合風險訊號
    if {"trade_night_ratio", "trade_total_amount"}.issubset(df.columns):
        df["trade_night_amount_cross"]  = df["trade_night_ratio"]  * df["trade_total_amount"]
    if {"crypto_night_ratio", "crypto_total_twd_value"}.issubset(df.columns):
        df["crypto_night_value_cross"]  = df["crypto_night_ratio"] * df["crypto_total_twd_value"]

    # KYC 完成時間 × 行為密度
    # 邏輯：KYC 完成越快（lvl2_minus_confirm_sec 越小）+ 交易越多 → 風險越高
    # 用 log1p 壓縮，避免極端值主導
    if {"lvl2_minus_confirm_sec", "trade_count"}.issubset(df.columns):
        df["kyc_trade_cross"]  = np.log1p(df["lvl2_minus_confirm_sec"].clip(lower=0)) * np.log1p(df["trade_count"])
    if {"lvl2_minus_confirm_sec", "crypto_txn_count"}.issubset(df.columns):
        df["kyc_crypto_cross"] = np.log1p(df["lvl2_minus_confirm_sec"].clip(lower=0)) * np.log1p(df["crypto_txn_count"])

    # 法幣 / 虛幣資金不對稱
    # twd_crypto_value_gap 接近 0：台幣入金幾乎全部換成虛幣（快進快出）
    # twd_crypto_value_ratio 接近 1：虛幣價值 ≈ 台幣入金（資金幾乎全部轉換）
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
        df["protocol_switch_fast_cross"] = df["protocol_switch_rate"] * (1 / (df["crypto_gap_min_sec"].clip(lower=1)))

    # 地址重用 × 共用 IP：地址重用 + 共用 IP 的複合風險（可能是同一批人操控多帳號）
    if {"trc20_addr_reuse_rate", "shared_ip_ratio"}.issubset(df.columns):
        df["trc20_addr_ip_cross"] = df["trc20_addr_reuse_rate"] * df["shared_ip_ratio"]
    if {"bsc_addr_reuse_rate", "shared_ip_ratio"}.issubset(df.columns):
        df["bsc_addr_ip_cross"]   = df["bsc_addr_reuse_rate"]   * df["shared_ip_ratio"]

    return df


# =========================
# 6. 主流程
# =========================

def build_feature_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    完整特徵工程主流程：抓資料 → 清洗 → 建特徵 → 合併 → 輸出。

    流程說明（共 8 步）：
    1. 分頁抓取所有原始資料表
    2. 偵測欄位名稱（相容脫敏版 API）
    3. 清洗各資料表，建立中間欄位
    4. 建立各渠道的基礎聚合特徵
    5. 建立進階特徵（網路、錢包、IP、時間、金額、資金流、序列）
    6. 合併所有特徵到 user_info 為基底的寬表
    7. 補缺值（數值欄位填 0，datetime 欄位保留）
    8. 依 train_label / predict_label 切出訓練集和測試集，輸出 CSV

    回傳：(train_feature, test_feature, feature_df)
    """

    print("\n[1/8] 抓取資料...")
    user_info        = fetch_table_paginated("user_info")
    twd_transfer     = fetch_table_paginated("twd_transfer")
    crypto_transfer  = fetch_table_paginated("crypto_transfer")
    usdt_twd_trading = fetch_table_paginated("usdt_twd_trading")
    usdt_swap        = fetch_table_paginated("usdt_swap")
    train_label      = fetch_table_paginated("train_label")    # 訓練集標籤（含 status）
    predict_label    = fetch_table_paginated("predict_label")  # 測試集（只有 user_id）

    print("\n[2/8] 欄位相容偵測...")
    # 優先使用脫敏版欄位（_hash），若不存在則退回原始欄位
    twd_ip_col      = get_existing_col(twd_transfer,     ["source_ip_hash", "source_ip"])
    crypto_ip_col   = get_existing_col(crypto_transfer,  ["source_ip_hash", "source_ip"])
    trade_ip_col    = get_existing_col(usdt_twd_trading, ["source_ip_hash", "source_ip"])
    from_wallet_col = get_existing_col(crypto_transfer,  ["from_wallet_hash", "from_wallet"])
    to_wallet_col   = get_existing_col(crypto_transfer,  ["to_wallet_hash",   "to_wallet"])

    print("twd_ip_col      :", twd_ip_col)
    print("crypto_ip_col   :", crypto_ip_col)
    print("trade_ip_col    :", trade_ip_col)
    print("from_wallet_col :", from_wallet_col)
    print("to_wallet_col   :", to_wallet_col)

    print("\n[3/8] 資料清洗...")
    user_info        = prepare_user_info(user_info)
    twd_transfer     = prepare_twd_transfer(twd_transfer, twd_ip_col)
    crypto_transfer  = prepare_crypto_transfer(crypto_transfer, crypto_ip_col, from_wallet_col, to_wallet_col)
    usdt_twd_trading = prepare_trade(usdt_twd_trading, trade_ip_col)
    usdt_swap        = prepare_swap(usdt_swap)

    print("\n[4/8] 建立基礎聚合特徵...")
    twd_feat   = build_twd_features(twd_transfer)
    crypto_feat= build_crypto_features(crypto_transfer)
    trade_feat = build_trade_features(usdt_twd_trading)
    swap_feat  = build_swap_features(usdt_swap)

    print("\n[5/8] 建立進階特徵...")
    network_feat  = build_network_features(crypto_transfer)
    wallet_risk_feat = analyze_wallet_risk(crypto_transfer)
    ip_feat       = analyze_ip_patterns(twd_transfer, crypto_transfer, usdt_twd_trading)
    temporal_feat = extract_temporal_anomalies(twd_transfer, crypto_transfer, usdt_twd_trading, usdt_swap)
    amount_feat   = detect_amount_anomalies(twd_transfer, crypto_transfer, usdt_twd_trading, usdt_swap)
    flow_feat     = calculate_fund_flow_patterns(twd_transfer, crypto_transfer, usdt_twd_trading, usdt_swap)
    seq_feat      = extract_sequence_features(twd_transfer, crypto_transfer)

    print("\n[6/8] 合併所有特徵...")
    # 以 user_info 為基底（保留所有用戶），依序 left join 各特徵表
    # 沒有交易紀錄的用戶，對應特徵欄位會是 NaN，後續步驟統一填 0
    feature_df = user_info.copy()

    # 先合併四個基礎聚合特徵（每個用戶都可能有）
    for feat in [twd_feat, crypto_feat, trade_feat, swap_feat]:
        feature_df = feature_df.merge(feat, on="user_id", how="left")

    # 再合併進階特徵（部分用戶可能沒有對應紀錄，如沒有內轉的用戶沒有 network_feat）
    for feat in [network_feat, wallet_risk_feat, ip_feat, temporal_feat, amount_feat, flow_feat, seq_feat]:
        feature_df = feature_df.merge(feat, on="user_id", how="left")

    # 建立交叉特徵（在合併後的完整特徵表上計算）
    feature_df = build_cross_features(feature_df)

    print("feature_df shape before fillna:", feature_df.shape)

    print("\n[7/8] 補缺值...")
    # datetime 欄位保留 NaT（不填 0，避免時間計算出錯）
    # 其他欄位（數值）填 0（代表該行為未發生）
    datetime_cols = feature_df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    for col in feature_df.columns:
        if col not in datetime_cols:
            feature_df[col] = feature_df[col].fillna(0)

    print("feature_df shape after fillna:", feature_df.shape)

    print("\n[8/8] 切出 train / test 並輸出 CSV...")
    # left join 確保 train/test 的用戶都有對應特徵
    # 若某個 user_id 在 feature_df 中找不到，特徵欄位會是 NaN（理論上不應發生）
    train_feature = train_label.merge(feature_df, on="user_id", how="left")
    test_feature  = predict_label.merge(feature_df, on="user_id", how="left")

    train_feature.to_csv("train_feature_modified.csv", index=False)
    test_feature.to_csv("test_feature_modified.csv",   index=False)
    feature_df.to_csv("feature_full_modified.csv",     index=False)

    print("train_feature shape:", train_feature.shape)
    print("test_feature  shape:", test_feature.shape)
    print("已輸出：train_feature_modified.csv / test_feature_modified.csv / feature_full_modified.csv")

    return train_feature, test_feature, feature_df


if __name__ == "__main__":
    build_feature_dataset()
