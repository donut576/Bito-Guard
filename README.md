# AML / 詐騙偵測專案（BitoPro）

從 BitoPro API 抓取交易資料，進行特徵工程後，使用 XGBoost / LightGBM / Random Forest 訓練人頭戶偵測模型；並提供 FastAPI 服務層供線上即時推論與視覺化 Dashboard。

主流程：**feature_engineering.py → run_all_models.py → aml-frontend**

---

## 專案結構

```
/
├── feature_engineering.py   # Step 1：API 抓取 → 清洗 → 特徵工程 → 輸出 CSV
├── run_all_models.py        # Step 2：一次跑 XGBoost / LightGBM / RF 三模型，輸出至 output_results/
├── model_xgboost.py         # 單獨跑 XGBoost（輸出至 output_xgb_v2/）
├── model_LightGBM.py        # 單獨跑 LightGBM
├── model_Rf.py              # 單獨跑 Random Forest
├── requirements.txt
│
├── train_feature.csv        # 訓練集特徵（feature_engineering.py 輸出）
├── test_feature.csv         # 測試集特徵（feature_engineering.py 輸出）
├── feature_full.csv         # 全量用戶特徵（含 IsolationForest 分數）
│
├── output_xgb_v2/           # model_xgboost.py 單獨輸出目錄
│   ├── compare_modes.csv
│   ├── full/ no_leak/ safe/
│       ├── feature_importance.csv
│       ├── best_params.csv
│       ├── metrics.csv
│       ├── threshold_analysis.csv
│       ├── valid_detail.csv
│       └── submission.csv
│
├── output_results/          # run_all_models.py 輸出目錄（三模型 × 三 mode）
│   ├── summary.csv          # 跨模型比較表
│   ├── xgb/ lgb/ rf/
│       ├── full/ no_leak/ safe/
│           ├── metrics.csv
│           ├── feature_importance.csv
│           ├── threshold_analysis.csv
│           ├── test_scores.csv
│           └── shap.json
│
├── aml-frontend/            # 視覺化 Dashboard（React + Vite + Recharts）
│   ├── server.py            # 輕量級 FastAPI server，讀取 output_results/ 並提供 API
│   ├── src/
│   │   ├── components/      # AMLDashboard、ShapPanel、UploadPanel、charts 等
│   │   ├── api/endpoints.js
│   │   └── main.jsx
│   └── package.json
│
├── frontend/                # 舊版 Dashboard（較簡單，可選用）
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/      # AlertFeed、CasesTab、SystemHealthTab 等
│   └── package.json
│
└── app/                     # FastAPI 進階服務層（線上推論 + 監控 + 案件管理）
    ├── main.py
    ├── config.py
    ├── models/              # Pydantic schemas（alert、case、drift、explain 等）
    ├── routers/             # 16 個 API 路由
    │   ├── predict.py       # 即時推論
    │   ├── explain.py       # SHAP 解釋
    │   ├── drift.py         # 特徵漂移偵測
    │   ├── monitoring.py    # 模型監控
    │   ├── alerts.py        # 警報管理
    │   ├── cases.py         # 案件管理
    │   ├── audit.py         # 稽核日誌
    │   ├── thresholds.py    # 動態閾值調整
    │   ├── clusters.py      # 身份聚類
    │   ├── graph.py         # 資金流圖譜
    │   ├── feature_store.py # 特徵倉儲
    │   ├── sequence.py      # 序列評分
    │   ├── copilot.py       # AI 助手
    │   ├── stream.py        # 串流消費
    │   └── model.py         # 模型管理
    ├── services/            # 業務邏輯層
    │   ├── predictor.py
    │   ├── shap_explainer.py
    │   ├── drift_detector.py
    │   ├── monitoring_system.py
    │   ├── alert_router.py
    │   ├── case_manager.py
    │   ├── audit_logger.py
    │   ├── threshold_controller.py
    │   ├── identity_clusterer.py
    │   ├── graph_engine.py
    │   ├── feature_store.py
    │   ├── sequence_scorer.py
    │   ├── ai_copilot.py
    │   ├── stream_consumer.py
    │   ├── ensemble_scorer.py
    │   └── model_loader.py
    └── migrations/          # 資料庫 schema（PostgreSQL）
        ├── 002_create_feature_store.sql
        ├── 003_create_graph_snapshots.sql
        ├── 004_create_identity_clusters.sql
        ├── 005_create_threshold_history.sql
        └── 006_create_cases.sql
```

---

## 使用方式

### ML Pipeline

**Step 1：產生特徵資料集**

```bash
python feature_engineering.py
```

輸出：
- `train_feature.csv`：訓練集（含 `status` 標籤）
- `test_feature.csv`：測試集
- `feature_full.csv`：全量用戶特徵表（含 IsolationForest 異常分數）

**Step 2a：一次跑三模型（建議）**

```bash
python run_all_models.py
```

輸出至 `output_results/{xgb,lgb,rf}/{full,no_leak,safe}/`，包含 metrics、feature_importance、threshold_analysis、test_scores、shap.json，以及跨模型比較的 `summary.csv`。

**Step 2b：單獨跑 XGBoost**

```bash
python model_xgboost.py
```

輸出至 `output_xgb_v2/`，包含三種 ablation mode 的評估結果與預測檔。

### Dashboard（aml-frontend）

先啟動 API server（讀取 `output_results/`）：

```bash
cd aml-frontend
python -m uvicorn server:app --port 8000 --reload
```

再啟動前端：

```bash
cd aml-frontend
npm install
npm run dev
```

開啟 `http://localhost:3000` 查看 Dashboard，包含模型比較、特徵重要性、SHAP 解釋、閾值分析、批次推論上傳等功能。

### FastAPI 進階服務層

```bash
uvicorn app.main:app --reload
```

啟動後可至 `http://localhost:8000/docs` 查看 Swagger UI。

主要設定透過環境變數或 `.env` 檔控制（參見 `app/config.py`）：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `MODEL_S3_URI` | `s3://aml-models/model_registry/latest` | 模型 artifact 路徑 |
| `DATABASE_URL` | `postgresql://...@localhost:5432/aml` | Audit log 資料庫 |
| `DEFAULT_MODE` | `safe` | 預設特徵版本 |
| `PSI_WARNING_THRESHOLD` | `0.1` | Drift 警告門檻 |
| `PSI_CRITICAL_THRESHOLD` | `0.2` | Drift 嚴重門檻 |

### 部署到 Render

本專案已提供 `render.yaml`（Render Blueprint），可一次建立：

- `aml-api`（FastAPI Web Service）
- `aml-frontend`（Vite Static Site）
- `aml-redis`（Redis）
- `aml-postgres`（PostgreSQL）

#### 1) 匯入 Blueprint

1. 將此 repo push 到 GitHub。
2. 到 Render 選 **New + → Blueprint**。
3. 選取此 repo，Render 會自動讀取 `render.yaml`。

#### 2) 設定必要環境變數

建立服務後，請在 Render Console 補上：

- `aml-api`：
    - `MODEL_S3_URI`（你的模型路徑，例如 `s3://aml-models/model_registry/latest`）
- `aml-frontend`：
    - `VITE_API_BASE_URL`（填 `aml-api` 的公開 URL，例如 `https://aml-api.onrender.com`）

> `DATABASE_URL` 與 `REDIS_URL` 會由 Blueprint 自動綁定到 Render 的 PostgreSQL/Redis 資源。

#### 3) 驗證部署

- API 健康檢查：`https://<your-api>/health`
- API 文件：`https://<your-api>/docs`
- 前端站點：`https://<your-frontend>`

---

## 特徵工程說明（feature_engineering.py）

資料來源（透過 BitoPro API 抓取）：

| 資料表 | 說明 |
|---|---|
| `user_info` | 用戶基本資料與 KYC 時間 |
| `twd_transfer` | 台幣出入金紀錄 |
| `crypto_transfer` | 虛擬貨幣轉帳紀錄 |
| `usdt_twd_trading` | USDT/TWD 撮合交易紀錄 |
| `usdt_swap` | 一鍵買賣（Swap）紀錄 |
| `train_label` / `predict_label` | 訓練/預測標籤 |

特徵類型：

- **KYC 時間差**：各 KYC 階段完成時間差（人頭戶通常極短）
- **基礎聚合特徵**：各渠道的交易筆數、金額統計、夜間/週末比例、IP 多樣性、時間熵
- **鏈別風險特徵**：TRC20 / BSC 使用比例、地址重用率、流向不平衡
- **網路特徵**：內轉網路的 in/out degree、資金流向不對稱
- **錢包風險**：高風險共用地址比例、地址重用率
- **IP 特徵**：跨渠道共用 IP 比例、IP 跳躍率
- **資金流動**：快進快出旗標（台幣入金 → 24 小時內虛幣提領）
- **行為序列**：可疑操作序列比例（twd_in → 1 小時內 → crypto_out）
- **金額異常**：整數金額比例、變異係數、偏態
- **交叉特徵**：深夜 × 金額、KYC 速度 × 交易密度、TRC20 × 夜間等複合訊號
- **IsolationForest 異常分數**：用 train+test 合併後 fit，確保分布一致

---

## 模型說明

### 支援模型

| 模型 | 腳本 | 說明 |
|---|---|---|
| XGBoost | `model_xgboost.py` / `run_all_models.py` | 主要模型，使用 `hist` tree method |
| LightGBM | `model_LightGBM.py` / `run_all_models.py` | 對照模型 |
| Random Forest | `model_Rf.py` / `run_all_models.py` | 對照模型 |

### Ablation Study（三種特徵版本）

| Mode | 說明 |
|---|---|
| `full` | 全部可用數值欄位（分數上限，可能含 leakage） |
| `no_leak` | 移除高風險可疑欄位（建議主要參考版本） |
| `safe` | 移除高風險欄位 + 人口學欄位（最接近真實部署情境） |

若 `full` 與 `safe` 分數差距 > 0.05，建議提交 `safe` 版本，避免線上線下落差。

### 切分策略

優先使用 **time-based split**（前 80% 訓練、後 20% 驗證），模擬「用歷史預測未來」的場景；找不到合適時間欄位時退回 random stratified split。

### 超參數調優

預設使用 **Optuna**（直接優化驗證集 F1），未安裝時自動退回手動參數。

---

## 環境需求

```bash
pip install pandas numpy requests scikit-learn xgboost lightgbm matplotlib seaborn
pip install optuna shap      # 選用
pip install fastapi uvicorn pydantic-settings  # FastAPI 服務層
```

或直接：

```bash
pip install -r requirements.txt
```

前端依賴：

```bash
cd aml-frontend
npm install
```
