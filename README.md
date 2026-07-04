# Titanic 生存預測機器學習平台

在既有 Flask + SQLite 乘客資料 CRUD 之上，擴充一套端到端機器學習平台：特徵工程、訓練（含超參數搜尋）、非同步 Job、模型 Registry，以及單筆／批次預測。前端頁面透過 **RESTful API + Ajax** 與後端溝通，操作全程無需重新整理頁面。

## 展示影片連結

[youtu.be/PetbbzxTKk8](https://youtu.be/PetbbzxTKk8)

## 目錄

- [功能特色](#功能特色)
- [技術棧](#技術棧)
- [安裝套件](#安裝套件)
- [執行方法](#執行方法)
- [專案結構](#專案結構)
- [API 端點](#api-端點)
- [說明：模型與超參數](#說明模型與超參數)
- [測試](#測試)
- [成果](#成果)
- [其它你想要補充的資訊](#其它你想要補充的資訊)

## 功能特色

- **乘客資料 CRUD**：網頁介面 + `/api/passengers` REST API（列表分頁、搜尋、新增、編輯、刪除）
- **特徵工程**：Title 萃取、FamilySize / IsAlone 衍生、分組中位數補值，訓練與預測共用同一條可序列化 Pipeline
- **模型訓練**：Logistic Regression / Random Forest，透過 `GridSearchCV` 做超參數搜尋，使用者可在網頁上自訂候選值
- **非同步訓練 Job**：訓練在背景執行緒執行，前端以 Ajax 輪詢 `job` 狀態與進度
- **模型 Registry**：訓練完成的 Pipeline 以 `joblib` 存檔，中繼資料（超參數、指標、特徵清單等）存入 SQLite `ml_model` 表，可切換啟用中模型
- **預測**：單筆表單預測、CSV 批次預測（可下載預測結果 CSV）

## 技術棧

| 層              | 技術                                                                                                       |
| :-------------- | :--------------------------------------------------------------------------------------------------------- |
| 環境 / 套件管理 | uv（`.venv` + `pyproject.toml` + `uv.lock`）                                                         |
| 展示層          | Flask + Jinja2 templates + REST API                                                                        |
| 前端互動        | 原生 JavaScript（Ajax／`fetch`）                                                                         |
| 服務層          | 純 Python（`src/ml/`，不依賴 Flask）                                                                     |
| 機器學習        | scikit-learn（`Pipeline`、`ColumnTransformer`、`GridSearchCV`、Logistic Regression / Random Forest） |
| 非同步處理      | `threading`（訓練 Job 狀態管理）                                                                         |
| 資料層          | SQLite（`titanic` / `ml_model` / `train_job` 表）+ `joblib` 模型檔                                 |
| 測試            | pytest + pytest-cov                                                                                        |

## 安裝套件

本專案使用 [uv](https://docs.astral.sh/uv/) 管理虛擬環境與套件，版本鎖定於 `uv.lock`。核心依賴（節錄自 `pyproject.toml` / `uv.lock`）：

- Flask==3.1.3
- pandas==3.0.3
- scikit-learn==1.9.0
- joblib==1.5.3
- numpy==2.5.0（scikit-learn / pandas 的相依套件）

開發／測試依賴：

- pytest==9.1.1
- pytest-cov==7.1.0

> 版本號可用 `uv pip list`、`pip list` 或 `conda list` 檢視。若你是 clone 這個 repo 並用 `uv sync` 安裝，版本會與 `uv.lock` 完全一致；若手動用 `pip install` 安裝，版本可能不同，請自行調整。

安裝方式：

```bash
uv sync
```

## 執行方法

```bash
# 1. 安裝依賴（第一次執行或 pyproject.toml 有變動時）
uv sync

# 2. 初始化資料庫（建立 titanic / ml_model / train_job 表，並匯入 titanic.csv）
uv run python init_db.py

# 3. 啟動 Flask 開發伺服器
uv run python app.py
```

啟動後開啟瀏覽器造訪 [http://127.0.0.1:5000](http://127.0.0.1:5000)：

| 頁面                | 路徑            |
| :------------------ | :-------------- |
| 首頁                | `/`           |
| 乘客資料管理        | `/passengers` |
| 模型訓練            | `/train`      |
| 模型清單            | `/models`     |
| 預測（單筆 / 批次） | `/predict`    |

> `init_db.py` 每次執行會 `DROP TABLE` 後重建 `titanic` 表並重新匯入 `titanic.csv`，`ml_model` / `train_job` 表則為冪等建立（不會清空既有訓練紀錄）。`my_db.db` 與 `models/` 目錄已列在 `.gitignore`，不會被提交版控，因此在新環境下**必須先執行 `init_db.py`** 才有資料可用。

## 專案結構

```
titanic_restful_project/
├── app.py                  # Flask 入口：CRUD 路由 + 掛載 ML blueprint
├── init_db.py               # 建表 + titanic.csv 匯入
├── src/
│   ├── web/                 # 展示層：Flask blueprint（輸入驗證、HTTP 狀態碼、JSON 序列化）
│   │   ├── ml_api.py         # /api/ml/* REST API
│   │   └── ml_pages.py       # /train /models /predict 頁面路由
│   └── ml/                  # 服務層：純 Python，不 import Flask（CON-4）
│       ├── config.py          # 集中設定：超參數格點、特徵清單、CV 設定、路徑
│       ├── features.py        # 特徵工程 Pipeline（訓練/預測共用，CON-2）
│       ├── hyperparams.py     # 使用者自訂超參數驗證 + GridSearch 格點建構
│       ├── training.py        # 訓練核心：GridSearchCV + 指標評估
│       ├── jobs.py            # 非同步訓練 Job 狀態管理（threading）
│       ├── registry.py        # 模型持久化（joblib）+ metadata（SQLite）+ 啟用切換
│       ├── prediction.py      # 單筆 / 批次預測
│       ├── validation.py      # 輸入驗證（單筆 / 批次）
│       ├── schema.py          # ml_model / train_job 建表
│       └── types.py           # 型別定義（dataclass）
├── models/                  # joblib 模型檔（不進版控）
├── templates/                # home / index(passengers) / new / edit / train / models / predict
├── static/                   # 前端 JS（Ajax）與圖片
├── tests/                    # unit + integration 測試
└── titanic.csv               # 原始資料集
```

## API 端點

### 乘客資料 CRUD

| 方法   | 路徑                                        | 說明                     |
| :----- | :------------------------------------------ | :----------------------- |
| GET    | `/api/passengers?page=&per_page=&search=` | 分頁查詢（可依姓名搜尋） |
| GET    | `/api/passengers/<id>`                    | 取得單筆乘客             |
| POST   | `/api/passengers`                         | 新增乘客                 |
| PUT    | `/api/passengers/<id>`                    | 更新乘客                 |
| DELETE | `/api/passengers/<id>`                    | 刪除乘客                 |

### 機器學習平台

| 方法   | 路徑                              | 說明                                                                                 |
| :----- | :-------------------------------- | :----------------------------------------------------------------------------------- |
| POST   | `/api/ml/train`                 | 送出訓練請求（`algorithm` + 可選 `hyperparameters`），立即回傳 `job_id`（202） |
| GET    | `/api/ml/train/status/<job_id>` | 查詢訓練 Job 狀態／進度；`done` 時附最佳超參數與指標                               |
| GET    | `/api/ml/models`                | 列出所有已訓練模型的中繼資料                                                         |
| POST   | `/api/ml/models/<id>/activate`  | 將指定模型設為啟用中（供預測使用）                                                   |
| DELETE | `/api/ml/models/<id>`           | 刪除模型                                                                             |
| POST   | `/api/ml/predict`               | 單筆預測（JSON body：乘客欄位）                                                      |
| POST   | `/api/ml/predict/batch`         | CSV 批次預測（multipart 表單欄位`file`；加 `?format=csv` 可直接下載結果 CSV）    |

## 說明：模型與超參數

### 共用前處理 Pipeline（`src/ml/features.py`）

訓練與預測共用同一條 `sklearn.Pipeline`，確保兩邊前處理邏輯完全一致（不會有「訓練一套、預測另一套」造成的偏差）：

1. **`TitanicFeatureEngineer`**（自訂 transformer）
   - 從 `Name` 以正規表示式萃取 `Title`（如 `Mr` / `Mrs` / `Miss` / `Master`），法文/英文同義詞（`Mlle`/`Ms`→`Miss`、`Mme`→`Mrs`）歸併，其餘罕見頭銜歸為 `Rare`
   - 衍生 `FamilySize = SibSp + Parch + 1`、`IsAlone = (FamilySize == 1)`
   - 缺值補值：`Age` 依 `Title` 分組中位數、`Fare` 依 `Pclass` 分組中位數、`Embarked` 用眾數；統計量僅在 `fit` 階段學習，`transform` 套用既學數值，確保訓練與預測一致
2. **`ColumnTransformer`**
   - 數值特徵（`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, `IsAlone`）→ `StandardScaler`
   - 類別特徵（`Sex`, `Embarked`, `Title`）→ `OneHotEncoder(handle_unknown="ignore")`（預測時遇到訓練未見過的類別不會報錯）

整條 Pipeline（前處理 + 分類器）以 `joblib` 序列化存檔，metadata（超參數、指標、特徵清單、資料集雜湊等）存入 SQLite `ml_model` 表。

### 支援演算法與超參數格點（`src/ml/config.py`）

訓練流程：`train_test_split`（`test_size=0.2`，`stratify=y`，`random_state=42`）→ 對訓練集做 `GridSearchCV`（`cv=5`）→ 在 held-out 測試集計算 Accuracy / Precision / Recall / F1 / ROC-AUC / 混淆矩陣。

| 演算法                        | 預設超參數格點                                                                                                                          |
| :---------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | `C: [0.01, 0.1, 1, 10]`、`penalty: [l1, l2]`、`solver: liblinear`                                                                 |
| **Random Forest**       | `n_estimators: [100, 300, 500]`、`max_depth: [None, 5, 10, 20]`、`min_samples_split: [2, 5, 10]`、`min_samples_leaf: [1, 2, 4]` |

使用者也可在訓練頁面自行輸入候選值，後端會依 `HYPERPARAM_SPECS` 驗證型別與合法範圍，並限制組合總數上限（避免笛卡兒積爆炸拖垮訓練時間）。

### 實測結果（使用預設格點，`titanic.csv` 891 筆資料，本機執行一次的結果）

| 演算法              | 最佳超參數                                                                             | CV 平均 Accuracy | Test Accuracy | Precision | Recall |   F1   | ROC-AUC |
| :------------------ | :------------------------------------------------------------------------------------- | :--------------: | :-----------: | :-------: | :----: | :----: | :-----: |
| Logistic Regression | `C=1`, `penalty=l2`, `solver=liblinear`                                          |      0.8203      |    0.8547    |  0.8413  | 0.7681 | 0.8030 | 0.8805 |
| Random Forest       | `n_estimators=100`, `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2` |      0.8273      |    0.8268    |  0.7969  | 0.7391 | 0.7669 | 0.8545 |

因為 `random_state=42` 固定，`train_test_split` 與各演算法的隨機性都是可重現的，重新訓練會得到相同結果。整體而言，在這份資料與這組格點下，兩個模型的表現相近：Random Forest 的交叉驗證分數略高，但 Logistic Regression 在 held-out 測試集上的各項指標略優，屬正常的模型變異範圍，並非其中一個模型明顯更好。實務上可在訓練頁面調整超參數格點，觀察指標變化。

## 測試

```bash
uv run pytest
uv run pytest --cov=src --cov-report=term-missing
```

測試涵蓋單元測試（特徵工程、超參數驗證、Job、Registry、訓練、預測驗證）與整合測試（ML REST API、ML 頁面、驗收情境）。

## 補充資訊

- **CON-2 共用管線**：訓練與預測嚴禁各寫一套前處理，一律透過 `features.build_feature_pipeline()`。
- **CON-4 框架解耦**：`src/ml/` 為純 Python 服務層，不 import 任何 Flask 物件，方便未來替換或獨立測試。
- **CON-3 模型持久化**：模型檔用 `joblib`，中繼資料存 SQLite `ml_model` 表，兩者以 `model_uid` 對應。
- **NFR-S2**：所有 SQL 一律使用參數化查詢，不做字串拼接。
