<div align="center">

# Titanic 生存預測機器學習平台

**一套把「資料管理 → 特徵工程 → 模型訓練 → 模型治理 → 線上預測」完整串起來的 Web ML 平台**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-3.0.3-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-144%20passed-success)](#測試)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)](#測試)

[展示影片](https://youtu.be/PetbbzxTKk8) · [快速開始](#快速開始) · [系統架構](#系統架構) · [設計決策](#技術亮點與設計決策) · [API 文件](#api-文件)

</div>

---

## 目錄

- [專案簡介](#專案簡介)
- [核心功能](#核心功能)
- [畫面截圖](#畫面截圖)
- [系統架構](#系統架構)
- [技術亮點與設計決策](#技術亮點與設計決策)
- [技術棧](#技術棧)
- [快速開始](#快速開始)
- [API 文件](#api-文件)
- [機器學習設計](#機器學習設計)
- [實測結果](#實測結果)
- [測試](#測試)
- [專案結構](#專案結構)
- [作者](#作者)

---

## 專案簡介

多數 Titanic 專案止步於 Notebook 裡的一次性訓練腳本。本專案的目標不同：**把模型從訓練到上線的完整生命週期，做成一個可操作、可重現、可測試的 Web 平台。**

具體來說，使用者可以在瀏覽器上完成以下流程，全程不需重新整理頁面：

1. 管理乘客資料（CRUD、分頁、搜尋）
2. 選擇演算法、自訂超參數候選值，送出訓練
3. 訓練在背景執行緒進行，前端輪詢 Job 狀態與進度
4. 訓練完成後模型自動登錄至 Registry，比較指標並切換啟用中模型
5. 以啟用中的模型做單筆預測或 CSV 批次預測，並下載結果

工程上的重點在於三件事：**訓練與預測共用同一條序列化 Pipeline**（避免 training/serving skew）、**服務層與 Web 框架完全解耦**（`src/ml/` 不 import 任何 Flask 物件）、以及 **98% 測試覆蓋率下的可重現訓練結果**。

---

## 核心功能

| 模組 | 功能 | 說明 |
| :--- | :--- | :--- |
| **資料管理** | 乘客 CRUD | 分頁、姓名搜尋、新增／編輯／刪除，全部走 REST API + Ajax |
| **特徵工程** | 衍生特徵與補值 | Title 萃取、FamilySize／IsAlone 衍生、分組中位數補值，封裝為可序列化 Pipeline |
| **模型訓練** | 超參數搜尋 | Logistic Regression／Random Forest，`GridSearchCV` 5-fold 交叉驗證，候選值可在網頁自訂 |
| **非同步 Job** | 背景訓練 | 送出訓練立即回傳 `job_id`（HTTP 202），前端輪詢狀態，不阻塞請求 |
| **模型 Registry** | 模型治理 | joblib 存 Pipeline、metadata 存 SQLite，支援清單／啟用切換／刪除，同時至多一個 active |
| **線上預測** | 單筆／批次 | 表單單筆預測回傳生存機率；CSV 批次預測支援結果下載 |

---

## 畫面截圖

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/home.png" alt="首頁"><br><sub><b>首頁</b>：平台功能介紹與操作導引</sub></td>
    <td width="50%"><img src="docs/screenshots/passengers.png" alt="乘客資料管理"><br><sub><b>資料管理</b>：分頁、搜尋、CRUD</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/screenshots/train.png" alt="模型訓練"><br><sub><b>模型訓練</b>：選演算法、自訂超參數候選值、輪詢 Job 進度</sub></td>
    <td width="50%"><img src="docs/screenshots/models.png" alt="模型管理"><br><sub><b>模型管理</b>：比較 CV／Accuracy，切換 active 模型</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/screenshots/predict.png" alt="生存預測"><br><sub><b>生存預測</b>：單筆表單預測與 CSV 批次預測</sub></td>
    <td width="50%"></td>
  </tr>
</table>

---

## 系統架構

### 分層架構

展示層負責 HTTP 契約，服務層負責業務邏輯，兩者以純資料結構溝通。服務層不依賴 Flask，可獨立測試或替換前端框架。

```mermaid
flowchart TB
    subgraph Client["瀏覽器"]
        UI["Jinja2 頁面 + 原生 JS（fetch/Ajax）"]
    end

    subgraph Web["展示層 src/web/（Flask）"]
        API["ml_api.py<br/>/api/ml/* JSON API<br/>輸入驗證 · 狀態碼 · 序列化"]
        PAGES["ml_pages.py<br/>/ml/* 頁面路由"]
        CRUD["app.py<br/>/api/passengers CRUD"]
    end

    subgraph Service["服務層 src/ml/（純 Python，不 import Flask）"]
        FEAT["features.py<br/>共用特徵管線"]
        TRAIN["training.py<br/>GridSearchCV + 指標評估"]
        JOBS["jobs.py<br/>非同步 Job 狀態"]
        REG["registry.py<br/>模型登錄 / active 切換"]
        PRED["prediction.py<br/>單筆 / 批次預測"]
        VAL["validation.py + hyperparams.py<br/>輸入與超參數驗證"]
    end

    subgraph Storage["資料層"]
        DB[("SQLite<br/>titanic · ml_model · train_job")]
        FILES[("models/*.joblib<br/>序列化 Pipeline")]
    end

    UI -->|HTTP| API
    UI -->|HTTP| PAGES
    UI -->|HTTP| CRUD
    API --> VAL
    API --> JOBS
    API --> REG
    API --> PRED
    JOBS --> TRAIN
    TRAIN --> FEAT
    PRED --> FEAT
    PRED --> REG
    TRAIN --> DB
    REG --> DB
    REG --> FILES
    CRUD --> DB
```

### 非同步訓練流程

訓練可能耗時數十秒（Random Forest 預設格點為 108 組參數 × 5 folds），因此採「立即回應 + 狀態輪詢」而非同步阻塞。

```mermaid
sequenceDiagram
    participant B as 瀏覽器
    participant A as ml_api（Flask）
    participant J as jobs.JobStore
    participant T as 背景執行緒
    participant S as SQLite + models/

    B->>A: POST /api/ml/train {algorithm, hyperparameters}
    A->>A: build_param_grid() 驗證型別／範圍／組合上限
    alt 驗證失敗
        A-->>B: 422 { error, fields }
    else 驗證通過
        A->>J: create job（status=pending）
        J->>T: 啟動 daemon thread
        A-->>B: 202 { job_id, status }
    end

    T->>S: load_training_data()
    T->>T: GridSearchCV 5-fold → held-out 測試集評估
    T->>S: joblib.dump(pipeline) + INSERT ml_model
    T->>J: mark_done(job_id, model_uid)

    loop 前端輪詢
        B->>A: GET /api/ml/train/status/{job_id}
        A-->>B: { status, progress, result: { best_params, metrics } }
    end
```

### 預測流程（共用管線）

預測不重寫任何前處理，而是載回訓練時序列化的整條 Pipeline，確保線上線下行為一致。

```mermaid
flowchart LR
    IN["POST /api/ml/predict<br/>乘客 JSON"] --> V{"validate_passenger()"}
    V -->|不合法| E422["422 欄位錯誤明細"]
    V -->|合法| ACT{"取得 active 模型"}
    ACT -->|不存在| E409["409 請先訓練並啟用模型"]
    ACT -->|存在| LOAD["joblib.load(pipeline)"]
    LOAD --> P["pipeline.predict_proba()<br/>（前處理與訓練時完全相同）"]
    P --> OUT["200 { survived, probability }"]
```

---

## 技術亮點與設計決策

### 1. 訓練與預測共用同一條序列化 Pipeline

**問題**：ML 系統最常見的線上事故是 training/serving skew — 訓練時用 pandas 算了一套補值，預測時在 API 層又寫了一套，兩邊統計量不同導致線上表現遠低於離線指標。

**做法**：把補值與衍生特徵封裝進自訂 transformer `TitanicFeatureEngineer`，**統計量只在 `fit` 階段學習並存於 estimator 內，`transform` 僅套用既學數值**，不在預測當下對輸入重算。整條 `Pipeline(前處理 + 分類器)` 以 joblib 序列化；預測端一律載回同一物件呼叫 `predict_proba`，不存在第二套前處理程式碼。

```python
# src/ml/features.py — fit 學統計量，transform 只套用
def fit(self, X, y=None):
    self.age_median_by_title_ = age.groupby(engineered["Title"]).median().to_dict()
    self.fare_median_by_pclass_ = fare.groupby(engineered["Pclass"]).median().to_dict()
    self.embarked_mode_ = self._safe_mode(engineered["Embarked"], default="S")
    return self
```

**額外處理**：類別編碼使用 `OneHotEncoder(handle_unknown="ignore")`，預測時遇到訓練未見過的類別不會拋錯。

### 2. 服務層與 Web 框架解耦

`src/ml/` 全部為純 Python，**不 import 任何 Flask 物件**。展示層 `src/web/` 只做三件事：輸入驗證、HTTP 狀態碼對應、JSON 序列化；業務邏輯一律委派服務層。

好處是服務層可獨立做單元測試（不需要 Flask test client），也能在未來換成 FastAPI 或 CLI 而不必改動任何 ML 程式碼。網域例外（如 `NoActiveModelError`）由服務層拋出，展示層負責翻譯成適當狀態碼。

### 3. 非同步 Job 採不可變狀態轉移

`JobStore` 以 `threading.Lock` 保護狀態，且每次轉移都用 `dataclasses.replace` **產生新的 Job 快照取代舊值**，而非就地修改欄位。這樣讀取端永遠拿到一致的完整快照，不會讀到「狀態已改成 done、但 metrics 還沒寫入」的中間態。

單一 Job 失敗只標記為 `failed` 並記錄錯誤訊息，不向外拋出，避免單次訓練失敗影響服務整體。

### 4. 超參數防呆：不讓不合理輸入流進 GridSearchCV

開放使用者自訂超參數的代價是「笛卡兒積爆炸」與非法值。`config.HYPERPARAM_SPECS` 集中定義每個參數的型別與安全範圍，並限制單一參數候選數（20）與**組合總數上限（200）**：

| 檢查項 | 行為 |
| :--- | :--- |
| 型別錯誤（如 `C="abc"`） | 422，回傳欄位層級錯誤明細 |
| 超出範圍（如 `n_estimators=99999`） | 422 |
| 組合數 > 200 | 422，避免訓練時間失控 |
| 未知演算法 | 422，錯誤訊息列出可用選項 |

驗證發生在建立 Job **之前**，因此使用者拿到的是同步的明確錯誤，而不是非同步的 failed 狀態。

### 5. 用資料結構取代應用層判斷

「同時至多一個 active 模型」不是靠應用層的 if 檢查，而是由 **SQLite partial unique index** 保證：

```sql
CREATE UNIQUE INDEX ux_ml_model_active ON ml_model (is_active) WHERE is_active = 1;
```

切換 active 時在單一交易內先清除舊 active 再設定新值，避免瞬間兩筆同為 active 而違反索引約束。約束寫在資料層，任何寫入路徑都繞不過去。

### 6. 模型可追溯性

每個登錄模型都記錄 `data_hash`（訓練資料內容的 SHA-256）、`feature_list`（編碼後模型實際消費的欄位）、`training_rows`、`training_duration_sec` 與完整超參數。當線上預測結果異常時，可回溯該模型究竟是用哪一份資料、哪些特徵訓練出來的。

### 7. 錯誤處理不裸奔 500

所有可預期的失敗都有明確狀態碼與訊息，不回傳未處理的 500：

| 狀況 | 狀態碼 | 回應 |
| :--- | :---: | :--- |
| 缺少必要欄位 / CSV 解析失敗 | 400 | 明確指出缺什麼 |
| 找不到 job／模型 | 404 | 附上查詢的 ID |
| 無 active 模型時預測 | 409 | 提示先完成訓練並啟用模型 |
| 資料列數超過 10,000 | 413 | 提示分批上傳 |
| 輸入驗證／超參數驗證失敗 | 422 | 回傳欄位層級（批次為列層級）錯誤明細 |

### 8. 安全性

所有 SQL 一律使用**參數化查詢**，不做字串拼接；使用者輸入在系統邊界即完成驗證（型別、範圍、長度、列數上限），不信任任何外部資料。

---

## 技術棧

| 層 | 技術 | 版本 |
| :--- | :--- | :--- |
| 環境／套件管理 | uv（`.venv` + `pyproject.toml` + `uv.lock`） | — |
| 語言 | Python | 3.11+（開發環境 3.12.10） |
| 展示層 | Flask + Jinja2 + REST API | 3.1.3 |
| 前端互動 | 原生 JavaScript（`fetch` / Ajax） | — |
| 機器學習 | scikit-learn（Pipeline、ColumnTransformer、GridSearchCV） | 1.9.0 |
| 資料處理 | pandas / numpy | 3.0.3 / 2.5.0 |
| 模型序列化 | joblib | 1.5.3 |
| 非同步處理 | threading（標準庫） | — |
| 資料庫 | SQLite（標準庫 sqlite3） | — |
| 測試 | pytest + pytest-cov | 9.1.1 / 7.1.0 |

---

## 快速開始

### 前置需求

- Python 3.11 以上
- [uv](https://docs.astral.sh/uv/)（套件與虛擬環境管理）

```bash
# macOS / Linux 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安裝與執行

```bash
# 1. 取得原始碼
git clone https://github.com/ychsieh725/titanic_restful_project.git
cd titanic_restful_project

# 2. 依 uv.lock 還原依賴（會自動建立 .venv）
uv sync

# 3. 初始化資料庫：建表並匯入 titanic.csv（891 筆）
uv run python init_db.py

# 4. 啟動開發伺服器
uv run python app.py
```

啟動後造訪 **http://127.0.0.1:5000**。

### 頁面路徑

| 頁面 | 路徑 |
| :--- | :--- |
| 首頁 | `/` |
| 乘客資料管理 | `/passengers` |
| 模型訓練 | `/ml/train` |
| 模型管理 | `/ml/models` |
| 生存預測 | `/ml/predict` |

### 首次使用建議流程

1. 進入 `/ml/train`，選擇 **Logistic Regression**（約 2 秒完成）送出訓練
2. 等待進度顯示完成後，到 `/ml/models` 點擊啟用該模型
3. 進入 `/ml/predict` 填寫乘客資料，即可取得生存機率

> **注意**：`init_db.py` 每次執行會 `DROP` 並重建 `titanic` 表後重新匯入 CSV；`ml_model` / `train_job` 表則為冪等建立，不會清空既有訓練紀錄。`my_db.db` 與 `models/` 已列入 `.gitignore`，因此在新環境必須先執行 `init_db.py`。

> **macOS 提醒**：系統的 AirPlay Receiver 預設佔用 5000 埠。若無法連線，請至「系統設定 → 一般 → AirDrop 與接力」關閉 AirPlay 接收器，或自行調整 `app.py` 的 port。

---

## API 文件

所有 API 皆回傳 JSON。

### 乘客資料 CRUD

| 方法 | 路徑 | 說明 |
| :--- | :--- | :--- |
| `GET` | `/api/passengers?page=&per_page=&search=` | 分頁查詢，可依姓名搜尋 |
| `GET` | `/api/passengers/<id>` | 取得單筆乘客 |
| `POST` | `/api/passengers` | 新增乘客 |
| `PUT` | `/api/passengers/<id>` | 更新乘客 |
| `DELETE` | `/api/passengers/<id>` | 刪除乘客 |

### 機器學習平台

| 方法 | 路徑 | 成功狀態碼 | 說明 |
| :--- | :--- | :---: | :--- |
| `POST` | `/api/ml/train` | 202 | 送出訓練（`algorithm` + 可選 `hyperparameters`），立即回傳 `job_id` |
| `GET` | `/api/ml/train/status/<job_id>` | 200 | 查詢 Job 狀態；完成時附最佳超參數與指標 |
| `GET` | `/api/ml/models` | 200 | 列出所有已登錄模型 |
| `POST` | `/api/ml/models/<id>/activate` | 200 | 將指定模型設為 active |
| `DELETE` | `/api/ml/models/<id>` | 200 | 刪除模型（含 joblib 檔） |
| `POST` | `/api/ml/predict` | 200 | 單筆預測 |
| `POST` | `/api/ml/predict/batch` | 200 | CSV 批次預測（表單欄位 `file`；加 `?format=csv` 直接下載結果） |

### 範例：送出訓練並輪詢結果

```bash
# 送出訓練（可省略 hyperparameters 使用預設格點）
curl -X POST http://127.0.0.1:5000/api/ml/train \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "logistic_regression"}'
# → 202 { "job_id": "cad70df6...", "status": "pending", "algorithm": "logistic_regression" }

# 輪詢狀態
curl http://127.0.0.1:5000/api/ml/train/status/cad70df6...
# → 200 { "status": "done", "result": { "best_params": {...}, "metrics": {...} } }
```

### 範例：單筆預測

```bash
curl -X POST http://127.0.0.1:5000/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Name": "Doe, Mr. John", "Pclass": 3, "Sex": "male",
        "Age": 29, "SibSp": 0, "Parch": 0,
        "Ticket": "A/5 21171", "Fare": 7.25, "Embarked": "S"
      }'
# → 200 { "survived": false, "probability": 0.07517962166555155 }
```

---

## 機器學習設計

### 特徵工程管線

```
原始欄位 → TitanicFeatureEngineer（衍生 + 補值）→ ColumnTransformer（縮放 + 編碼）→ 分類器
```

**衍生特徵**

| 特徵 | 規則 |
| :--- | :--- |
| `Title` | 以正則 `,\s*([^\.]+)\.` 自 `Name` 萃取頭銜；`Mlle`/`Ms` → `Miss`、`Mme` → `Mrs`，其餘罕見頭銜歸為 `Rare` |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `FamilySize == 1` |

**缺值補值**（統計量僅於 `fit` 學習）

| 欄位 | 策略 |
| :--- | :--- |
| `Age` | 依 `Title` 分組中位數，全域中位數為 fallback |
| `Fare` | 依 `Pclass` 分組中位數，全域中位數為 fallback |
| `Embarked` | 眾數 |

**編碼**

- 數值特徵（`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, `IsAlone`）→ `StandardScaler`
- 類別特徵（`Sex`, `Embarked`, `Title`）→ `OneHotEncoder(handle_unknown="ignore")`

### 訓練設定

| 項目 | 值 |
| :--- | :--- |
| 資料切分 | `train_test_split(test_size=0.2, stratify=y, random_state=42)` → 訓練 712 筆 / 測試 179 筆 |
| 交叉驗證 | `GridSearchCV(cv=5)`，於訓練集上執行 |
| 評估 | 於 held-out 測試集計算 Accuracy / Precision / Recall / F1 / ROC-AUC / 混淆矩陣 |
| 可重現性 | `random_state=42` 全程固定 |

### 預設超參數格點

| 演算法 | 格點 | 組合數 |
| :--- | :--- | :---: |
| Logistic Regression | `C: [0.01, 0.1, 1, 10]`、`penalty: [l1, l2]`、`solver: [liblinear]` | 8 |
| Random Forest | `n_estimators: [100, 300, 500]`、`max_depth: [None, 5, 10, 20]`、`min_samples_split: [2, 5, 10]`、`min_samples_leaf: [1, 2, 4]` | 108 |

使用者可於訓練頁面自訂候選值，後端依 `HYPERPARAM_SPECS` 驗證型別與範圍，並限制組合總數上限 200。

---

## 實測結果

資料集 `titanic.csv`（891 筆，生還 342 筆），使用預設格點，於本機執行的結果：

| 演算法 | 最佳超參數 | CV Accuracy | Test Accuracy | Precision | Recall | F1 | ROC-AUC | 訓練耗時 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=1`, `penalty=l2`, `solver=liblinear` | 0.8203 | **0.8547** | 0.8413 | 0.7681 | 0.8030 | **0.8805** | 2.4 s |
| **Random Forest** | `n_estimators=300`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2` | **0.8273** | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8534 | 15.8 s |

**混淆矩陣**（測試集 179 筆，格式 `[[TN, FP], [FN, TP]]`）

| 演算法 | 混淆矩陣 |
| :--- | :--- |
| Logistic Regression | `[[100, 10], [16, 53]]` |
| Random Forest | `[[98, 12], [21, 48]]` |

**觀察**：Random Forest 的交叉驗證分數略高（0.8273 vs 0.8203），但在 held-out 測試集上 Logistic Regression 的各項指標反而較優。這是資料量有限（891 筆）時常見的模型變異，不足以斷言任一模型明顯較佳；以泛化表現與訓練成本（2.4s vs 15.8s）綜合考量，Logistic Regression 在此資料集上是更務實的選擇。

由於 `random_state=42` 固定，同一環境重複訓練可得到相同結果（Random Forest 已重跑驗證一致）。惟 GridSearchCV 在多組參數 CV 分數並列時的選擇會受 scikit-learn 版本影響，跨版本執行可能選到不同但分數相同的參數組合。

---

## 測試

```bash
# 執行全部測試
uv run pytest

# 含覆蓋率報告
uv run pytest --cov=src --cov-report=term-missing
```

**測試結果：144 passed，服務層與展示層整體覆蓋率 98%**

| 模組 | 覆蓋率 | 模組 | 覆蓋率 |
| :--- | :---: | :--- | :---: |
| `src/ml/features.py` | 100% | `src/ml/registry.py` | 100% |
| `src/ml/training.py` | 100% | `src/ml/jobs.py` | 100% |
| `src/ml/prediction.py` | 100% | `src/ml/validation.py` | 100% |
| `src/ml/schema.py` | 100% | `src/ml/types.py` | 100% |
| `src/web/ml_pages.py` | 100% | `src/web/ml_api.py` | 98% |
| `src/ml/config.py` | 95% | `src/ml/hyperparams.py` | 90% |

測試範圍涵蓋：

- **單元測試** — 特徵工程（fit/transform 一致性）、超參數驗證、Job 狀態轉移、Registry CRUD、訓練流程、輸入驗證
- **整合測試** — ML REST API（含錯誤狀態碼）、頁面路由
- **驗收測試** — 端到端情境：訓練 → 登錄 → 啟用 → 預測

---

## 專案結構

```
titanic_restful_project/
├── app.py                     # Flask 入口：乘客 CRUD 路由 + 掛載 ML blueprint
├── init_db.py                 # 建表 + titanic.csv 匯入
├── src/
│   ├── web/                   # 展示層：輸入驗證、HTTP 狀態碼、JSON 序列化
│   │   ├── ml_api.py          #   /api/ml/* REST API
│   │   └── ml_pages.py        #   /ml/* 頁面路由
│   └── ml/                    # 服務層：純 Python，不 import Flask
│       ├── config.py          #   集中設定：超參數格點、特徵清單、CV 設定、路徑
│       ├── features.py        #   特徵工程 Pipeline（訓練／預測共用）
│       ├── hyperparams.py     #   使用者自訂超參數驗證 + 格點建構
│       ├── training.py        #   訓練核心：GridSearchCV + 指標評估
│       ├── jobs.py            #   非同步 Job 狀態管理（threading）
│       ├── registry.py        #   模型持久化 + metadata + active 切換
│       ├── prediction.py      #   單筆／批次預測
│       ├── validation.py      #   輸入驗證
│       ├── schema.py          #   ml_model / train_job 建表
│       └── types.py           #   型別定義（dataclass）
├── templates/                 # Jinja2 頁面
├── static/                    # 前端 JS 與圖片
├── tests/                     # 單元 + 整合 + 驗收測試（144 個）
├── docs/screenshots/          # README 截圖
├── models/                    # joblib 模型檔（不進版控）
└── titanic.csv                # 原始資料集
```

### 資料模型

| 表 | 用途 | 關鍵欄位 |
| :--- | :--- | :--- |
| `titanic` | 乘客資料 | `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked` |
| `ml_model` | 模型登錄 | `model_uid`, `algorithm`, `hyperparameters`, `best_cv_score`, `metrics`, `feature_list`, `data_hash`, `file_path`, `is_active` |
| `train_job` | 訓練工作 | `job_id`, `status`, `algorithm`, `progress`, `result_model_uid`, `error_message` |

---

## 作者

**YCHsieh** · [GitHub @ychsieh725](https://github.com/ychsieh725)

本專案為個人作品集，展示端到端機器學習平台的設計與實作能力。
