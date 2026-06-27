# 開發進度紀錄 — Titanic ML 平台

> **分支:** `feat/ml-platform`
> **最後更新:** 2026-06-27 15:56
> **狀態:** M1 訓練流程就緒（特徵 → 訓練 → 非同步 job → 持久化），待 4.3 訓練 API
> **測試總計:** 40 passed / 覆蓋率 99%
> **開發流程:** `/task-next → /plan → /tdd → /verify`，每任務 TDD（RED→GREEN）+ 獨立 commit

---

## 1. 里程碑進度

| 里程碑 | 目標 | 狀態 |
|---|---|---|
| M1 核心管線可訓練 | 一鍵訓練 + 超參數 + 模型儲存 | 🔄 訓練流程完成（4.3 API 未做） |
| M2 預測可用 | 單筆 + CSV 批次預測 | ⏳ 未開始 |
| M3 MVP 驗收 | UI 串接 + 6 大必做驗收 | ⏳ 未開始 |

## 2. 任務完成總覽

| # | 任務 | 狀態 | Commit |
|---|---|---|---|
| 1.x / 2.1 / 2.2 | 初始化 / 需求 / 分支 / 服務層骨架 | ✅ | （前置） |
| 3.1 | 特徵工程管線（可序列化、共用） | ✅ | `2df9da9` |
| 3.2 | 管線單元測試 | ✅ | 併入 3.1 |
| 4.1 | 訓練服務（LR+RF / GridSearchCV / 指標） | ✅ | `46b9513` |
| 2.3 | DB schema（ml_model + train_job） | ✅ | `0404e02` |
| 5.1 | 模型持久化 + metadata 寫入 | ✅ | `8ccc52c` |
| 4.2 | 非同步 job（threading） | ✅ | `495ffe5` |
| 4.3 | 訓練 API | ⏳ | — |
| 5.2 | 模型清單 + active 切換 | ⏳ | — |
| 6.x | 單筆 / CSV 批次預測 | ⏳ | — |
| 7.x / 8.x | UI / 整合測試 / 驗收 | ⏳ | — |

---

## 3. Commit 時間軸

| 時間 | Hash | 摘要 |
|---|---|---|
| 15:14 | `2df9da9` | feat(ml): 共用可序列化特徵工程管線 |
| 15:26 | `46b9513` | feat(ml): 訓練服務 + GridSearchCV 超參數搜尋 |
| 15:34 | `0404e02` | feat(ml): ml_model / train_job DB schema |
| 15:39 | `8ccc52c` | feat(ml): 模型 Registry 持久化（joblib + metadata） |
| 15:56 | `495ffe5` | feat(ml): 非同步訓練 job 執行器（threading + 狀態生命週期） |

---

## 4. 各任務細節

### 3.1 特徵工程管線 — `src/ml/features.py`

**對應需求:** FR-2.1~2.4, 2.7 / CON-2

- `TitanicFeatureEngineer`（stateful transformer）：`fit` 學習統計量、`transform` 套用
  - Age：Title 分組中位數補值（全域中位數 fallback）
  - Embarked：眾數補值；Fare：Pclass 分組中位數補值
  - Name 萃取 Title 並歸併稀有頭銜（Mr/Mrs/Miss/Master/Rare）
  - 衍生 FamilySize / IsAlone
  - `get_feature_names_out`：使整條 pipeline 可內省
- `build_feature_pipeline()`：engineer → ColumnTransformer（StandardScaler + OneHotEncoder `handle_unknown="ignore"`）
- `passenger_to_frame()`：PassengerInput → 單列 DataFrame
- **核心保證（CON-2）:** 統計量於 fit 學習、transform 套用 → 單筆預測不會「對自己算中位數」

### 4.1 訓練服務 — `src/ml/training.py`

**對應需求:** FR-3.2~3.6, 3.10

- `train_model(algorithm, X, y) -> (Pipeline, TrainResult)`
  - Pipeline = features + clf；train/test split（stratify, random_state=42）
  - GridSearchCV（cv=5, n_jobs=config.N_JOBS）；best_estimator_ 在 held-out 測試集評估
- `_evaluate()`：accuracy / precision / recall / F1 / ROC-AUC + 2×2 混淆矩陣
- `_build_estimator()`：LR / RF 工廠；未知演算法 → 明確 ValueError
- `load_training_data()`：sqlite3 參數化讀 titanic 表（NFR-S2）
- `_hash_frame()`：sha256 訓練資料雜湊（可追溯）

### 2.3 DB schema — `src/ml/schema.py`

**對應需求:** FR-4.2 / SRS §6.2, §6.3

- `init_ml_schema(db_path)`：冪等（CREATE IF NOT EXISTS），非破壞
- `ml_model` 表：JSON 欄存 TEXT、is_active 存 INTEGER 0/1
- **partial unique index** `ux_ml_model_active`（`WHERE is_active=1`）→ DB 保證至多一個 active 模型（FR-4.5）
- `train_job` 表：status CHECK 限定 pending/running/done/failed
- `init_db.py` 末尾 lazy import 呼叫（單一初始化入口，不影響 titanic 重匯）

### 5.1 模型持久化 — `src/ml/registry.py`

**對應需求:** FR-4.1, 4.2

- `save_model(pipeline, result, db_path, models_dir) -> ModelMetadata`
  - mkdir models/ → `joblib.dump` 整條 fit pipeline → 參數化 INSERT → 回讀組 ModelMetadata
- `get_model_by_uid()`：依 uid 取 metadata，不存在回 None
- `load_model_file()`：joblib.load 整條 pipeline（供預測共用，CON-2）
- save 一律 `is_active=0`（active 切換留給 5.2）

### 4.2 非同步 job 執行器 — `src/ml/jobs.py`

**對應需求:** FR-3.7~3.9 / NFR-R1

- `run_training(job_id, algorithm, store, db_path, models_dir)`：背景執行緒目標
  - mark_running → load_training_data → train_model → save_model → mark_done(model_uid)
  - 任一步失敗統一 mark_failed 並記錄錯誤，**不向外拋出**（NFR-R1）
  - training/registry 延遲匯入，避免 JobStore 純狀態管理載入 sklearn
- `start_training_job(algorithm, store, ...)`：建立 job → 啟動 daemon thread → 立即回傳 pending（FR-3.7）
  - 先驗證演算法，未知時同步拋 ValueError 且不建 job（讓 API 取得明確錯誤）
- 既有 `JobStore`（2.2）：thread-safe、不可變狀態轉移（replace 產生新 Job 快照）

---

## 5. 測試細節（40 passed）

| 測試檔 | 數量 | 覆蓋重點 |
|---|---|---|
| `tests/test_features.py` | 10 | Title 萃取/歸併、FamilySize/IsAlone、補值無 NaN、**單筆==批次一致性**、**joblib 往返不變**、未見類別韌性、PassengerInput 轉換 |
| `tests/test_training.py` | 9 | LR/RF 訓練、best_params `clf__` 前綴、指標邊界 [0,1]、混淆矩陣 2×2、**可重現性**、joblib 往返可預測、未知演算法 ValueError、DB 載入整合 |
| `tests/test_schema.py` | 7 | 建表、欄位齊全（PRAGMA 比對）、冪等、**單一 active 約束（IntegrityError）**、status CHECK |
| `tests/test_registry.py` | 7 | joblib 檔+DB 列、JSON 欄往返、預設 inactive、載回可預測、uid 查詢、UNIQUE(model_uid) 約束 |
| `tests/test_jobs.py` | 7 | JobStore create/get/list 排序/`_transition` KeyError、**run_training 成功 done**、**失敗 failed 不中斷（NFR-R1）**、start_training_job 立即回傳並完成、未知演算法 ValueError 不建 job |

**測試策略**
- 全程 TDD：先寫失敗測試（RED）→ 最小實作（GREEN）→ 重構
- 訓練測試用 monkeypatch 縮小超參數格點 + 40 列 synthetic 資料（每類 ≥ 5 筆滿足 cv=5/stratify）
- DB/檔案測試一律用 `tmp_path`，不碰專案 `my_db.db` 或 `models/`

### 覆蓋率（`pytest --cov=src`）

| 模組 | Stmts | Miss | Cover | 備註 |
|---|---|---|---|---|
| `src/ml/features.py` | 53 | 0 | **100%** | |
| `src/ml/training.py` | 55 | 0 | **100%** | |
| `src/ml/schema.py` | 15 | 0 | **100%** | |
| `src/ml/registry.py` | 41 | 0 | **100%** | |
| `src/ml/jobs.py` | 59 | 0 | **100%** | 4.2 補滿（原 50%） |
| `src/ml/types.py` | 70 | 0 | **100%** | |
| `src/ml/config.py` | 26 | 2 | 92% | 未覆蓋：`get_param_grid` 冗餘錯誤分支 |
| **TOTAL** | **321** | **2** | **99%** | 門檻 80% ✅ |

### 已知警告（非專案程式碼）
- 3× `DeprecationWarning`（NumPy 2.5 shape API），來自 joblib 內部序列化，不影響功能。

---

## 6. 專案約束驗證

| 約束 | 狀態 | 驗證方式 |
|---|---|---|
| CON-2 共用序列化管線 | ✅ | 存/載皆整條 pipeline；單筆==批次、joblib 往返測試 |
| CON-4 服務層框架解耦 | ✅ | `src/ml/` 無 import flask（grep 驗證） |
| CON-3 模型持久化 | ✅ | joblib 存模型、metadata 入 ml_model 表 |
| NFR-S2 參數化查詢 | ✅ | 所有 SQL 用 `?` 佔位，無字串拼接 |
| testing 80%+ | ✅ | 93% |

---

## 7. 下一步

| 順序 | 任務 | 依賴 | 說明 |
|---|---|---|---|
| 1 | **4.3 訓練 API** | 4.2 ✅ | POST /api/ml/train（呼叫 start_training_job）、GET /train/status/{job_id} |
| 2 | 5.2 模型清單 + active 切換 | 5.1 ✅ | 完成後預測端可取 active 模型 |
| 3 | 6.x 單筆 / CSV 批次預測 | 5.x | M2 |
