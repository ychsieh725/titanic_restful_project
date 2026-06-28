# 開發進度紀錄 — Titanic ML 平台

> **分支:** `feat/ml-platform`
> **最後更新:** 2026-06-28 12:20
> **狀態:** ✅ M1 + M2 + M3 全達成；MVP 六大必做項逐項綠燈，待開 PR 合回 main
> **測試總計:** 116 passed / 覆蓋率 99%
> **開發流程:** `/task-next → /plan → /tdd → /verify`，每任務 TDD（RED→GREEN）+ 獨立 commit

---

## 1. 里程碑進度

| 里程碑 | 目標 | 狀態 |
|---|---|---|
| M1 核心管線可訓練 | 一鍵訓練 + 超參數 + 模型儲存 | ✅ 完成 |
| M2 預測可用 | 單筆 + CSV 批次預測 | ✅ 完成（5.2 + 6.1 + 6.2 + 6.3） |
| M3 MVP 驗收 | UI 串接 + 6 大必做驗收 | ✅ 完成（7.x UI + 8.x 整合/驗收） |

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
| 4.3 | 訓練 API（POST train / GET status） | ✅ | `a356094` |
| 5.2 | 模型清單 + active 切換 API | ✅ | `a4aecd0` |
| 6.1 | 單筆預測服務 + API | ✅ | `3890a15` |
| 6.3 | 輸入驗證（schema-based） | ✅ | `2a72f3d` |
| 6.2 | CSV 批次預測 + 結果 CSV 下載 | ✅ | `d3ad13f` |
| 7.1–7.4 | UI 頁面（訓練/模型/預測 + 導覽列） | ✅ | `505c92d` |
| 8.1 / 8.2 | 端到端整合測試 + 六大必做驗收 | ✅ | (本次) |

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

### 4.3 訓練 API — `src/web/ml_api.py`（新增展示層 package）

**對應需求:** SRS §4.2 / FR-3.7~3.9

- `create_ml_blueprint(store, db_path, models_dir)`：blueprint 工廠，依賴注入便於測試
- `POST /api/ml/train`：缺 body/欄位 → 400；未知演算法 → 422；成功 → 202 + job_id
- `GET /api/ml/train/status/<job_id>`：不存在 → 404；存在 → 200，done 時附
  `result.{best_params, best_cv_score, metrics}`（FR-3.8）
- **分層**：blueprint 屬展示層（可 import flask + src.ml），`src/ml/` 維持
  框架無關（CON-4）；`app.py` 以 register_blueprint 掛載

### 5.2 模型清單 + active 切換 — `src/ml/registry.py` + `src/web/ml_api.py`

**對應需求:** FR-4.3, 4.4, 4.5

- registry：`list_models()`（新→舊）、`get_active_model()`（供 6.x 預測）、
  `set_active_model(id)`（單一交易內先清後設，避免 partial unique index 衝突；
  不存在回 None）
- API：`GET /api/ml/models` → 200 清單；`POST /api/ml/models/<id>/activate`
  → 200 / 不存在 404；`_metadata_payload` 序列化（datetime → isoformat）
- **至多一個 active** 由 DB 的 partial unique index + 交易順序共同保證

### 6.1 單筆預測 — `src/ml/prediction.py` + `src/web/ml_api.py`

**對應需求:** FR-5.1, 5.2, 4.5

- `predict_one(passenger, db_path)`：get_active_model → load_model_file →
  passenger_to_frame → `predict_proba`[:,1] → PredictionResult(survived, probability)
- `NoActiveModelError`：服務層網域例外，無 active 模型時拋出（不裸奔 500）
- API `POST /api/ml/predict`：缺 body/必填欄位 → 400（指出缺哪欄）；
  無 active 模型 → **409**；成功 → 200 {survived, probability}
- 共用 active pipeline（CON-2），預測端不另寫前處理

### 6.3 輸入驗證 — `src/ml/validation.py`

**對應需求:** FR-5.5 / NFR-S1, S2

- `validate_passenger(data) -> PassengerInput`：逐欄驗證缺漏/型別/範圍/允許值，
  **一次蒐集所有錯誤**（非 fail-fast），訊息以使用者語氣指出哪欄、如何修正
- `ValidationError(errors: dict[欄位, 訊息])`：服務層網域例外
- 規則對齊 SRS §6.1 + init_db CHECK（Pclass∈{1,2,3}、Sex∈{male,female}、
  Age 0–120、SibSp/Parch≥0、Fare≥0、Embarked∈{C,Q,S}、長度上限），常數集中 config
- **契約精修**：`/predict` 欄位驗證失敗由 400 改 **422**（格式對但語意錯，
  含 `fields` 明細）；400 僅保留給無法解析的 body

### 6.2 CSV 批次預測 — `src/ml/prediction.py` + `validation.py` + `src/web/ml_api.py`

**對應需求:** FR-5.3, 5.4

- **服務層**
  - `features.passengers_to_frame(list)`：多筆轉管線契約 DataFrame；
    `passenger_to_frame` 改委派此函式（DRY）
  - `prediction.predict_batch(passengers, db_path) -> DataFrame`：一次載 active
    pipeline 整批 `predict_proba`（共用管線 CON-2），附 `probability`/`survived`
    兩欄；機率/判定以**原生 Python 型別**輸出（供 jsonify / CSV）；空清單回空結果
  - `validation.validate_passengers(records)`：逐列重用 `validate_passenger`
    蒐集所有問題列，全合法回 list；否則拋 `BatchValidationError(row_errors)`
  - `config.MAX_BATCH_ROWS = 10000`：單次上限
- **展示層** `POST /api/ml/predict/batch`（multipart 上傳）
  - 無檔 / 非 `.csv` / 解析失敗 / 0 列 → **400**；超 `MAX_BATCH_ROWS` → **413**
  - 驗證失敗 → **422** `{rows:[{row, fields}]}`（標出每列）；無 active → **409**
  - 預設 **200** JSON `{results, total}`；`?format=csv` → **200** `text/csv` 附件下載
  - `_csv_to_records`：在資料邊界把 pandas NaN → None、numpy 純量 → 原生型別，
    讓服務層維持函式庫無關（CON-4）；修正 `where` 無法在 float 欄存 None 之陷阱

---

## 5. 測試細節（100 passed）

| 測試檔 | 數量 | 覆蓋重點 |
|---|---|---|
| `tests/test_features.py` | 10 | Title 萃取/歸併、FamilySize/IsAlone、補值無 NaN、**單筆==批次一致性**、**joblib 往返不變**、未見類別韌性、PassengerInput 轉換 |
| `tests/test_training.py` | 9 | LR/RF 訓練、best_params `clf__` 前綴、指標邊界 [0,1]、混淆矩陣 2×2、**可重現性**、joblib 往返可預測、未知演算法 ValueError、DB 載入整合 |
| `tests/test_schema.py` | 7 | 建表、欄位齊全（PRAGMA 比對）、冪等、**單一 active 約束（IntegrityError）**、status CHECK |
| `tests/test_registry.py` | 12 | joblib 檔+DB 列、JSON 欄往返、預設 inactive、載回可預測、uid 查詢、UNIQUE(model_uid) 約束、**清單排序、active 切換唯一性、get_active** |
| `tests/test_jobs.py` | 7 | JobStore create/get/list 排序/`_transition` KeyError、**run_training 成功 done**、**失敗 failed 不中斷（NFR-R1）**、start_training_job 立即回傳並完成、未知演算法 ValueError 不建 job |
| `tests/test_ml_api.py` | 25 | 訓練（400/422/202）、status（404/done含指標/failed）、模型清單、activate（200/404/切換唯一）、單筆預測（400/422含 fields/409/200）、**批次（無檔/非csv/空檔/壞CSV→400、超列→413、壞列→422含 row、無 active→409、成功→200 JSON、`?format=csv`→text/csv 附件）** |
| `tests/test_prediction.py` | 8 | predict_one 回 PredictionResult、survived 對應門檻、無 active→例外；**predict_batch：survived/probability 欄、門檻一致、原生型別、空清單、無 active→例外** |
| `tests/test_validation.py` | 10（含參數化 共 30 例） | 合法→PassengerInput、可空欄位、**一次蒐集所有缺漏**、範圍/允許值、型別、長度上限；**批次：全合法→list、空→[]、依列索引蒐集錯誤** |

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
| `src/ml/registry.py` | 73 | 0 | **100%** | 含 5.2 清單/active |
| `src/ml/jobs.py` | 59 | 0 | **100%** | 4.2 補滿（原 50%） |
| `src/ml/prediction.py` | 14 | 0 | **100%** | 6.1 單筆預測 |
| `src/ml/validation.py` | 93 | 0 | **100%** | 6.3 輸入驗證 |
| `src/ml/types.py` | 70 | 0 | **100%** | |
| `src/web/ml_api.py` | 61 | 0 | **100%** | 4.3 + 5.2 + 6.1 + 6.3 API |
| `src/ml/config.py` | 34 | 2 | 94% | 未覆蓋：`get_param_grid` 冗餘錯誤分支 |
| **TOTAL** | **529** | **2** | **99%** | 門檻 80% ✅ |

### 已知警告（非專案程式碼）
- 3× `DeprecationWarning`（NumPy 2.5 shape API），來自 joblib 內部序列化，不影響功能。

---

## 6. 專案約束驗證

| 約束 | 狀態 | 驗證方式 |
|---|---|---|
| CON-2 共用序列化管線 | ✅ | 存/載皆整條 pipeline；單筆==批次、joblib 往返測試 |
| CON-4 服務層框架解耦 | ✅ | `src/ml/` 無 import flask；展示層獨立於 `src/web/`（grep 驗證） |
| CON-3 模型持久化 | ✅ | joblib 存模型、metadata 入 ml_model 表 |
| NFR-S2 參數化查詢 | ✅ | 所有 SQL 用 `?` 佔位，無字串拼接 |
| testing 80%+ | ✅ | 99% |

---

## 7. SRS §8 六大必做項驗收（test_acceptance.py 端到端）

| # | 驗收項 | 狀態 | 對應測試 |
|---|---|---|---|
| AC1 | 一鍵訓練且能得知完成（running→done） | ✅ | `test_train_to_registry_flow` |
| AC2 | 超參數調校且顯示最佳超參數與指標 | ✅ | 同上（斷言 best_params + metrics） |
| AC3 | 模型被儲存且可於模型頁列出 | ✅ | 同上（joblib 檔存在 + /models 列出） |
| AC4 | 單筆預測輸出存活判定與生存機率 | ✅ | `test_single_prediction_outputs_survival_and_probability` |
| AC5 | CSV 批次預測並可下載含機率結果 | ✅ | `test_batch_prediction_json_and_csv_download` |
| AC6 | 預測與訓練前處理一致（同一序列化管線） | ✅ | `test_shared_pipeline_single_equals_batch` |

> 全程以完整 app（API + 頁面兩 blueprint）對臨時 DB 跑端到端流程；MVP 必做需求全數通過。

---

## 8. 下一步

| 順序 | 任務 | 說明 |
|---|---|---|
| 1 | 開 PR 合回 main | MVP 完成；`gh pr create`（WHY/WHAT/IMPACT） |
| 2 | （Optional 加分）FR-6 EDA 視覺化 | 性別×艙等存活率熱力圖、特徵重要度 |
