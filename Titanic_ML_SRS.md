# Titanic 生存預測機器學習平台
## 軟體需求規格書（Software Requirements Specification, SRS）

> 建議專案代號：**Project Manifest**（取自船舶「乘客名冊／登船名單 manifest」之意，呼應預測結果以名冊卡呈現的設計主題；可自行更換）

| 項目 | 內容 |
|---|---|
| 文件版本 | v1.0 |
| 文件狀態 | Draft |
| 撰寫基準 | IEEE Std 830-1998 / ISO/IEC/IEEE 29148:2018 |
| 適用範圍 | 從零開始之獨立 Web 應用程式（不依附既有 base 專案） |
| 目標讀者 | 開發者本人、課程指導老師、作品集審閱者（潛在雇主） |

### 版本歷程

| 版本 | 日期 | 修改摘要 |
|---|---|---|
| v1.0 | （填入） | 初版，涵蓋訓練／預測／模型管理／EDA 全功能需求 |

---

## 1. 緒論（Introduction）

### 1.1 目的（Purpose）

本文件定義「Titanic 生存預測機器學習平台」的完整軟體需求。系統提供一個 Web 介面，讓使用者能對 Titanic 乘客資料進行**機器學習模型訓練（含超參數調校）、訓練狀態監控、模型儲存與管理、單筆與批次存活預測（含生存機率）**，並提供一組**資料分析視覺化頁面**作為作品集展示。本 SRS 作為設計、實作、測試與驗收的共同依據。

### 1.2 產品範圍（Scope）

**包含（In Scope）**

- 以 `titanic` 資料表為訓練來源的端到端機器學習流程。
- 共用的特徵工程管線（feature engineering pipeline），確保訓練與預測前處理一致。
- 一鍵訓練、超參數搜尋、最佳參數顯示。
- 非同步訓練工作（job）與訓練狀態查詢。
- 訓練完成模型之持久化儲存與迷你模型登錄（Model Registry）。
- 單筆表單預測與 CSV 批次預測，輸出存活判定與生存機率。
- 資料分析與模型可解釋性視覺化頁面（Optional，作品集導向）。

**不包含（Out of Scope）**

- 使用者帳號系統、權限管理、多租戶。
- 分散式訓練、GPU 訓練、大規模資料管線。
- 線上模型自動再訓練排程（僅保留手動觸發；自動化列為未來工作）。
- 與外部生產系統之整合部署（僅需可於本機或單一伺服器運行）。

### 1.3 名詞、縮寫與定義（Definitions, Acronyms, Abbreviations）

| 縮寫／名詞 | 說明 |
|---|---|
| SRS | Software Requirements Specification，軟體需求規格書 |
| EDA | Exploratory Data Analysis，探索性資料分析 |
| Pipeline | sklearn 之前處理＋模型串接物件，可整體序列化 |
| GridSearchCV | 網格搜尋＋交叉驗證的超參數調校方法 |
| Job | 一次非同步訓練工作，具獨立 `job_id` 與狀態 |
| Model Registry | 模型登錄，記錄各訓練模型 metadata 並標示啟用中模型 |
| Active Model | 目前供預測使用的啟用中模型 |
| HITL | Human-in-the-Loop，人在迴路 |
| `predict_proba` | sklearn 取得分類機率的方法，用於輸出生存機率 |
| Title | 由 `Name` 萃取之頭銜（Mr/Mrs/Miss/Master/Rare） |

### 1.4 參考資料（References）

- IEEE Std 830-1998, Recommended Practice for Software Requirements Specifications.
- ISO/IEC/IEEE 29148:2018, Systems and software engineering — Life cycle processes — Requirements engineering.
- Kaggle, *Titanic - Machine Learning from Disaster* 資料集欄位定義。
- scikit-learn 官方文件：`Pipeline`、`ColumnTransformer`、`GridSearchCV`。

### 1.5 文件慣例（Conventions）

- 需求以「系統應（shall）…」描述，每條具唯一編號（FR-x / NFR-x）。
- 優先級：**M（Must，必做，對應作業必做需求）**、**S（Should，建議）**、**C（Could，加分／Optional）**。
- 需求編號一經發布不重用，新增以遞增編號處理。

---

## 2. 整體描述（Overall Description）

### 2.1 產品觀點（Product Perspective）

本系統為**獨立、自包含的 Web 應用程式**，採三層邏輯切分：

```
┌─────────────────────────────────────────────┐
│  展示層 Presentation                          │
│  Web 頁面（首頁 / 訓練 / 模型 / 預測 / EDA）   │
│  + REST API 端點                              │
├─────────────────────────────────────────────┤
│  服務層 Service（框架無關）                    │
│  特徵工程 / 訓練 / 模型登錄 / 預測 / EDA 運算  │
├─────────────────────────────────────────────┤
│  資料層 Data                                  │
│  titanic 表 / ml_model 表 / 模型檔（joblib）   │
└─────────────────────────────────────────────┘
```

> 設計約束：**服務層不得依賴 Web 框架**（純 Python 模組），使展示層可在 Flask / FastAPI 之間替換而不影響核心邏輯（對應可攜性需求 NFR）。

### 2.2 產品功能摘要（Product Functions）

1. 載入並驗證 `titanic` 訓練資料。
2. 透過共用管線進行缺失值處理與特徵萃取。
3. 一鍵訓練、超參數搜尋、回報最佳參數與評估指標。
4. 以非同步 job 執行訓練並可查詢完成狀態。
5. 儲存訓練後模型並於 Model Registry 管理、切換啟用模型。
6. 單筆與 CSV 批次預測，輸出存活判定與生存機率。
7. 多面向資料分析與模型可解釋性視覺化。

### 2.3 使用者類別與特性（User Classes and Characteristics）

| 使用者類別 | 描述 | 主要使用功能 |
|---|---|---|
| 一般操作者 | 課堂示範、作品集瀏覽者，無 ML 背景 | 預測、瀏覽 EDA |
| 資料／開發者 | 專案作者、需調校模型者 | 訓練、超參數調校、模型管理 |
| 審閱者 | 老師、面試官 | 全功能，著重設計品質與工程嚴謹度 |

### 2.4 運作環境（Operating Environment）

- 伺服器端：Python 3.11+，可於本機（localhost）或單一 Linux/Windows 主機運行。
- 用戶端：現代瀏覽器（Chrome / Edge / Firefox / Safari 最新版）。
- 資料庫：開發用 SQLite；可切換至 PostgreSQL（與 CareLoop 技術線一致，便於作品集敘事）。
- 模型檔儲存於本機檔案系統 `models/` 目錄。

### 2.5 設計與實作限制（Design and Implementation Constraints）

| 編號 | 限制 |
|---|---|
| CON-1 | 後端語言為 Python；ML 使用 scikit-learn 生態系。 |
| CON-2 | 訓練與預測**必須共用同一條序列化管線**，禁止預測時另寫前處理。 |
| CON-3 | 模型以 `joblib` 持久化；metadata 存於 `ml_model` 資料表。 |
| CON-4 | 服務層與 Web 框架解耦（純 Python，不 import 框架物件）。 |
| CON-5 | 視覺化建議使用 Plotly（互動式）；可退用 matplotlib/seaborn 出圖。 |
| CON-6 | 非同步訓練至少以標準庫 `threading` 實作；可選用 RQ/Celery 升級。 |

### 2.6 假設與相依（Assumptions and Dependencies）

- 假設 `titanic` 資料表保留近似原始 Kaggle 欄位，特別是 `Name`、`Cabin`、`Embarked`，否則 `Title`、`Deck` 等特徵無法萃取。
- 假設資料量為單機可處理規模（約 891 列；批次預測上限見 NFR）。
- 相依套件：`scikit-learn`、`pandas`、`joblib`、`plotly`、Web 框架（Flask 或 FastAPI）、ORM（SQLAlchemy）。

---

## 3. 功能需求（Functional Requirements）

> 優先級欄位：M＝必做、S＝建議、C＝加分／Optional。

### 3.1 FR-1 資料管理（Data Management）

| 編號 | 需求敘述 | 優先級 |
|---|---|---|
| FR-1.1 | 系統應提供 `titanic` 資料表作為訓練資料來源，欄位定義見 §6.1。 | M |
| FR-1.2 | 系統應能自 `titanic` 表載入完整訓練資料集供訓練使用。 | M |
| FR-1.3 | 系統應在載入時檢查必要欄位是否存在，缺漏時回報明確錯誤。 | M |
| FR-1.4 | 系統應計算並記錄該次訓練資料的筆數與資料雜湊（data hash），供模型可追溯。 | S |

### 3.2 FR-2 特徵工程（Feature Engineering）

| 編號 | 需求敘述 | 優先級 |
|---|---|---|
| FR-2.1 | 系統應對 `Age` 缺失值以 `Title`（或 `Pclass`×`Sex`）分組中位數補值。 | M |
| FR-2.2 | 系統應對 `Embarked` 缺失值補眾數、對 `Fare` 缺失值以同 `Pclass` 中位數補值。 | M |
| FR-2.3 | 系統應自 `Name` 萃取 `Title` 並將稀有頭銜歸併（Mr/Mrs/Miss/Master/Rare）。 | M |
| FR-2.4 | 系統應衍生 `FamilySize`(=SibSp+Parch+1) 與 `IsAlone`。 | M |
| FR-2.5 | 系統應自 `Cabin` 衍生 `HasCabin`，並可萃取 `Deck`（首字母）。 | S |
| FR-2.6 | 系統應衍生 `FarePerPerson`（以同票號人數平攤票價）。 | C |
| FR-2.7 | 系統應將上述前處理與類別編碼封裝為單一可序列化管線，訓練與預測共用同一物件。 | M |

### 3.3 FR-3 模型訓練與超參數調校（Training & Hyperparameter Tuning）

| 編號 | 需求敘述 | 優先級 |
|---|---|---|
| FR-3.1 | 系統應提供「一鍵訓練」操作，由 `titanic` 表資料啟動完整訓練流程。 | M |
| FR-3.2 | 系統應支援至少兩種演算法供選擇：Logistic Regression 與 Random Forest。 | M |
| FR-3.3 | 系統應使用 GridSearchCV（或 RandomizedSearchCV）執行超參數搜尋，交叉驗證折數 `cv ≥ 5`。 | M |
| FR-3.4 | 系統應允許設定每種演算法之超參數搜尋格點（參數數量由開發者決定，至少 2 個超參數）。 | M |
| FR-3.5 | 系統應於訓練完成後顯示**最佳超參數**（`best_params_`）與交叉驗證最佳分數。 | M |
| FR-3.6 | 系統應將資料切分為訓練／測試集，並於 held-out 測試集計算 accuracy、precision、recall、F1、ROC-AUC 及混淆矩陣。 | M |
| FR-3.7 | 系統應以非同步工作（job）執行訓練，立即回傳 `job_id`，不阻塞使用者介面。 | M |
| FR-3.8 | 系統應提供查詢端點，依 `job_id` 回報狀態：`pending` / `running` / `done` / `failed`，並於完成時附上指標摘要。 | M |
| FR-3.9 | 訓練失敗時，系統應將 job 標記為 `failed` 並記錄錯誤訊息，不得使整體服務中斷。 | M |
| FR-3.10 | 系統應支援設定隨機種子（random_state），確保訓練結果可重現。 | S |

### 3.4 FR-4 模型儲存與管理（Model Persistence & Registry）

| 編號 | 需求敘述 | 優先級 |
|---|---|---|
| FR-4.1 | 系統應於訓練完成後，以 `joblib` 將**整條管線（含前處理＋模型）**持久化至 `models/` 目錄。 | M |
| FR-4.2 | 系統應於 `ml_model` 表記錄模型 metadata（演算法、最佳超參數、各項指標、特徵清單、訓練筆數、資料雜湊、檔案路徑、訓練時間、建立時間），定義見 §6.2。 | M |
| FR-4.3 | 系統應提供模型清單頁，列出所有已訓練模型及其指標與建立時間。 | M |
| FR-4.4 | 系統應允許將任一模型設定為「啟用中模型（active）」，同時間至多一個 active。 | S |
| FR-4.5 | 預測功能應一律使用 active 模型；若無 active 模型，應提示使用者先完成訓練。 | M |
| FR-4.6 | 系統應顯示每個模型的可追溯資訊（用何參數、何資料、何時訓練），呼應 MLOps 可追溯原則。 | S |

### 3.5 FR-5 預測（Prediction）

| 編號 | 需求敘述 | 優先級 |
|---|---|---|
| FR-5.1 | 系統應提供單筆輸入表單（Pclass、Sex、Age、SibSp、Parch、Fare、Embarked、Name 等欄位）進行預測。 | M |
| FR-5.2 | 系統應回傳**存活判定（是／否）**與**生存機率**（以 `predict_proba` 取得，0–1 或百分比）。 | M |
| FR-5.3 | 系統應提供 CSV 上傳介面進行批次預測。 | M |
| FR-5.4 | 批次預測結果應於頁面以表格呈現，並提供含 `survived` 與 `probability` 欄位之結果 CSV 下載。 | M |
| FR-5.5 | 系統應驗證輸入（單筆與 CSV），對缺漏欄位、型別錯誤、超出範圍給予以介面語氣描述的明確錯誤（指出缺哪欄、如何修正），不得回傳未處理的 500。 | M |
| FR-5.6 | 系統應限制上傳檔案格式為 CSV 並限制大小（見 NFR-S）。 | S |
| FR-5.7 | 系統宜以視覺化方式呈現單筆預測結果（如名冊卡／登船證樣式，標示存活機率），強化作品集記憶點。 | C |

### 3.6 FR-6 資料分析視覺化（EDA & Explainability，Optional 作品集導向）

| 編號 | 需求敘述 | 優先級 |
|---|---|---|
| FR-6.1 | 系統應提供人口統計分析：性別存活率（驗證「女士與兒童優先」）、年齡分佈與各年齡層存活率。 | C |
| FR-6.2 | 系統應提供社會階級分析：各 `Pclass` 存活率、`Fare` 與存活關係、**性別×艙等存活率熱力圖**。 | C |
| FR-6.3 | 系統應提供家庭結構分析：`SibSp`／`Parch`／`FamilySize` 存活率，並標示最佳生存家庭規模。 | C |
| FR-6.4 | 系統應提供登船港口分析：各 `Embarked` 存活率，並交叉 `Pclass` 說明組成差異。 | C |
| FR-6.5 | 系統應提供特徵工程驗證視覺化：缺失值分佈、各 `Title` 存活率、`HasCabin` 與存活關係。 | C |
| FR-6.6 | 系統應提供模型可解釋性視覺化：Random Forest 特徵重要度、ROC 曲線、混淆矩陣。 | C |
| FR-6.7 | 系統宜提供互動式 what-if 介面（調整年齡／艙等／性別等即時更新預測機率），展示 HITL 互動設計。 | C |

### 3.7 FR-7 使用者介面與導覽（UI & Navigation）

| 編號 | 需求敘述 | 優先級 |
|---|---|---|
| FR-7.1 | 系統應提供首頁，含專案簡介、資料集概況與各功能入口。 | M |
| FR-7.2 | 各頁面應具一致的導覽列，可於訓練／模型／預測／EDA 間切換。 | M |
| FR-7.3 | 介面應為響應式，於桌機與行動裝置皆可正常使用。 | S |
| FR-7.4 | 訓練頁應即時（輪詢或推播）反映 job 狀態變化，使用者無需手動重整即可得知是否完成。 | M |

---

## 4. 外部介面需求（External Interface Requirements）

### 4.1 使用者介面（User Interfaces）

- 首頁、訓練頁、模型管理頁、預測頁、EDA 頁。
- 訓練頁需含：演算法選擇、開始訓練按鈕、狀態指示（pending/running/done/failed）、結果區（最佳超參數＋指標）。
- 預測頁需含：單筆表單區、CSV 上傳區、結果表格與下載按鈕。

### 4.2 應用程式介面（API，REST）

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/api/ml/train` | 啟動訓練 job，body 含演算法與選用設定，回傳 `job_id` |
| GET | `/api/ml/train/status/{job_id}` | 查詢訓練 job 狀態與結果摘要 |
| GET | `/api/ml/models` | 取得模型清單與 metadata |
| POST | `/api/ml/models/{id}/activate` | 設定 active 模型 |
| POST | `/api/ml/predict` | 單筆預測，回傳存活判定＋機率 |
| POST | `/api/ml/predict/batch` | CSV 批次預測，回傳結果集／下載連結 |
| GET | `/api/eda/{chart}` | 取得指定分析圖表資料／圖（Optional） |

> 回應格式統一為 JSON；錯誤回應含 `error` 訊息與適當 HTTP 狀態碼（400 輸入錯誤、404 找不到、422 驗證失敗、500 內部錯誤）。

### 4.3 軟體介面（Software Interfaces）

- scikit-learn：`Pipeline`、`ColumnTransformer`、`GridSearchCV`、各分類器。
- pandas：資料載入與特徵運算。
- joblib：模型序列化。
- Plotly（或 matplotlib/seaborn）：視覺化。
- SQLAlchemy：ORM；SQLite／PostgreSQL。

---

## 5. 非功能需求（Non-Functional Requirements）

| 編號 | 類別 | 需求敘述 | 優先級 |
|---|---|---|---|
| NFR-P1 | 效能 | 在 891 列規模下，單次超參數搜尋訓練應於合理時間內完成（建議 < 60 秒）。 | M |
| NFR-P2 | 效能 | 單筆預測回應時間應 < 1 秒（已載入模型情況下）。 | M |
| NFR-P3 | 效能 | 批次預測應支援至少 10,000 列，並對過大檔案給出限制提示。 | S |
| NFR-U1 | 易用性 | 一般使用者無需閱讀說明即可完成一次預測；錯誤訊息以使用者語氣、具可行修正指引。 | M |
| NFR-R1 | 可靠性 | 單一 job 失敗不得影響其他 job 或整體服務；狀態須正確反映 failed。 | M |
| NFR-R2 | 可靠性 | 服務重啟後，已儲存模型與其 metadata 應仍可被列出與使用。 | M |
| NFR-S1 | 安全性 | 檔案上傳僅接受 CSV、限制大小、限制列數，並防止以惡意內容觸發程式執行。 | M |
| NFR-S2 | 安全性 | 不得將使用者輸入直接拼接為 SQL；一律使用 ORM／參數化查詢。 | M |
| NFR-M1 | 可維護性 | 服務層與展示層分離；特徵工程、訓練、預測各自為獨立模組。 | M |
| NFR-M2 | 可維護性 | 超參數格點、特徵清單等以設定集中管理，便於調整。 | S |
| NFR-Po1 | 可攜性 | 服務層不依賴特定 Web 框架，可在 Flask／FastAPI 間替換。 | S |
| NFR-Po2 | 可攜性 | 資料庫可在 SQLite 與 PostgreSQL 間切換而不需改動服務層邏輯。 | S |
| NFR-C1 | 相容性 | 支援主流現代瀏覽器最新版本。 | S |

---

## 6. 資料需求（Data Requirements）

### 6.1 `titanic` 資料表（資料字典）

| 欄位 | 型別 | 可空 | 說明 |
|---|---|---|---|
| PassengerId | INTEGER (PK) | 否 | 乘客編號 |
| Survived | INTEGER (0/1) | 否 | 目標變數：0＝罹難、1＝生還 |
| Pclass | INTEGER (1/2/3) | 否 | 艙等（社經階級 proxy） |
| Name | TEXT | 否 | 姓名（含頭銜，供 Title 萃取） |
| Sex | TEXT (male/female) | 否 | 性別 |
| Age | REAL | 是 | 年齡（約 177 筆缺失） |
| SibSp | INTEGER | 否 | 同行兄弟姊妹／配偶數 |
| Parch | INTEGER | 否 | 同行父母／子女數 |
| Ticket | TEXT | 否 | 票號 |
| Fare | REAL | 是 | 票價（少數缺失） |
| Cabin | TEXT | 是 | 艙房（大量缺失，供 HasCabin/Deck 萃取） |
| Embarked | TEXT (C/Q/S) | 是 | 登船港口（少數缺失） |

### 6.2 `ml_model` 資料表（模型登錄字典）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INTEGER (PK) | 流水號 |
| model_uid | TEXT | 模型唯一識別碼（UUID） |
| algorithm | TEXT | 演算法名稱（logistic_regression / random_forest …） |
| hyperparameters | JSON | 最佳超參數 `best_params_` |
| best_cv_score | REAL | 交叉驗證最佳分數 |
| metrics | JSON | accuracy / precision / recall / f1 / roc_auc |
| feature_list | JSON | 訓練所用特徵欄位 |
| training_rows | INTEGER | 訓練資料筆數 |
| data_hash | TEXT | 訓練資料雜湊，供可追溯 |
| file_path | TEXT | 模型 joblib 檔路徑 |
| training_duration_sec | REAL | 訓練耗時（秒） |
| is_active | BOOLEAN | 是否為啟用中模型 |
| created_at | DATETIME | 建立時間 |

### 6.3 `train_job` 資料（訓練工作狀態，可存記憶體或 DB）

| 欄位 | 型別 | 說明 |
|---|---|---|
| job_id | TEXT | 工作唯一識別碼（UUID） |
| status | TEXT | pending / running / done / failed |
| algorithm | TEXT | 本次訓練演算法 |
| progress | TEXT/REAL | 進度描述或百分比 |
| result_model_uid | TEXT | 完成後對應的模型 UID |
| error_message | TEXT | 失敗原因 |
| created_at / finished_at | DATETIME | 起訖時間 |

---

## 7. 使用案例（Use Cases）

| 編號 | 名稱 | 主要流程摘要 |
|---|---|---|
| UC-01 | 訓練模型 | 使用者選演算法 → 按一鍵訓練 → 系統建立 job 回傳 `job_id` → 背景執行超參數搜尋 → 完成後寫入 Registry 並儲存模型檔。 |
| UC-02 | 監控訓練狀態 | 使用者於訓練頁觀察狀態由 running → done，並看到最佳超參數與指標；失敗則顯示 failed 與原因。 |
| UC-03 | 管理／切換模型 | 使用者於模型頁瀏覽清單 → 選定某模型設為 active → 後續預測改用該模型。 |
| UC-04 | 單筆預測 | 使用者填表 → 送出 → 系統以 active 模型回傳存活判定與生存機率（含結果視覺化）。 |
| UC-05 | 批次預測 | 使用者上傳 CSV → 系統驗證並逐列預測 → 顯示結果表格並提供下載。 |
| UC-06 | 瀏覽資料分析 | 使用者瀏覽各 EDA 頁，獲得人口、階級、家庭、港口、特徵工程與模型可解釋性洞察。 |

---

## 8. 驗收準則（Acceptance Criteria，對應必做需求）

系統通過驗收須同時滿足：

1. 可於頁面**一鍵**啟動訓練，並於介面**得知訓練是否完成**（狀態由 running 變為 done）。【FR-3.1, FR-3.7, FR-3.8, FR-7.4】
2. 訓練使用**超參數調校**且**顯示最佳超參數**與評估指標。【FR-3.3, FR-3.4, FR-3.5, FR-3.6】
3. 訓練完成之模型**被儲存**且可於模型頁列出。【FR-4.1, FR-4.2, FR-4.3】
4. 可進行**單筆輸入預測**，輸出**存活判定與生存機率**。【FR-5.1, FR-5.2】
5. 可進行 **CSV 批次預測**並可下載含機率之結果。【FR-5.3, FR-5.4】
6. 預測前處理與訓練前處理**一致**（同一序列化管線）。【FR-2.7, CON-2】

> Optional（作品集加分）視覺化頁面（FR-6 系列）不影響必做驗收，但建議至少完成「性別×艙等存活率熱力圖」與「特徵重要度」兩張。

---

## 9. 需求追溯矩陣（作業要求 → 需求編號）

| 作業要求 | 對應需求 | 性質 |
|---|---|---|
| 新增一個或多個機器學習相關頁面 | FR-7.1, FR-7.2 + 訓練/模型/預測各頁 | 必做 |
| 一鍵將 titanic 資料訓練模型 | FR-1.2, FR-3.1 | 必做 |
| 調整超參數並顯示最佳超參數資訊 | FR-3.3, FR-3.4, FR-3.5 | 必做 |
| 能觀察／知道模型是否訓練完成 | FR-3.7, FR-3.8, FR-7.4 | 必做 |
| 將訓練好的模型儲存起來 | FR-4.1, FR-4.2 | 必做 |
| 輸入資料預測是否存活＋生存機率 | FR-5.1, FR-5.2 | 必做 |
| 上傳 CSV 批次處理 | FR-5.3, FR-5.4 | 必做 |
| 人口統計與生存機率關聯 | FR-6.1 | Optional |
| 社會階級與資源分配 | FR-6.2 | Optional |
| 家庭結構與同行人數 | FR-6.3 | Optional |
| 登船地點與登船特徵 | FR-6.4 | Optional |
| 特徵工程與機器學習建模 | FR-2 系列, FR-6.5, FR-6.6 | Optional／必做交集 |
| 其它視覺化（feature importance／ROC／what-if 等） | FR-6.6, FR-6.7 | Optional 加分 |

---

## 10. 附錄（Appendix）

### 10.1 建議超參數格點（範例，可自行增減）

- **Logistic Regression**：`C ∈ {0.01, 0.1, 1, 10}`、`penalty ∈ {l1, l2}`、`solver ∈ {liblinear, saga}`。
- **Random Forest**：`n_estimators ∈ {100, 300, 500}`、`max_depth ∈ {None, 5, 10, 20}`、`min_samples_split ∈ {2, 5, 10}`、`min_samples_leaf ∈ {1, 2, 4}`。

### 10.2 待確認事項（Open Issues）

- 是否需保留多版本資料集（如 train/test 分檔）或單表即可。
- 非同步工作是否升級為 RQ/Celery（與既有 RabbitMQ 經驗整合）。
- EDA 頁圖表數量上限與互動程度（影響時程）。

### 10.3 未來工作（Future Work）

- 模型自動再訓練排程與資料飛輪（data flywheel）。
- 使用者帳號與權限。
- 模型 A/B 測試與線上監控。
