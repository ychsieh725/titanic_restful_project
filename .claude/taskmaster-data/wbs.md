# WBS - Titanic 生存預測機器學習平台

**建立日期:** 2026-06-27
**最後更新:** 2026-06-27（6.3 輸入驗證 完成；M2 待 6.2 CSV 批次）
**開發模式:** MVP（SRS 必做需求 / Must）
**主要語言:** Python 3.11+ / Flask / scikit-learn
**專案描述:** 在現有 Flask CRUD（titanic 表）之上，擴充端到端 ML 平台：共用特徵工程管線、一鍵訓練+超參數調校、非同步 job 狀態查詢、模型持久化與 Registry、單筆/CSV 批次預測（含生存機率）。

> 範圍依據：`Titanic_ML_SRS.md` §8 驗收準則（6 大必做項）與 §9 需求追溯矩陣。
> 架構約束：服務層（`src/ml/`）為純 Python，不依賴 Web 框架（CON-4）；訓練與預測共用同一條序列化管線（CON-2）。

---

## 任務清單

| # | 任務 | 狀態 | 優先級 | 依賴 | 預估 | 對應需求 / 備註 |
|---|------|------|--------|------|------|------|
| 1.1 | 專案初始化 | ✅ 完成 | 高 | - | 0.5h | 自動完成（task-init） |
| 1.2 | 需求分析（SRS 已備） | ✅ 完成 | 高 | - | 1h | 來源 Titanic_ML_SRS.md |
| 2.1 | 開功能分支 `feat/ml-platform` | ✅ 完成 | 高 | 1.2 | 0.1h | 已於 feat/ml-platform 工作 |
| 2.2 | 服務層骨架 `src/ml/`（框架解耦） | ✅ 完成 | 高 | 2.1 | 1.5h | types/config/jobs 完成，CON-4 已驗證 |
| 2.3 | DB schema：`ml_model` + `train_job` | ✅ 完成 | 高 | 2.1 | 1.5h | FR-4.2, §6.2/§6.3 |
| 3.1 | 特徵工程管線（補值/Title/FamilySize/編碼，可序列化） | ✅ 完成 | 高 | 2.2 | 3h | FR-2.1~2.4, 2.7（共用管線） |
| 3.2 | 管線單元測試（RED→GREEN） | ✅ 完成 | 高 | 3.1 | 1.5h | 併入 3.1：10 測試 / features.py 100% |
| 4.1 | 訓練服務：LR+RF, GridSearchCV cv≥5, train/test split, 指標 | ✅ 完成 | 高 | 3.1 | 3h | FR-3.2~3.6 |
| 4.2 | 非同步 job（threading）+ 狀態管理 pending/running/done/failed | ✅ 完成 | 高 | 4.1 | 2h | FR-3.7~3.9, NFR-R1 |
| 4.3 | 訓練 API：POST /api/ml/train、GET /train/status/{job_id} | ✅ 完成 | 高 | 4.2 | 1.5h | §4.2 API |
| 5.1 | 模型持久化（joblib 整條管線）+ metadata 寫入 ml_model | ✅ 完成 | 高 | 4.1, 2.3 | 2h | FR-4.1, 4.2 |
| 5.2 | 模型清單 API + active 模型切換 | ✅ 完成 | 高 | 5.1 | 1.5h | FR-4.3, 4.5（預測用 active） |
| 6.1 | 單筆預測服務 + API（predict_proba 機率） | ✅ 完成 | 高 | 5.1 | 2h | FR-5.1, 5.2 |
| 6.2 | CSV 批次預測 + 結果 CSV 下載 | ⏳ 待處理 | 高 | 6.1 | 2h | FR-5.3, 5.4 |
| 6.3 | 輸入驗證（schema-based，明確錯誤訊息，不回 500） | ✅ 完成 | 高 | 6.1 | 1.5h | FR-5.5, NFR-S1/S2 |
| 7.1 | 訓練頁（演算法選擇 + job 輪詢狀態） | ⏳ 待處理 | 高 | 4.3 | 2h | FR-7.4 |
| 7.2 | 模型管理頁（清單 + 設 active） | ⏳ 待處理 | 中 | 5.2 | 1.5h | FR-4.3 |
| 7.3 | 預測頁（單筆表單 + CSV 上傳結果表格） | ⏳ 待處理 | 高 | 6.2 | 2.5h | FR-5.1, 5.3 |
| 7.4 | 導覽列整合（首頁/訓練/模型/預測） | ⏳ 待處理 | 中 | 7.1 | 1h | FR-7.1, 7.2 |
| 8.1 | 整合測試（API 端點 + DB） | ⏳ 待處理 | 高 | 6.3, 5.2 | 2h | 整合測試 |
| 8.2 | 驗收檢查（6 大必做項逐項驗證） | ⏳ 待處理 | 高 | 7.3, 8.1 | 1h | §8 驗收準則 |

**MVP 總預估：約 38h**

### 狀態說明
- ✅ 完成
- 🔄 進行中
- ⏳ 待處理
- 🚫 阻塞
- ⏭️ 跳過

---

## 里程碑

| 里程碑 | 目標 | 包含任務 | 狀態 |
|--------|------|----------|------|
| M1: 核心管線可訓練 | 一鍵訓練 + 超參數 + 模型儲存 | 2.x, 3.x, 4.x, 5.1 | ✅ 完成 |
| M2: 預測可用 | 單筆 + CSV 批次預測（含機率） | 5.2, 6.x | ⏳ 待處理 |
| M3: MVP 驗收 | UI 串接 + 6 大必做驗收通過 | 7.x, 8.x | ⏳ 待處理 |

---

## 風險與阻塞

| 風險 | 影響 | 緩解策略 |
|------|------|----------|
| 訓練/預測前處理不一致 | 預測結果錯誤（CON-2 違規） | 強制共用同一序列化 Pipeline，禁止預測端另寫前處理；3.2 單元測試覆蓋 |
| threading 與 SQLite 多執行緒衝突 | job 寫入競態 | job 狀態與 DB 寫入加鎖；沿用 check_same_thread=False 但集中寫入點 |
| GridSearchCV 超時（NFR-P1 < 60s） | UI 等待過久 | 控制格點大小、n_jobs=-1；非同步 job 不阻塞 UI |
| 服務層誤 import Flask 物件 | 違反 CON-4 可攜性 | code review 檢查 src/ml/ 不得 import flask |
| 在 master 直接改 code | 違反 dev-workflow 鐵律 | 任務 2.1 先開 feat/ml-platform 分支 |

---

## 現有資產（可重用）

- `app.py` — Flask CRUD（titanic 表 GET/POST/PUT/DELETE + 分頁/搜尋）
- `init_db.py` — SQLite 建表 + titanic.csv 匯入（含 CHECK 約束）
- `templates/` — index/new/edit（含 noJS 版本）
- `titanic.csv` — 891 列 Kaggle 原始欄位
