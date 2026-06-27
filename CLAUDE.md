# CLAUDE.md - Titanic 生存預測機器學習平台

> **專案:** Titanic 生存預測機器學習平台
> **描述:** 在現有 Flask CRUD 之上擴充端到端 ML 平台（特徵工程 / 訓練+超參數 / 非同步 job / 模型 Registry / 單筆+批次預測）
> **語言:** Python 3.11+ / Flask / scikit-learn
> **建立:** 2026-06-27
> **需求來源:** Titanic_ML_SRS.md
> **WBS:** .claude/taskmaster-data/wbs.md

## 開發流程

遵循標準流程：/task-next → /plan → /tdd → /verify

## 專案規則

已載入 `.claude/rules/` 中的通用規則（自動生效）：
- coding-style: 不可變性、檔案大小
- development-workflow: 研究先行、Plan-TDD-Review、先開分支再動 code
- security: commit 前安全檢查
- testing: 80%+ 覆蓋率
- git-workflow: Conventional Commits

## 專案專屬約束（來自 SRS）

- **CON-2 共用管線**: 訓練與預測必須共用同一條序列化 Pipeline，禁止預測時另寫前處理。
- **CON-4 框架解耦**: 服務層 `src/ml/` 為純 Python，不得 import Flask 物件。
- **CON-3 模型持久化**: joblib 存模型，metadata 存 `ml_model` 表。
- **NFR-S2**: 一律使用參數化查詢，禁止字串拼接 SQL。

## 禁止事項

- 不在根目錄建立新原始碼檔案 → ML 邏輯放 `src/ml/`
- 不在 master 分支直接改 code → 先開 `feat/*` 分支
- 不建立重複檔案 (v2, enhanced_, new_) → 擴展現有
- 不硬編碼可配置的值（超參數格點、特徵清單）→ 集中設定
- 不靜默吞噬錯誤、不回未處理的 500 → 明確驗證錯誤訊息

## 強制要求

- 每完成一個功能後 commit（WHY/WHAT/IMPACT）
- 先搜尋現有實作再建立新檔案
- 3 步驟以上的任務先用 TodoWrite 拆解

## 技術棧

| 層 | 技術 |
|---|---|
| 環境/套件 | **uv**（虛擬環境 `.venv` + `pyproject.toml` + `uv.lock`） |
| 展示層 | Flask + Jinja2 templates + REST API |
| 服務層 | 純 Python（feature engineering / training / registry / prediction） |
| ML | scikit-learn（Pipeline, ColumnTransformer, GridSearchCV, LR/RF） |
| 非同步 | threading（job 狀態管理） |
| 資料層 | SQLite（titanic / ml_model / train_job）+ joblib 模型檔 |

## 環境與套件管理（uv）

- 安裝依賴：`uv sync`（依 `uv.lock` 還原）
- 加套件：`uv add <pkg>`／開發套件：`uv add --dev <pkg>`
- 執行：`uv run python app.py`、`uv run pytest`
- **禁止** `pip install`／手動改 `.venv`；一律透過 uv 並提交 `uv.lock`

## 專案結構（標準型，目標）

```
titanic_restful_project/
├── app.py                  # Flask 入口（現有 CRUD + 掛載 ML blueprint）
├── init_db.py              # 建表 + CSV 匯入
├── src/
│   └── ml/                 # 服務層（純 Python，框架無關）
│       ├── features.py     # 特徵工程管線
│       ├── training.py     # 訓練 + 超參數搜尋
│       ├── jobs.py         # 非同步 job 狀態
│       ├── registry.py     # 模型登錄 / active 切換
│       └── prediction.py   # 單筆 / 批次預測
├── models/                 # joblib 模型檔
├── templates/              # 訓練 / 模型 / 預測 / EDA 頁
└── tests/                  # unit + integration
```
