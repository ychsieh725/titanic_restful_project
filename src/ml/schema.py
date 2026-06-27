"""ML 資料表 schema（任務 2.3 / FR-4.2, SRS §6.2、§6.3）。

定義 ml_model（模型登錄）與 train_job（訓練工作狀態）兩張表的 DDL，
並提供冪等的 init_ml_schema()。所有建表均用 CREATE ... IF NOT EXISTS，
不影響既有 titanic 表，可安全重複執行。

SQLite 無原生 JSON/BOOLEAN/DATETIME：
- JSON 欄（hyperparameters/metrics/feature_list）存為 TEXT
- 布林（is_active）存為 INTEGER 0/1（CHECK 約束）
- 時間（created_at/finished_at）存為 ISO 字串 TEXT

純 Python sqlite3，不 import 任何 Web 框架物件（CON-4 / NFR-M1）。
"""

from __future__ import annotations

import sqlite3

from . import config

# --- ml_model（SRS §6.2）-------------------------------------------------

CREATE_ML_MODEL_TABLE = """
CREATE TABLE IF NOT EXISTS ml_model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_uid TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    hyperparameters TEXT NOT NULL,
    best_cv_score REAL NOT NULL,
    metrics TEXT NOT NULL,
    feature_list TEXT NOT NULL,
    training_rows INTEGER NOT NULL,
    data_hash TEXT NOT NULL,
    file_path TEXT NOT NULL,
    training_duration_sec REAL NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# 至多一個 active 模型：以 partial unique index 由資料結構保證唯一性
# （FR-4.5 active 切換），免除應用層的 if 檢查。
CREATE_ACTIVE_MODEL_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_ml_model_active
ON ml_model (is_active) WHERE is_active = 1;
"""

# --- train_job（SRS §6.3）------------------------------------------------

CREATE_TRAIN_JOB_TABLE = """
CREATE TABLE IF NOT EXISTS train_job (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'done', 'failed')),
    algorithm TEXT NOT NULL,
    progress TEXT NOT NULL DEFAULT '',
    result_model_uid TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);
"""

_DDL_STATEMENTS = (
    CREATE_ML_MODEL_TABLE,
    CREATE_ACTIVE_MODEL_INDEX,
    CREATE_TRAIN_JOB_TABLE,
)


def init_ml_schema(db_path: str | None = None) -> None:
    """建立 ml_model 與 train_job 表（冪等，非破壞）。

    Args:
        db_path: SQLite 檔案路徑；省略時用 config.DATABASE_PATH。
    """
    path = str(db_path or config.DATABASE_PATH)
    connection = sqlite3.connect(path)
    try:
        for statement in _DDL_STATEMENTS:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
