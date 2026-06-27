"""ML schema 單元測試（任務 2.3 / FR-4.2, §6.2/§6.3）。

驗證 ml_model 與 train_job 表建立、欄位齊全、約束生效與冪等性。
全程使用臨時 SQLite，不碰專案資料庫。
"""

from __future__ import annotations

import sqlite3

import pytest

from src.ml.schema import init_ml_schema


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row[0] for row in rows}


def _column_names(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "ml.db")


# --- 表建立與欄位 ---------------------------------------------------------

def test_creates_both_tables(db_path: str) -> None:
    init_ml_schema(db_path)
    connection = sqlite3.connect(db_path)
    try:
        tables = _table_names(connection)
    finally:
        connection.close()
    assert {"ml_model", "train_job"} <= tables


def test_ml_model_has_required_columns(db_path: str) -> None:
    init_ml_schema(db_path)
    connection = sqlite3.connect(db_path)
    try:
        columns = _column_names(connection, "ml_model")
    finally:
        connection.close()
    expected = {
        "id", "model_uid", "algorithm", "hyperparameters", "best_cv_score",
        "metrics", "feature_list", "training_rows", "data_hash", "file_path",
        "training_duration_sec", "is_active", "created_at",
    }
    assert expected <= columns


def test_train_job_has_required_columns(db_path: str) -> None:
    init_ml_schema(db_path)
    connection = sqlite3.connect(db_path)
    try:
        columns = _column_names(connection, "train_job")
    finally:
        connection.close()
    expected = {
        "job_id", "status", "algorithm", "progress",
        "result_model_uid", "error_message", "created_at", "finished_at",
    }
    assert expected <= columns


# --- 冪等性 ---------------------------------------------------------------

def test_idempotent(db_path: str) -> None:
    init_ml_schema(db_path)
    init_ml_schema(db_path)  # 第二次不得報錯
    connection = sqlite3.connect(db_path)
    try:
        assert {"ml_model", "train_job"} <= _table_names(connection)
    finally:
        connection.close()


# --- 約束：至多一個 active 模型 ------------------------------------------

def _insert_model(connection: sqlite3.Connection, uid: str, is_active: int) -> None:
    connection.execute(
        """
        INSERT INTO ml_model (
            model_uid, algorithm, hyperparameters, best_cv_score, metrics,
            feature_list, training_rows, data_hash, file_path,
            training_duration_sec, is_active
        ) VALUES (?, ?, '{}', 0.8, '{}', '[]', 100, 'h', 'p', 1.0, ?)
        """,
        (uid, "random_forest", is_active),
    )


def test_single_active_model_enforced(db_path: str) -> None:
    init_ml_schema(db_path)
    connection = sqlite3.connect(db_path)
    try:
        _insert_model(connection, "uid-1", 1)
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            _insert_model(connection, "uid-2", 1)  # 第二個 active 違反唯一索引
            connection.commit()
    finally:
        connection.close()


def test_multiple_inactive_models_allowed(db_path: str) -> None:
    init_ml_schema(db_path)
    connection = sqlite3.connect(db_path)
    try:
        _insert_model(connection, "uid-1", 0)
        _insert_model(connection, "uid-2", 0)
        connection.commit()
        count = connection.execute("SELECT COUNT(*) FROM ml_model").fetchone()[0]
        assert count == 2
    finally:
        connection.close()


# --- 約束：status 限定值 -------------------------------------------------

def test_train_job_status_check(db_path: str) -> None:
    init_ml_schema(db_path)
    connection = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO train_job (job_id, status, algorithm) VALUES (?, ?, ?)",
                ("j1", "bogus", "random_forest"),
            )
            connection.commit()
    finally:
        connection.close()
