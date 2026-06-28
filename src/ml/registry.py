"""模型 Registry（任務 5.1）。

將訓練產出（整條 fit pipeline + TrainResult）持久化：以 joblib 存模型檔到
models/，並把 metadata 寫入 ml_model 表（FR-4.1, 4.2）。提供回讀與載入輔助，
供預測（6.1）與清單/active 切換（5.2）使用。

純 Python，不 import 任何 Web 框架物件（CON-4 / NFR-M1）。所有 SQL 均參數化
（NFR-S2）。active 切換不在本模組職責（留給 5.2），save 一律存 is_active=0。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
from sklearn.pipeline import Pipeline

from . import config
from .types import ModelMetadata, TrainResult

_INSERT_MODEL_SQL = """
INSERT INTO ml_model (
    model_uid, algorithm, hyperparameters, best_cv_score, metrics,
    feature_list, training_rows, data_hash, file_path,
    training_duration_sec, is_active
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
"""

_SELECT_BY_ID_SQL = "SELECT * FROM ml_model WHERE id = ?"
_SELECT_BY_UID_SQL = "SELECT * FROM ml_model WHERE model_uid = ?"
_SELECT_ALL_SQL = "SELECT * FROM ml_model ORDER BY created_at DESC, id DESC"
_SELECT_ACTIVE_SQL = "SELECT * FROM ml_model WHERE is_active = 1"


def save_model(
    pipeline: Pipeline,
    result: TrainResult,
    db_path: str | None = None,
    models_dir: str | Path | None = None,
) -> ModelMetadata:
    """持久化整條 pipeline 與 metadata，回傳登錄後的 ModelMetadata。

    Args:
        pipeline: 4.1 回傳的已 fit pipeline（含前處理＋分類器，CON-2）。
        result: 對應的訓練結果摘要。
        db_path: SQLite 路徑；省略時用 config.DATABASE_PATH。
        models_dir: 模型檔目錄；省略時用 config.MODELS_DIR。

    Returns:
        含 DB 指派的 id 與 created_at 的 ModelMetadata。

    Raises:
        sqlite3.IntegrityError: model_uid 重複時（UNIQUE 約束）。
    """
    path = str(db_path or config.DATABASE_PATH)
    directory = Path(models_dir or config.MODELS_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / f"{result.model_uid}.joblib"
    joblib.dump(pipeline, file_path)

    parameters = (
        result.model_uid,
        result.algorithm,
        json.dumps(result.best_params),
        result.best_cv_score,
        json.dumps(asdict(result.metrics)),
        json.dumps(result.feature_list),
        result.training_rows,
        result.data_hash,
        str(file_path),
        result.training_duration_sec,
    )

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(_INSERT_MODEL_SQL, parameters)
        connection.commit()
        row = connection.execute(_SELECT_BY_ID_SQL, (cursor.lastrowid,)).fetchone()
    finally:
        connection.close()

    return _row_to_metadata(row)


def get_model_by_uid(
    model_uid: str,
    db_path: str | None = None,
) -> Optional[ModelMetadata]:
    """依 model_uid 取得登錄 metadata；不存在回 None。"""
    path = str(db_path or config.DATABASE_PATH)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(_SELECT_BY_UID_SQL, (model_uid,)).fetchone()
    finally:
        connection.close()
    return _row_to_metadata(row) if row is not None else None


def load_model_file(metadata: ModelMetadata) -> Pipeline:
    """依 metadata.file_path 載回整條 pipeline（供預測共用，CON-2）。"""
    return joblib.load(metadata.file_path)


def list_models(db_path: str | None = None) -> list[ModelMetadata]:
    """列出所有登錄模型，建立時間新→舊（FR-4.3）。"""
    path = str(db_path or config.DATABASE_PATH)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(_SELECT_ALL_SQL).fetchall()
    finally:
        connection.close()
    return [_row_to_metadata(row) for row in rows]


def get_active_model(db_path: str | None = None) -> Optional[ModelMetadata]:
    """取得 active 模型；無則回 None（供預測判斷，FR-4.5）。"""
    path = str(db_path or config.DATABASE_PATH)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(_SELECT_ACTIVE_SQL).fetchone()
    finally:
        connection.close()
    return _row_to_metadata(row) if row is not None else None


def set_active_model(
    model_id: int,
    db_path: str | None = None,
) -> Optional[ModelMetadata]:
    """將指定模型設為 active，同時間至多一個（FR-4.4）。

    於單一交易內先清除既有 active 再設定目標，避免 partial unique index
    在「新舊皆為 active」瞬間衝突。目標不存在時回 None（不變更任何資料）。
    """
    path = str(db_path or config.DATABASE_PATH)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        exists = connection.execute(_SELECT_BY_ID_SQL, (model_id,)).fetchone()
        if exists is None:
            return None
        with connection:  # 交易：全成功才提交
            connection.execute("UPDATE ml_model SET is_active = 0 WHERE is_active = 1")
            connection.execute(
                "UPDATE ml_model SET is_active = 1 WHERE id = ?", (model_id,)
            )
        row = connection.execute(_SELECT_BY_ID_SQL, (model_id,)).fetchone()
    finally:
        connection.close()
    return _row_to_metadata(row)


def delete_model(
    model_id: int,
    db_path: str | None = None,
) -> Optional[ModelMetadata]:
    """刪除指定模型的 DB 列與其 joblib 檔，回傳被刪除的 metadata。

    先讀回 metadata 以取得 file_path，再刪 DB 列、清模型檔（檔案不存在不報錯）。
    刪除 active 模型後即無 active 模型，預測端會據此回提示（FR-4.5），屬預期行為。

    Args:
        model_id: 目標模型 id。
        db_path: SQLite 路徑；省略時用 config.DATABASE_PATH。

    Returns:
        被刪除的 ModelMetadata；目標不存在時回 None（不變更任何資料）。
    """
    path = str(db_path or config.DATABASE_PATH)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(_SELECT_BY_ID_SQL, (model_id,)).fetchone()
        if row is None:
            return None
        metadata = _row_to_metadata(row)
        with connection:  # 交易：DB 列刪除
            connection.execute("DELETE FROM ml_model WHERE id = ?", (model_id,))
    finally:
        connection.close()

    Path(metadata.file_path).unlink(missing_ok=True)
    return metadata


def _row_to_metadata(row: sqlite3.Row) -> ModelMetadata:
    """sqlite Row → ModelMetadata，還原 JSON 欄、布林與時間型別。"""
    return ModelMetadata(
        id=row["id"],
        model_uid=row["model_uid"],
        algorithm=row["algorithm"],
        hyperparameters=json.loads(row["hyperparameters"]),
        best_cv_score=row["best_cv_score"],
        metrics=json.loads(row["metrics"]),
        feature_list=json.loads(row["feature_list"]),
        training_rows=row["training_rows"],
        data_hash=row["data_hash"],
        file_path=row["file_path"],
        training_duration_sec=row["training_duration_sec"],
        is_active=bool(row["is_active"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
