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
