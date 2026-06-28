"""模型 Registry 持久化單元測試（任務 5.1 / FR-4.1, 4.2）。

驗證整條 pipeline 以 joblib 持久化、metadata 寫入 ml_model 表並可正確回讀。
全程使用臨時 SQLite 與臨時 models 目錄，不碰專案資料庫或 models/。
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.ml import config
from src.ml.registry import (
    delete_model,
    get_active_model,
    get_model_by_uid,
    list_models,
    load_model_file,
    save_model,
    set_active_model,
)
from src.ml.schema import init_ml_schema
from src.ml.training import train_model
from src.ml.types import ModelMetadata


@pytest.fixture(autouse=True)
def tiny_param_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        config.PARAM_GRIDS, config.LOGISTIC_REGRESSION,
        {"clf__C": [0.1, 1.0], "clf__solver": ["liblinear"]},
    )


@pytest.fixture
def trained(tmp_path):
    """訓練一個真實小模型，回傳 (pipeline, result, db_path, models_dir, X)。"""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        survived = i % 2
        rows.append({
            "Pclass": int(rng.integers(1, 4)),
            "Sex": "female" if survived else "male",
            "Age": float(rng.integers(1, 70)) if i % 7 else None,
            "SibSp": int(rng.integers(0, 3)),
            "Parch": int(rng.integers(0, 3)),
            "Ticket": f"T{i}",
            "Fare": float(rng.uniform(5, 80)) if i % 5 else None,
            "Cabin": None,
            "Embarked": ["C", "Q", "S", None][i % 4],
            "Name": f"Doe, {'Mrs' if survived else 'Mr'}. Person {i}",
        })
    X = pd.DataFrame(rows)
    y = pd.Series([i % 2 for i in range(40)])
    pipeline, result = train_model(config.LOGISTIC_REGRESSION, X, y)

    db_path = str(tmp_path / "ml.db")
    models_dir = tmp_path / "models"
    init_ml_schema(db_path)
    return pipeline, result, db_path, models_dir, X


# --- save_model：joblib 檔 + DB 列 ---------------------------------------

def test_save_model_writes_joblib_and_row(trained) -> None:
    pipeline, result, db_path, models_dir, _ = trained
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)

    assert isinstance(metadata, ModelMetadata)
    assert metadata.id > 0
    assert metadata.created_at is not None
    assert (models_dir / f"{result.model_uid}.joblib").exists()

    connection = sqlite3.connect(db_path)
    try:
        count = connection.execute("SELECT COUNT(*) FROM ml_model").fetchone()[0]
    finally:
        connection.close()
    assert count == 1


def test_metadata_json_fields_roundtrip(trained) -> None:
    pipeline, result, db_path, models_dir, _ = trained
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)

    assert metadata.hyperparameters == result.best_params
    assert metadata.metrics["accuracy"] == pytest.approx(result.metrics.accuracy)
    assert metadata.feature_list == result.feature_list
    assert metadata.algorithm == result.algorithm
    assert metadata.data_hash == result.data_hash
    assert metadata.training_rows == result.training_rows


def test_save_model_defaults_inactive(trained) -> None:
    pipeline, result, db_path, models_dir, _ = trained
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    assert metadata.is_active is False


# --- load：往返後可預測（接軌 6.1）--------------------------------------

def test_load_model_file_predicts(trained) -> None:
    pipeline, result, db_path, models_dir, X = trained
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    loaded = load_model_file(metadata)
    preds = loaded.predict(X)
    np.testing.assert_array_equal(preds, pipeline.predict(X))


# --- get_model_by_uid -----------------------------------------------------

def test_get_model_by_uid_returns_metadata(trained) -> None:
    pipeline, result, db_path, models_dir, _ = trained
    save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    fetched = get_model_by_uid(result.model_uid, db_path=db_path)
    assert fetched is not None
    assert fetched.model_uid == result.model_uid


def test_get_model_by_uid_missing_returns_none(trained) -> None:
    _, _, db_path, _, _ = trained
    assert get_model_by_uid("does-not-exist", db_path=db_path) is None


# --- delete_model ---------------------------------------------------------

def test_delete_model_removes_row_and_file(trained) -> None:
    pipeline, result, db_path, models_dir, _ = trained
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    model_file = models_dir / f"{result.model_uid}.joblib"
    assert model_file.exists()

    deleted = delete_model(metadata.id, db_path=db_path)

    assert deleted is not None
    assert deleted.id == metadata.id
    assert not model_file.exists()                       # joblib 檔已清除
    assert get_model_by_uid(result.model_uid, db_path=db_path) is None  # DB 列已刪


def test_delete_model_missing_returns_none(trained) -> None:
    _, _, db_path, _, _ = trained
    assert delete_model(999, db_path=db_path) is None


def test_delete_active_model_leaves_no_active(trained) -> None:
    pipeline, result, db_path, models_dir, _ = trained
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    set_active_model(metadata.id, db_path=db_path)

    delete_model(metadata.id, db_path=db_path)

    assert get_active_model(db_path=db_path) is None


# --- UNIQUE(model_uid) ----------------------------------------------------

def test_duplicate_model_uid_rejected(trained) -> None:
    pipeline, result, db_path, models_dir, _ = trained
    save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    with pytest.raises(sqlite3.IntegrityError):
        save_model(pipeline, result, db_path=db_path, models_dir=models_dir)


# --- 清單 / active 切換（5.2）-------------------------------------------

@pytest.fixture
def empty_db(tmp_path) -> str:
    db_path = str(tmp_path / "ml.db")
    init_ml_schema(db_path)
    return db_path


def _seed_model(db_path: str, uid: str, is_active: int = 0) -> int:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.execute(
            """
            INSERT INTO ml_model (
                model_uid, algorithm, hyperparameters, best_cv_score, metrics,
                feature_list, training_rows, data_hash, file_path,
                training_duration_sec, is_active
            ) VALUES (?, 'random_forest', '{}', 0.8, '{}', '[]', 100, 'h', 'p', 1.0, ?)
            """,
            (uid, is_active),
        )
        connection.commit()
        return cursor.lastrowid
    finally:
        connection.close()


def test_list_models_empty(empty_db: str) -> None:
    assert list_models(db_path=empty_db) == []


def test_list_models_newest_first(empty_db: str) -> None:
    _seed_model(empty_db, "uid-1")
    _seed_model(empty_db, "uid-2")
    models = list_models(db_path=empty_db)
    assert [m.model_uid for m in models] == ["uid-2", "uid-1"]


def test_set_active_model_marks_single_active(empty_db: str) -> None:
    id1 = _seed_model(empty_db, "uid-1")
    id2 = _seed_model(empty_db, "uid-2")

    activated = set_active_model(id1, db_path=empty_db)
    assert activated is not None and activated.is_active is True

    # 切換到另一個 → 只有後者 active
    set_active_model(id2, db_path=empty_db)
    actives = [m.id for m in list_models(db_path=empty_db) if m.is_active]
    assert actives == [id2]


def test_set_active_model_missing_returns_none(empty_db: str) -> None:
    assert set_active_model(999, db_path=empty_db) is None


def test_get_active_model(empty_db: str) -> None:
    assert get_active_model(db_path=empty_db) is None
    model_id = _seed_model(empty_db, "uid-1")
    set_active_model(model_id, db_path=empty_db)
    active = get_active_model(db_path=empty_db)
    assert active is not None and active.id == model_id
