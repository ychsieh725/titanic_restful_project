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
from src.ml.registry import get_model_by_uid, load_model_file, save_model
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


# --- UNIQUE(model_uid) ----------------------------------------------------

def test_duplicate_model_uid_rejected(trained) -> None:
    pipeline, result, db_path, models_dir, _ = trained
    save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    with pytest.raises(sqlite3.IntegrityError):
        save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
