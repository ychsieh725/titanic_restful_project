"""訓練服務單元測試（任務 4.1 / FR-3.2~3.6, 3.10）。

驗證 LR/RF 訓練、GridSearchCV 超參數搜尋、held-out 指標與可重現性。
使用 synthetic 小資料 + 縮小格點，避免真實 DB 與長時間搜尋。
"""

from __future__ import annotations

import sqlite3

import joblib
import numpy as np
import pandas as pd
import pytest

from src.ml import config
from src.ml.training import _build_estimator, load_training_data, train_model
from src.ml.types import Metrics, TrainResult


@pytest.fixture(autouse=True)
def tiny_param_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    """縮小超參數格點，加速測試（仍保留每演算法 ≥ 2 個超參數值組合）。"""
    monkeypatch.setitem(
        config.PARAM_GRIDS, config.LOGISTIC_REGRESSION,
        {"clf__C": [0.1, 1.0], "clf__solver": ["liblinear"]},
    )
    monkeypatch.setitem(
        config.PARAM_GRIDS, config.RANDOM_FOREST,
        {"clf__n_estimators": [10, 20], "clf__max_depth": [3]},
    )


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """40 列 synthetic 資料，每類 ≥ 5 筆以滿足 cv=5 與 stratify。"""
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
    df = pd.DataFrame(rows)
    y = pd.Series([i % 2 for i in range(40)], name=config.TARGET_COLUMN)
    return df, y


# --- estimator 工廠 -------------------------------------------------------

def test_build_estimator_supports_lr_and_rf() -> None:
    assert _build_estimator(config.LOGISTIC_REGRESSION) is not None
    assert _build_estimator(config.RANDOM_FOREST) is not None


def test_build_estimator_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="不支援"):
        _build_estimator("svm")


# --- train_model：兩種演算法 ---------------------------------------------

@pytest.mark.parametrize("algorithm", [config.LOGISTIC_REGRESSION, config.RANDOM_FOREST])
def test_train_model_returns_fitted_pipeline_and_result(
    algorithm: str, training_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = training_data
    pipeline, result = train_model(algorithm, X, y)

    # pipeline 已 fit，可直接預測
    preds = pipeline.predict(X.iloc[[0]])
    assert preds.shape == (1,)

    assert isinstance(result, TrainResult)
    assert result.algorithm == algorithm
    assert 0.0 <= result.best_cv_score <= 1.0
    # best_params 鍵對齊 "clf__" 前綴（FR-3.5）
    assert all(k.startswith("clf__") for k in result.best_params)
    assert result.feature_list
    assert result.training_rows == int(len(X) * (1 - config.TEST_SIZE))
    assert result.training_duration_sec >= 0.0
    assert result.model_uid


# --- 指標（FR-3.6）-------------------------------------------------------

def test_metrics_within_bounds_and_confusion_shape(
    training_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = training_data
    _, result = train_model(config.RANDOM_FOREST, X, y)
    m = result.metrics
    assert isinstance(m, Metrics)
    for value in (m.accuracy, m.precision, m.recall, m.f1, m.roc_auc):
        assert 0.0 <= value <= 1.0
    assert len(m.confusion_matrix) == 2
    assert all(len(row) == 2 for row in m.confusion_matrix)
    assert all(isinstance(v, int) for row in m.confusion_matrix for v in row)


# --- 可重現性（FR-3.10）--------------------------------------------------

def test_training_is_reproducible(
    training_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = training_data
    _, r1 = train_model(config.RANDOM_FOREST, X, y)
    _, r2 = train_model(config.RANDOM_FOREST, X, y)
    assert r1.best_params == r2.best_params
    assert r1.metrics.accuracy == pytest.approx(r2.metrics.accuracy)


# --- 序列化往返（接軌 5.1 持久化）---------------------------------------

def test_pipeline_joblib_roundtrip_predicts(
    training_data: tuple[pd.DataFrame, pd.Series], tmp_path
) -> None:
    X, y = training_data
    pipeline, _ = train_model(config.LOGISTIC_REGRESSION, X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    before = pipeline.predict(X)
    after = loaded.predict(X)
    np.testing.assert_array_equal(before, after)


def test_train_model_rejects_unknown_algorithm(
    training_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = training_data
    with pytest.raises(ValueError, match="不支援"):
        train_model("svm", X, y)


# --- load_training_data：SQLite 整合 -------------------------------------

def test_load_training_data_reads_features_and_target(
    training_data: tuple[pd.DataFrame, pd.Series], tmp_path
) -> None:
    X, y = training_data
    db_path = tmp_path / "titanic.db"
    frame = X.copy()
    frame[config.TARGET_COLUMN] = y.values
    connection = sqlite3.connect(db_path)
    try:
        frame.to_sql("titanic", connection, index=False)
    finally:
        connection.close()

    loaded_X, loaded_y = load_training_data(str(db_path))
    assert list(loaded_X.columns) == list(config.RAW_FEATURE_COLUMNS)
    assert loaded_y.name == config.TARGET_COLUMN
    assert len(loaded_X) == len(X)
