"""單筆預測服務單元測試（任務 6.1 / FR-5.1, 5.2, 4.5）。

驗證以 active 模型對單筆輸入預測、回傳存活判定與生存機率，
以及無 active 模型時拋出明確網域例外。
共用 active pipeline（CON-2），不另寫前處理。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import config
from src.ml.prediction import NoActiveModelError, predict_batch, predict_one
from src.ml.registry import save_model, set_active_model
from src.ml.schema import init_ml_schema
from src.ml.training import train_model
from src.ml.types import PassengerInput, PredictionResult


@pytest.fixture(autouse=True)
def tiny_param_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        config.PARAM_GRIDS, config.LOGISTIC_REGRESSION,
        {"clf__C": [0.1, 1.0], "clf__solver": ["liblinear"]},
    )


@pytest.fixture
def db_with_active_model(tmp_path):
    """訓練→儲存→設 active，回傳 (db_path, models_dir)。"""
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

    db_path = str(tmp_path / "ml.db")
    models_dir = tmp_path / "models"
    init_ml_schema(db_path)
    pipeline, result = train_model(config.LOGISTIC_REGRESSION, X, y)
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    set_active_model(metadata.id, db_path=db_path)
    return db_path, models_dir


def _passenger() -> PassengerInput:
    return PassengerInput(
        Pclass=1, Sex="female", SibSp=0, Parch=0,
        Name="Test, Mrs. Example", Ticket="X",
        Age=29.0, Fare=80.0, Cabin="C20", Embarked="C",
    )


# --- 有 active 模型 -------------------------------------------------------

def test_predict_one_returns_result(db_with_active_model) -> None:
    db_path, _ = db_with_active_model
    result = predict_one(_passenger(), db_path=db_path)

    assert isinstance(result, PredictionResult)
    assert isinstance(result.survived, bool)
    assert 0.0 <= result.probability <= 1.0


def test_predict_one_survived_matches_probability(db_with_active_model) -> None:
    db_path, _ = db_with_active_model
    result = predict_one(_passenger(), db_path=db_path)
    assert result.survived == (result.probability >= 0.5)


# --- 無 active 模型 -------------------------------------------------------

def test_predict_one_without_active_model_raises(tmp_path) -> None:
    db_path = str(tmp_path / "ml.db")
    init_ml_schema(db_path)  # 有表但無 active 模型
    with pytest.raises(NoActiveModelError):
        predict_one(_passenger(), db_path=db_path)


# --- 批次預測（6.2）------------------------------------------------------

def _passengers(count: int) -> list[PassengerInput]:
    return [
        PassengerInput(
            Pclass=(i % 3) + 1,
            Sex="female" if i % 2 else "male",
            SibSp=0, Parch=0,
            Name=f"Doe, {'Mrs' if i % 2 else 'Mr'}. Person {i}",
            Ticket=f"T{i}",
            Age=float(20 + i) if i % 4 else None,
            Fare=float(10 + i) if i % 3 else None,
            Cabin=None,
            Embarked=["C", "Q", "S", None][i % 4],
        )
        for i in range(count)
    ]


def test_predict_batch_returns_frame_with_survived_and_probability(db_with_active_model) -> None:
    db_path, _ = db_with_active_model
    passengers = _passengers(5)

    result = predict_batch(passengers, db_path=db_path)

    assert len(result) == 5
    assert "survived" in result.columns
    assert "probability" in result.columns
    assert result["probability"].between(0.0, 1.0).all()


def test_predict_batch_survived_matches_threshold(db_with_active_model) -> None:
    db_path, _ = db_with_active_model
    result = predict_batch(_passengers(4), db_path=db_path)

    expected = result["probability"] >= 0.5
    assert list(result["survived"]) == list(expected)


def test_predict_batch_returns_native_python_types(db_with_active_model) -> None:
    """to_dict 結果須為 JSON-friendly 原生型別（供展示層 jsonify）。"""
    db_path, _ = db_with_active_model
    record = predict_batch(_passengers(1), db_path=db_path).to_dict(orient="records")[0]

    assert isinstance(record["survived"], bool)
    assert isinstance(record["probability"], float)


def test_predict_batch_empty_returns_empty_frame(db_with_active_model) -> None:
    db_path, _ = db_with_active_model
    result = predict_batch([], db_path=db_path)
    assert len(result) == 0


def test_predict_batch_without_active_model_raises(tmp_path) -> None:
    db_path = str(tmp_path / "ml.db")
    init_ml_schema(db_path)
    with pytest.raises(NoActiveModelError):
        predict_batch(_passengers(3), db_path=db_path)
