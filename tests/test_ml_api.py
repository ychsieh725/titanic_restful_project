"""訓練 API 整合測試（任務 4.3 / SRS §4.2, FR-3.7~3.9）。

以 Flask test client 驗證 POST /api/ml/train 與 GET /api/ml/train/status/<id>：
輸入驗證、立即回 job_id、輪詢至 done 含指標摘要、失敗回 failed。
blueprint 工廠注入臨時 DB / models 目錄，不碰專案 my_db.db。
"""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pandas as pd
import pytest
from flask import Flask

from src.ml import config
from src.ml.jobs import JobStore
from src.ml.schema import init_ml_schema
from src.web.ml_api import create_ml_blueprint


@pytest.fixture(autouse=True)
def tiny_param_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        config.PARAM_GRIDS, config.LOGISTIC_REGRESSION,
        {"clf__C": [0.1, 1.0], "clf__solver": ["liblinear"]},
    )


@pytest.fixture
def seeded_db(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        survived = i % 2
        rows.append({
            "Survived": survived,
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
    db_path = str(tmp_path / "ml.db")
    connection = sqlite3.connect(db_path)
    try:
        pd.DataFrame(rows).to_sql("titanic", connection, index=False)
    finally:
        connection.close()
    init_ml_schema(db_path)
    return db_path, tmp_path / "models"


def _make_client(db_path, models_dir):
    app = Flask(__name__)
    store = JobStore()
    app.register_blueprint(
        create_ml_blueprint(store=store, db_path=db_path, models_dir=models_dir)
    )
    return app.test_client()


def _poll_status(client, job_id, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/ml/train/status/{job_id}")
        body = response.get_json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job 未在時限內結束")


# --- POST /train 輸入驗證 -------------------------------------------------

def test_train_missing_body_returns_400(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.post("/api/ml/train")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_train_missing_algorithm_returns_400(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.post("/api/ml/train", json={"foo": "bar"})
    assert response.status_code == 400


def test_train_unknown_algorithm_returns_422(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.post("/api/ml/train", json={"algorithm": "svm"})
    assert response.status_code == 422
    assert "error" in response.get_json()


# --- POST /train 成功 -----------------------------------------------------

def test_train_valid_returns_202_with_job_id(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.post(
        "/api/ml/train", json={"algorithm": config.LOGISTIC_REGRESSION}
    )
    assert response.status_code == 202
    body = response.get_json()
    assert body["job_id"]
    assert body["status"] in ("pending", "running")
    assert body["algorithm"] == config.LOGISTIC_REGRESSION


# --- GET /train/status ----------------------------------------------------

def test_status_unknown_job_returns_404(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.get("/api/ml/train/status/nope")
    assert response.status_code == 404


def test_status_done_includes_metrics_summary(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    job_id = client.post(
        "/api/ml/train", json={"algorithm": config.LOGISTIC_REGRESSION}
    ).get_json()["job_id"]

    body = _poll_status(client, job_id)
    assert body["status"] == "done"
    assert body["result_model_uid"]
    assert "result" in body
    assert "accuracy" in body["result"]["metrics"]
    assert body["result"]["best_params"] is not None


def test_status_failure_returns_failed(tmp_path) -> None:
    empty_db = str(tmp_path / "empty.db")  # 無 titanic 表 → 訓練失敗
    client = _make_client(empty_db, tmp_path / "models")
    job_id = client.post(
        "/api/ml/train", json={"algorithm": config.LOGISTIC_REGRESSION}
    ).get_json()["job_id"]

    body = _poll_status(client, job_id)
    assert body["status"] == "failed"
    assert body["error_message"]
