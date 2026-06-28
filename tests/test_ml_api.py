"""訓練 API 整合測試（任務 4.3 / SRS §4.2, FR-3.7~3.9）。

以 Flask test client 驗證 POST /api/ml/train 與 GET /api/ml/train/status/<id>：
輸入驗證、立即回 job_id、輪詢至 done 含指標摘要、失敗回 failed。
blueprint 工廠注入臨時 DB / models 目錄，不碰專案 my_db.db。
"""

from __future__ import annotations

import io
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest
from flask import Flask

from src.ml import config
from src.ml.jobs import JobStore
from src.ml.registry import save_model, set_active_model
from src.ml.schema import init_ml_schema
from src.ml.training import train_model
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


# --- 模型清單 / active 切換（5.2）----------------------------------------

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


def test_list_models_returns_seeded(seeded_db) -> None:
    db_path, models_dir = seeded_db
    _seed_model(db_path, "uid-1")
    _seed_model(db_path, "uid-2")
    client = _make_client(db_path, models_dir)

    response = client.get("/api/ml/models")
    assert response.status_code == 200
    models = response.get_json()["models"]
    assert {m["model_uid"] for m in models} == {"uid-1", "uid-2"}


def test_activate_model_returns_200_and_marks_active(seeded_db) -> None:
    db_path, models_dir = seeded_db
    model_id = _seed_model(db_path, "uid-1")
    client = _make_client(db_path, models_dir)

    response = client.post(f"/api/ml/models/{model_id}/activate")
    assert response.status_code == 200
    assert response.get_json()["model"]["is_active"] is True


def test_activate_missing_model_returns_404(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.post("/api/ml/models/999/activate")
    assert response.status_code == 404


def test_activate_switches_active_model(seeded_db) -> None:
    db_path, models_dir = seeded_db
    id1 = _seed_model(db_path, "uid-1")
    id2 = _seed_model(db_path, "uid-2")
    client = _make_client(db_path, models_dir)

    client.post(f"/api/ml/models/{id1}/activate")
    client.post(f"/api/ml/models/{id2}/activate")

    models = client.get("/api/ml/models").get_json()["models"]
    actives = [m["id"] for m in models if m["is_active"]]
    assert actives == [id2]


# --- 單筆預測（6.1）------------------------------------------------------

def _train_and_activate(seeded_db) -> None:
    """以 seeded_db 訓練一個模型並設為 active（供預測測試）。"""
    db_path, models_dir = seeded_db
    X = pd.read_sql_query(
        "SELECT * FROM titanic", sqlite3.connect(db_path)
    ).drop(columns=["Survived"])
    y = pd.read_sql_query(
        "SELECT Survived FROM titanic", sqlite3.connect(db_path)
    )["Survived"]
    pipeline, result = train_model(config.LOGISTIC_REGRESSION, X, y)
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    set_active_model(metadata.id, db_path=db_path)


_VALID_PASSENGER = {
    "Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0,
    "Name": "Test, Mrs. Example", "Ticket": "X",
    "Age": 29.0, "Fare": 80.0, "Cabin": "C20", "Embarked": "C",
}


def test_predict_missing_body_returns_400(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.post("/api/ml/predict")
    assert response.status_code == 400


def test_predict_missing_required_field_returns_422(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    payload = {k: v for k, v in _VALID_PASSENGER.items() if k != "Sex"}
    response = client.post("/api/ml/predict", json=payload)
    assert response.status_code == 422
    assert "Sex" in response.get_json()["fields"]


def test_predict_invalid_value_returns_422_with_fields(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    payload = {**_VALID_PASSENGER, "Pclass": 9, "Embarked": "Z"}
    response = client.post("/api/ml/predict", json=payload)
    assert response.status_code == 422
    fields = response.get_json()["fields"]
    assert "Pclass" in fields and "Embarked" in fields


def test_predict_without_active_model_returns_409(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.post("/api/ml/predict", json=_VALID_PASSENGER)
    assert response.status_code == 409
    assert "error" in response.get_json()


def test_predict_valid_returns_200_with_probability(seeded_db) -> None:
    db_path, models_dir = seeded_db
    _train_and_activate(seeded_db)
    client = _make_client(db_path, models_dir)

    response = client.post("/api/ml/predict", json=_VALID_PASSENGER)
    assert response.status_code == 200
    body = response.get_json()
    assert isinstance(body["survived"], bool)
    assert 0.0 <= body["probability"] <= 1.0


# --- CSV 批次預測（6.2 / FR-5.3, 5.4）------------------------------------

_CSV_HEADER = "Pclass,Sex,SibSp,Parch,Name,Ticket,Age,Fare,Cabin,Embarked"


def _csv_bytes(rows: list[str]) -> bytes:
    return ("\n".join([_CSV_HEADER, *rows]) + "\n").encode("utf-8")


_VALID_ROW = '1,female,0,0,"Doe, Mrs. A",X1,29,80,,C'


def _upload(client, data_bytes: bytes, filename: str = "passengers.csv", query: str = ""):
    return client.post(
        f"/api/ml/predict/batch{query}",
        data={"file": (io.BytesIO(data_bytes), filename)},
        content_type="multipart/form-data",
    )


def test_batch_missing_file_returns_400(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = client.post(
        "/api/ml/predict/batch", content_type="multipart/form-data"
    )
    assert response.status_code == 400


def test_batch_non_csv_extension_returns_400(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = _upload(client, _csv_bytes([_VALID_ROW]), filename="data.txt")
    assert response.status_code == 400


def test_batch_empty_file_returns_400(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = _upload(client, _csv_bytes([]))  # 僅表頭，無資料列
    assert response.status_code == 400


def test_batch_unparseable_csv_returns_400(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    # 欄位數不一致 → pandas 解析錯誤
    malformed = b'a,b,c\n1,2\n3,4,5,6,7\n'
    response = _upload(client, malformed)
    assert response.status_code == 400


def test_batch_invalid_rows_return_422_with_row_index(seeded_db) -> None:
    db_path, models_dir = seeded_db
    _train_and_activate(seeded_db)
    client = _make_client(db_path, models_dir)

    bad_row = '9,female,0,0,"Doe, Mrs. B",X2,29,80,,Z'  # Pclass/Embarked 非法
    response = _upload(client, _csv_bytes([_VALID_ROW, bad_row]))

    assert response.status_code == 422
    rows = response.get_json()["rows"]
    assert rows[0]["row"] == 1
    assert "Pclass" in rows[0]["fields"]


def test_batch_without_active_model_returns_409(seeded_db) -> None:
    db_path, models_dir = seeded_db
    client = _make_client(db_path, models_dir)
    response = _upload(client, _csv_bytes([_VALID_ROW]))
    assert response.status_code == 409


def test_batch_exceeding_max_rows_returns_413(seeded_db, monkeypatch) -> None:
    db_path, models_dir = seeded_db
    monkeypatch.setattr(config, "MAX_BATCH_ROWS", 2)
    client = _make_client(db_path, models_dir)
    response = _upload(client, _csv_bytes([_VALID_ROW] * 3))
    assert response.status_code == 413


def test_batch_valid_returns_200_json(seeded_db) -> None:
    db_path, models_dir = seeded_db
    _train_and_activate(seeded_db)
    client = _make_client(db_path, models_dir)

    response = _upload(client, _csv_bytes([_VALID_ROW, _VALID_ROW]))
    assert response.status_code == 200
    body = response.get_json()
    assert body["total"] == 2
    assert len(body["results"]) == 2
    first = body["results"][0]
    assert isinstance(first["survived"], bool)
    assert 0.0 <= first["probability"] <= 1.0


def test_batch_format_csv_returns_attachment(seeded_db) -> None:
    db_path, models_dir = seeded_db
    _train_and_activate(seeded_db)
    client = _make_client(db_path, models_dir)

    response = _upload(client, _csv_bytes([_VALID_ROW]), query="?format=csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "attachment" in response.headers.get("Content-Disposition", "")
    text = response.get_data(as_text=True)
    assert "survived" in text and "probability" in text
