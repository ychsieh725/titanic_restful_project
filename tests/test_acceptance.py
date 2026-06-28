"""端到端整合與驗收測試（任務 8.1 / 8.2，對應 SRS §8 六大必做項）。

以完整 Flask app（同時掛載 /api/ml JSON 與 /ml 頁面兩個 blueprint）對臨時 DB
跑端到端流程，逐項驗證 SRS §8 驗收準則：

  AC1 一鍵訓練且能得知完成（running→done）          【FR-3.1/3.7/3.8/7.4】
  AC2 超參數調校且顯示最佳超參數與指標              【FR-3.3~3.6】
  AC3 模型被儲存且可於模型頁列出                    【FR-4.1~4.3】
  AC4 單筆預測輸出存活判定與生存機率                【FR-5.1/5.2】
  AC5 CSV 批次預測並可下載含機率結果                【FR-5.3/5.4】
  AC6 預測與訓練前處理一致（同一序列化管線）        【FR-2.7/CON-2】

訓練測試以 monkeypatch 縮小超參數格點 + 40 列 synthetic 資料保持快速；
全程不碰專案 my_db.db 或 models/，一律用 tmp_path。
"""

from __future__ import annotations

import io
import sqlite3
import time

import joblib
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
from src.web.ml_pages import create_ml_pages_blueprint


@pytest.fixture(autouse=True)
def tiny_param_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    """縮小格點：仍經 GridSearchCV（多組候選）但速度快、結果穩定。"""
    monkeypatch.setitem(
        config.PARAM_GRIDS, config.LOGISTIC_REGRESSION,
        {"clf__C": [0.1, 1.0, 10.0], "clf__solver": ["liblinear"]},
    )


def _synthetic_rows(count: int = 40) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(count):
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
    return rows


@pytest.fixture
def app_env(tmp_path):
    """建立 seeded 臨時 DB + 完整 app（API + 頁面），回傳 (client, db_path, models_dir)。"""
    db_path = str(tmp_path / "ml.db")
    models_dir = tmp_path / "models"

    connection = sqlite3.connect(db_path)
    try:
        pd.DataFrame(_synthetic_rows()).to_sql("titanic", connection, index=False)
    finally:
        connection.close()
    init_ml_schema(db_path)

    app = Flask(__name__, template_folder="../templates")
    app.register_blueprint(
        create_ml_blueprint(store=JobStore(), db_path=db_path, models_dir=models_dir)
    )
    app.register_blueprint(create_ml_pages_blueprint())
    return app.test_client(), db_path, models_dir


@pytest.fixture
def trained_env(app_env):
    """在 app_env 上以服務層同步訓練並設 active（快速，供預測類驗收）。"""
    client, db_path, models_dir = app_env
    frame = pd.read_sql_query("SELECT * FROM titanic", sqlite3.connect(db_path))
    X = frame.drop(columns=["Survived"])
    y = frame["Survived"]
    pipeline, result = train_model(config.LOGISTIC_REGRESSION, X, y)
    metadata = save_model(pipeline, result, db_path=db_path, models_dir=models_dir)
    set_active_model(metadata.id, db_path=db_path)
    return client, db_path, models_dir


def _poll(client, job_id, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/ml/train/status/{job_id}").get_json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("訓練 job 未在時限內結束")


_VALID_PASSENGER = {
    "Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0,
    "Name": "Test, Mrs. Example", "Ticket": "X",
    "Age": 29.0, "Fare": 80.0, "Cabin": "C20", "Embarked": "C",
}


# --- AC1 + AC2 + AC3：一鍵訓練 → 完成 → 最佳超參數 → 儲存且列出 ----------

def test_train_to_registry_flow(app_env) -> None:
    """端到端：POST 訓練 → 輪詢至 done → 顯示最佳超參數/指標 → 模型已存且可列出。"""
    client, db_path, models_dir = app_env

    # AC1：一鍵啟動，立即取得 job 並進入非終態
    start = client.post("/api/ml/train", json={"algorithm": config.LOGISTIC_REGRESSION})
    assert start.status_code == 202
    job = start.get_json()
    assert job["status"] in ("pending", "running")

    # AC1：可得知訓練完成（running → done）
    final = _poll(client, job["job_id"])
    assert final["status"] == "done"

    # AC2：顯示最佳超參數與評估指標
    result = final["result"]
    assert result["best_params"]                       # 非空 → 確有超參數搜尋
    assert any(k.startswith("clf__") for k in result["best_params"])
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0

    # AC3：模型被儲存（joblib 檔存在）且可於模型頁列出
    models = client.get("/api/ml/models").get_json()["models"]
    assert len(models) == 1
    saved = models[0]
    assert saved["model_uid"] == final["result_model_uid"]
    assert (models_dir).exists()
    assert joblib.load(saved["file_path"]) is not None


# --- AC4：單筆預測輸出存活判定與生存機率 --------------------------------

def test_single_prediction_outputs_survival_and_probability(trained_env) -> None:
    client, _, _ = trained_env
    response = client.post("/api/ml/predict", json=_VALID_PASSENGER)

    assert response.status_code == 200
    body = response.get_json()
    assert isinstance(body["survived"], bool)
    assert 0.0 <= body["probability"] <= 1.0


# --- AC5：CSV 批次預測並可下載含機率結果 --------------------------------

_CSV_HEADER = "Pclass,Sex,SibSp,Parch,Name,Ticket,Age,Fare,Cabin,Embarked"
_CSV_ROW = '1,female,0,0,"Doe, Mrs. A",X1,29,80,,C'


def _csv_bytes(rows: list[str]) -> bytes:
    return ("\n".join([_CSV_HEADER, *rows]) + "\n").encode("utf-8")


def _upload(client, data_bytes: bytes, query: str = ""):
    return client.post(
        f"/api/ml/predict/batch{query}",
        data={"file": (io.BytesIO(data_bytes), "passengers.csv")},
        content_type="multipart/form-data",
    )


def test_batch_prediction_json_and_csv_download(trained_env) -> None:
    client, _, _ = trained_env

    # JSON 結果含機率
    json_response = _upload(client, _csv_bytes([_CSV_ROW, _CSV_ROW]))
    assert json_response.status_code == 200
    body = json_response.get_json()
    assert body["total"] == 2
    assert all(0.0 <= r["probability"] <= 1.0 for r in body["results"])

    # 可下載含機率之 CSV
    csv_response = _upload(client, _csv_bytes([_CSV_ROW]), query="?format=csv")
    assert csv_response.status_code == 200
    assert csv_response.mimetype == "text/csv"
    assert "attachment" in csv_response.headers.get("Content-Disposition", "")
    text = csv_response.get_data(as_text=True)
    assert "probability" in text and "survived" in text


# --- AC6：預測與訓練前處理一致（同一序列化管線，CON-2）-----------------

def test_shared_pipeline_single_equals_batch(trained_env) -> None:
    """同一乘客走單筆與 1 列批次，機率須一致 → 證明共用同一序列化管線。"""
    client, _, _ = trained_env

    single = client.post("/api/ml/predict", json=_VALID_PASSENGER).get_json()
    batch_row = '1,female,0,0,"Test, Mrs. Example",X,29,80,C20,C'
    batch = _upload(client, _csv_bytes([batch_row])).get_json()

    assert single["probability"] == pytest.approx(batch["results"][0]["probability"])
    assert single["survived"] == batch["results"][0]["survived"]


# --- FR-7：三個 ML 頁面皆由完整 app 正常服務 ----------------------------

@pytest.mark.parametrize("path", ["/ml/train", "/ml/models", "/ml/predict"])
def test_ml_pages_served_by_full_app(app_env, path: str) -> None:
    client, _, _ = app_env
    assert client.get(path).status_code == 200
