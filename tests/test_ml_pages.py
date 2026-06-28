"""ML 頁面路由整合測試（任務 7.x / FR-7.x）。

以 Flask test client 驗證訓練 / 模型 / 預測頁的 GET 路由回 200，
且頁面含關鍵互動元素（演算法選擇、清單容器、表單、CSV 上傳、導覽列）。
頁面僅 render_template（無業務邏輯）；JS 互動流程留待 8.x E2E。
"""

from __future__ import annotations

import pytest
from flask import Flask

from src.web.ml_pages import create_ml_pages_blueprint


@pytest.fixture
def client():
    app = Flask(__name__, template_folder="../templates")
    app.register_blueprint(create_ml_pages_blueprint())
    return app.test_client()


# --- 訓練頁（7.1）--------------------------------------------------------

def test_train_page_returns_200(client) -> None:
    response = client.get("/ml/train")
    assert response.status_code == 200


def test_train_page_has_algorithm_select_and_train_action(client) -> None:
    html = client.get("/ml/train").get_data(as_text=True)
    assert "logistic_regression" in html
    assert "random_forest" in html
    assert "/api/ml/train" in html  # 前端呼叫訓練 API


# --- 模型管理頁（7.2）----------------------------------------------------

def test_models_page_returns_200(client) -> None:
    assert client.get("/ml/models").status_code == 200


def test_models_page_calls_models_api(client) -> None:
    html = client.get("/ml/models").get_data(as_text=True)
    assert "/api/ml/models" in html


# --- 預測頁（7.3）--------------------------------------------------------

def test_predict_page_returns_200(client) -> None:
    assert client.get("/ml/predict").status_code == 200


def test_predict_page_has_single_and_batch_forms(client) -> None:
    html = client.get("/ml/predict").get_data(as_text=True)
    assert "/api/ml/predict" in html          # 單筆
    assert "/api/ml/predict/batch" in html    # CSV 批次
    assert 'type="file"' in html               # CSV 上傳


# --- 導覽列（7.4）--------------------------------------------------------

@pytest.mark.parametrize("path", ["/ml/train", "/ml/models", "/ml/predict"])
def test_pages_share_navbar(client, path: str) -> None:
    html = client.get(path).get_data(as_text=True)
    for link in ("/ml/train", "/ml/models", "/ml/predict"):
        assert link in html
