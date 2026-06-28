"""ML 平台頁面路由（任務 7.x / SRS §7）。

提供訓練 / 模型管理 / 預測三個 HTML 頁面。頁面為薄客戶端：僅 render_template，
所有資料互動由前端 JS 呼叫 /api/ml 既有 JSON API（共用展示層契約，不重複業務邏輯）。

與 ml_api（JSON，url_prefix=/api/ml）分離為獨立 blueprint，避免頁面路由與 API
路由前綴衝突，並保持每個 blueprint 單一職責。
"""

from __future__ import annotations

from flask import Blueprint, render_template


def create_ml_pages_blueprint() -> Blueprint:
    """建立 ML 頁面 blueprint（純展示，無依賴注入需求）。"""
    blueprint = Blueprint("ml_pages", __name__, url_prefix="/ml")

    @blueprint.get("/train")
    def train_page():
        return render_template("train.html", active_page="train")

    @blueprint.get("/models")
    def models_page():
        return render_template("models.html", active_page="models")

    @blueprint.get("/predict")
    def predict_page():
        return render_template("predict.html", active_page="predict")

    return blueprint
