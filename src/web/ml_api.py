"""ML 訓練 REST API（任務 4.3 / SRS §4.2）。

提供 POST /api/ml/train 與 GET /api/ml/train/status/<job_id>，把 HTTP 請求
翻譯為 jobs 服務層呼叫。採 blueprint 工廠注入依賴（store / db_path /
models_dir），使正式環境與測試可用不同資料來源。

展示層：負責輸入驗證、HTTP 狀態碼與 JSON 序列化；業務邏輯一律委派 src.ml。
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from src.ml.jobs import JobStore, job_store, start_training_job
from src.ml.registry import get_model_by_uid
from src.ml.types import Job, JobStatus


def create_ml_blueprint(
    store: JobStore | None = None,
    db_path: str | None = None,
    models_dir: str | Path | None = None,
) -> Blueprint:
    """建立 ML API blueprint，依賴以參數注入便於測試。

    Args:
        store: job 狀態存放；省略時用程序層級 job_store。
        db_path: 訓練資料與 registry 的 SQLite 路徑；省略時用設定預設。
        models_dir: 模型檔目錄；省略時用設定預設。
    """
    active_store = store or job_store
    blueprint = Blueprint("ml_api", __name__, url_prefix="/api/ml")

    @blueprint.post("/train")
    def train():
        data = request.get_json(silent=True)
        if not data or "algorithm" not in data:
            return jsonify({"error": "缺少必要欄位 'algorithm'"}), 400

        try:
            job = start_training_job(
                data["algorithm"],
                store=active_store,
                db_path=db_path,
                models_dir=models_dir,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422

        return (
            jsonify(
                {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "algorithm": job.algorithm,
                }
            ),
            202,
        )

    @blueprint.get("/train/status/<job_id>")
    def train_status(job_id: str):
        job = active_store.get(job_id)
        if job is None:
            return jsonify({"error": f"找不到 job：{job_id}"}), 404
        return jsonify(_job_payload(job, db_path)), 200

    return blueprint


def _job_payload(job: Job, db_path: str | None) -> dict:
    """組裝 job 狀態回應；done 時附最佳超參數與指標摘要（FR-3.8）。"""
    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "algorithm": job.algorithm,
        "progress": job.progress,
        "result_model_uid": job.result_model_uid,
        "error_message": job.error_message,
    }

    if job.status is JobStatus.DONE and job.result_model_uid:
        metadata = get_model_by_uid(job.result_model_uid, db_path=db_path)
        if metadata is not None:
            payload["result"] = {
                "best_params": metadata.hyperparameters,
                "best_cv_score": metadata.best_cv_score,
                "metrics": metadata.metrics,
            }

    return payload
