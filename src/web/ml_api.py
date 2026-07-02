"""ML 訓練 REST API（任務 4.3 / SRS §4.2）。

提供 POST /api/ml/train 與 GET /api/ml/train/status/<job_id>，把 HTTP 請求
翻譯為 jobs 服務層呼叫。採 blueprint 工廠注入依賴（store / db_path /
models_dir），使正式環境與測試可用不同資料來源。

展示層：負責輸入驗證、HTTP 狀態碼與 JSON 序列化；業務邏輯一律委派 src.ml。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, Response, jsonify, request

from src.ml import config
from src.ml.hyperparams import HyperparameterError, build_param_grid
from src.ml.jobs import JobStore, job_store, start_training_job
from src.ml.prediction import NoActiveModelError, predict_batch, predict_one
from src.ml.registry import (
    delete_model,
    get_model_by_uid,
    list_models,
    set_active_model,
)
from src.ml.types import Job, JobStatus, ModelMetadata
from src.ml.validation import (
    BatchValidationError,
    ValidationError,
    validate_passenger,
    validate_passengers,
)


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

        # 驗證使用者自訂超參數（防呆）：不合法回 422 帶欄位明細，
        # 不讓不合理輸入流入 GridSearchCV 造成訓練失敗或卡死。
        try:
            param_grid = build_param_grid(
                data["algorithm"], data.get("hyperparameters")
            )
        except HyperparameterError as exc:
            return jsonify({"error": "超參數驗證失敗", "fields": exc.errors}), 422
        except ValueError as exc:  # 未知演算法
            return jsonify({"error": str(exc)}), 422

        try:
            job = start_training_job(
                data["algorithm"],
                store=active_store,
                db_path=db_path,
                models_dir=models_dir,
                param_grid=param_grid,
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

    @blueprint.get("/models")
    def models():
        items = [_metadata_payload(m) for m in list_models(db_path=db_path)]
        return jsonify({"models": items}), 200

    @blueprint.post("/models/<int:model_id>/activate")
    def activate_model(model_id: int):
        metadata = set_active_model(model_id, db_path=db_path)
        if metadata is None:
            return jsonify({"error": f"找不到模型：{model_id}"}), 404
        return jsonify({"model": _metadata_payload(metadata)}), 200

    @blueprint.delete("/models/<int:model_id>")
    def remove_model(model_id: int):
        metadata = delete_model(model_id, db_path=db_path)
        if metadata is None:
            return jsonify({"error": f"找不到模型：{model_id}"}), 404
        return jsonify({"deleted": _metadata_payload(metadata)}), 200

    @blueprint.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "請提供乘客資料（JSON 物件）"}), 400

        try:
            passenger = validate_passenger(data)
        except ValidationError as exc:
            return jsonify({"error": str(exc), "fields": exc.errors}), 422

        try:
            result = predict_one(passenger, db_path=db_path)
        except NoActiveModelError as exc:
            return jsonify({"error": str(exc)}), 409

        return jsonify(
            {"survived": result.survived, "probability": result.probability}
        ), 200

    @blueprint.post("/predict/batch")
    def predict_batch_endpoint():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"error": "請上傳 CSV 檔（表單欄位名 file）。"}), 400
        if not upload.filename.lower().endswith(".csv"):
            return jsonify({"error": "僅接受副檔名為 .csv 的檔案。"}), 400

        try:
            frame = pd.read_csv(upload.stream)
        except Exception:
            return jsonify({"error": "CSV 解析失敗，請確認檔案為有效的 CSV 格式。"}), 400

        if frame.empty:
            return jsonify({"error": "CSV 沒有任何資料列。"}), 400
        if len(frame) > config.MAX_BATCH_ROWS:
            return jsonify(
                {"error": f"資料列數超過上限 {config.MAX_BATCH_ROWS}，請分批上傳。"}
            ), 413

        try:
            passengers = validate_passengers(_frame_to_records(frame))
        except BatchValidationError as exc:
            rows = [
                {"row": index, "fields": fields}
                for index, fields in sorted(exc.row_errors.items())
            ]
            return jsonify({"error": "批次輸入驗證失敗。", "rows": rows}), 422

        try:
            result = predict_batch(passengers, db_path=db_path)
        except NoActiveModelError as exc:
            return jsonify({"error": str(exc)}), 409

        if request.args.get("format") == "csv":
            return Response(
                result.to_csv(index=False),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=predictions.csv"},
            )

        return jsonify(
            {"results": _frame_to_records(result), "total": len(result)}
        ), 200

    return blueprint


def _frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """將 DataFrame 正規化為 JSON/驗證層可用的原生 Python record 清單。

    pandas 會把缺值讀成 NaN、數值欄讀成 numpy 純量：NaN 直接 jsonify 會產生
    非法 JSON（`NaN` token），numpy 純量也非驗證層預期的原生型別。此處在資料
    邊界一次轉換：NaN → None、numpy 純量 → 原生型別，同時服務於 CSV 輸入正規化
    與批次預測結果序列化，維持框架/函式庫無關（CON-4）。
    """
    cleaned = frame.astype(object).where(pd.notnull(frame), None)
    return [
        {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in record.items()
        }
        for record in cleaned.to_dict(orient="records")
    ]


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


def _metadata_payload(metadata: ModelMetadata) -> dict:
    """序列化 ModelMetadata 為 JSON-friendly dict（datetime → isoformat）。"""
    return {
        "id": metadata.id,
        "model_uid": metadata.model_uid,
        "algorithm": metadata.algorithm,
        "hyperparameters": metadata.hyperparameters,
        "best_cv_score": metadata.best_cv_score,
        "metrics": metadata.metrics,
        "feature_list": metadata.feature_list,
        "training_rows": metadata.training_rows,
        "data_hash": metadata.data_hash,
        "file_path": metadata.file_path,
        "training_duration_sec": metadata.training_duration_sec,
        "is_active": metadata.is_active,
        "created_at": metadata.created_at.isoformat(),
    }
