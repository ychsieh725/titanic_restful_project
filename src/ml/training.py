"""訓練服務（任務 4.1）。

以共用特徵管線（features.build_feature_pipeline）外包分類器，透過 GridSearchCV
執行超參數搜尋，於 held-out 測試集計算指標，回傳已 fit 的整條 pipeline 與
TrainResult 摘要（FR-3.2~3.6, 3.10）。

純 Python，不 import 任何 Web 框架物件（CON-4 / NFR-M1）。非同步包裝（4.2）、
持久化與 DB 寫入（5.1）由各自模組負責；本模組只提供同步訓練核心。
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .features import build_feature_pipeline
from .types import Metrics, TrainResult


def train_model(
    algorithm: str,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict[str, list] | None = None,
) -> tuple[Pipeline, TrainResult]:
    """訓練單一演算法並回傳已 fit 的整條 pipeline 與結果摘要。

    Args:
        algorithm: SUPPORTED_ALGORITHMS 之一。
        X: 含 RAW_FEATURE_COLUMNS 的原始特徵 DataFrame。
        y: 目標欄位（0/1）。
        param_grid: GridSearchCV 格點；省略時用 config 預設。已驗證的使用者
            自訂格點由呼叫端（hyperparams.build_param_grid）提供。

    Returns:
        (fitted_pipeline, TrainResult)。pipeline 含前處理＋最佳分類器，
        可直接 predict 或交由 registry 持久化（5.1）。

    Raises:
        ValueError: 演算法不在支援清單時（訊息明確列出可用選項）。
    """
    estimator = _build_estimator(algorithm)
    if param_grid is None:
        param_grid = config.get_param_grid(algorithm)

    pipeline = Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("clf", estimator),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    started = time.perf_counter()
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=config.CV_FOLDS,
        n_jobs=config.N_JOBS,
    )
    search.fit(X_train, y_train)
    duration = time.perf_counter() - started

    best_pipeline: Pipeline = search.best_estimator_
    metrics = _evaluate(best_pipeline, X_test, y_test)

    result = TrainResult(
        model_uid=uuid.uuid4().hex,
        algorithm=algorithm,
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
        metrics=metrics,
        feature_list=_feature_names(best_pipeline),
        training_rows=len(X_train),
        data_hash=_hash_frame(X),
        training_duration_sec=duration,
    )
    return best_pipeline, result


def load_training_data(
    database_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """從 SQLite titanic 表載入訓練資料（特徵 X 與目標 y）。

    僅選取原始特徵欄與目標欄；使用唯讀查詢，不拼接 SQL（NFR-S2）。
    """
    path = str(database_path or config.DATABASE_PATH)
    columns = ", ".join((*config.RAW_FEATURE_COLUMNS, config.TARGET_COLUMN))
    connection = sqlite3.connect(path)
    try:
        frame = pd.read_sql_query(f"SELECT {columns} FROM titanic", connection)
    finally:
        connection.close()

    y = frame[config.TARGET_COLUMN]
    X = frame[list(config.RAW_FEATURE_COLUMNS)]
    return X, y


def _build_estimator(algorithm: str) -> ClassifierMixin:
    """依演算法名稱建立分類器（超參數由 GridSearch 覆寫）。"""
    if algorithm == config.LOGISTIC_REGRESSION:
        return LogisticRegression(
            max_iter=config.LR_MAX_ITER,
            random_state=config.RANDOM_STATE,
        )
    if algorithm == config.RANDOM_FOREST:
        return RandomForestClassifier(random_state=config.RANDOM_STATE)

    supported = ", ".join(config.SUPPORTED_ALGORITHMS)
    raise ValueError(f"不支援的演算法 '{algorithm}'，可用選項：{supported}")


def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Metrics:
    """於 held-out 測試集計算評估指標（FR-3.6）。"""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return Metrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
    )


def _feature_names(pipeline: Pipeline) -> list[str]:
    """取編碼後模型實際消費的特徵欄位名稱（SRS §6.2 feature_list）。"""
    features_step = pipeline.named_steps["features"]
    return [str(name) for name in features_step.get_feature_names_out()]


def _hash_frame(frame: pd.DataFrame) -> str:
    """對訓練資料內容計算 sha256，供模型追溯（SRS §6.2 data_hash）。"""
    row_hashes = pd.util.hash_pandas_object(frame, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()
