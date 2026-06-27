"""非同步訓練 job 單元測試（任務 4.2 / FR-3.7~3.9, NFR-R1）。

涵蓋 JobStore 狀態轉移與 run_training / start_training_job 執行器：
成功標記 done、失敗標記 failed 不中斷服務、立即回傳 job_id 不阻塞。
DB/檔案一律用 tmp_path。
"""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

from src.ml import config
from src.ml.jobs import JobStore, run_training, start_training_job
from src.ml.types import JobStatus


@pytest.fixture(autouse=True)
def tiny_param_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        config.PARAM_GRIDS, config.LOGISTIC_REGRESSION,
        {"clf__C": [0.1, 1.0], "clf__solver": ["liblinear"]},
    )


@pytest.fixture
def seeded_db(tmp_path):
    """臨時 DB：含 titanic 資料 + ml schema；回傳 (db_path, models_dir)。"""
    from src.ml.schema import init_ml_schema

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


def _wait_for_terminal(store: JobStore, job_id: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job is not None and job.status in (JobStatus.DONE, JobStatus.FAILED):
            return
        time.sleep(0.05)
    raise AssertionError("job 未在時限內結束")


# --- JobStore 狀態轉移 ----------------------------------------------------

def test_jobstore_create_and_get() -> None:
    store = JobStore()
    job = store.create(config.RANDOM_FOREST)
    assert job.status is JobStatus.PENDING
    assert store.get(job.job_id) == job
    assert store.get("missing") is None


def test_jobstore_list_sorted_newest_first() -> None:
    store = JobStore()
    first = store.create(config.RANDOM_FOREST)
    time.sleep(0.02)  # 確保 created_at 有可分辨差距
    second = store.create(config.LOGISTIC_REGRESSION)
    listed = store.list_all()
    assert listed[0].job_id == second.job_id
    assert {j.job_id for j in listed} == {first.job_id, second.job_id}


def test_jobstore_transition_missing_raises() -> None:
    store = JobStore()
    with pytest.raises(KeyError, match="找不到 job"):
        store.mark_running("missing")


# --- run_training：成功 ---------------------------------------------------

def test_run_training_success_marks_done(seeded_db) -> None:
    db_path, models_dir = seeded_db
    store = JobStore()
    job = store.create(config.LOGISTIC_REGRESSION)

    run_training(job.job_id, config.LOGISTIC_REGRESSION, store, db_path, models_dir)

    done = store.get(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.result_model_uid
    assert done.finished_at is not None
    assert (models_dir / f"{done.result_model_uid}.joblib").exists()

    connection = sqlite3.connect(db_path)
    try:
        count = connection.execute("SELECT COUNT(*) FROM ml_model").fetchone()[0]
    finally:
        connection.close()
    assert count == 1


# --- run_training：失敗不中斷（NFR-R1）----------------------------------

def test_run_training_failure_marks_failed(tmp_path) -> None:
    empty_db = str(tmp_path / "empty.db")  # 無 titanic 表 → load 失敗
    store = JobStore()
    job = store.create(config.LOGISTIC_REGRESSION)

    # 不得拋出例外（服務不中斷）
    run_training(job.job_id, config.LOGISTIC_REGRESSION, store, empty_db, tmp_path)

    failed = store.get(job.job_id)
    assert failed.status is JobStatus.FAILED
    assert failed.error_message
    assert failed.finished_at is not None


# --- start_training_job：非同步、立即回傳 -------------------------------

def test_start_training_job_returns_and_completes(seeded_db) -> None:
    db_path, models_dir = seeded_db
    store = JobStore()

    job = start_training_job(
        config.LOGISTIC_REGRESSION, store=store,
        db_path=db_path, models_dir=models_dir,
    )
    # 立即回傳 job_id（尚未完成）
    assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)

    _wait_for_terminal(store, job.job_id)
    assert store.get(job.job_id).status is JobStatus.DONE


def test_start_training_job_rejects_unknown_algorithm() -> None:
    store = JobStore()
    with pytest.raises(ValueError, match="不支援"):
        start_training_job("svm", store=store)
    assert store.list_all() == []  # 未建立 job
