"""服務層資料契約。

集中定義跨模組流動的資料結構。所有 dataclass 預設為 frozen（不可變），
狀態變更一律建立新物件（見 coding-style 不可變性原則），避免隱藏副作用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    """訓練 job 的生命週期狀態（對應 SRS §6.3）。

    繼承 str 使其可直接序列化為 JSON 字串。
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class PassengerInput:
    """單筆預測的輸入欄位（對應 FR-5.1 / SRS §6.1 原始欄位）。

    僅含預測所需的原始特徵；前處理由共用管線負責（CON-2），
    呼叫端不得自行前處理。可空欄位（Age/Fare/Cabin/Embarked）允許 None，
    由管線補值。
    """

    Pclass: int
    Sex: str
    SibSp: int
    Parch: int
    Name: str
    Ticket: str
    Age: Optional[float] = None
    Fare: Optional[float] = None
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None


@dataclass(frozen=True)
class Metrics:
    """held-out 測試集評估指標（對應 FR-3.6）。"""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class TrainResult:
    """一次訓練完成的產出摘要（供寫入 Model Registry）。"""

    model_uid: str
    algorithm: str
    best_params: dict[str, Any]
    best_cv_score: float
    metrics: Metrics
    feature_list: list[str]
    training_rows: int
    data_hash: str
    training_duration_sec: float


@dataclass(frozen=True)
class ModelMetadata:
    """Model Registry 中的模型登錄資料（對應 ml_model 表 / SRS §6.2）。"""

    id: int
    model_uid: str
    algorithm: str
    hyperparameters: dict[str, Any]
    best_cv_score: float
    metrics: dict[str, Any]
    feature_list: list[str]
    training_rows: int
    data_hash: str
    file_path: str
    training_duration_sec: float
    is_active: bool
    created_at: datetime


@dataclass(frozen=True)
class Job:
    """非同步訓練 job 的不可變快照（對應 SRS §6.3）。

    狀態轉移由 jobs.JobStore 以「建立新 Job」方式處理，不在原物件上修改。
    """

    job_id: str
    algorithm: str
    status: JobStatus = JobStatus.PENDING
    progress: str = ""
    result_model_uid: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None


@dataclass(frozen=True)
class PredictionResult:
    """單筆預測輸出（對應 FR-5.2）。

    survived 為存活判定（True/False），probability 為生存機率（0–1）。
    """

    survived: bool
    probability: float
