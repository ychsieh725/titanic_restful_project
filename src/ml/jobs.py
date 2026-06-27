"""非同步訓練 job 的執行緒安全狀態管理。

訓練以 threading 在背景執行（CON-6），本模組提供集中、執行緒安全的
job 狀態存取。狀態轉移採不可變更新：每次以 dataclasses.replace 產生
新的 Job 快照取代舊值，避免並行讀寫到部分更新的物件（coding-style 不可變性）。

NFR-R1：單一 job 失敗（標記 failed）不影響其他 job 或服務整體。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Optional

from .types import Job, JobStatus


class JobStore:
    """執行緒安全的記憶體 job 登錄。

    服務重啟後 job 狀態不保留（已完成模型仍由 Registry 持久化，符合 NFR-R2）；
    job 本身為短生命週期的進行中狀態，置於記憶體即可。
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, algorithm: str) -> Job:
        """建立一個 pending job 並回傳其快照。

        Args:
            algorithm: 本次訓練演算法名稱。

        Returns:
            新建立的 Job（status=pending），含自動產生的 job_id。
        """
        job = Job(job_id=uuid.uuid4().hex, algorithm=algorithm)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """依 job_id 取得 job 快照；不存在時回傳 None。"""
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        """回傳所有 job 快照（建立時間新→舊）。"""
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def mark_running(self, job_id: str, progress: str = "training") -> Job:
        """將 job 標記為 running。"""
        return self._transition(
            job_id, status=JobStatus.RUNNING, progress=progress
        )

    def mark_done(self, job_id: str, result_model_uid: str) -> Job:
        """將 job 標記為 done，並記錄產出的模型 UID 與完成時間。"""
        return self._transition(
            job_id,
            status=JobStatus.DONE,
            progress="completed",
            result_model_uid=result_model_uid,
            finished_at=datetime.now(),
        )

    def mark_failed(self, job_id: str, error_message: str) -> Job:
        """將 job 標記為 failed，並記錄錯誤訊息與完成時間（NFR-R1）。"""
        return self._transition(
            job_id,
            status=JobStatus.FAILED,
            progress="failed",
            error_message=error_message,
            finished_at=datetime.now(),
        )

    def _transition(self, job_id: str, **changes) -> Job:
        """以不可變更新方式套用狀態變更並寫回。

        Raises:
            KeyError: job_id 不存在時，明確指出。
        """
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise KeyError(f"找不到 job：{job_id}")
            updated = replace(current, **changes)
            self._jobs[job_id] = updated
        return updated


# 程序層級單一實例：供展示層與訓練背景執行緒共用同一份 job 狀態。
job_store = JobStore()
