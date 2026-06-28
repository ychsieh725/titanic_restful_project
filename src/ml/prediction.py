"""單筆預測服務（任務 6.1）。

以 active 模型對單筆乘客輸入預測，回傳存活判定與生存機率（FR-5.1, 5.2）。
預測一律使用 active 模型載回的整條 pipeline（CON-2），不另寫前處理；
無 active 模型時拋出明確網域例外，由展示層翻譯為適當狀態碼（不裸奔 500）。

純 Python，不 import 任何 Web 框架物件（CON-4 / NFR-M1）。
"""

from __future__ import annotations

import pandas as pd

from .features import passenger_to_frame, passengers_to_frame
from .registry import get_active_model, load_model_file
from .types import PassengerInput, PredictionResult

# 存活判定門檻：生存機率 >= 此值即判定存活。
SURVIVAL_THRESHOLD = 0.5


class NoActiveModelError(Exception):
    """無 active 模型可供預測時拋出（FR-4.5）。

    展示層應據此提示使用者先完成訓練，而非回傳未處理的 500。
    """


def predict_one(
    passenger: PassengerInput,
    db_path: str | None = None,
) -> PredictionResult:
    """以 active 模型對單筆輸入預測。

    Args:
        passenger: 單筆乘客原始特徵（前處理由共用管線負責，CON-2）。
        db_path: registry 的 SQLite 路徑；省略時用設定預設。

    Returns:
        PredictionResult（survived 判定 + probability 生存機率）。

    Raises:
        NoActiveModelError: 目前無 active 模型時。
    """
    metadata = get_active_model(db_path=db_path)
    if metadata is None:
        raise NoActiveModelError("目前沒有啟用中的模型，請先完成訓練並設定 active 模型。")

    pipeline = load_model_file(metadata)
    frame = passenger_to_frame(passenger)
    probability = float(pipeline.predict_proba(frame)[0][1])

    return PredictionResult(
        survived=probability >= SURVIVAL_THRESHOLD,
        probability=probability,
    )


def predict_batch(
    passengers: list[PassengerInput],
    db_path: str | None = None,
) -> pd.DataFrame:
    """以 active 模型對多筆輸入批次預測（FR-5.3）。

    一次載回 active pipeline 並整批 predict_proba（共用管線 CON-2），
    回傳含原始欄位加上 survived / probability 兩欄的 DataFrame。
    機率與判定以原生 Python 型別輸出，供展示層直接 jsonify / CSV 下載。

    Args:
        passengers: 已驗證的乘客清單；空清單回傳僅含欄位的空結果。
        db_path: registry 的 SQLite 路徑；省略時用設定預設。

    Returns:
        DataFrame，欄位 = 原始輸入 + ["probability", "survived"]。

    Raises:
        NoActiveModelError: 目前無 active 模型時。
    """
    metadata = get_active_model(db_path=db_path)
    if metadata is None:
        raise NoActiveModelError("目前沒有啟用中的模型，請先完成訓練並設定 active 模型。")

    frame = passengers_to_frame(passengers)
    result = frame.copy()

    if frame.empty:
        result["probability"] = pd.Series(dtype=float)
        result["survived"] = pd.Series(dtype=bool)
        return result

    pipeline = load_model_file(metadata)
    probabilities = pipeline.predict_proba(frame)[:, 1]
    result["probability"] = [float(value) for value in probabilities]
    result["survived"] = [bool(value >= SURVIVAL_THRESHOLD) for value in probabilities]
    return result
