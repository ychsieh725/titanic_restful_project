"""單筆預測服務（任務 6.1）。

以 active 模型對單筆乘客輸入預測，回傳存活判定與生存機率（FR-5.1, 5.2）。
預測一律使用 active 模型載回的整條 pipeline（CON-2），不另寫前處理；
無 active 模型時拋出明確網域例外，由展示層翻譯為適當狀態碼（不裸奔 500）。

純 Python，不 import 任何 Web 框架物件（CON-4 / NFR-M1）。
"""

from __future__ import annotations

from .features import passenger_to_frame
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
