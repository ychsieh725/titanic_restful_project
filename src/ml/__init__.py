"""Titanic ML 服務層（框架無關）。

此套件為純 Python，不得 import 任何 Web 框架物件（CON-4 / NFR-M1）。
展示層（Flask）透過此套件的公開介面操作，使服務層可在 Flask / FastAPI
之間替換而不影響核心邏輯。

模組邊界：
- types       資料契約（dataclass / enum），定義跨模組流動的資料結構
- config      集中設定：超參數格點、特徵清單、CV 折數、路徑（NFR-M2）
- jobs        非同步訓練 job 的執行緒安全狀態管理
- features    特徵工程管線（任務 3.1）
- training    訓練與超參數搜尋（任務 4.1）
- registry    模型登錄與 active 切換（任務 5.1）
- prediction  單筆 / 批次預測（任務 6.1）
"""

from . import config, features, jobs, types

__all__ = ["config", "features", "jobs", "types"]
