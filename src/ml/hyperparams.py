"""使用者超參數驗證與 GridSearch 格點建構（FR-3.3 防呆）。

把使用者於訓練頁輸入的超參數候選值，安全地轉成 GridSearchCV 用的格點：
- 僅接受該演算法 HYPERPARAM_SPECS 內的已知參數，未知鍵明確拒絕。
- 逐值檢查型別與範圍（如 C 需正數、n_estimators 為界內整數）。
- 未提供或留空的參數回退至 config.PARAM_GRIDS 預設；固定參數（如 solver）保留。
- 限制單一參數候選數與整體組合總數，避免笛卡兒積爆炸使訓練卡死（NFR-P1）。

任一不合法即拋 HyperparameterError（含逐欄位訊息），由展示層翻成 422，
不讓不合理輸入流入 GridSearchCV 造成未處理的 500（FR-3.9 / coding-style）。

純 Python，不 import 任何 Web 框架物件（CON-4）。
"""

from __future__ import annotations

from math import prod
from typing import Optional

from . import config

# 代表「不限深度」的字面值；前端以此字串表達 max_depth=None。
_NONE_TOKENS = {"none", "null", ""}


class HyperparameterError(Exception):
    """超參數驗證失敗，errors 為「欄位 → 錯誤訊息」對應（含整體 _grid）。"""

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        super().__init__("超參數驗證失敗：" + "；".join(errors.values()))


def build_param_grid(
    algorithm: str,
    overrides: Optional[dict] = None,
) -> dict[str, list]:
    """依使用者覆寫建構並驗證 GridSearchCV 格點。

    Args:
        algorithm: SUPPORTED_ALGORITHMS 之一。
        overrides: {參數名: 候選值清單} 覆寫；None/缺項回退預設。

    Returns:
        以 "clf__" 前綴對應分類器步驟的格點，可直接交給 GridSearchCV。

    Raises:
        ValueError: 演算法不支援時（訊息列出可用選項）。
        HyperparameterError: 任一覆寫不合法或格點過大時，errors 含所有問題。
    """
    if algorithm not in config.HYPERPARAM_SPECS:
        supported = ", ".join(config.SUPPORTED_ALGORITHMS)
        raise ValueError(f"不支援的演算法 '{algorithm}'，可用選項：{supported}")

    overrides = overrides or {}
    specs = config.HYPERPARAM_SPECS[algorithm]
    defaults = config.PARAM_GRIDS[algorithm]
    errors: dict[str, str] = {}
    grid: dict[str, list] = {}

    # 未知參數鍵：明確拒絕，避免靜默忽略使用者意圖。
    for key in overrides:
        if key not in specs:
            errors[key] = f"未知的超參數 '{key}'。"

    for name, spec in specs.items():
        raw = overrides.get(name)
        if raw in (None, "", []):
            # 回退預設；若預設未涵蓋此參數則略過，交由 estimator 自身預設。
            default = defaults.get(f"clf__{name}")
            if default is not None:
                grid[f"clf__{name}"] = default
            continue
        values, error = _parse_values(name, raw, spec)
        if error:
            errors[name] = error
        else:
            grid[f"clf__{name}"] = values

    # 保留 specs 未涵蓋的固定參數（如 LR 的 solver=liblinear，l1 必需）。
    for key, value in defaults.items():
        grid.setdefault(key, value)

    if errors:
        raise HyperparameterError(errors)

    _check_grid_size(grid, errors)
    if errors:
        raise HyperparameterError(errors)

    return grid


def _parse_values(name: str, raw, spec: dict) -> tuple[list, Optional[str]]:
    """將單一參數的候選值清單驗證並轉型；回 (values, error_or_None)。"""
    values = raw if isinstance(raw, list) else [raw]
    if len(values) > config.MAX_VALUES_PER_PARAM:
        return [], f"{spec['label']} 候選值過多（上限 {config.MAX_VALUES_PER_PARAM} 個）。"

    parsed: list = []
    for item in values:
        value, error = _coerce(item, spec)
        if error:
            return [], f"{spec['label']}：{error}"
        parsed.append(value)
    return parsed, None


def _coerce(item, spec: dict) -> tuple[object, Optional[str]]:
    """依 spec 型別檢查並轉型單一值；回 (value, error_or_None)。"""
    kind = spec["type"]

    if kind == "choice":
        if item not in spec["choices"]:
            return None, f"需為 {spec['choices']} 之一，收到「{item}」。"
        return item, None

    if kind == "int_or_none" and _is_none_token(item):
        return None, None

    if kind in ("int", "int_or_none"):
        number, error = _as_int(item)
    else:  # float
        number, error = _as_float(item)
    if error:
        return None, error

    if number < spec["min"] or number > spec["max"]:
        return None, f"需介於 {spec['min']}–{spec['max']}，收到「{item}」。"
    return number, None


def _check_grid_size(grid: dict[str, list], errors: dict[str, str]) -> None:
    """檢查 GridSearch 組合總數是否超過上限（防笛卡兒積爆炸）。"""
    combinations = prod(len(values) for values in grid.values())
    if combinations > config.MAX_GRID_COMBINATIONS:
        errors["_grid"] = (
            f"參數組合數 {combinations} 超過上限 {config.MAX_GRID_COMBINATIONS}，"
            "請減少候選值數量。"
        )


def _is_none_token(item) -> bool:
    return item is None or (isinstance(item, str) and item.strip().lower() in _NONE_TOKENS)


def _as_int(item) -> tuple[int, Optional[str]]:
    if isinstance(item, bool):
        return 0, "需為整數。"
    if isinstance(item, int):
        return item, None
    if isinstance(item, float):
        if item.is_integer():
            return int(item), None
        return 0, f"需為整數，收到「{item}」。"
    try:
        text = str(item).strip()
        number = float(text)
    except (TypeError, ValueError):
        return 0, f"需為整數，收到「{item}」。"
    if not number.is_integer():
        return 0, f"需為整數，收到「{item}」。"
    return int(number), None


def _as_float(item) -> tuple[float, Optional[str]]:
    if isinstance(item, bool):
        return 0.0, "需為數值。"
    if isinstance(item, (int, float)):
        return float(item), None
    try:
        return float(str(item).strip()), None
    except (TypeError, ValueError):
        return 0.0, f"需為數值，收到「{item}」。"
