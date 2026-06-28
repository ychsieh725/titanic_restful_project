"""單筆輸入驗證（任務 6.3 / FR-5.5, NFR-S1/S2）。

對單筆乘客輸入做 schema-based 驗證：缺漏、型別、範圍與允許值，並一次蒐集
所有錯誤（非 fail-fast），回報以使用者語氣、具修正指引的明確訊息。驗證通過
回傳不可變的 PassengerInput；展示層據此回 422，避免裸奔 500。

規則對齊 SRS §6.1 與 init_db 的 CHECK 約束，常數集中於 config（NFR-M2）。
純 Python，不 import 任何 Web 框架物件（CON-4）。
"""

from __future__ import annotations

from typing import Optional

from . import config
from .types import PassengerInput


class ValidationError(Exception):
    """輸入驗證失敗，attribute errors 為「欄位 → 錯誤訊息」對應。"""

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        super().__init__("輸入驗證失敗：" + "；".join(errors.values()))


class BatchValidationError(Exception):
    """批次驗證失敗，row_errors 為「列索引（0 起算）→ 欄位錯誤」對應（FR-5.3）。

    展示層據此回 422 並標出每一列的問題欄位，讓使用者一次修正整份 CSV。
    """

    def __init__(self, row_errors: dict[int, dict[str, str]]) -> None:
        self.row_errors = row_errors
        super().__init__(f"批次驗證失敗：{len(row_errors)} 列含不合法欄位。")


def validate_passenger(data: dict) -> PassengerInput:
    """驗證單筆輸入並轉為 PassengerInput。

    Raises:
        ValidationError: 任一欄位不合法時，errors 含所有問題欄位。
    """
    errors: dict[str, str] = {}

    pclass = _check_choice(data, "Pclass", config.VALID_PCLASS, errors, is_int=True)
    sex = _check_choice(data, "Sex", config.VALID_SEX, errors)
    sibsp = _check_non_negative_int(data, "SibSp", errors)
    parch = _check_non_negative_int(data, "Parch", errors)
    name = _check_text(data, "Name", config.NAME_MAX_LEN, errors)
    ticket = _check_text(data, "Ticket", config.TICKET_MAX_LEN, errors)

    age = _check_optional_number(
        data, "Age", errors, minimum=config.AGE_MIN, maximum=config.AGE_MAX
    )
    fare = _check_optional_number(data, "Fare", errors, minimum=0.0)
    cabin = _check_optional_text(data, "Cabin", config.CABIN_MAX_LEN, errors)
    embarked = _check_optional_choice(data, "Embarked", config.VALID_EMBARKED, errors)

    if errors:
        raise ValidationError(errors)

    return PassengerInput(
        Pclass=pclass, Sex=sex, SibSp=sibsp, Parch=parch,
        Name=name, Ticket=ticket,
        Age=age, Fare=fare, Cabin=cabin, Embarked=embarked,
    )


def validate_passengers(records: list[dict]) -> list[PassengerInput]:
    """逐列驗證批次輸入，蒐集所有問題列後一次回報（非 fail-fast）。

    重用 validate_passenger 確保單筆與批次規則一致（DRY）。

    Args:
        records: 每列一個 dict 的乘客原始輸入；空清單回傳空清單。

    Returns:
        全部合法時，依序回傳對應的 PassengerInput 清單。

    Raises:
        BatchValidationError: 任一列不合法時，row_errors 含所有問題列。
    """
    validated: list[PassengerInput] = []
    row_errors: dict[int, dict[str, str]] = {}

    for index, record in enumerate(records):
        try:
            validated.append(validate_passenger(record))
        except ValidationError as exc:
            row_errors[index] = exc.errors

    if row_errors:
        raise BatchValidationError(row_errors)

    return validated


# --- 欄位驗證輔助（缺漏即記錄並回 None，避免後續再次報錯）----------------

def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _missing(data: dict, field: str, errors: dict) -> bool:
    if data.get(field) is None:
        errors[field] = f"{field} 為必填欄位，請提供。"
        return True
    return False


def _check_choice(data, field, choices, errors, *, is_int=False):
    if _missing(data, field, errors):
        return None
    value = data[field]
    if is_int and not _is_int(value):
        errors[field] = f"{field} 需為整數，且為 {choices} 之一。"
        return None
    if value not in choices:
        errors[field] = f"{field} 需為 {choices} 之一，收到「{value}」。"
        return None
    return value


def _check_non_negative_int(data, field, errors):
    if _missing(data, field, errors):
        return None
    value = data[field]
    if not _is_int(value) or value < 0:
        errors[field] = f"{field} 需為 0 或正整數，收到「{value}」。"
        return None
    return value


def _check_text(data, field, max_len, errors):
    if _missing(data, field, errors):
        return None
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        errors[field] = f"{field} 不可為空白。"
        return None
    if len(value) > max_len:
        errors[field] = f"{field} 長度不可超過 {max_len} 字元。"
        return None
    return value


def _check_optional_number(data, field, errors, *, minimum, maximum=None) -> Optional[float]:
    value = data.get(field)
    if value is None:
        return None
    if not _is_number(value):
        errors[field] = f"{field} 需為數值，收到「{value}」。"
        return None
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}–{maximum}" if maximum is not None else f"≥ {minimum}"
        errors[field] = f"{field} 需介於 {bound}，收到「{value}」。"
        return None
    return float(value)


def _check_optional_text(data, field, max_len, errors) -> Optional[str]:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        errors[field] = f"{field} 需為字串。"
        return None
    if len(value) > max_len:
        errors[field] = f"{field} 長度不可超過 {max_len} 字元。"
        return None
    return value


def _check_optional_choice(data, field, choices, errors) -> Optional[str]:
    value = data.get(field)
    if value is None:
        return None
    if value not in choices:
        errors[field] = f"{field} 需為 {choices} 之一，收到「{value}」。"
        return None
    return value
