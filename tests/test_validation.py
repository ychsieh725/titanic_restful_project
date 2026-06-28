"""單筆輸入驗證單元測試（任務 6.3 / FR-5.5, NFR-S1/S2）。

驗證 validate_passenger 對缺漏、型別錯誤、超出範圍給明確訊息，
並一次蒐集所有錯誤（非 fail-fast），且合法輸入回傳 PassengerInput。
"""

from __future__ import annotations

import pytest

from src.ml.validation import (
    BatchValidationError,
    ValidationError,
    validate_passenger,
    validate_passengers,
)
from src.ml.types import PassengerInput


def _valid() -> dict:
    return {
        "Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0,
        "Name": "Test, Mrs. Example", "Ticket": "X",
        "Age": 29.0, "Fare": 80.0, "Cabin": "C20", "Embarked": "C",
    }


# --- 合法 -----------------------------------------------------------------

def test_valid_returns_passenger_input() -> None:
    result = validate_passenger(_valid())
    assert isinstance(result, PassengerInput)
    assert result.Pclass == 1
    assert result.Sex == "female"


def test_optional_fields_may_be_none() -> None:
    data = _valid()
    for field in ("Age", "Fare", "Cabin", "Embarked"):
        data[field] = None
    result = validate_passenger(data)
    assert result.Age is None and result.Embarked is None


# --- 缺漏：一次蒐集所有錯誤 ----------------------------------------------

def test_missing_required_fields_collects_all() -> None:
    from src.ml import config

    data = {  # 完全空 → 應一次回報所有必填欄缺漏
        "Age": None, "Fare": None, "Cabin": None, "Embarked": None,
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_passenger(data)
    errors = exc_info.value.errors
    for field in config.REQUIRED_PASSENGER_FIELDS:
        assert field in errors


# --- 範圍 / 允許值 --------------------------------------------------------

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("Pclass", 5),
        ("Sex", "unknown"),
        ("Age", -1),
        ("Age", 999),
        ("SibSp", -1),
        ("Fare", -5),
        ("Embarked", "Z"),
    ],
)
def test_invalid_values_rejected(field: str, bad_value) -> None:
    data = _valid()
    data[field] = bad_value
    with pytest.raises(ValidationError) as exc_info:
        validate_passenger(data)
    assert field in exc_info.value.errors


# --- 型別錯誤 -------------------------------------------------------------

def test_non_numeric_age_rejected() -> None:
    data = _valid()
    data["Age"] = "old"
    with pytest.raises(ValidationError) as exc_info:
        validate_passenger(data)
    assert "Age" in exc_info.value.errors


def test_empty_name_rejected() -> None:
    data = _valid()
    data["Name"] = "   "
    with pytest.raises(ValidationError) as exc_info:
        validate_passenger(data)
    assert "Name" in exc_info.value.errors


# --- 型別 / 長度邊界（NFR-S1 惡意輸入路徑）-------------------------------

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("Pclass", "1"),                 # 非整數型別
        ("SibSp", "2"),                  # 非整數型別
        ("Name", "x" * 101),             # 超長
        ("Ticket", "T" * 31),            # 超長
        ("Fare", "free"),                # 非數值
        ("Cabin", 123),                  # 非字串
        ("Cabin", "C" * 31),             # 超長
    ],
)
def test_type_and_length_violations_rejected(field: str, bad_value) -> None:
    data = _valid()
    data[field] = bad_value
    with pytest.raises(ValidationError) as exc_info:
        validate_passenger(data)
    assert field in exc_info.value.errors


# --- 批次驗證（6.2 / FR-5.3）---------------------------------------------

def test_validate_passengers_all_valid_returns_list() -> None:
    records = [_valid(), _valid(), _valid()]
    result = validate_passengers(records)
    assert len(result) == 3
    assert all(isinstance(p, PassengerInput) for p in result)


def test_validate_passengers_empty_returns_empty_list() -> None:
    assert validate_passengers([]) == []


def test_validate_passengers_collects_errors_by_row_index() -> None:
    bad = _valid()
    bad["Pclass"] = 9          # row 1 非法
    worse = {k: v for k, v in _valid().items() if k != "Sex"}  # row 2 缺 Sex

    with pytest.raises(BatchValidationError) as exc_info:
        validate_passengers([_valid(), bad, worse])

    row_errors = exc_info.value.row_errors
    assert set(row_errors.keys()) == {1, 2}
    assert "Pclass" in row_errors[1]
    assert "Sex" in row_errors[2]
