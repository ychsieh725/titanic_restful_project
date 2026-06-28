"""超參數驗證與格點建構單元測試（FR-3.3 防呆）。

驗證 build_param_grid 將使用者覆寫安全地轉為 GridSearchCV 格點：型別/範圍/允許值
檢查、未知鍵拒絕、空值回退預設、固定參數（solver）保留，以及組合數上限防呆。
"""

from __future__ import annotations

import pytest

from src.ml import config
from src.ml.hyperparams import HyperparameterError, build_param_grid


# --- 預設與回退 -----------------------------------------------------------

def test_no_overrides_returns_default_grid() -> None:
    grid = build_param_grid(config.LOGISTIC_REGRESSION, None)
    # 與預設等價（含 clf__ 前綴與固定 solver）
    assert grid["clf__C"] == config.PARAM_GRIDS[config.LOGISTIC_REGRESSION]["clf__C"]
    assert grid["clf__solver"] == ["liblinear"]


def test_partial_override_keeps_defaults_for_others() -> None:
    grid = build_param_grid(config.LOGISTIC_REGRESSION, {"C": [0.5, 2]})
    assert grid["clf__C"] == [0.5, 2.0]
    # penalty 未覆寫 → 沿用預設
    assert grid["clf__penalty"] == config.PARAM_GRIDS[config.LOGISTIC_REGRESSION]["clf__penalty"]
    assert grid["clf__solver"] == ["liblinear"]


def test_empty_value_falls_back_to_default() -> None:
    grid = build_param_grid(config.LOGISTIC_REGRESSION, {"C": []})
    assert grid["clf__C"] == config.PARAM_GRIDS[config.LOGISTIC_REGRESSION]["clf__C"]


# --- 型別 / 範圍 / 允許值防呆 --------------------------------------------

def test_negative_C_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.LOGISTIC_REGRESSION, {"C": [-1]})
    assert "C" in exc.value.errors


def test_non_numeric_C_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.LOGISTIC_REGRESSION, {"C": ["abc"]})
    assert "C" in exc.value.errors


def test_invalid_penalty_choice_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.LOGISTIC_REGRESSION, {"penalty": ["l3"]})
    assert "penalty" in exc.value.errors


def test_n_estimators_out_of_range_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.RANDOM_FOREST, {"n_estimators": [99999]})
    assert "n_estimators" in exc.value.errors


def test_non_integer_n_estimators_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.RANDOM_FOREST, {"n_estimators": [1.5]})
    assert "n_estimators" in exc.value.errors


def test_unknown_hyperparameter_key_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.LOGISTIC_REGRESSION, {"bogus": [1]})
    assert "bogus" in exc.value.errors


def test_collects_multiple_errors() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.RANDOM_FOREST, {"n_estimators": [-1], "min_samples_leaf": [0]})
    assert {"n_estimators", "min_samples_leaf"} <= set(exc.value.errors)


# --- max_depth 允許 None --------------------------------------------------

def test_max_depth_accepts_none() -> None:
    grid = build_param_grid(config.RANDOM_FOREST, {"max_depth": ["none", 10]})
    assert None in grid["clf__max_depth"]
    assert 10 in grid["clf__max_depth"]


# --- 組合數上限防呆 -------------------------------------------------------

def test_grid_combination_cap_rejected() -> None:
    # 5 × 5 × 5 × 5 = 625 > 200 上限
    big = {
        "n_estimators": [1, 2, 3, 4, 5],
        "max_depth": [1, 2, 3, 4, 5],
        "min_samples_split": [2, 3, 4, 5, 6],
        "min_samples_leaf": [1, 2, 3, 4, 5],
    }
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.RANDOM_FOREST, big)
    assert any("組合" in msg for msg in exc.value.errors.values())


def test_too_many_values_for_single_param_rejected() -> None:
    too_many = list(range(1, config.MAX_VALUES_PER_PARAM + 2))
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.RANDOM_FOREST, {"n_estimators": too_many})
    assert "n_estimators" in exc.value.errors


# --- 型別轉換邊界 ---------------------------------------------------------

def test_float_string_value_accepted() -> None:
    grid = build_param_grid(config.LOGISTIC_REGRESSION, {"C": ["0.5", "2"]})
    assert grid["clf__C"] == [0.5, 2.0]


def test_single_non_list_value_accepted() -> None:
    grid = build_param_grid(config.LOGISTIC_REGRESSION, {"penalty": "l1"})
    assert grid["clf__penalty"] == ["l1"]


def test_max_depth_out_of_range_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.RANDOM_FOREST, {"max_depth": [9999]})
    assert "max_depth" in exc.value.errors


def test_boolean_value_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.RANDOM_FOREST, {"n_estimators": [True]})
    assert "n_estimators" in exc.value.errors


def test_non_numeric_float_rejected() -> None:
    with pytest.raises(HyperparameterError) as exc:
        build_param_grid(config.LOGISTIC_REGRESSION, {"C": ["wat"]})
    assert "C" in exc.value.errors


# --- 未知演算法 -----------------------------------------------------------

def test_unknown_algorithm_raises_value_error() -> None:
    with pytest.raises(ValueError):
        build_param_grid("svm", {"C": [1]})
