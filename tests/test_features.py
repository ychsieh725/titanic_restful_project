"""特徵工程管線單元測試（任務 3.1 / 3.2）。

核心驗證 CON-2：訓練與預測共用同一條序列化管線，且單筆與批次前處理一致。
統計量（分組中位數 / 眾數）必須於 fit 學習、transform 套用，
而非在 transform 當下對輸入重算——否則單筆預測會「對自己算中位數」。
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.ml import config
from src.ml.features import (
    TitanicFeatureEngineer,
    build_feature_pipeline,
    passenger_to_frame,
)
from src.ml.types import PassengerInput


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """小型訓練資料，涵蓋缺值、稀有頭銜、不同 Pclass 與 Embarked。"""
    return pd.DataFrame(
        [
            {"Pclass": 1, "Sex": "female", "Age": 38.0, "SibSp": 1, "Parch": 0,
             "Ticket": "PC 17599", "Fare": 71.28, "Cabin": "C85", "Embarked": "C",
             "Name": "Cumings, Mrs. John Bradley"},
            {"Pclass": 3, "Sex": "male", "Age": 22.0, "SibSp": 1, "Parch": 0,
             "Ticket": "A/5 21171", "Fare": 7.25, "Cabin": None, "Embarked": "S",
             "Name": "Braund, Mr. Owen Harris"},
            {"Pclass": 3, "Sex": "female", "Age": 26.0, "SibSp": 0, "Parch": 0,
             "Ticket": "STON/O2", "Fare": 7.92, "Cabin": None, "Embarked": "S",
             "Name": "Heikkinen, Miss. Laina"},
            {"Pclass": 1, "Sex": "male", "Age": 54.0, "SibSp": 0, "Parch": 0,
             "Ticket": "17463", "Fare": 51.86, "Cabin": "E46", "Embarked": "S",
             "Name": "McCarthy, Dr. Timothy J"},
            {"Pclass": 3, "Sex": "male", "Age": None, "SibSp": 3, "Parch": 1,
             "Ticket": "349909", "Fare": 21.07, "Cabin": None, "Embarked": "S",
             "Name": "Palsson, Master. Gosta Leonard"},
            {"Pclass": 2, "Sex": "female", "Age": 14.0, "SibSp": 1, "Parch": 0,
             "Ticket": "237736", "Fare": None, "Cabin": None, "Embarked": None,
             "Name": "Nasser, Mrs. Nicholas"},
        ]
    )


# --- TitanicFeatureEngineer：衍生特徵 -------------------------------------

def test_title_extracted_and_rare_grouped(raw_df: pd.DataFrame) -> None:
    eng = TitanicFeatureEngineer().fit(raw_df)
    out = eng.transform(raw_df)
    titles = out["Title"].tolist()
    assert titles[0] == "Mrs"
    assert titles[1] == "Mr"
    assert titles[2] == "Miss"
    assert titles[3] == config.TITLE_RARE  # Dr -> Rare
    assert titles[4] == "Master"


def test_family_size_and_is_alone(raw_df: pd.DataFrame) -> None:
    out = TitanicFeatureEngineer().fit_transform(raw_df)
    # row1: SibSp1+Parch0+1 = 2, 非獨身
    assert out.loc[0, "FamilySize"] == 2
    assert out.loc[0, "IsAlone"] == 0
    # row2(idx2): SibSp0+Parch0+1 = 1, 獨身
    assert out.loc[2, "FamilySize"] == 1
    assert out.loc[2, "IsAlone"] == 1


# --- 補值（fit 學統計量、transform 套用）---------------------------------

def test_no_nan_after_transform(raw_df: pd.DataFrame) -> None:
    out = TitanicFeatureEngineer().fit_transform(raw_df)
    for col in (*config.NUMERIC_FEATURES, *config.CATEGORICAL_FEATURES):
        assert out[col].isna().sum() == 0, f"{col} 仍有缺值"


def test_age_imputed_uses_fit_statistics_not_row(raw_df: pd.DataFrame) -> None:
    """CON-2：單筆缺 Age 補的是 fit 統計量，而非對該單列重算。"""
    eng = TitanicFeatureEngineer().fit(raw_df)
    single = raw_df.iloc[[4]].copy()  # Master，Age 缺
    out_single = eng.transform(single)
    out_full = eng.transform(raw_df)
    assert out_single.loc[4, "Age"] == pytest.approx(out_full.loc[4, "Age"])
    assert not np.isnan(out_single.loc[4, "Age"])


def test_embarked_mode_and_fare_pclass_median(raw_df: pd.DataFrame) -> None:
    out = TitanicFeatureEngineer().fit_transform(raw_df)
    assert out.loc[5, "Embarked"] == "S"     # 眾數補值
    assert not np.isnan(out.loc[5, "Fare"])  # 同 Pclass 中位數補值


# --- 完整管線：編碼輸出全數值 --------------------------------------------

def test_pipeline_output_all_numeric_no_nan(raw_df: pd.DataFrame) -> None:
    pipe = build_feature_pipeline()
    matrix = pipe.fit_transform(raw_df)
    arr = np.asarray(matrix)
    assert arr.dtype.kind in "fiu"
    assert not np.isnan(arr).any()


# --- CON-2：單筆 == 批次一致性 -------------------------------------------

def test_single_row_matches_batch(raw_df: pd.DataFrame) -> None:
    pipe = build_feature_pipeline().fit(raw_df)
    batch = np.asarray(pipe.transform(raw_df))
    single = np.asarray(pipe.transform(raw_df.iloc[[4]]))
    np.testing.assert_allclose(single[0], batch[4])


# --- CON-2：序列化往返不變 -----------------------------------------------

def test_joblib_roundtrip_preserves_transform(raw_df: pd.DataFrame, tmp_path) -> None:
    pipe = build_feature_pipeline().fit(raw_df)
    before = np.asarray(pipe.transform(raw_df))
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    after = np.asarray(loaded.transform(raw_df))
    np.testing.assert_allclose(before, after)


# --- 預測韌性：未見類別不爆錯 --------------------------------------------

def test_unseen_title_does_not_crash(raw_df: pd.DataFrame) -> None:
    pipe = build_feature_pipeline().fit(raw_df)
    unseen = pd.DataFrame(
        [{"Pclass": 2, "Sex": "male", "Age": 40.0, "SibSp": 0, "Parch": 0,
          "Ticket": "X", "Fare": 13.0, "Cabin": None, "Embarked": "Q",
          "Name": "Reverend, Rev. John"}]
    )
    matrix = np.asarray(pipe.transform(unseen))
    assert not np.isnan(matrix).any()


# --- PassengerInput 轉換 -------------------------------------------------

def test_passenger_to_frame_roundtrips_through_pipeline(raw_df: pd.DataFrame) -> None:
    pipe = build_feature_pipeline().fit(raw_df)
    passenger = PassengerInput(
        Pclass=3, Sex="male", SibSp=0, Parch=0,
        Name="Test, Mr. Example", Ticket="X",
        Age=None, Fare=None, Cabin=None, Embarked=None,
    )
    frame = passenger_to_frame(passenger)
    assert list(frame.columns) == list(config.RAW_FEATURE_COLUMNS)
    assert len(frame) == 1
    matrix = np.asarray(pipe.transform(frame))
    assert not np.isnan(matrix).any()
