"""特徵工程管線（任務 3.1）。

封裝補值、Title 萃取、FamilySize/IsAlone 衍生與類別編碼為單一可序列化
sklearn Pipeline，訓練與預測共用同一物件（FR-2.1~2.4, 2.7 / CON-2）。

關鍵設計（CON-2）：分組中位數與眾數等統計量於 `fit` 學習並存於 estimator
內，`transform` 僅套用既學統計量——而非在 transform 當下對輸入重算。
如此單筆預測與批次訓練前處理一致，且 joblib 序列化後行為不變。

純 Python，不 import 任何 Web 框架物件（CON-4 / NFR-M1）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .types import PassengerInput


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """衍生特徵 + 缺值補值的 stateful 轉換器。

    fit 階段學習以下統計量（供 transform 套用，確保訓練/預測一致）：
    - 各 Title 的 Age 中位數（FR-2.1），全域中位數為 fallback
    - Embarked 眾數（FR-2.2）
    - 各 Pclass 的 Fare 中位數（FR-2.2），全域中位數為 fallback

    transform 輸出含 NUMERIC_FEATURES + CATEGORICAL_FEATURES 欄位的 DataFrame，
    保證無缺值，供下游編碼器使用。
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TitanicFeatureEngineer":
        engineered = self._engineer(X)

        age = engineered["Age"]
        self.age_median_by_title_: dict[str, float] = (
            age.groupby(engineered["Title"]).median().dropna().to_dict()
        )
        self.age_median_global_: float = float(age.median())

        self.embarked_mode_: str = self._safe_mode(engineered["Embarked"], default="S")

        fare = engineered["Fare"]
        self.fare_median_by_pclass_: dict[int, float] = (
            fare.groupby(engineered["Pclass"]).median().dropna().to_dict()
        )
        self.fare_median_global_: float = float(fare.median())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        engineered = self._engineer(X)

        engineered["Age"] = self._impute_by_group(
            engineered["Age"], engineered["Title"],
            self.age_median_by_title_, self.age_median_global_,
        )
        engineered["Fare"] = self._impute_by_group(
            engineered["Fare"], engineered["Pclass"],
            self.fare_median_by_pclass_, self.fare_median_global_,
        )
        engineered["Embarked"] = engineered["Embarked"].fillna(self.embarked_mode_)

        columns = [*config.NUMERIC_FEATURES, *config.CATEGORICAL_FEATURES]
        return engineered[columns]

    # --- 內部：純衍生（不含補值），fit/transform 共用 --------------------

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["Title"] = self._extract_title(df["Name"])
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        return df

    @staticmethod
    def _extract_title(names: pd.Series) -> pd.Series:
        raw = names.str.extract(config.TITLE_REGEX, expand=False).str.strip()
        return raw.map(lambda t: config.TITLE_MAPPING.get(t, config.TITLE_RARE))

    @staticmethod
    def _safe_mode(series: pd.Series, default: str) -> str:
        modes = series.dropna().mode()
        return str(modes.iloc[0]) if not modes.empty else default

    @staticmethod
    def _impute_by_group(
        values: pd.Series,
        groups: pd.Series,
        group_medians: dict,
        global_median: float,
    ) -> pd.Series:
        filled = values.fillna(groups.map(group_medians))
        return filled.fillna(global_median)


def build_feature_pipeline() -> Pipeline:
    """組裝完整前處理管線（不含分類器）。

    結構：engineer（衍生+補值）→ encode（數值縮放 + 類別 OneHot）。
    類別編碼用 handle_unknown="ignore"，使預測端遇到訓練未見的類別不爆錯。

    訓練（4.1）再外包分類器：Pipeline([("features", build_feature_pipeline()),
    ("clf", estimator)])，對齊 config.PARAM_GRIDS 的 "clf__" 前綴。
    """
    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(config.NUMERIC_FEATURES)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(config.CATEGORICAL_FEATURES)),
        ]
    )
    return Pipeline(
        steps=[
            ("engineer", TitanicFeatureEngineer()),
            ("encode", encoder),
        ]
    )


def passenger_to_frame(passenger: PassengerInput) -> pd.DataFrame:
    """將單筆預測輸入轉為符合管線契約的單列 DataFrame（供 FR-5.1 預測）。

    欄位順序對齊 config.RAW_FEATURE_COLUMNS，缺漏欄交由管線補值（CON-2）。
    """
    row = {column: getattr(passenger, column) for column in config.RAW_FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=list(config.RAW_FEATURE_COLUMNS))
