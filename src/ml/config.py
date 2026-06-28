"""服務層集中設定（NFR-M2）。

超參數格點、特徵清單、CV 設定與路徑集中於此，便於調整且避免散落硬編碼。
數值來源：SRS §10.1 建議超參數格點、§6.1 資料字典。
"""

from __future__ import annotations

from pathlib import Path

# --- 演算法 ---------------------------------------------------------------

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"

SUPPORTED_ALGORITHMS: tuple[str, ...] = (LOGISTIC_REGRESSION, RANDOM_FOREST)

# --- 超參數格點（SRS §10.1）-----------------------------------------------
# 鍵以 "clf__" 前綴對應 sklearn Pipeline 中分類器步驟名稱 "clf"，
# 供 GridSearchCV 直接使用。

PARAM_GRIDS: dict[str, dict[str, list]] = {
    LOGISTIC_REGRESSION: {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    },
    RANDOM_FOREST: {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    },
}

# --- 訓練設定（FR-3.3 / FR-3.6 / FR-3.10）---------------------------------

CV_FOLDS = 5          # 交叉驗證折數，需 >= 5
TEST_SIZE = 0.2       # held-out 測試集比例
RANDOM_STATE = 42     # 可重現性

# GridSearchCV 平行度（NFR-P1）。-1 用滿核心；4.2 在 threading job 內可調為 1
# 以避免 loky 巢狀平行的資源競爭。集中於此便於調整（NFR-M2）。
N_JOBS = -1

# LogisticRegression 最大迭代數，設足夠大以避免縮放後仍不收斂警告（FR-3.2）。
LR_MAX_ITER = 1000

# --- 資料欄位（SRS §6.1）--------------------------------------------------

TARGET_COLUMN = "Survived"

# 從 titanic 表載入、進入特徵管線前的原始輸入欄位
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
    "Name",
)

# --- 特徵工程設定（SRS §3.2 FR-2 / NFR-M2 集中設定）----------------------

# 自 Name 萃取 Title 的正則：抓 "," 之後、"." 之前的頭銜詞（FR-2.3）。
# 例："Braund, Mr. Owen Harris" -> "Mr"
TITLE_REGEX = r",\s*([^\.]+)\."

# 稀有頭銜歸併對應（FR-2.3）。表內為標準四類；未列出者一律歸 "Rare"。
# 法/英文同義頭銜（Mlle/Ms→Miss、Mme→Mrs）顯式對應，其餘職銜歸 Rare。
TITLE_MAPPING: dict[str, str] = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}
TITLE_RARE = "Rare"

# 經 TitanicFeatureEngineer.transform 後，進入編碼器的特徵欄位分組。
# 數值欄走 StandardScaler，類別欄走 OneHotEncoder(handle_unknown="ignore")。
NUMERIC_FEATURES: tuple[str, ...] = (
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Sex",
    "Embarked",
    "Title",
)

# --- 輸入驗證規則（FR-5.5 / 對齊 init_db CHECK 約束）----------------------

# 單筆預測必填欄位；其餘可空，由共用管線補值。
REQUIRED_PASSENGER_FIELDS: tuple[str, ...] = (
    "Pclass",
    "Sex",
    "SibSp",
    "Parch",
    "Name",
    "Ticket",
)

VALID_PCLASS: tuple[int, ...] = (1, 2, 3)
VALID_SEX: tuple[str, ...] = ("male", "female")
VALID_EMBARKED: tuple[str, ...] = ("C", "Q", "S")

AGE_MIN, AGE_MAX = 0.0, 120.0
NAME_MAX_LEN = 100
TICKET_MAX_LEN = 30
CABIN_MAX_LEN = 30

# CSV 批次預測單次上限（FR-5.3）；超過即拒收，避免單請求過大拖垮服務。
MAX_BATCH_ROWS = 10000

# --- 路徑 -----------------------------------------------------------------

# 專案根目錄（src/ml/config.py → 上溯三層）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# joblib 模型檔儲存目錄（CON-3 / FR-4.1）；已列入 .gitignore
MODELS_DIR = PROJECT_ROOT / "models"

# SQLite 資料庫（與 app.py 一致）
DATABASE_PATH = PROJECT_ROOT / "my_db.db"


def get_param_grid(algorithm: str) -> dict[str, list]:
    """取得指定演算法的超參數格點。

    Args:
        algorithm: 演算法名稱，需為 SUPPORTED_ALGORITHMS 之一。

    Returns:
        對應的超參數格點。

    Raises:
        ValueError: 演算法不在支援清單時，明確指出可用選項。
    """
    if algorithm not in PARAM_GRIDS:
        supported = ", ".join(SUPPORTED_ALGORITHMS)
        raise ValueError(
            f"不支援的演算法 '{algorithm}'，可用選項：{supported}"
        )
    return PARAM_GRIDS[algorithm]
