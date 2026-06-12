import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths  (DVC-tracked)
# ---------------------------------------------------------------------------
DATA_DIR: str = os.path.join(os.path.expanduser("~"), "data", "inputs")
OUTPUT_DIR: str = os.path.join(os.path.expanduser("~"), "data", "outputs")

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")  
MLFLOW_EXPERIMENT_NAME: str = "home-credit-default-risk"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42
N_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Target Encoding
# ---------------------------------------------------------------------------
TE_SMOOTHING: int = 40
TE_MIN_SAMPLES: int = 80

# ---------------------------------------------------------------------------
# Sub-Model (row-level LightGBM used to build meta-features)
# ---------------------------------------------------------------------------
SUB_MODEL_PARAMS: dict = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=100,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
SUB_MODEL_EARLY_STOPPING: int = 50
SUB_MODEL_HIGH_RISK_THRESHOLD: float = 0.15

# ---------------------------------------------------------------------------
# KNN Feature Generation
# ---------------------------------------------------------------------------
KNN_COLS: list = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_ANNUITY_RATIO"]
KNN_K_VALUES: list = [200, 500]

# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------
CORR_THRESHOLD: float = 0.985
NULL_IMP_PERCENTILE: int = 80
NULL_IMP_SCORE_THRESHOLD: float = 1.0
NULL_IMP_N_RUNS: int = 5
NULL_IMP_SAMPLE: int = 60_000
CORR_SAMPLE: int = 30_000

# ---------------------------------------------------------------------------
# Groupby / Frequency feature columns
# ---------------------------------------------------------------------------
GROUPBY_CAT_COLS: list = [
    "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE", "OCCUPATION_TYPE",
    "NAME_INCOME_TYPE", "CODE_GENDER", "AGE_RANGE",
]
GROUPBY_NUM_COLS: list = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "EXT_MEAN", "CREDIT_ANNUITY_RATIO", "ANNUITY_INCOME_RATIO",
    "DAYS_EMPLOYED_YRS",
]
COMBO_CAT_PAIRS: list = [
    ("NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"),
    ("CODE_GENDER", "NAME_FAMILY_STATUS"),
    ("OCCUPATION_TYPE", "ORGANIZATION_TYPE"),
    ("AGE_RANGE", "NAME_EDUCATION_TYPE"),
]
TARGET_ENCODING_COLS: list = [
    "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE", "OCCUPATION_TYPE",
    "NAME_INCOME_TYPE", "CODE_GENDER", "NAME_HOUSING_TYPE",
    "AGE_RANGE", "NAME_EDUCATION_TYPE__NAME_INCOME_TYPE",
    "CODE_GENDER__NAME_FAMILY_STATUS", "OCCUPATION_TYPE__ORGANIZATION_TYPE",
    "AGE_RANGE__NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_CONTRACT_TYPE",
]

# ---------------------------------------------------------------------------
# Model A: LightGBM primary
# ---------------------------------------------------------------------------
LGB_PARAMS_A: dict = dict(
    objective="binary", metric="auc", boosting_type="gbdt",
    n_estimators=10000, learning_rate=0.01, num_leaves=48, max_depth=7,
    subsample=0.8, colsample_bytree=0.25, reg_alpha=0.05, reg_lambda=0.1,
    min_child_samples=40, min_child_weight=30,
    random_state=SEED, n_jobs=-1, verbose=-1,
)
LGB_A_EARLY_STOPPING: int = 200
LGB_A_LOG_EVAL: int = 500

# ---------------------------------------------------------------------------
# Model B: LightGBM top-400 features
# ---------------------------------------------------------------------------
LGB_PARAMS_B: dict = dict(
    objective="binary", metric="auc", boosting_type="gbdt",
    n_estimators=10000, learning_rate=0.008, num_leaves=34, max_depth=5,
    subsample=0.75, colsample_bytree=0.35, reg_alpha=0.05, reg_lambda=0.2,
    min_child_samples=60, min_child_weight=50,
    random_state=123, n_jobs=-1, verbose=-1,
)
LGB_B_SEED: int = 123
LGB_B_TOP_N_FEATURES: int = 400

# ---------------------------------------------------------------------------
# Model C: LightGBM seed averaging
# ---------------------------------------------------------------------------
LGB_PARAMS_C: dict = dict(
    objective="binary", metric="auc", boosting_type="gbdt",
    n_estimators=10000, learning_rate=0.01, num_leaves=40, max_depth=6,
    subsample=0.8, colsample_bytree=0.3, reg_alpha=0.1, reg_lambda=0.15,
    min_child_samples=50, min_child_weight=40,
    n_jobs=-1, verbose=-1,
)
LGB_C_SEEDS: list = [456, 789, 1234]
LGB_C_EARLY_STOPPING: int = 200

# ---------------------------------------------------------------------------
# Model D: XGBoost
# ---------------------------------------------------------------------------
XGB_PARAMS: dict = dict(
    objective="binary:logistic", eval_metric="auc",
    n_estimators=10000, learning_rate=0.01, max_depth=5,
    subsample=0.8, colsample_bytree=0.3, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=40, gamma=0.1,
    random_state=SEED, n_jobs=-1, verbosity=0,
    early_stopping_rounds=200,
)

# ---------------------------------------------------------------------------
# Model E: CatBoost
# ---------------------------------------------------------------------------
CAT_PARAMS_BASE: dict = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=10000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    verbose=500,
    early_stopping_rounds=300,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    grow_policy="SymmetricTree",
    leaf_estimation_iterations=3,
)
CAT_CPU_RSM: float = 0.3  

# ---------------------------------------------------------------------------
# Ensemble / Stacking
# ---------------------------------------------------------------------------
STACK_LR_C: float = 0.35
STACK_LR_MAX_ITER: int = 2000
STACK_LR_SEED: int = 789
STACK_RAW_COLS: list = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CREDIT_ANNUITY_RATIO", "KNN_TARGET_500", "ANNUITY_INCOME_RATIO",
]
BLEND_WEIGHT_MAX: float = 0.60
SUBMISSION_FILENAME: str = "submission.csv"