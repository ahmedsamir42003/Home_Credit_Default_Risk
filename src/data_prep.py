import gc
import json
import logging
import os
import re
from typing import NamedTuple, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

from src import config
from src.features import (
    application_features,
    bureau_and_balance_features,
    credit_card_features,
    installments_features,
    pos_cash_features,
    previous_application_features,
)
from src.utils import (
    add_frequency_features,
    add_groupby_ratio_features,
    add_target_encoding,
    read_csv,
    reduce_memory_usage,
    validate_dataframe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public output type
# ---------------------------------------------------------------------------

class PreparedData(NamedTuple):

    train: pd.DataFrame          # LGB/XGB numeric features (no TARGET, no SK_ID_CURR)
    test: pd.DataFrame           # LGB/XGB numeric features
    cat_train: pd.DataFrame      # CatBoost version (string cats preserved, no TARGET)
    cat_test: pd.DataFrame       # CatBoost version
    cat_feature_names: List[str] # Categorical column names for CatBoost Pool
    cat_feature_indices: List[int]
    target: pd.Series            # Binary labels aligned with train rows
    cat_y: pd.Series             # Binary labels aligned with cat_train rows
    train_ids: pd.Series         # SK_ID_CURR for train
    test_ids: pd.Series          # SK_ID_CURR for test
    feat_all: List[str]          # All feature names after selection
    feat_top400: List[str]       # Top-400 by LGB importance
    feat_no_gp: List[str]        # All features excluding GP* columns


# ---------------------------------------------------------------------------
# Sub-model meta-features (row-level LGB → aggregate predictions per client)
# ---------------------------------------------------------------------------

def _sub_model_features(
    table_path: str,
    target_df: pd.DataFrame,
    prefix: str,
    params: dict,
    early_stopping: int,
    high_risk_threshold: float,
    seed: int,
) -> pd.DataFrame:

    df = read_csv(table_path)
    for col in df.columns:
        if "DAYS_" in col:
            df[col] = df[col].replace(365243, np.nan)

    feat_cols = [
        c for c in df.columns
        if df[c].dtype != "object" and c not in ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]
    ]

    train_ids_set = set(target_df["SK_ID_CURR"].values)
    mask_train = df["SK_ID_CURR"].isin(train_ids_set)
    df_tr = df[mask_train].copy()
    df_te = df[~mask_train].copy()

    df_tr = df_tr.merge(target_df[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR", how="inner")

    X_tr = df_tr[feat_cols].replace([np.inf, -np.inf], np.nan)
    y_tr = df_tr["TARGET"].astype(int)
    groups = df_tr["SK_ID_CURR"]

    oof_preds = np.zeros(len(df_tr))
    fitted_models = []

    gkf = GroupKFold(n_splits=5)
    for tr_idx, va_idx in gkf.split(X_tr, y_tr, groups=groups):
        m = lgb.LGBMClassifier(**params, random_state=seed)
        m.fit(
            X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
            eval_set=[(X_tr.iloc[va_idx], y_tr.iloc[va_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        oof_preds[va_idx] = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
        fitted_models.append(m)

    auc = roc_auc_score(y_tr, oof_preds)
    logger.info("%s sub-model OOF AUC: %.4f", prefix, auc)

    df_tr["_SUB_PRED"] = oof_preds

    X_te = df_te[feat_cols].replace([np.inf, -np.inf], np.nan)
    test_preds = np.mean([m.predict_proba(X_te)[:, 1] for m in fitted_models], axis=0)
    df_te["_SUB_PRED"] = test_preds

    all_rows = pd.concat([df_tr[["SK_ID_CURR", "_SUB_PRED"]], df_te[["SK_ID_CURR", "_SUB_PRED"]]])
    sub_agg = all_rows.groupby("SK_ID_CURR")["_SUB_PRED"].agg(
        **{
            f"{prefix}_SUB_MEAN": "mean",
            f"{prefix}_SUB_MAX": "max",
            f"{prefix}_SUB_MIN": "min",
            f"{prefix}_SUB_STD": "std",
        }
    ).reset_index()
    high_risk = (
        all_rows[all_rows["_SUB_PRED"] > high_risk_threshold]
        .groupby("SK_ID_CURR")
        .size()
        .reset_index(name=f"{prefix}_SUB_HIGHRISK")
    )
    sub_agg = sub_agg.merge(high_risk, on="SK_ID_CURR", how="left")
    sub_agg[f"{prefix}_SUB_HIGHRISK"] = sub_agg[f"{prefix}_SUB_HIGHRISK"].fillna(0)

    del df, df_tr, df_te, all_rows
    gc.collect()
    return sub_agg


# ---------------------------------------------------------------------------
# KNN target features
# ---------------------------------------------------------------------------

def _knn_target_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    knn_cols: List[str],
    k_values: List[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    valid_cols = [c for c in knn_cols if c in train.columns and c in test.columns]
    X_all = pd.concat([train[valid_cols], test[valid_cols]], axis=0).fillna(-999).values
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    X_tr, X_te = X_all[: len(train)], X_all[len(train) :]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for k in k_values:
        oof_k = np.zeros(len(train))
        test_k = np.zeros(len(test))

        for tr_idx, va_idx in skf.split(X_tr, target):
            knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
            knn.fit(X_tr[tr_idx], target.values[tr_idx])
            oof_k[va_idx] = knn.predict_proba(X_tr[va_idx])[:, 1]
            test_k += knn.predict_proba(X_te)[:, 1] / 5

        train[f"KNN_TARGET_{k}"] = oof_k.astype("float32")
        test[f"KNN_TARGET_{k}"] = test_k.astype("float32")
        logger.info("KNN k=%d OOF AUC: %.6f", k, roc_auc_score(target, oof_k))

    del X_all, X_tr, X_te
    gc.collect()
    return train, test


# ---------------------------------------------------------------------------
# Feature importance helper (used for null-importance selection)
# ---------------------------------------------------------------------------

def _get_importances(X: pd.DataFrame, y: pd.Series, shuffle: bool = False, seed: int = 0) -> np.ndarray:

    if shuffle:
        y = y.sample(frac=1, random_state=seed).reset_index(drop=True)

    fold_imp = np.zeros(X.shape[1])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    for ti, vi in skf.split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=40, max_depth=5,
            subsample=0.8, colsample_bytree=0.3, min_child_samples=50,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        m.fit(
            X.iloc[ti], y.iloc[ti],
            eval_set=[(X.iloc[vi], y.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        fold_imp += m.feature_importances_

    return fold_imp / 3


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def prepare_data(cfg: object = config) -> PreparedData:

    data_dir = cfg.DATA_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ load
    logger.info("Loading application tables …")
    try:
        app_train_raw = read_csv(os.path.join(data_dir, "application_train.csv"))
        app_test_raw = read_csv(os.path.join(data_dir, "application_test.csv"))
    except FileNotFoundError as exc:
        logger.error("Could not load application CSVs from %s", data_dir)
        raise

    validate_dataframe(app_train_raw, "application_train", ["SK_ID_CURR", "TARGET"])
    validate_dataframe(app_test_raw, "application_test", ["SK_ID_CURR"])

    app_train = application_features(app_train_raw)
    app_test = application_features(app_test_raw)
    del app_train_raw, app_test_raw
    gc.collect()

    # -------------------------------------------------- auxiliary table feats
    logger.info("Building bureau features …")
    buro_feat = bureau_and_balance_features(data_dir)

    logger.info("Building previous application features …")
    prev_feat = previous_application_features(data_dir)

    logger.info("Building POS cash features …")
    pos_feat = pos_cash_features(data_dir)

    logger.info("Building credit card features …")
    cc_feat = credit_card_features(data_dir)

    logger.info("Building installments features …")
    ins_feat = installments_features(data_dir)

    # ---------------------------------------------------------- sub-models
    target_df = app_train[["SK_ID_CURR", "TARGET"]].copy()

    logger.info("Training sub-model on previous_application rows …")
    prev_sub = _sub_model_features(
        os.path.join(data_dir, "previous_application.csv"), target_df, "PREV",
        cfg.SUB_MODEL_PARAMS, cfg.SUB_MODEL_EARLY_STOPPING,
        cfg.SUB_MODEL_HIGH_RISK_THRESHOLD, cfg.SEED,
    )

    logger.info("Training sub-model on bureau rows …")
    buro_sub = _sub_model_features(
        os.path.join(data_dir, "bureau.csv"), target_df, "BURO",
        cfg.SUB_MODEL_PARAMS, cfg.SUB_MODEL_EARLY_STOPPING,
        cfg.SUB_MODEL_HIGH_RISK_THRESHOLD, cfg.SEED,
    )

    logger.info("Training sub-model on installments rows …")
    ins_sub = _sub_model_features(
        os.path.join(data_dir, "installments_payments.csv"), target_df, "INS",
        cfg.SUB_MODEL_PARAMS, cfg.SUB_MODEL_EARLY_STOPPING,
        cfg.SUB_MODEL_HIGH_RISK_THRESHOLD, cfg.SEED,
    )

    # --------------------------------------------------------------- merge
    logger.info("Merging all feature tables …")
    train = app_train.copy()
    test = app_test.copy()

    for feat_df in [buro_feat, prev_feat, pos_feat, cc_feat, ins_feat, prev_sub, buro_sub, ins_sub]:
        train = train.merge(feat_df, on="SK_ID_CURR", how="left")
        test = test.merge(feat_df, on="SK_ID_CURR", how="left")

    logger.info("After merge: train %s  test %s", train.shape, test.shape)
    del app_train, app_test, buro_feat, prev_feat, pos_feat, cc_feat, ins_feat
    del prev_sub, buro_sub, ins_sub
    gc.collect()

    # --------------------------------------- frequency / groupby / TE feats
    logger.info("Adding frequency / groupby / target-encoding features …")
    cat_cols_for_freq = [c for c in train.columns if train[c].dtype == "str"]
    train, test = add_frequency_features(train, test, cat_cols_for_freq)

    for c1, c2 in cfg.COMBO_CAT_PAIRS:
        if c1 in train.columns and c2 in train.columns:
            new_col = f"{c1}__{c2}"
            train[new_col] = train[c1].astype(str) + "__" + train[c2].astype(str)
            test[new_col] = test[c1].astype(str) + "__" + test[c2].astype(str)

    gp_cat = [c for c in cfg.GROUPBY_CAT_COLS if c in train.columns]
    gp_num = [c for c in cfg.GROUPBY_NUM_COLS if c in train.columns]
    train, test = add_groupby_ratio_features(train, test, gp_cat, gp_num)

    te_cols = [c for c in cfg.TARGET_ENCODING_COLS if c in train.columns]
    train, test = add_target_encoding(
        train, test, "TARGET", te_cols,
        n_splits=cfg.N_FOLDS, smoothing=cfg.TE_SMOOTHING,
        min_samples_leaf=cfg.TE_MIN_SAMPLES, seed=cfg.SEED,
    )

    train = reduce_memory_usage(train)
    test = reduce_memory_usage(test)
    logger.info("After encoding block: train %s  test %s", train.shape, test.shape)

    # --------------------------------------------------------------- KNN
    target = train["TARGET"].astype("int8")
    train_ids = train["SK_ID_CURR"].copy()
    test_ids = test["SK_ID_CURR"].copy()

    logger.info("Computing KNN target features …")
    train, test = _knn_target_features(train, test, target, cfg.KNN_COLS, cfg.KNN_K_VALUES, cfg.SEED)
    logger.info("After KNN: train %s  test %s", train.shape, test.shape)

    # ---------------------------------------- prepare CatBoost version first
    cat_train = train.drop(columns=["SK_ID_CURR"]).copy()
    cat_test = test.drop(columns=["SK_ID_CURR"]).copy()
    cat_feature_names = [c for c in cat_train.columns if c != "TARGET" and cat_train[c].dtype == "str"]

    for col in cat_feature_names:
        cat_train[col] = cat_train[col].fillna("__nan__").astype(str)
        cat_test[col] = cat_test[col].fillna("__nan__").astype(str)

    cat_train = cat_train.replace([np.inf, -np.inf], np.nan)
    cat_test = cat_test.replace([np.inf, -np.inf], np.nan)

    cat_train_feats = cat_train.drop(columns=["TARGET"]).copy()
    cat_train_feats, cat_test = cat_train_feats.align(cat_test, join="inner", axis=1)
    cat_train = pd.concat([cat_train[["TARGET"]].reset_index(drop=True), cat_train_feats.reset_index(drop=True)], axis=1)
    cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]

    for col in cat_train.columns:
        if col not in cat_feature_names and col != "TARGET":
            cat_train[col] = pd.to_numeric(cat_train[col], errors="coerce")
            cat_test[col] = pd.to_numeric(cat_test[col], errors="coerce")

    cat_clean_names = {c: re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in cat_train.columns}
    cat_train = cat_train.rename(columns=cat_clean_names)
    cat_test = cat_test.rename(columns=cat_clean_names)
    cat_feature_names = [cat_clean_names.get(c, c) for c in cat_feature_names]

    # ---------------------------------------- prepare LGB/XGB version
    train = train.drop(columns=["TARGET", "SK_ID_CURR"])
    test = test.drop(columns=["SK_ID_CURR"])

    obj_cols = [c for c in train.columns if train[c].dtype == "str"]
    for col in obj_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train[col], test[col]], axis=0).astype(str).fillna("nan")
        le.fit(all_vals)
        train[col] = le.transform(train[col].astype(str).fillna("nan")).astype("int32")
        test[col] = le.transform(test[col].astype(str).fillna("nan")).astype("int32")

    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    train, test = train.align(test, join="inner", axis=1)

    clean_names = {c: re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in train.columns}
    train = train.rename(columns=clean_names)
    test = test.rename(columns=clean_names)

    # ---------------------------------------- drop constant / high-corr cols
    logger.info("Dropping constant and highly-correlated features …")
    drop_cols = [c for c in train.columns if train[c].isna().all() or train[c].nunique(dropna=False) <= 1]
    if drop_cols:
        train = train.drop(columns=drop_cols)
        test = test.drop(columns=drop_cols)
        cat_drop = [c for c in drop_cols if c in cat_train.columns]
        if cat_drop:
            cat_train = cat_train.drop(columns=cat_drop)
            cat_test = cat_test.drop(columns=cat_drop)

    sample = train.sample(min(cfg.CORR_SAMPLE, len(train)), random_state=42)
    corr = sample.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > cfg.CORR_THRESHOLD)]
    if high_corr:
        train = train.drop(columns=high_corr)
        test = test.drop(columns=high_corr)
        cat_drop = [c for c in high_corr if c in cat_train.columns]
        if cat_drop:
            cat_train = cat_train.drop(columns=cat_drop)
            cat_test = cat_test.drop(columns=cat_drop)
        cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]
    logger.info("After correlation filter: train %s  test %s", train.shape, test.shape)

    # ---------------------------------------- null-importance selection
    logger.info("Running null-importance feature selection …")
    feat_names = list(train.columns)
    ni_idx = train.sample(min(cfg.NULL_IMP_SAMPLE, len(train)), random_state=42).index
    X_ni = train.loc[ni_idx].reset_index(drop=True)
    y_ni = target.loc[ni_idx].reset_index(drop=True)

    actual_imp = _get_importances(X_ni, y_ni, shuffle=False, seed=42)
    null_imps = np.column_stack([
        _get_importances(X_ni, y_ni, shuffle=True, seed=100 + i)
        for i in range(cfg.NULL_IMP_N_RUNS)
    ])
    null_80 = np.percentile(null_imps, cfg.NULL_IMP_PERCENTILE, axis=1)
    score_vs_null = actual_imp / (null_80 + 1)
    drop_null = [feat_names[j] for j in range(len(feat_names)) if score_vs_null[j] < cfg.NULL_IMP_SCORE_THRESHOLD]

    if drop_null:
        train = train.drop(columns=drop_null)
        test = test.drop(columns=drop_null)
        feat_names = list(train.columns)
    logger.info("Null-importance dropped %d features. Remaining: %d", len(drop_null), len(feat_names))
    del X_ni, y_ni
    gc.collect()

    # ---------------------------------------- feature subsets for diversity
    logger.info("Computing feature importance subsets …")
    m_imp = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=48, max_depth=6,
        subsample=0.8, colsample_bytree=0.3, min_child_samples=50,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    imp_arr = np.zeros(len(feat_names))
    for ti, vi in StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(train, target):
        m_imp.fit(
            train.iloc[ti], target.iloc[ti],
            eval_set=[(train.iloc[vi], target.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        imp_arr += m_imp.feature_importances_
    imp_arr /= 3

    imp_rank = pd.DataFrame({"feature": feat_names, "imp": imp_arr}).sort_values("imp", ascending=False)
    feat_all = feat_names
    feat_top400 = imp_rank.head(min(cfg.LGB_B_TOP_N_FEATURES, len(feat_names)))["feature"].tolist()
    feat_no_gp = [f for f in feat_names if not f.startswith("GP")]
    del m_imp
    gc.collect()

    # ---------------------------------------- finalise CatBoost targets
    cat_y = cat_train["TARGET"].astype("int8").copy()
    cat_train = cat_train.drop(columns=["TARGET"])
    cat_train, cat_test = cat_train.align(cat_test, join="inner", axis=1)
    cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]
    cat_feature_indices = [cat_train.columns.get_loc(c) for c in cat_feature_names]

    logger.info(
        "Data preparation complete.  train=%s  test=%s  cat_train=%s  cat_test=%s",
        train.shape, test.shape, cat_train.shape, cat_test.shape,
    )

    result = PreparedData(
        train=train, test=test,
        cat_train=cat_train, cat_test=cat_test,
        cat_feature_names=cat_feature_names, cat_feature_indices=cat_feature_indices,
        target=target, cat_y=cat_y,
        train_ids=train_ids, test_ids=test_ids,
        feat_all=feat_all, feat_top400=feat_top400, feat_no_gp=feat_no_gp,
    )

    save_prepared_data(result, cfg)
    return result


# ---------------------------------------------------------------------------
# Persistence helpers — save / load PreparedData to data/outputs/prepared/
# ---------------------------------------------------------------------------

PREPARED_DIR_KEY = "prepared"  


def _prep_dir(cfg: object) -> str:

    path = os.path.join(cfg.OUTPUT_DIR, PREPARED_DIR_KEY)
    os.makedirs(path, exist_ok=True)
    return path


def save_prepared_data(data: PreparedData, cfg: object) -> None:

    d = _prep_dir(cfg)
    logger.info("Saving prepared data to %s …", d)

    try:
        data.train.to_parquet(os.path.join(d, "train.parquet"),         index=False)
        data.test.to_parquet(os.path.join(d, "test.parquet"),           index=False)
        data.cat_train.to_parquet(os.path.join(d, "cat_train.parquet"), index=False)
        data.cat_test.to_parquet(os.path.join(d, "cat_test.parquet"),   index=False)

        data.target.to_frame("TARGET").to_parquet(os.path.join(d, "target.parquet"),   index=False)
        data.cat_y.to_frame("TARGET").to_parquet(os.path.join(d,  "cat_y.parquet"),    index=False)
        data.train_ids.to_frame("SK_ID_CURR").to_parquet(os.path.join(d, "train_ids.parquet"), index=False)
        data.test_ids.to_frame("SK_ID_CURR").to_parquet(os.path.join(d,  "test_ids.parquet"),  index=False)

        meta = {
            "feat_all":            data.feat_all,
            "feat_top400":         data.feat_top400,
            "feat_no_gp":          data.feat_no_gp,
            "cat_feature_names":   data.cat_feature_names,
            "cat_feature_indices": data.cat_feature_indices,
        }
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Prepared data saved successfully.")
    except OSError as exc:
        logger.error("Failed to save prepared data: %s", exc)
        raise

""" called by train """
def load_prepared_data(cfg: object) -> PreparedData:

    d = _prep_dir(cfg)
    logger.info("Loading prepared data from %s …", d)

    expected = ["train.parquet", "test.parquet", "cat_train.parquet", "cat_test.parquet",
                "target.parquet", "cat_y.parquet", "train_ids.parquet", "test_ids.parquet", "meta.json"]
    missing = [f for f in expected if not os.path.exists(os.path.join(d, f))]
    if missing:
        raise FileNotFoundError(
            f"Prepared data incomplete — missing files: {missing}\n"
            f"Run the prepare stage first:  python main.py --stage prepare"
        )

    try:
        train     = pd.read_parquet(os.path.join(d, "train.parquet"))
        test      = pd.read_parquet(os.path.join(d, "test.parquet"))
        cat_train = pd.read_parquet(os.path.join(d, "cat_train.parquet"))
        cat_test  = pd.read_parquet(os.path.join(d, "cat_test.parquet"))
        target    = pd.read_parquet(os.path.join(d, "target.parquet"))["TARGET"].astype("int8")
        cat_y     = pd.read_parquet(os.path.join(d, "cat_y.parquet"))["TARGET"].astype("int8")
        train_ids = pd.read_parquet(os.path.join(d, "train_ids.parquet"))["SK_ID_CURR"]
        test_ids  = pd.read_parquet(os.path.join(d, "test_ids.parquet"))["SK_ID_CURR"]

        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
    except Exception as exc:
        logger.error("Failed to load prepared data: %s", exc)
        raise

    logger.info("Prepared data loaded.  train=%s  test=%s", train.shape, test.shape)

    return PreparedData(
        train=train, test=test,
        cat_train=cat_train, cat_test=cat_test,
        cat_feature_names=meta["cat_feature_names"],
        cat_feature_indices=meta["cat_feature_indices"],
        target=target, cat_y=cat_y,
        train_ids=train_ids, test_ids=test_ids,
        feat_all=meta["feat_all"],
        feat_top400=meta["feat_top400"],
        feat_no_gp=meta["feat_no_gp"],
    )