import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:

    try:
        df = pd.read_csv(path, usecols=usecols)
    except FileNotFoundError:
        logger.error("CSV not found: %s", path)
        raise

    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            if (
                df[col].min() >= np.iinfo(np.int32).min
                and df[col].max() <= np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype("int32")

    logger.debug("Loaded %s  shape=%s", path, df.shape)
    return df


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif str(col_type).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def validate_dataframe(df: pd.DataFrame, name: str, required_cols: List[str]) -> None:

    if df.empty:
        raise ValueError(f"Dataframe '{name}' is empty after loading.")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe '{name}' is missing columns: {missing}")
    logger.info(
        "%s  rows=%d  cols=%d  null_pct=%.1f%%",
        name,
        len(df),
        len(df.columns),
        100 * df.isna().mean().mean(),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def time_weighted_agg(
    df: pd.DataFrame,
    group_col: str,
    value_cols: List[str],
    time_col: str,
    prefix: str,
    decay: float = 0.002,
) -> pd.DataFrame:

    ids = df[group_col].unique()
    result = pd.DataFrame({group_col: ids})

    w = np.exp(decay * df[time_col].values.astype("float64"))

    for vc in value_cols:
        mask = df[vc].notna().values
        if mask.sum() == 0:
            result[f"{prefix}_{vc}_TWMEAN"] = np.nan
            continue

        grp_vals = df[group_col].values[mask]
        wv = df[vc].values[mask].astype("float64") * w[mask]
        ww = w[mask]

        temp = pd.DataFrame({group_col: grp_vals, "_wv": wv, "_w": ww})
        agg = temp.groupby(group_col)[["_wv", "_w"]].sum()
        col_name = f"{prefix}_{vc}_TWMEAN"
        agg[col_name] = (agg["_wv"] / agg["_w"]).astype("float32")

        result = result.merge(agg[[col_name]].reset_index(), on=group_col, how="left")

    return result


def compute_trend(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    time_col: str,
    prefix: str,
) -> pd.DataFrame:

    trend_col = f"{prefix}_{value_col}_TREND"
    empty_result = pd.DataFrame(
        {group_col: df[group_col].unique(), trend_col: np.nan}
    )

    temp = df[[group_col, value_col, time_col]].dropna().copy()
    if len(temp) == 0:
        return empty_result

    gcounts = temp.groupby(group_col)[value_col].transform("count")
    temp = temp[gcounts >= 3].copy()
    if len(temp) == 0:
        return empty_result

    g = temp.groupby(group_col)
    mt = g[time_col].transform("mean").astype("float64")
    mv = g[value_col].transform("mean").astype("float64")
    dt = temp[time_col].astype("float64") - mt
    dv = temp[value_col].astype("float64") - mv

    temp["_dtdv"] = dt * dv
    temp["_dt2"] = dt ** 2

    agg = temp.groupby(group_col)[["_dtdv", "_dt2"]].sum()
    agg[trend_col] = (agg["_dtdv"] / (agg["_dt2"] + 1e-8)).astype("float32")

    return agg[[trend_col]].reset_index()


def agg_time_window(
    df: pd.DataFrame,
    group_col: str,
    cols: List[str],
    time_col: str,
    cutoff: float,
    prefix: str,
) -> pd.DataFrame:

    sub = df[df[time_col] >= cutoff]
    if len(sub) == 0:
        return pd.DataFrame({group_col: df[group_col].unique()})

    valid_cols = [c for c in cols if c in sub.columns]
    if not valid_cols:
        return pd.DataFrame({group_col: df[group_col].unique()})

    agg = sub.groupby(group_col)[valid_cols].agg(["mean", "max", "sum"])
    agg.columns = [f"{prefix}_{c[0]}_{c[1].upper()}" for c in agg.columns]

    cnt = (
        sub.groupby(group_col)
        .size()
        .reset_index(name=f"{prefix}_COUNT")
    )
    return agg.reset_index().merge(cnt, on=group_col, how="left")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def add_frequency_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    full = pd.concat(
        [train_df.drop(columns=["TARGET"], errors="ignore"), test_df],
        axis=0,
        ignore_index=True,
    )

    for col in cols:
        if col not in full.columns:
            continue
        vc = full[col].fillna("__nan__").value_counts(dropna=False)
        full[f"{col}_FREQ"] = full[col].fillna("__nan__").map(vc).astype("float32")
        full[f"{col}_FREQ_NORM"] = (full[f"{col}_FREQ"] / len(full)).astype("float32")

    out_train = full.iloc[: len(train_df)].copy()
    out_test = full.iloc[len(train_df) :].copy()

    if "TARGET" in train_df.columns:
        out_train["TARGET"] = train_df["TARGET"].values

    return out_train, out_test


def add_groupby_ratio_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    full = pd.concat(
        [train_df.drop(columns=["TARGET"], errors="ignore"), test_df],
        axis=0,
        ignore_index=True,
    )

    for cat in cat_cols:
        if cat not in full.columns:
            continue
        for num in num_cols:
            if num not in full.columns:
                continue
            gp = full.groupby(cat)[num].agg(["mean", "median", "std"]).reset_index()
            gp.columns = [
                cat,
                f"GB_{cat}_{num}_MEAN",
                f"GB_{cat}_{num}_MEDIAN",
                f"GB_{cat}_{num}_STD",
            ]
            full = full.merge(gp, on=cat, how="left")
            full[f"GB_{cat}_{num}_DIFF"] = full[num] - full[f"GB_{cat}_{num}_MEAN"]
            full[f"GB_{cat}_{num}_RATIO"] = full[num] / (
                full[f"GB_{cat}_{num}_MEAN"] + 1e-6
            )

    full = reduce_memory_usage(full)
    out_train = full.iloc[: len(train_df)].copy()
    out_test = full.iloc[len(train_df) :].copy()

    if "TARGET" in train_df.columns:
        out_train["TARGET"] = train_df["TARGET"].values

    return out_train, out_test


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    cols: List[str],
    n_splits: int = 5,
    smoothing: int = 40,
    min_samples_leaf: int = 80,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_df = train_df.copy()
    test_df = test_df.copy()

    global_mean = train_df[target_col].mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cols:
        if col not in train_df.columns:
            continue

        new_col = f"{col}_TE"
        if new_col in train_df.columns:
            continue  # already computed in a prior call

        tr_enc = np.zeros(len(train_df), dtype="float32")

        for tr_idx, va_idx in skf.split(train_df, train_df[target_col]):
            tr_fold = train_df.iloc[tr_idx]
            stats = tr_fold.groupby(col)[target_col].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            smooth[stats["count"] < min_samples_leaf] = global_mean
            tr_enc[va_idx] = (
                train_df.iloc[va_idx][col]
                .map(smooth)
                .fillna(global_mean)
                .values.astype("float32")
            )

        # Full-data encoding for test set
        full_stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        full_smooth = (
            full_stats["count"] * full_stats["mean"] + smoothing * global_mean
        ) / (full_stats["count"] + smoothing)
        full_smooth[full_stats["count"] < min_samples_leaf] = global_mean

        train_df[new_col] = tr_enc
        test_df[new_col] = (
            test_df[col].map(full_smooth).fillna(global_mean).values.astype("float32")
        )

    return train_df, test_df
