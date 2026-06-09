"""
features.py
-----------
Pipeline order (called from data_prep.py):
  1. application_features        – main table
  2. bureau_and_balance_features – bureau.csv + bureau_balance.csv
  3. previous_application_features
  4. pos_cash_features
  5. credit_card_features
  6. installments_features
"""

import gc
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import (
    agg_time_window,
    compute_trend,
    read_csv,
    reduce_memory_usage,
    time_weighted_agg,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Application table
# ---------------------------------------------------------------------------

def application_features(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    # ------------------------------------------------------------------ anomaly
    out["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)

    def _row_sum_numeric(frame: pd.DataFrame, cols: list, dtype: str = "float32") -> pd.Series:
        cols = [c for c in cols if c in frame.columns]
        if not cols:
            return pd.Series(np.zeros(len(frame), dtype=dtype), index=frame.index)
        return frame[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).astype(dtype)

    # --------------------------------------------------------- EXT_SOURCE block
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    out["EXT_MEAN"] = out[ext].mean(axis=1)
    out["EXT_STD"] = out[ext].std(axis=1)
    out["EXT_PROD"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
    out["EXT_MIN"] = out[ext].min(axis=1)
    out["EXT_MAX"] = out[ext].max(axis=1)
    out["EXT_NANCOUNT"] = out[ext].isna().sum(axis=1)
    out["EXT_S1xS2"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"]
    out["EXT_S1xS3"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_3"]
    out["EXT_S2xS3"] = out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
    out["EXT_S2divS3"] = out["EXT_SOURCE_2"] / (out["EXT_SOURCE_3"] + 1e-4)
    out["EXT_S1divS2"] = out["EXT_SOURCE_1"] / (out["EXT_SOURCE_2"] + 1e-4)
    for i in [1, 2, 3]:
        col = f"EXT_SOURCE_{i}"
        out[f"{col}_SQ"] = out[col] ** 2
        out[f"{col}_CB"] = out[col] ** 3

    out["EXT_S2xBIRTH"] = out["EXT_SOURCE_2"] * out["DAYS_BIRTH"]
    out["EXT_S1xBIRTH"] = out["EXT_SOURCE_1"] * out["DAYS_BIRTH"]
    out["EXT_S3xBIRTH"] = out["EXT_SOURCE_3"] * out["DAYS_BIRTH"]
    out["EXT_S2xEMPL"] = out["EXT_SOURCE_2"] * out["DAYS_EMPLOYED"]
    out["EXT_S3xEMPL"] = out["EXT_SOURCE_3"] * out["DAYS_EMPLOYED"]

    # GP-learned expressions (genetic programming from notebook)
    out["GP1"] = out["EXT_SOURCE_2"] ** 2 * out["EXT_SOURCE_3"]
    out["GP2"] = out["EXT_SOURCE_1"] * out["DAYS_BIRTH"] / (out["AMT_ANNUITY"] + 1)
    out["GP3"] = out["EXT_SOURCE_2"] * out["REGION_RATING_CLIENT_W_CITY"]
    out["GP4"] = out["EXT_SOURCE_3"] * np.log1p(np.abs(out["DAYS_BIRTH"]))
    out["GP5"] = out["AMT_ANNUITY"] * out["EXT_SOURCE_3"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["GP6"] = out["EXT_SOURCE_1"] * out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["GP7"] = out["EXT_SOURCE_2"] * out["AMT_CREDIT"] / (out["AMT_GOODS_PRICE"] + 1)
    out["GP8"] = (out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]) / (out["AMT_CREDIT"] + 1)
    out["GP9"] = out["EXT_MEAN"] * out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] - 1)
    out["GP10"] = (out["AMT_GOODS_PRICE"] - out["AMT_CREDIT"]) * out["EXT_SOURCE_2"] / (out["AMT_ANNUITY"] + 1)

    # -------------------------------------------------------------- financial
    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["CREDIT_ANNUITY_RATIO"] = out["AMT_CREDIT"] / (out["AMT_ANNUITY"] + 1)
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / (out["AMT_GOODS_PRICE"] + 1)
    out["GOODS_INCOME_RATIO"] = out["AMT_GOODS_PRICE"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["INCOME_PER_CHILD"] = out["AMT_INCOME_TOTAL"] / (out["CNT_CHILDREN"] + 1)
    out["INCOME_PER_FAM"] = out["AMT_INCOME_TOTAL"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_CREDIT"] + 1)
    out["PAYMENT_LENGTH"] = out["AMT_CREDIT"] / (out["AMT_ANNUITY"] + 1)
    out["DOWN_PAYMENT"] = out["AMT_GOODS_PRICE"] - out["AMT_CREDIT"]
    out["DOWN_PAYMENT_RATIO"] = out["DOWN_PAYMENT"] / (out["AMT_GOODS_PRICE"] + 1)
    out["INCOME_CREDIT_PERC"] = out["AMT_INCOME_TOTAL"] / (out["AMT_CREDIT"] + 1)
    out["INCOME_ANNUITY_PERC"] = out["AMT_INCOME_TOTAL"] / (out["AMT_ANNUITY"] + 1)
    out["CREDIT_TERM"] = out["AMT_ANNUITY"] / (out["AMT_CREDIT"] + 1)
    out["INCOME_CREDIT_PERC2"] = out["AMT_INCOME_TOTAL"] / (out["AMT_CREDIT"] + 1)
    out["EXT_WEIGHTED"] = 2 * out["EXT_SOURCE_2"] + out["EXT_SOURCE_3"] + 0.5 * out["EXT_SOURCE_1"]

    # --------------------------------------------------------- age/employment
    out["DAYS_BIRTH_YRS"] = out["DAYS_BIRTH"] / -365.25
    out["DAYS_EMPLOYED_YRS"] = out["DAYS_EMPLOYED"] / -365.25
    out["EMPLOYED_TO_BIRTH"] = out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] + 1)
    out["CAR_AGE_TO_BIRTH"] = out["OWN_CAR_AGE"] / (out["DAYS_BIRTH_YRS"] + 1)
    out["ID_PUBLISH_TO_BIRTH"] = out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["PHONE_TO_BIRTH"] = out["DAYS_LAST_PHONE_CHANGE"] / (out["DAYS_BIRTH"] + 1)
    out["PHONE_TO_EMPLOYED"] = out["DAYS_LAST_PHONE_CHANGE"] / (out["DAYS_EMPLOYED"] + 1)
    out["REG_TO_BIRTH"] = out["DAYS_REGISTRATION"] / (out["DAYS_BIRTH"] + 1)
    out["AGE_RANGE"] = pd.cut(
        out["DAYS_BIRTH_YRS"],
        bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],
        labels=False,
    )
    out["INCOME_EMPLOYED"] = out["AMT_INCOME_TOTAL"] * out["DAYS_EMPLOYED_YRS"]
    out["EMPLOYED_TO_ID"] = out["DAYS_EMPLOYED"] / (out["DAYS_ID_PUBLISH"] + 1)
    out["ID_TO_BIRTH_RATIO"] = out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["REG_TO_EMPLOYED_RATIO"] = out["DAYS_REGISTRATION"] / (out["DAYS_EMPLOYED"] + 1)
    out["DAYS_EMPLOYED_PERC"] = out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] + 1)
    out["PHONE_MINUS_REG"] = out["DAYS_LAST_PHONE_CHANGE"] - out["DAYS_REGISTRATION"]
    out["CAR_EMPLOYED_RATIO"] = out["OWN_CAR_AGE"] / (out["DAYS_EMPLOYED_YRS"] + 1)

    # ------------------------------------------------------- family / credit
    out["CREDIT_PER_PERSON"] = out["AMT_CREDIT"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["ANNUITY_PER_PERSON"] = out["AMT_ANNUITY"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["CHILDREN_RATIO"] = out["CNT_CHILDREN"] / (out["CNT_FAM_MEMBERS"] + 1)

    # --------------------------------------------------------- social circle
    out["DEF_30_RATIO"] = out["DEF_30_CNT_SOCIAL_CIRCLE"] / (out["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
    out["DEF_60_RATIO"] = out["DEF_60_CNT_SOCIAL_CIRCLE"] / (out["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
    out["OBS_30_60_RATIO"] = out["OBS_30_CNT_SOCIAL_CIRCLE"] / (out["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
    out["DEF_30_60_RATIO"] = out["DEF_30_CNT_SOCIAL_CIRCLE"] / (out["DEF_60_CNT_SOCIAL_CIRCLE"] + 1)

    # --------------------------------------------------------- misc counters
    doc_cols = [c for c in out.columns if "FLAG_DOCUMENT" in c]
    out["DOCUMENT_COUNT"] = _row_sum_numeric(out, doc_cols)

    amt_req_cols = [c for c in out.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    out["AMT_REQ_SUM"] = _row_sum_numeric(out, amt_req_cols)

    contact_cols = [
        c for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                     "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
        if c in out.columns
    ]
    if contact_cols:
        out["FLAG_CONTACTS_SUM"] = _row_sum_numeric(out, contact_cols)

    out["APP_NULLS"] = out.isna().sum(axis=1).astype("int16")
    out["CITY_RATING_x_EXT2"] = out["REGION_RATING_CLIENT_W_CITY"] * out["EXT_SOURCE_2"]
    out["EXT_RANGE"] = out["EXT_MAX"] - out["EXT_MIN"]
    out["EXT_SOURCE_SPREAD"] = out["EXT_STD"] / (out["EXT_MEAN"] + 1e-4)
    out["REGION_POP_x_EXT"] = out["REGION_POPULATION_RELATIVE"] * out["EXT_MEAN"]
    out["HOUR_APPR_x_EXT2"] = out["HOUR_APPR_PROCESS_START"] * out["EXT_SOURCE_2"]
    out["LIVE_REGION_DIFF"] = (
        out["REG_REGION_NOT_LIVE_REGION"].astype(float)
        + out["REG_REGION_NOT_WORK_REGION"].astype(float)
        + out.get("LIVE_REGION_NOT_WORK_REGION", pd.Series(0, index=out.index)).astype(float)
    )

    logger.info("application_features: output shape %s", out.shape)
    return out


# ---------------------------------------------------------------------------
# 2. Bureau + Bureau Balance
# ---------------------------------------------------------------------------

def bureau_and_balance_features(data_dir: str) -> pd.DataFrame:

    import os
    bureau = read_csv(os.path.join(data_dir, "bureau.csv"))
    bb = read_csv(os.path.join(data_dir, "bureau_balance.csv"))

    # ----------------------------------------- bureau_balance pivot
    bb_counts = bb.pivot_table(
        index="SK_ID_BUREAU", columns="STATUS",
        values="MONTHS_BALANCE", aggfunc="count", fill_value=0,
    )
    bb_counts.columns = [f"BB_STATUS_{c}" for c in bb_counts.columns]
    bb_counts = bb_counts.reset_index()

    bb_months = (
        bb.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"]
        .agg(BB_MONTHS_MIN="min", BB_MONTHS_MAX="max", BB_MONTHS_SIZE="size")
        .reset_index()
    )
    bb_agg = bb_months.merge(bb_counts, on="SK_ID_BUREAU", how="left")
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    del bb, bb_counts, bb_months, bb_agg
    gc.collect()

    # ----------------------------------------- derived ratios
    bureau["CREDIT_DURATION"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    bureau["ENDDATE_DIFF"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_ENDDATE_FACT"]
    bureau["DEBT_CREDIT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["OVERDUE_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM_DEBT"] + 1)
    bureau["AMT_ANNUITY_CREDIT"] = bureau["AMT_ANNUITY"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["CREDIT_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["DAYS_CREDIT_UPDATE_DIFF"] = bureau["DAYS_CREDIT_UPDATE"] - bureau["DAYS_CREDIT"]

    # ----------------------------------------- numeric aggregation
    num_cols = [
        c for c in bureau.columns
        if bureau[c].dtype != "str" and c not in ["SK_ID_BUREAU", "SK_ID_CURR"]
    ]
    buro_num = bureau.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    buro_num.columns = [f"BURO_{c[0]}_{c[1].upper()}" for c in buro_num.columns]
    buro_feat = buro_num.reset_index()

    # ----------------------------------------- categorical (OHE + mean)
    cat_cols = [c for c in bureau.columns if bureau[c].dtype == "str"]
    if cat_cols:
        buro_cat = pd.get_dummies(bureau[["SK_ID_CURR"] + cat_cols], columns=cat_cols, dummy_na=True)
        buro_cat = buro_cat.groupby("SK_ID_CURR").mean().reset_index()
        buro_cat.columns = ["SK_ID_CURR"] + [
            f"BURO_{c}" for c in buro_cat.columns if c != "SK_ID_CURR"
        ]
        buro_feat = buro_feat.merge(buro_cat, on="SK_ID_CURR", how="left")

    buro_feat = buro_feat.merge(
        bureau.groupby("SK_ID_CURR").size().reset_index(name="BURO_COUNT"),
        on="SK_ID_CURR", how="left",
    )

    # ----------------------------------------- active / closed splits
    for status in ["Active", "Closed"]:
        sub = bureau[bureau["CREDIT_ACTIVE"] == status]
        if len(sub) == 0:
            continue
        key = [c for c in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT",
                             "DAYS_CREDIT_ENDDATE", "DEBT_CREDIT_RATIO"] if c in sub.columns]
        sa = sub.groupby("SK_ID_CURR")[key].agg(["mean", "sum", "max", "min"])
        sa.columns = [f"BURO_{status.upper()}_{c[0]}_{c[1].upper()}" for c in sa.columns]
        sc = sub.groupby("SK_ID_CURR").size().reset_index(name=f"BURO_{status.upper()}_COUNT")
        buro_feat = buro_feat.merge(sa.reset_index().merge(sc, on="SK_ID_CURR"), on="SK_ID_CURR", how="left")

    # ----------------------------------------- time-window aggregations
    tw_cols = [c for c in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE", "DEBT_CREDIT_RATIO"]
               if c in bureau.columns]
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y"), (-1095, "3Y"), (-1825, "5Y")]:
        tw = agg_time_window(bureau, "SK_ID_CURR", tw_cols, "DAYS_CREDIT", days, f"BURO_{label}")
        buro_feat = buro_feat.merge(tw, on="SK_ID_CURR", how="left")

    # ----------------------------------------- time-weighted & trend
    tw_feats = time_weighted_agg(
        bureau, "SK_ID_CURR",
        ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO"],
        "DAYS_CREDIT", "BURO", decay=0.001,
    )
    buro_feat = buro_feat.merge(tw_feats, on="SK_ID_CURR", how="left")

    for col in ["AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO"]:
        trend = compute_trend(bureau, "SK_ID_CURR", col, "DAYS_CREDIT", "BURO")
        buro_feat = buro_feat.merge(trend, on="SK_ID_CURR", how="left")

    # ----------------------------------------- last record snapshot
    last_bureau = bureau.sort_values("DAYS_CREDIT", ascending=False).groupby("SK_ID_CURR").first().reset_index()
    for col in ["DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO", "CREDIT_DAY_OVERDUE"]:
        if col in last_bureau.columns:
            buro_feat = buro_feat.merge(
                last_bureau[["SK_ID_CURR", col]].rename(columns={col: f"BURO_LAST_{col}"}),
                on="SK_ID_CURR", how="left",
            )

    buro_feat = buro_feat.merge(
        bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique().reset_index(name="BURO_CREDIT_TYPE_NUNIQUE"),
        on="SK_ID_CURR", how="left",
    )
    buro_feat = buro_feat.merge(
        (bureau.groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"].max() > 0)
        .astype("int8").reset_index(name="BURO_OVERDUE_EVER"),
        on="SK_ID_CURR", how="left",
    )

    del bureau
    gc.collect()
    logger.info("bureau_and_balance_features: output shape %s", buro_feat.shape)
    return buro_feat


# ---------------------------------------------------------------------------
# 3. Previous Applications
# ---------------------------------------------------------------------------

def previous_application_features(data_dir: str) -> pd.DataFrame:

    import os
    prev = read_csv(os.path.join(data_dir, "previous_application.csv"))

    for col in [c for c in prev.columns if "DAYS_" in c]:
        prev[col] = prev[col].replace(365243, np.nan)

    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"] + 1)
    prev["CREDIT_GOODS_P"] = prev["AMT_CREDIT"] / (prev["AMT_GOODS_PRICE"] + 1)
    prev["APP_GOODS_RATIO"] = prev["AMT_APPLICATION"] / (prev["AMT_GOODS_PRICE"] + 1)
    prev["DAYS_FIRST_DUE_DIFF"] = prev["DAYS_FIRST_DUE"] - prev["DAYS_FIRST_DRAWING"]
    prev["DAYS_LAST_DUE_DIFF"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]
    prev["DOWN_PAYMENT_P"] = prev["AMT_DOWN_PAYMENT"] / (prev["AMT_CREDIT"] + 1)
    prev["INTEREST_SHARE"] = prev["CNT_PAYMENT"] * prev["AMT_ANNUITY"] - prev["AMT_CREDIT"]
    prev["INTEREST_RATE"] = prev["INTEREST_SHARE"] / (prev["AMT_CREDIT"] + 1)
    num_cols = [c for c in prev.columns if prev[c].dtype != "str" and c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    prev_num = prev.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    prev_num.columns = [f"PREV_{c[0]}_{c[1].upper()}" for c in prev_num.columns]
    prev_feat = prev_num.reset_index()

    cat_cols = [c for c in prev.columns if prev[c].dtype == "str"]
    if cat_cols:
        prev_cat = pd.get_dummies(prev[["SK_ID_CURR"] + cat_cols], columns=cat_cols, dummy_na=True)
        prev_cat = prev_cat.groupby("SK_ID_CURR").mean().reset_index()
        prev_cat.columns = ["SK_ID_CURR"] + [f"PREV_{c}" for c in prev_cat.columns if c != "SK_ID_CURR"]
        prev_feat = prev_feat.merge(prev_cat, on="SK_ID_CURR", how="left")

    prev_feat = prev_feat.merge(
        prev.groupby("SK_ID_CURR").size().reset_index(name="PREV_COUNT"),
        on="SK_ID_CURR", how="left",
    )

    for status in ["Approved", "Refused", "Canceled"]:
        sub = prev[prev["NAME_CONTRACT_STATUS"] == status]
        if len(sub) == 0:
            continue
        sa = sub.groupby("SK_ID_CURR")[["AMT_CREDIT", "AMT_APPLICATION", "AMT_ANNUITY", "DAYS_DECISION"]].agg(["mean", "max", "min"])
        sa.columns = [f"PREV_{status.upper()}_{c[0]}_{c[1].upper()}" for c in sa.columns]
        sc = sub.groupby("SK_ID_CURR").size().reset_index(name=f"PREV_{status.upper()}_COUNT")
        prev_feat = prev_feat.merge(sa.reset_index().merge(sc, on="SK_ID_CURR"), on="SK_ID_CURR", how="left")

    for ctype, label in [("Cash loans", "CASH"), ("Revolving loans", "REVOLV")]:
        sub = prev[prev["NAME_CONTRACT_TYPE"] == ctype]
        if len(sub) == 0:
            continue
        sa = sub.groupby("SK_ID_CURR")[["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO"]].agg(["mean", "sum", "max"])
        sa.columns = [f"PREV_{label}_{c[0]}_{c[1].upper()}" for c in sa.columns]
        sc = sub.groupby("SK_ID_CURR").size().reset_index(name=f"PREV_{label}_COUNT")
        prev_feat = prev_feat.merge(sa.reset_index().merge(sc, on="SK_ID_CURR"), on="SK_ID_CURR", how="left")

    tw_cols = [c for c in ["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO", "INTEREST_RATE"] if c in prev.columns]
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y"), (-1095, "3Y")]:
        tw = agg_time_window(prev, "SK_ID_CURR", tw_cols, "DAYS_DECISION", days, f"PREV_{label}")
        prev_feat = prev_feat.merge(tw, on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(prev, "SK_ID_CURR", ["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO"],
                                  "DAYS_DECISION", "PREV", decay=0.001)
    prev_feat = prev_feat.merge(tw_feats, on="SK_ID_CURR", how="left")

    app_rate = (
        prev.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"]
        .apply(lambda x: (x == "Approved").mean())
        .reset_index(name="PREV_APPROVAL_RATE")
    )
    prev_feat = prev_feat.merge(app_rate, on="SK_ID_CURR", how="left")

    last_prev = prev.sort_values("DAYS_DECISION", ascending=False).groupby("SK_ID_CURR").first().reset_index()
    for col in ["DAYS_DECISION", "AMT_CREDIT", "APP_CREDIT_RATIO", "INTEREST_RATE"]:
        if col in last_prev.columns:
            prev_feat = prev_feat.merge(
                last_prev[["SK_ID_CURR", col]].rename(columns={col: f"PREV_LAST_{col}"}),
                on="SK_ID_CURR", how="left",
            )

    del prev
    gc.collect()
    logger.info("previous_application_features: output shape %s", prev_feat.shape)
    return prev_feat


# ---------------------------------------------------------------------------
# 4. POS Cash Balance
# ---------------------------------------------------------------------------

def pos_cash_features(data_dir: str) -> pd.DataFrame:

    import os
    pos = read_csv(os.path.join(data_dir, "POS_CASH_balance.csv"))

    pos["SK_DPD_RATIO"] = pos["SK_DPD"] / (pos["SK_DPD_DEF"] + 1)
    pos["LATE_POS"] = (pos["SK_DPD"] > 0).astype("int8")

    num_cols = [c for c in pos.columns if pos[c].dtype != "str" and c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    pn = pos.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    pn.columns = [f"POS_{c[0]}_{c[1].upper()}" for c in pn.columns]
    pn = pn.reset_index()

    if "NAME_CONTRACT_STATUS" in pos.columns:
        pc = pd.get_dummies(pos[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]], columns=["NAME_CONTRACT_STATUS"], dummy_na=True)
        pc = pc.groupby("SK_ID_CURR").mean().reset_index()
        pc.columns = ["SK_ID_CURR"] + [f"POS_{c}" for c in pc.columns if c != "SK_ID_CURR"]
        pn = pn.merge(pc, on="SK_ID_CURR", how="left")

    pn = pn.merge(pos.groupby("SK_ID_CURR").size().reset_index(name="POS_COUNT"), on="SK_ID_CURR", how="left")
    pn = pn.merge(pos.groupby("SK_ID_CURR")["LATE_POS"].mean().reset_index(name="POS_LATE_RATE"), on="SK_ID_CURR", how="left")

    tw_cols_pos = ["SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"]
    for months, label in [(-3, "3M"), (-6, "6M"), (-12, "12M"), (-24, "24M")]:
        tw = agg_time_window(pos, "SK_ID_CURR", tw_cols_pos, "MONTHS_BALANCE", months, f"POS_{label}")
        pn = pn.merge(tw, on="SK_ID_CURR", how="left")

    loan_agg = pos.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        POS_PL_DPD_MAX=("SK_DPD", "max"),
        POS_PL_DPD_MEAN=("SK_DPD", "mean"),
        POS_PL_LATE_RATE=("LATE_POS", "mean"),
        POS_PL_MONTHS=("MONTHS_BALANCE", "count"),
    ).reset_index()
    pl_cols = [c for c in loan_agg.columns if c.startswith("POS_PL_")]
    pos_pl = loan_agg.groupby("SK_ID_CURR")[pl_cols].agg(["mean", "max", "std"])
    pos_pl.columns = [f"{c[0]}_{c[1].upper()}" for c in pos_pl.columns]
    pn = pn.merge(pos_pl.reset_index(), on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(pos, "SK_ID_CURR", ["SK_DPD", "CNT_INSTALMENT_FUTURE"], "MONTHS_BALANCE", "POS", decay=0.02)
    pn = pn.merge(tw_feats, on="SK_ID_CURR", how="left")

    trend = compute_trend(pos, "SK_ID_CURR", "SK_DPD", "MONTHS_BALANCE", "POS")
    pn = pn.merge(trend, on="SK_ID_CURR", how="left")

    if "NAME_CONTRACT_STATUS" in pos.columns:
        completed = pos[pos["NAME_CONTRACT_STATUS"] == "Completed"]
        comp_rate = (completed.groupby("SK_ID_CURR").size() / pos.groupby("SK_ID_CURR").size()).reset_index(name="POS_COMPLETED_RATE")
        pn = pn.merge(comp_rate, on="SK_ID_CURR", how="left")

    del pos, loan_agg
    gc.collect()
    logger.info("pos_cash_features: output shape %s", pn.shape)
    return pn


# ---------------------------------------------------------------------------
# 5. Credit Card Balance
# ---------------------------------------------------------------------------

def credit_card_features(data_dir: str) -> pd.DataFrame:

    import os
    cc = read_csv(os.path.join(data_dir, "credit_card_balance.csv"))

    cc["CC_BAL_LIM_RATIO"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_PAY_TOTAL_RATIO"] = cc["AMT_PAYMENT_TOTAL_CURRENT"] / (cc["AMT_TOTAL_RECEIVABLE"] + 1)
    cc["CC_DRAW_LIM"] = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_LATE"] = (cc["SK_DPD"] > 0).astype("int8")
    cc["CC_MIN_PAY_RATIO"] = cc["AMT_INST_MIN_REGULARITY"] / (cc["AMT_PAYMENT_CURRENT"] + 1)

    num_cols = [c for c in cc.columns if cc[c].dtype != "str" and c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    cn = cc.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    cn.columns = [f"CC_{c[0]}_{c[1].upper()}" for c in cn.columns]
    cn = cn.reset_index()
    cn = cn.merge(cc.groupby("SK_ID_CURR").size().reset_index(name="CC_COUNT"), on="SK_ID_CURR", how="left")
    cn = cn.merge(cc.groupby("SK_ID_CURR")["CC_LATE"].mean().reset_index(name="CC_LATE_RATE"), on="SK_ID_CURR", how="left")

    tw_cols_cc = ["AMT_BALANCE", "CC_BAL_LIM_RATIO", "CC_DRAW_LIM", "SK_DPD"]
    for months, label in [(-3, "3M"), (-6, "6M"), (-12, "12M"), (-24, "24M")]:
        tw = agg_time_window(cc, "SK_ID_CURR", tw_cols_cc, "MONTHS_BALANCE", months, f"CC_{label}")
        cn = cn.merge(tw, on="SK_ID_CURR", how="left")

    loan_agg = cc.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        CC_PL_BAL_LIM_MAX=("CC_BAL_LIM_RATIO", "max"),
        CC_PL_BAL_LIM_MEAN=("CC_BAL_LIM_RATIO", "mean"),
        CC_PL_DRAW_MEAN=("CC_DRAW_LIM", "mean"),
        CC_PL_DPD_MAX=("SK_DPD", "max"),
        CC_PL_LATE_RATE=("CC_LATE", "mean"),
    ).reset_index()
    pl_cols = [c for c in loan_agg.columns if c.startswith("CC_PL_")]
    cc_pl = loan_agg.groupby("SK_ID_CURR")[pl_cols].agg(["mean", "max", "std"])
    cc_pl.columns = [f"{c[0]}_{c[1].upper()}" for c in cc_pl.columns]
    cn = cn.merge(cc_pl.reset_index(), on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(cc, "SK_ID_CURR", ["AMT_BALANCE", "CC_BAL_LIM_RATIO", "SK_DPD"], "MONTHS_BALANCE", "CC", decay=0.02)
    cn = cn.merge(tw_feats, on="SK_ID_CURR", how="left")

    for col in ["AMT_BALANCE", "CC_BAL_LIM_RATIO"]:
        trend = compute_trend(cc, "SK_ID_CURR", col, "MONTHS_BALANCE", "CC")
        cn = cn.merge(trend, on="SK_ID_CURR", how="left")

    del cc, loan_agg
    gc.collect()
    logger.info("credit_card_features: output shape %s", cn.shape)
    return cn


# ---------------------------------------------------------------------------
# 6. Installments Payments
# ---------------------------------------------------------------------------

def installments_features(data_dir: str) -> pd.DataFrame:

    import os
    ins = read_csv(os.path.join(data_dir, "installments_payments.csv"))

    ins["PAYMENT_PERC"] = (ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 0.001)).replace([np.inf, -np.inf], np.nan).astype("float32")
    ins["PAYMENT_DIFF"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).astype("float32")
    ins["DPD"] = np.maximum(ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"], 0).astype("float32")
    ins["DBD"] = np.maximum(ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"], 0).astype("float32")
    ins["LATE_PAYMENT"] = (ins["DPD"] > 0).astype("int8")
    ins["SIGNIFICANT_UNDERPAY"] = (ins["PAYMENT_DIFF"] > 100).astype("int8")

    num_cols = [c for c in ins.columns if ins[c].dtype != "str" and c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    iN = ins.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    iN.columns = [f"INS_{c[0]}_{c[1].upper()}" for c in iN.columns]
    iN = iN.reset_index()
    iN = iN.merge(ins.groupby("SK_ID_CURR").size().reset_index(name="INS_COUNT"), on="SK_ID_CURR", how="left")
    iN = iN.merge(ins.groupby("SK_ID_CURR")["LATE_PAYMENT"].mean().reset_index(name="INS_LATE_RATE"), on="SK_ID_CURR", how="left")
    iN = iN.merge(ins.groupby("SK_ID_CURR")["SIGNIFICANT_UNDERPAY"].mean().reset_index(name="INS_SIGUNDERPAY_RATE"), on="SK_ID_CURR", how="left")

    tw_cols_ins = ["DPD", "PAYMENT_PERC", "PAYMENT_DIFF", "LATE_PAYMENT"]
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y")]:
        tw = agg_time_window(ins, "SK_ID_CURR", tw_cols_ins, "DAYS_INSTALMENT", days, f"INS_{label}")
        iN = iN.merge(tw, on="SK_ID_CURR", how="left")

    loan_agg = ins.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        INS_PL_DPD_MEAN=("DPD", "mean"),
        INS_PL_DPD_MAX=("DPD", "max"),
        INS_PL_LATE_SUM=("LATE_PAYMENT", "sum"),
        INS_PL_LATE_RATE=("LATE_PAYMENT", "mean"),
        INS_PL_PAYPERC_MEAN=("PAYMENT_PERC", "mean"),
        INS_PL_PAYPERC_MIN=("PAYMENT_PERC", "min"),
        INS_PL_PAYDIFF_MAX=("PAYMENT_DIFF", "max"),
        INS_PL_COUNT=("DPD", "size"),
    ).reset_index()
    pl_cols = [c for c in loan_agg.columns if c.startswith("INS_PL_")]
    ins_pl = loan_agg.groupby("SK_ID_CURR")[pl_cols].agg(["mean", "max", "std"])
    ins_pl.columns = [f"{c[0]}_{c[1].upper()}" for c in ins_pl.columns]
    iN = iN.merge(ins_pl.reset_index(), on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(ins, "SK_ID_CURR", ["DPD", "PAYMENT_PERC", "PAYMENT_DIFF"], "DAYS_INSTALMENT", "INS", decay=0.001)
    iN = iN.merge(tw_feats, on="SK_ID_CURR", how="left")

    for col in ["DPD", "PAYMENT_PERC"]:
        trend = compute_trend(ins, "SK_ID_CURR", col, "DAYS_INSTALMENT", "INS")
        iN = iN.merge(trend, on="SK_ID_CURR", how="left")

    # Last-k installments
    ins_sorted = ins.sort_values("DAYS_INSTALMENT", ascending=False)
    for k in [3, 5, 10, 30]:
        last_k = ins_sorted.groupby("SK_ID_CURR").head(k)
        lk_agg = last_k.groupby("SK_ID_CURR").agg(
            **{
                f"INS_LAST{k}_DPD_MEAN": ("DPD", "mean"),
                f"INS_LAST{k}_DPD_MAX": ("DPD", "max"),
                f"INS_LAST{k}_PAYPERC_MEAN": ("PAYMENT_PERC", "mean"),
                f"INS_LAST{k}_PAYDIFF_MEAN": ("PAYMENT_DIFF", "mean"),
                f"INS_LAST{k}_LATE_RATE": ("LATE_PAYMENT", "mean"),
            }
        ).reset_index()
        iN = iN.merge(lk_agg, on="SK_ID_CURR", how="left")

    if "NUM_INSTALMENT_VERSION" in ins.columns:
        ver_agg = ins.groupby("SK_ID_CURR")["NUM_INSTALMENT_VERSION"].agg(
            INS_VERSION_NUNIQUE="nunique", INS_VERSION_MAX="max", INS_VERSION_MEAN="mean"
        ).reset_index()
        iN = iN.merge(ver_agg, on="SK_ID_CURR", how="left")

    del ins, loan_agg, ins_sorted
    gc.collect()
    logger.info("installments_features: output shape %s", iN.shape)
    return iN
