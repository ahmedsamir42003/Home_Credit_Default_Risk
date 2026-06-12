import json
import logging
import os
from typing import Dict, List, Tuple

import joblib
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

from src import config
from src.data_prep import PreparedData

logger = logging.getLogger(__name__)


def _detect_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _models_dir(cfg: object) -> str:
    """Return path to the saved-models directory, creating it if needed."""
    path = os.path.join(cfg.OUTPUT_DIR, "models")
    os.makedirs(path, exist_ok=True)
    return path


def _save_model(name: str, fold: int, model: object, cfg: object) -> str:

    d = _models_dir(cfg)
    path = os.path.join(d, f"{name}_fold{fold}.pkl")
    joblib.dump(model, path)
    logger.info("Model saved: %s fold %d → %s", name, fold, path)
    return path


def load_model(name: str, fold: int, cfg: object) -> object:

    path = os.path.join(_models_dir(cfg), f"{name}_fold{fold}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Run training first.")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Checkpoint helpers — TWO levels:
#   1. Fold-level  (_save/_load_fold_ckpt)  — saved after every single fold
#   2. Model-level (_save/_load_ckpt)       — saved after all folds of a model
# On restart: model-level check first (skip whole model), then fold-level
# (skip already-done folds within a model).
# ---------------------------------------------------------------------------

def _ckpt_dir(cfg: object) -> str:
    path = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path


def _save_fold_ckpt(name: str, fold: int, oof: np.ndarray, test_acc: np.ndarray, cfg: object) -> None:
    d = _ckpt_dir(cfg)
    np.save(os.path.join(d, f"{name}_fold{fold}_oof.npy"),     oof)
    np.save(os.path.join(d, f"{name}_fold{fold}_testacc.npy"), test_acc)
    logger.info("Fold checkpoint saved: %s fold %d", name, fold)


def _load_fold_ckpt(name: str, fold: int, cfg: object):
    d = _ckpt_dir(cfg)
    op = os.path.join(d, f"{name}_fold{fold}_oof.npy")
    tp = os.path.join(d, f"{name}_fold{fold}_testacc.npy")
    if os.path.exists(op) and os.path.exists(tp):
        logger.info("Fold checkpoint found — skipping: %s fold %d", name, fold)
        return np.load(op), np.load(tp)
    return None


def _save_ckpt(name: str, oof: np.ndarray, test: np.ndarray, cfg: object) -> None:
    d = _ckpt_dir(cfg)
    np.save(os.path.join(d, f"{name}_oof.npy"),  oof)
    np.save(os.path.join(d, f"{name}_test.npy"), test)
    logger.info("Model checkpoint saved: %s", name)


def _load_ckpt(name: str, cfg: object):
    d = _ckpt_dir(cfg)
    op = os.path.join(d, f"{name}_oof.npy")
    tp = os.path.join(d, f"{name}_test.npy")
    if os.path.exists(op) and os.path.exists(tp):
        logger.info("Model checkpoint found — skipping all folds for: %s", name)
        return np.load(op), np.load(tp)
    return None


def _clear_ckpts(cfg: object) -> None:
    import shutil
    shutil.rmtree(_ckpt_dir(cfg), ignore_errors=True)
    logger.info("Checkpoints cleared.")


# ---------------------------------------------------------------------------
# Model trainers — all with fold-level checkpointing
# ---------------------------------------------------------------------------

def train_lgb_a(data: PreparedData, cfg: object) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(data.train))
    test_preds = np.zeros(len(data.test))
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)

    for fold, (ti, vi) in enumerate(skf.split(data.train, data.target)):
        ckpt = _load_fold_ckpt("lgb_a", fold, cfg)
        if ckpt:
            oof, test_preds = ckpt
            continue
        logger.info("LGB-A fold %d/%d", fold + 1, cfg.N_FOLDS)
        m = lgb.LGBMClassifier(**cfg.LGB_PARAMS_A)
        m.fit(
            data.train.iloc[ti], data.target.iloc[ti],
            eval_set=[(data.train.iloc[vi], data.target.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(cfg.LGB_A_EARLY_STOPPING, verbose=True),
                       lgb.log_evaluation(cfg.LGB_A_LOG_EVAL)],
        )
        oof[vi]     = m.predict_proba(data.train.iloc[vi])[:, 1]
        test_preds += m.predict_proba(data.test)[:, 1] / cfg.N_FOLDS
        _save_model("lgb_a", fold, m, cfg)
        _save_fold_ckpt("lgb_a", fold, oof, test_preds, cfg)

    logger.info("LGB-A OOF AUC: %.6f", roc_auc_score(data.target, oof))
    return oof, test_preds


def train_lgb_b(data: PreparedData, cfg: object) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(data.train))
    test_preds = np.zeros(len(data.test))
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.LGB_B_SEED)
    train_b = data.train[[f for f in data.feat_top400 if f in data.train.columns]]
    test_b  = data.test[[f  for f in data.feat_top400 if f in data.test.columns]]

    for fold, (ti, vi) in enumerate(skf.split(train_b, data.target)):
        ckpt = _load_fold_ckpt("lgb_b", fold, cfg)
        if ckpt:
            oof, test_preds = ckpt
            continue
        logger.info("LGB-B fold %d/%d", fold + 1, cfg.N_FOLDS)
        m = lgb.LGBMClassifier(**cfg.LGB_PARAMS_B)
        m.fit(
            train_b.iloc[ti], data.target.iloc[ti],
            eval_set=[(train_b.iloc[vi], data.target.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(cfg.LGB_A_EARLY_STOPPING, verbose=True),
                       lgb.log_evaluation(cfg.LGB_A_LOG_EVAL)],
        )
        oof[vi]     = m.predict_proba(train_b.iloc[vi])[:, 1]
        test_preds += m.predict_proba(test_b)[:, 1] / cfg.N_FOLDS
        _save_model("lgb_b", fold, m, cfg)
        _save_fold_ckpt("lgb_b", fold, oof, test_preds, cfg)

    logger.info("LGB-B OOF AUC: %.6f", roc_auc_score(data.target, oof))
    return oof, test_preds


def train_lgb_c(data: PreparedData, cfg: object) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(data.train))
    test_preds = np.zeros(len(data.test))
    n_seeds = len(cfg.LGB_C_SEEDS)
    train_c = data.train[[f for f in data.feat_no_gp if f in data.train.columns]]
    test_c  = data.test[[f  for f in data.feat_no_gp if f in data.test.columns]]

    for seed in cfg.LGB_C_SEEDS:
        skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=seed)
        oof_seed  = np.zeros(len(data.train))
        test_seed = np.zeros(len(data.test))
        ckpt_key  = f"lgb_c_seed{seed}"

        for fold, (ti, vi) in enumerate(skf.split(train_c, data.target)):
            ckpt = _load_fold_ckpt(ckpt_key, fold, cfg)
            if ckpt:
                oof_seed, test_seed = ckpt
                continue
            params = {**cfg.LGB_PARAMS_C, "random_state": seed}
            m = lgb.LGBMClassifier(**params)
            m.fit(
                train_c.iloc[ti], data.target.iloc[ti],
                eval_set=[(train_c.iloc[vi], data.target.iloc[vi])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(cfg.LGB_C_EARLY_STOPPING, verbose=False)],
            )
            oof_seed[vi]  = m.predict_proba(train_c.iloc[vi])[:, 1]
            test_seed    += m.predict_proba(test_c)[:, 1] / cfg.N_FOLDS
            _save_model(ckpt_key, fold, m, cfg)
            _save_fold_ckpt(ckpt_key, fold, oof_seed, test_seed, cfg)

        oof        += oof_seed / n_seeds
        test_preds += test_seed / n_seeds
        logger.info("LGB-C seed %d OOF AUC: %.6f", seed, roc_auc_score(data.target, oof_seed))

    logger.info("LGB-C (avg) OOF AUC: %.6f", roc_auc_score(data.target, oof))
    return oof, test_preds


def train_xgb(data: PreparedData, cfg: object) -> Tuple[np.ndarray, np.ndarray]:
    has_gpu = _detect_gpu()
    xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    if xgb_version >= (2, 0):
        extra = {"device": "cuda", "tree_method": "hist"} if has_gpu else {"tree_method": "hist"}
    else:
        extra = {"tree_method": "gpu_hist"} if has_gpu else {"tree_method": "hist"}

    oof = np.zeros(len(data.train))
    test_preds = np.zeros(len(data.test))
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)

    for fold, (ti, vi) in enumerate(skf.split(data.train, data.target)):
        ckpt = _load_fold_ckpt("xgb", fold, cfg)
        if ckpt:
            oof, test_preds = ckpt
            continue
        logger.info("XGB fold %d/%d", fold + 1, cfg.N_FOLDS)
        m = xgb.XGBClassifier(**cfg.XGB_PARAMS, **extra)
        m.fit(
            data.train.iloc[ti], data.target.iloc[ti],
            eval_set=[(data.train.iloc[vi], data.target.iloc[vi])],
            verbose=500,
        )
        oof[vi]     = m.predict_proba(data.train.iloc[vi])[:, 1]
        test_preds += m.predict_proba(data.test)[:, 1] / cfg.N_FOLDS
        _save_model("xgb", fold, m, cfg)
        _save_fold_ckpt("xgb", fold, oof, test_preds, cfg)

    logger.info("XGB OOF AUC: %.6f", roc_auc_score(data.target, oof))
    return oof, test_preds


def train_catboost(data: PreparedData, cfg: object) -> Tuple[np.ndarray, np.ndarray]:
    has_gpu  = _detect_gpu()
    cat_task = "GPU" if has_gpu else "CPU"
    oof = np.zeros(len(data.cat_train))
    test_preds = np.zeros(len(data.cat_test))
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=42)

    for fold, (ti, vi) in enumerate(skf.split(data.cat_train, data.cat_y)):
        ckpt = _load_fold_ckpt("cat", fold, cfg)
        if ckpt:
            oof, test_preds = ckpt
            continue
        logger.info("CatBoost fold %d/%d", fold + 1, cfg.N_FOLDS)
        cat_params = {**cfg.CAT_PARAMS_BASE, "task_type": cat_task, "random_seed": 42 + fold}
        if cat_task == "CPU":
            cat_params["rsm"] = cfg.CAT_CPU_RSM
        train_pool = Pool(data.cat_train.iloc[ti], label=data.cat_y.iloc[ti], cat_features=data.cat_feature_indices)
        valid_pool = Pool(data.cat_train.iloc[vi], label=data.cat_y.iloc[vi], cat_features=data.cat_feature_indices)
        test_pool  = Pool(data.cat_test, cat_features=data.cat_feature_indices)
        m = CatBoostClassifier(**cat_params)
        m.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        oof[vi]     = m.predict_proba(valid_pool)[:, 1]
        test_preds += m.predict_proba(test_pool)[:, 1] / cfg.N_FOLDS
        _save_model("cat", fold, m, cfg)
        _save_fold_ckpt("cat", fold, oof, test_preds, cfg)

    logger.info("CatBoost OOF AUC: %.6f", roc_auc_score(data.cat_y, oof))
    return oof, test_preds


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def _rank_norm(a: np.ndarray) -> np.ndarray:
    return rankdata(a) / len(a)


def build_ensemble(models, data, cfg):
    target = data.target
    names  = list(models.keys())

    logger.info("=== Individual Model OOF AUCs ===")
    for name, (oof, _) in models.items():
        logger.info("  %s: %.6f", name, roc_auc_score(target, oof))

    raw_stack_cols = [c for c in cfg.STACK_RAW_COLS if c in data.train.columns]
    oof_stack  = np.column_stack([models[n][0] for n in names])
    test_stack = np.column_stack([models[n][1] for n in names])
    if raw_stack_cols:
        oof_stack  = np.column_stack([oof_stack,  data.train[raw_stack_cols].fillna(-999).values.astype("float32")])
        test_stack = np.column_stack([test_stack, data.test[raw_stack_cols].fillna(-999).values.astype("float32")])

    oof_stack_lr  = np.zeros(len(data.train))
    test_stack_lr = np.zeros(len(data.test))
    skf_stack = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.STACK_LR_SEED)
    for fold, (ti, vi) in enumerate(skf_stack.split(oof_stack, target)):
        lr = LogisticRegression(C=cfg.STACK_LR_C, max_iter=cfg.STACK_LR_MAX_ITER,
                                solver="lbfgs", random_state=cfg.STACK_LR_SEED)
        lr.fit(oof_stack[ti], target.values[ti])
        oof_stack_lr[vi]  = lr.predict_proba(oof_stack[vi])[:, 1]
        test_stack_lr    += lr.predict_proba(test_stack)[:, 1] / cfg.N_FOLDS

    stack_lr_auc = roc_auc_score(target, oof_stack_lr)
    logger.info("Logistic stacked OOF AUC: %.6f", stack_lr_auc)

    blend_oof   = [_rank_norm(models[n][0]) for n in names] + [_rank_norm(oof_stack_lr)]
    blend_test  = [_rank_norm(models[n][1]) for n in names] + [_rank_norm(test_stack_lr)]
    blend_names = names + ["stack_lr"]
    n_models    = len(blend_names)
    w0     = np.ones(n_models) / n_models
    bounds = [(0.0, cfg.BLEND_WEIGHT_MAX)] * n_models

    def neg_auc(w):
        w = np.clip(w, 0, 1) / (np.clip(w, 0, 1).sum() + 1e-12)
        return -roc_auc_score(target, sum(wi * ri for wi, ri in zip(w, blend_oof)))

    best_result = None
    for method in ["SLSQP", "Powell"]:
        try:
            cons = ({"type": "eq", "fun": lambda w: np.sum(np.clip(w, 0, 1)) - 1.0},) if method == "SLSQP" else ()
            r = minimize(neg_auc, w0, method=method, bounds=bounds, constraints=cons,
                         options={"maxiter": 3000, "ftol": 1e-10})
            if best_result is None or r.fun < best_result.fun:
                best_result = r
        except Exception as exc:
            logger.warning("%s optimisation failed: %s", method, exc)

    best_w  = np.clip(best_result.x, 0, 1)
    best_w /= best_w.sum() + 1e-12
    blend_auc = -best_result.fun
    logger.info("Optimised weights:")
    for name, w in zip(blend_names, best_w):
        logger.info("  %-15s: %.4f", name, w)
    logger.info("Optimised blend OOF AUC: %.6f", blend_auc)

    test_pred_blend = sum(w * r for w, r in zip(best_w, blend_test))
    candidates = {"blend": (blend_auc, test_pred_blend), "stack_lr": (stack_lr_auc, test_stack_lr)}
    best_name  = max(candidates, key=lambda k: candidates[k][0])
    final_auc, test_pred = candidates[best_name]
    logger.info("Selected ensemble: %s  OOF AUC %.6f", best_name, final_auc)
    return final_auc, test_pred


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_and_predict(data: PreparedData, cfg: object = config) -> pd.DataFrame:
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="ensemble") as run:
        logger.info("MLflow run id: %s", run.info.run_id)
        mlflow.log_params({
            "n_folds": cfg.N_FOLDS, "seed": cfg.SEED,
            "te_smoothing": cfg.TE_SMOOTHING, "te_min_samples": cfg.TE_MIN_SAMPLES,
            "corr_threshold": cfg.CORR_THRESHOLD, "null_imp_n_runs": cfg.NULL_IMP_N_RUNS,
            "null_imp_score_thr": cfg.NULL_IMP_SCORE_THRESHOLD,
            "lgb_a_lr": cfg.LGB_PARAMS_A["learning_rate"], "lgb_a_leaves": cfg.LGB_PARAMS_A["num_leaves"],
            "lgb_b_lr": cfg.LGB_PARAMS_B["learning_rate"], "lgb_b_leaves": cfg.LGB_PARAMS_B["num_leaves"],
            "xgb_lr": cfg.XGB_PARAMS["learning_rate"], "xgb_max_depth": cfg.XGB_PARAMS["max_depth"],
            "cat_lr": cfg.CAT_PARAMS_BASE["learning_rate"], "cat_depth": cfg.CAT_PARAMS_BASE["depth"],
            "stack_lr_C": cfg.STACK_LR_C,
        })

        for model_name, train_fn, target_ref in [
            ("lgb_a", train_lgb_a, data.target),
            ("lgb_b", train_lgb_b, data.target),
            ("lgb_c", train_lgb_c, data.target),
            ("xgb",   train_xgb,   data.target),
            ("cat",   train_catboost, data.cat_y),
        ]:
            ckpt = _load_ckpt(model_name, cfg)
            if ckpt:
                oof, test = ckpt
            else:
                logger.info("===== Training Model: %s =====", model_name.upper())
                oof, test = train_fn(data, cfg)
                _save_ckpt(model_name, oof, test, cfg)
            auc = roc_auc_score(target_ref, oof)
            mlflow.log_metric(f"oof_auc_{model_name}", auc)
            logger.info("%s OOF AUC: %.6f", model_name.upper(), auc)

            if model_name == "lgb_a":  oof_lgb1, test_lgb1, auc_lgb_a = oof, test, auc
            elif model_name == "lgb_b": oof_lgb2, test_lgb2, auc_lgb_b = oof, test, auc
            elif model_name == "lgb_c": oof_lgb3, test_lgb3, auc_lgb_c = oof, test, auc
            elif model_name == "xgb":   oof_xgb,  test_xgb,  auc_xgb   = oof, test, auc
            elif model_name == "cat":   oof_cat,  test_cat,  auc_cat   = oof, test, auc

        models = {
            "lgb_a":    (oof_lgb1, test_lgb1),
            "lgb_b":    (oof_lgb2, test_lgb2),
            "lgb_seed": (oof_lgb3, test_lgb3),
            "xgb":      (oof_xgb,  test_xgb),
            "cat":      (oof_cat,   test_cat),
        }

        logger.info("===== Building Ensemble =====")
        final_auc, test_pred = build_ensemble(models, data, cfg)
        mlflow.log_metric("oof_auc_ensemble", final_auc)

        # ── log all saved model files to MLflow ──
        models_dir = _models_dir(cfg)
        mlflow.log_artifacts(models_dir, artifact_path="models")
        logger.info("All models logged to MLflow artifacts.")

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        submission_path = os.path.join(cfg.OUTPUT_DIR, cfg.SUBMISSION_FILENAME)
        submission = pd.DataFrame({"SK_ID_CURR": data.test_ids.astype(int), "TARGET": test_pred})
        try:
            submission.to_csv(submission_path, index=False)
            logger.info("Submission saved to %s  shape=%s", submission_path, submission.shape)
        except OSError as exc:
            logger.error("Failed to save submission: %s", exc)
            raise

        mlflow.log_artifact(submission_path, artifact_path="submission")

        metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
        metrics = {
            "oof_auc_lgb_a": round(auc_lgb_a, 6), "oof_auc_lgb_b": round(auc_lgb_b, 6),
            "oof_auc_lgb_c": round(auc_lgb_c, 6), "oof_auc_xgb":   round(auc_xgb, 6),
            "oof_auc_catboost": round(auc_cat, 6), "oof_auc_ensemble": round(final_auc, 6),
        }
        try:
            with open(metrics_path, "w") as fh:
                json.dump(metrics, fh, indent=2)
            logger.info("Metrics written to %s", metrics_path)
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
        except OSError as exc:
            logger.warning("Could not write metrics.json: %s", exc)

        logger.info("===== FINAL OOF AUC: %.6f =====", final_auc)

    _clear_ckpts(cfg)
    return submission
