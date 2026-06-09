import argparse
import logging
import sys
import types

from src import config as _default_cfg


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Home Credit Default Risk – production ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", default="all", choices=["all", "prepare", "train"],
                   help="'prepare' = feature engineering only (saves to disk). "
                        "'train' = load prepared data and train models. "
                        "'all' = run both stages end-to-end.")
    p.add_argument("--data-dir", default=_default_cfg.DATA_DIR,
                   help="Path to directory containing raw CSV files.")
    p.add_argument("--output-dir", default=_default_cfg.OUTPUT_DIR,
                   help="Path where outputs (submission CSV, prepared data, etc.) are written.")
    p.add_argument("--n-folds", type=int, default=_default_cfg.N_FOLDS,
                   help="Number of cross-validation folds.")
    p.add_argument("--seed", type=int, default=_default_cfg.SEED,
                   help="Global random seed.")
    p.add_argument("--submission", default=_default_cfg.SUBMISSION_FILENAME,
                   help="Filename for the submission CSV.")
    p.add_argument("--mlflow-uri", default=_default_cfg.MLFLOW_TRACKING_URI,
                   help="MLflow tracking URI (local path or remote server).")
    p.add_argument("--mlflow-experiment", default=_default_cfg.MLFLOW_EXPERIMENT_NAME,
                   help="MLflow experiment name.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity.")
    return p


def _override_config(args: argparse.Namespace) -> types.SimpleNamespace:
    cfg = types.SimpleNamespace(**{k: getattr(_default_cfg, k) for k in dir(_default_cfg) if not k.startswith("__")})
    cfg.DATA_DIR               = args.data_dir
    cfg.OUTPUT_DIR             = args.output_dir
    cfg.N_FOLDS                = args.n_folds
    cfg.SEED                   = args.seed
    cfg.SUBMISSION_FILENAME    = args.submission
    cfg.MLFLOW_TRACKING_URI    = args.mlflow_uri
    cfg.MLFLOW_EXPERIMENT_NAME = args.mlflow_experiment
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting Home Credit pipeline  stage=%s", args.stage)

    cfg = _override_config(args)

    # Lazy imports so each stage only loads what it needs
    if args.stage in ("all", "prepare"):
        from src.data_prep import prepare_data
        logger.info("── Stage 1/2: Feature engineering & data preparation ──")
        data = prepare_data(cfg)              

    if args.stage in ("all", "train"):
        from src.train import train_and_predict
        if args.stage == "train":

            from src.data_prep import load_prepared_data
            logger.info("── Stage 2/2: Loading prepared data from disk ──")
            data = load_prepared_data(cfg)
        logger.info("── Stage 2/2: Model training & ensemble ──")
        submission = train_and_predict(data, cfg)
        logger.info("Pipeline finished.  Submission shape: %s", submission.shape)


if __name__ == "__main__":
    main()