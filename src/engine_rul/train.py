"""rul-train: the notebook's Section 3 as a command. Trains on FD001, scores the official FD001 test files once."""
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from engine_rul.baseline import health_sensors, healthy_baseline, normalise
from engine_rul.config import load_config
from engine_rul.data import SENSORS, assign_regime, holdout_sha256, load_cmapss
from engine_rul.features import features
from engine_rul.model import fit, out_of_fold, save_bundle
from engine_rul.scoring import scores

logger = logging.getLogger(__name__)


def run(config):
    np.random.seed(config["seed"])
    cm = load_cmapss()
    cm["regime"] = assign_regime(cm)
    fd1 = cm[cm.subset == config["train_set"]]
    tr, te = (fd1.part == "train").values, (fd1.part == "test").values

    health = health_sensors(fd1[tr], SENSORS)
    base1 = healthy_baseline(fd1[tr], health)
    X1 = features(fd1, normalise(fd1, base1))
    y1 = np.minimum(fd1.rul.values, config["rul_cap"])
    groups = fd1.engine.values
    is_last = (fd1.groupby("engine").cycle.transform("max") == fd1.cycle).values
    logger.info("%d health sensors, %d features for %d cycles", len(health), X1.shape[1], len(X1))

    t0 = time.time()
    oof = out_of_fold(X1[tr], y1[tr], groups[tr], config)
    model = fit(X1[tr], y1[tr], config)
    train_seconds = time.time() - t0

    pred_te = model.predict(X1[te])
    last_te = is_last[te]
    y_te_last = fd1.rul.values[te][last_te]
    test = scores(pred_te[last_te], y_te_last, "LightGBM, rolling features")
    zone = fd1.rul.values[tr] <= 50

    sha = holdout_sha256(config["holdout_files"])
    if sha != config["holdout_sha256"]:
        logger.error("the frozen holdout has changed: %s", sha)
    report = {
        "test RMSE": float(test["RMSE"]),
        "test PHM score (sum)": float(test["PHM score (sum)"]),
        "test late share": float(test["late share"]),
        "OOF RMSE (capped, all cycles)": float(np.sqrt(np.mean((oof - y1[tr]) ** 2))),
        "OOF RMSE (true RUL <= 50)": float(np.sqrt(np.mean((oof[zone] - fd1.rul.values[tr][zone]) ** 2))),
        "train seconds (OOF + final fit)": round(train_seconds, 1),
        "holdout sha256": sha,
        "holdout unchanged": sha == config["holdout_sha256"],
        "machine": platform.machine(),
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "numpy": np.__version__},
    }
    bundle = {"model": model, "baseline": base1, "health": health, "feature_names": list(X1.columns),
              "rul_cap": config["rul_cap"], "window": config["window"], "holdout_sha256": sha}
    return report, bundle


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config()
        report, bundle = run(config)
        save_bundle(os.environ.get("MODEL_PATH", config["model_path"]), **bundle)
        Path(config["report_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(config["report_path"]).write_text(json.dumps(report, indent=2))
        logger.info("test RMSE %.1f · PHM %.0f · late %.0f%% · OOF RMSE %.1f · trained in %.1fs",
                    report["test RMSE"], report["test PHM score (sum)"], 100 * report["test late share"],
                    report["OOF RMSE (capped, all cycles)"], report["train seconds (OOF + final fit)"])
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
