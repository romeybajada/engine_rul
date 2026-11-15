"""The given LightGBM model, split by engine. Notebook Section 3, with the bundle the service will load."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold


def lgb_params(config):
    return {**config["lgb_params"], "random_state": config["seed"]}


def engine_folds(X, y, groups, n_folds, seed):
    """Group k-fold by engine: every cycle of an engine lands in the same fold."""
    return GroupKFold(n_folds, shuffle=True, random_state=seed).split(X, y, groups)


def out_of_fold(X, y, groups, config):
    oof = np.zeros(len(y))
    for a, b in engine_folds(X, y, groups, config["n_folds"], config["seed"]):
        oof[b] = lgb.LGBMRegressor(**lgb_params(config)).fit(X.iloc[a], y[a]).predict(X.iloc[b])
    return oof


def fit(X, y, config):
    return lgb.LGBMRegressor(**lgb_params(config)).fit(X, y)


def save_bundle(path, **bundle):
    """Model plus everything serving needs to rebuild the same features: the baseline, the sensors, the column order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path):
    return joblib.load(path)
