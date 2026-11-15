"""As-of features. Notebook Section 3: features, unchanged."""
import numpy as np
import pandas as pd

from engine_rul.config import load_config

WINDOW = load_config()["window"]


def features(d: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    """As-of features: current value, rolling mean / std / slope over WINDOW past cycles of the same engine. d sorted by engine, cycle."""
    t = d.cycle.astype(float)
    cols = list(Z.columns)
    stack = pd.concat([Z, Z.mul(t, axis=0).add_suffix("_xt"), t.rename("t"), (t * t).rename("t2")], axis=1)
    roll = stack.groupby(d.engine.values, sort=False).rolling(WINDOW, min_periods=1)
    m = roll.mean().reset_index(level=0, drop=True)
    sd = Z.groupby(d.engine.values, sort=False).rolling(WINDOW, min_periods=2).std().reset_index(level=0, drop=True)
    var_t = (m["t2"] - m["t"] ** 2).replace(0, np.nan)
    X = {"cycle": d.cycle.values}
    for c in cols:
        X[c] = Z[c].values
        X[c + "_mean"] = m[c].values
        X[c + "_std"] = sd[c].values
        X[c + "_slope"] = ((m[c + "_xt"] - m[c] * m["t"]) / var_t).values
    return pd.DataFrame(X, index=d.index)
