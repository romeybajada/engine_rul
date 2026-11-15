"""The healthy baseline and z-scoring against it. Notebook Section 3: healthy_baseline, normalise."""
import pandas as pd

from engine_rul.config import load_config

BASELINE_CYCLES = load_config()["baseline_cycles"]


def health_sensors(train: pd.DataFrame, sensors) -> list:
    """Notebook's HEALTH rule: drop sensors constant (or two-valued) in the training fleet."""
    return [c for c in sensors if train[c].nunique() > 2]


def healthy_baseline(d: pd.DataFrame, cols) -> pd.DataFrame:
    """Mean and std of each sensor per regime over every engine's first BASELINE_CYCLES cycles (no labels used)."""
    return d[d.cycle <= BASELINE_CYCLES].groupby("regime")[cols].agg(["mean", "std"])


def normalise(d: pd.DataFrame, base: pd.DataFrame, by_regime: bool = True) -> pd.DataFrame:
    """z-score against the healthy baseline; by_regime=False applies the first regime's statistics to every row."""
    reg = d.regime if by_regime else pd.Series(base.index[0], index=d.index)
    cols = base.columns.get_level_values(0).unique()
    return pd.DataFrame({c: (d[c] - reg.map(base[(c, "mean")])) / reg.map(base[(c, "std")]) for c in cols}, index=d.index)
