import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fleet():
    """Three synthetic engines in one regime, 60 cycles each, sorted by engine and cycle like load_cmapss."""
    rng = np.random.default_rng(0)
    rows = []
    for e in ["E1", "E2", "E3"]:
        for c in range(1, 61):
            rows.append({"engine": e, "cycle": c, "regime": 0, "s2": 640 + 0.02 * c + rng.normal(0, 0.5),
                         "s4": 1400 + 0.1 * c + rng.normal(0, 5), "s1": 518.67, "s6": [21.60, 21.61][c % 2]})
    return pd.DataFrame(rows)
