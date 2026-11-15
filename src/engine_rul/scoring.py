"""RMSE and the asymmetric PHM08 score. Notebook Section 3: phm_score, scores, unchanged."""
import numpy as np


def phm_score(err):
    err = np.asarray(err, float)                    # err = predicted − true; late (err > 0) costs more
    return np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)


def scores(pred, true, name):
    e = np.asarray(pred) - np.asarray(true)
    return {"model": name, "RMSE": np.sqrt(np.mean(e ** 2)), "PHM score (sum)": phm_score(e).sum(),
            "PHM score / engine": phm_score(e).mean(), "late share": (e > 0).mean(), "bias (cycles)": e.mean()}
