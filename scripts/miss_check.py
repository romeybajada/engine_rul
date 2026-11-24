"""How often would a weekly batch job miss an engine? Diary Entry 1, Table 3 ("Engine missed").

An engine flies about 35 cycles a week. If the Monday list says "predicted 60", batch misses it when the engine
really has 35 cycles or fewer left: it fails before the next run. This counts that share from the out-of-fold
predictions on the FD001 training engines (every cycle, engine-grouped folds, so no engine scores itself).

python scripts/miss_check.py -> reports/miss_check.json
"""
import json
from pathlib import Path

import numpy as np

from engine_rul.baseline import health_sensors, healthy_baseline, normalise
from engine_rul.config import load_config
from engine_rul.data import SENSORS, assign_regime, load_cmapss
from engine_rul.features import features
from engine_rul.model import out_of_fold

WEEK = 35          # cycles flown per week (~5 flights a day)
BAND = (55, 65)    # "predicted about 60"


def main():
    config = load_config()
    np.random.seed(config["seed"])
    cm = load_cmapss()
    cm["regime"] = assign_regime(cm)
    fd1 = cm[(cm.subset == config["train_set"]) & (cm.part == "train")]
    health = health_sensors(fd1, SENSORS)
    X = features(fd1, normalise(fd1, healthy_baseline(fd1, health)))
    y = np.minimum(fd1.rul.values, config["rul_cap"])
    oof = out_of_fold(X, y, fd1.engine.values, config)
    true = fd1.rul.values

    near = (oof >= BAND[0]) & (oof <= BAND[1])
    missed = near & (true <= WEEK)
    late = oof - true
    report = {
        "cycles scored (OOF)": int(len(oof)),
        "predicted in 55-65": int(near.sum()),
        "of those, true RUL <= 35 (fails before next weekly run)": int(missed.sum()),
        "miss share": round(float(missed.sum() / near.sum()), 4),
        "late error at true RUL 35-65: RMSE": round(float(np.sqrt(np.mean(late[(true >= 35) & (true <= 65)] ** 2))), 2),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/miss_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
