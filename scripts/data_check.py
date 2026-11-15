"""Entry 1, Step 2: check the data against its card. Writes reports/data_check.md; every count in the diary comes from here.

Run after rul-train (it reloads the trained bundle to score the RUL-above-cap engines).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul.baseline import normalise
from engine_rul.config import load_config
from engine_rul.data import RAW, SENSORS, SUBSETS, assign_regime, load_cmapss
from engine_rul.features import features
from engine_rul.model import load_bundle


def main():
    cfg = load_config()
    cm = load_cmapss()
    cm["regime"] = assign_regime(cm)
    train = cm[cm.part == "train"]
    f1 = train[train.subset == "FD001"]
    out = {}

    # card quirk 3: constant sensors in FD001, which move in FD002
    nu1 = f1[SENSORS].nunique()
    flat = list(nu1[nu1 <= 2].index)
    out["FD001 constant sensors"] = [c for c in flat if nu1[c] == 1]
    out["FD001 two-valued sensors"] = [c for c in flat if nu1[c] == 2]
    out["FD002 distinct values of those"] = train[train.subset == "FD002"][flat].nunique().to_dict()

    # card quirk 4: the readme's counts against the files
    readme = (RAW / "CMaps/readme.txt").read_text(errors="replace")
    fd4 = readme.split("Data Set: FD004")[1]
    out["readme FD004 train/test"] = [int(fd4.split("Train trjectories:")[1].split()[0]), int(fd4.split("Test trajectories:")[1].split()[0])]
    out["files FD004 train/test"] = [int(cm[(cm.subset == "FD004") & (cm.part == p)].unit.nunique()) for p in ["train", "test"]]
    out["readme last sensor number"] = 26
    out["sensor columns in files"] = len(SENSORS)

    # card quirk 1: test trajectories end before failure
    rul1 = pd.read_csv(RAW / "CMaps/RUL_FD001.txt", sep=r"\s+", header=None)[0]
    t1 = cm[(cm.subset == "FD001") & (cm.part == "test")]
    out["FD001 RUL lines / test engines"] = [len(rul1), int(t1.unit.nunique())]
    out["FD001 true RUL at last test cycle min/median/max"] = [int(rul1.min()), float(rul1.median()), int(rul1.max())]

    # card quirk 7: lives vary
    life = train.groupby(["subset", "engine"]).cycle.max()
    out["FD001 life min/median/max"] = [int(life["FD001"].min()), float(life["FD001"].median()), int(life["FD001"].max())]

    # card: the upload holds the folder twice
    out["CMaps folders in this download"] = len(list(RAW.glob("**/CMaps")))

    # not on the card: x.txt
    x = pd.read_csv(RAW / "CMaps/x.txt", sep=r"\s+", header=None)[0].values
    rul2 = pd.read_csv(RAW / "CMaps/RUL_FD002.txt", sep=r"\s+", header=None)[0].values
    out["x.txt lines"] = len(x)
    out["x.txt == first lines of RUL_FD002.txt"] = bool((rul2[: len(x)] == x).all())
    out["RUL_FD002 lines"] = len(rul2)

    # not on the card: raw format
    out["columns with sep=' ' (trailing spaces)"] = pd.read_csv(RAW / "CMaps/train_FD001.txt", sep=" ", header=None).shape[1]
    out["integer-valued sensors"] = [c for c in SENSORS if pd.api.types.is_integer_dtype(
        pd.read_csv(RAW / "CMaps/train_FD001.txt", sep=r"\s+", header=None)[SENSORS.index(c) + 5])]
    out["s17 distinct values FD001 train"] = int(f1.s17.nunique())
    out["s17 corr with RUL FD001"] = round(float(f1.s17.corr(f1.rul)), 2)

    # not on the card: test engines shorter than the rolling window
    tl = cm[cm.part == "test"].groupby(["subset", "engine"]).cycle.max()
    out[f"test engines shorter than WINDOW={cfg['window']}"] = {fd: int((tl[fd] < cfg["window"]).sum()) for fd in SUBSETS}
    out["shortest test engine"] = {fd: int(tl[fd].min()) for fd in SUBSETS}

    # not on the card: true RUL above the cap on the FD001 test files
    b = load_bundle(cfg["model_path"])
    fd1 = cm[cm.subset == "FD001"]
    X = features(fd1, normalise(fd1, b["baseline"]))
    te = (fd1.part == "test").values
    last = (fd1.groupby("engine").cycle.transform("max") == fd1.cycle).values & te
    p, y = b["model"].predict(X[last]), fd1.rul.values[last]
    e, above = p - y, y > cfg["rul_cap"]
    out["FD001 test engines with RUL > cap"] = int(above.sum())
    out["max prediction"] = round(float(p.max()), 1)
    out["share of squared error from them"] = round(float((e[above] ** 2).sum() / (e ** 2).sum()), 2)
    out["test RMSE all / without them"] = [round(float(np.sqrt((e ** 2).mean())), 2), round(float(np.sqrt((e[~above] ** 2).mean())), 2)]

    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
