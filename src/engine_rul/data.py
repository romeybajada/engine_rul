"""Loading C-MAPSS and assigning flight regimes. Notebook Section 1: get_raw, load_cmapss, assign_regime, unchanged."""
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul.config import data_dir, load_config

logger = logging.getLogger(__name__)

_CFG = load_config()
KAGGLE_HANDLE = _CFG["kaggle_handle"]
RAW, PROCESSED = data_dir(_CFG) / "raw" / "cmapss", data_dir(_CFG) / "processed"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + SENSORS


def get_raw(filename: str) -> Path:
    local = RAW / filename
    if local.exists():
        return local
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_HANDLE, path=filename))


def load_cmapss() -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED / "22_cmapss.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t = time.time()
    frames = []
    for fd in SUBSETS:
        rul_end = pd.read_csv(get_raw(f"CMaps/RUL_{fd}.txt"), sep=r"\s+", header=None)[0].values
        for part in ["train", "test"]:
            d = pd.read_csv(get_raw(f"CMaps/{part}_{fd}.txt"), sep=r"\s+", header=None, names=COLS)
            assert d.notna().all().all() and len(d.columns) == 26, f"{part}_{fd}: schema changed"
            last = d.groupby("unit").cycle.transform("max")
            # train engines run to failure; test engines stop early and the RUL at their last cycle is in RUL_FD00x.txt
            d["rul"] = (last - d.cycle) + (rul_end[d.unit.values - 1] if part == "test" else 0)
            if part == "test":
                assert d.unit.nunique() == len(rul_end), f"RUL_{fd}: one line per test engine expected"
            d.insert(0, "subset", fd); d.insert(1, "part", part)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["engine"] = df.subset + "-" + df.part + "-" + df.unit.astype(str).str.zfill(3)
    df = df.sort_values(["subset", "part", "unit", "cycle"]).reset_index(drop=True)
    df.to_parquet(cache)
    logger.info("loaded + cached in %.0fs", time.time() - t)
    return df


# ---- reference tables: the paper's Table 2 and the flight regimes ----
SENSOR_TABLE = pd.DataFrame([
    ("s1", "T2", "Total temperature at fan inlet", "°R", "condition", ""),
    ("s2", "T24", "Total temperature at LPC outlet", "°R", "health", "LPC"),
    ("s3", "T30", "Total temperature at HPC outlet", "°R", "health", "HPC"),
    ("s4", "T50", "Total temperature at LPT outlet", "°R", "health", "LPT"),
    ("s5", "P2", "Pressure at fan inlet", "psia", "condition", ""),
    ("s6", "P15", "Total pressure in bypass duct", "psia", "health", "Fan"),
    ("s7", "P30", "Total pressure at HPC outlet", "psia", "health", "HPC"),
    ("s8", "Nf", "Physical fan speed", "rpm", "health", "Fan"),
    ("s9", "Nc", "Physical core speed", "rpm", "health", "HPC;HPT"),
    ("s10", "epr", "Engine pressure ratio (P50/P2)", "-", "health", "LPT"),
    ("s11", "Ps30", "Static pressure at HPC outlet", "psia", "health", "HPC"),
    ("s12", "phi", "Ratio of fuel flow to Ps30", "pps/psi", "health", "Burner;HPC"),
    ("s13", "NRf", "Corrected fan speed", "rpm", "health", "Fan"),
    ("s14", "NRc", "Corrected core speed", "rpm", "health", "HPC;HPT"),
    ("s15", "BPR", "Bypass ratio", "-", "health", "Fan"),
    ("s16", "farB", "Burner fuel-air ratio", "-", "health", "Burner"),
    ("s17", "htBleed", "Bleed enthalpy", "-", "health", "HPC"),
    ("s18", "Nf_dmd", "Demanded fan speed", "rpm", "control", ""),
    ("s19", "PCNfR_dmd", "Demanded corrected fan speed", "rpm", "control", ""),
    ("s20", "W31", "HPT coolant bleed", "lbm/s", "health", "HPT"),
    ("s21", "W32", "LPT coolant bleed", "lbm/s", "health", "LPT"),
], columns=["col", "symbol", "description", "unit", "role", "observes"]).set_index("col")
REGIMES = pd.DataFrame({"altitude_kft": [0, 10, 20, 25, 35, 42], "mach": [0.0, 0.25, 0.70, 0.62, 0.84, 0.84],
                        "tra": [100, 100, 100, 60, 100, 100]}, index=pd.Index(range(6), name="regime"))
SYM = SENSOR_TABLE.symbol.to_dict()


def assign_regime(d: pd.DataFrame) -> np.ndarray:
    """Nearest flight regime in the reference table (settings scaled by their ranges)."""
    scale = np.array([42.0, 0.84, 100.0])
    dist = (((d[["setting1", "setting2", "setting3"]].values[:, None, :] - REGIMES.values[None]) / scale) ** 2).sum(-1)
    assert np.sqrt(dist.min(1)).max() < 0.01, "a row matches no known flight regime"
    return dist.argmin(1)


def holdout_sha256(files) -> str:
    """Fingerprint of the frozen holdout (notebook Section 1's HOLDOUT_SHA)."""
    return hashlib.sha256(b"".join(get_raw(f).read_bytes() for f in files)).hexdigest()
