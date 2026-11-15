"""rul-validate: the data contract on the raw C-MAPSS files. Exit code 0 if every check passes, 1 otherwise.

Each check returns a list of problems, so one run reports all of them. Real checks, not asserts: `python -O` strips asserts.
"""
import logging
import sys

import pandas as pd
import pandera.pandas as pa

from engine_rul.config import load_config
from engine_rul.data import COLS, SENSORS, SUBSETS, get_raw, holdout_sha256

logger = logging.getLogger(__name__)

SCHEMA = pa.DataFrameSchema(
    {
        "unit": pa.Column(int, pa.Check.ge(1)),
        "cycle": pa.Column(int, pa.Check.ge(1)),
        "setting1": pa.Column(float),
        "setting2": pa.Column(float),
        "setting3": pa.Column(float, pa.Check.isin([60.0, 100.0])),   # throttle angle: two values in the paper
        **{s: pa.Column(float) for s in SENSORS},                     # nullable=False by default
    },
    strict=True,
    coerce=True,   # s17 (htBleed) and s18 (Nf_dmd) are written as integers in every file
)


def read_raw(path) -> pd.DataFrame:
    """Read without names first: with names=COLS, a 27th column would silently become the index."""
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] != len(COLS):
        raise ValueError(f"{path}: {raw.shape[1]} columns, expected {len(COLS)}")
    raw.columns = COLS
    return raw


def check_cycles(d: pd.DataFrame, name: str) -> list:
    """Every engine starts at cycle 1 and counts up by one, with no gap or repeat."""
    problems = []
    g = d.groupby("unit").cycle
    if (bad := int((g.min() != 1).sum())):
        problems.append(f"{name}: {bad} engines do not start at cycle 1")
    if (bad := int((g.diff().dropna() != 1).sum())):
        problems.append(f"{name}: {bad} cycle steps are not +1 (gap, repeat or unsorted)")
    return problems


def check_rul_file(test: pd.DataFrame, rul: pd.Series, name: str) -> list:
    """One RUL line per test engine, engines numbered 1..n: a truncated file (like x.txt) leaves engines unlabelled."""
    problems = []
    units = sorted(test.unit.unique())
    if len(rul) != len(units):
        problems.append(f"{name}: {len(rul)} RUL lines for {len(units)} test engines")
    if units != list(range(1, len(units) + 1)):
        problems.append(f"{name}: test engines are not numbered 1..{len(units)}")
    if (rul < 0).any():
        problems.append(f"{name}: {int((rul < 0).sum())} negative RUL values")
    return problems


def validate_subset(fd: str) -> list:
    problems = []
    frames = {}
    for part in ["train", "test"]:
        name = f"{part}_{fd}"
        try:
            frames[part] = SCHEMA.validate(read_raw(get_raw(f"CMaps/{name}.txt")), lazy=True)
        except pa.errors.SchemaErrors as err:
            problems.append(f"{name}: schema failed\n{err.failure_cases.head(10).to_string()}")
            continue
        except ValueError as err:
            problems.append(str(err))
            continue
        problems += check_cycles(frames[part], name)
    if "test" in frames:
        rul = pd.read_csv(get_raw(f"CMaps/RUL_{fd}.txt"), sep=r"\s+", header=None)[0]
        problems += check_rul_file(frames["test"], rul, f"RUL_{fd}")
    return problems


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    problems = []
    for fd in SUBSETS:
        found = validate_subset(fd)
        logger.info("%s: %s", fd, "valid" if not found else f"{len(found)} problems")
        problems += found
    sha = holdout_sha256(config["holdout_files"])
    if sha != config["holdout_sha256"]:
        problems.append(f"the frozen holdout has changed: {sha[:12]}… against {config['holdout_sha256'][:12]}…")
    for p in problems:
        logger.error(p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
