"""Each test guards one quirk of the dataset card (or one we found), on small files written here: no real data needed."""
import numpy as np
import pandas as pd
import pytest

import engine_rul.data as data
from engine_rul.data import COLS
from engine_rul.validate import check_cycles, check_rul_file, read_raw


def write_raw(path, units_cycles, trailing="  "):
    """C-MAPSS style: space-separated, no header, two trailing spaces on every line."""
    lines = []
    for unit, n in units_cycles:
        for c in range(1, n + 1):
            row = [unit, c, 0.0, 0.0, 100.0] + [518.67 + 0.01 * c] * 21
            lines.append(" ".join(str(v) for v in row) + trailing)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """One subset, FD001: two training engines run to failure, two test engines cut early."""
    cm = tmp_path / "raw" / "cmapss" / "CMaps"
    cm.mkdir(parents=True)
    write_raw(cm / "train_FD001.txt", [(1, 40), (2, 50)])
    write_raw(cm / "test_FD001.txt", [(1, 20), (2, 35)])
    (cm / "RUL_FD001.txt").write_text("112 \n98 \n")
    monkeypatch.setattr(data, "RAW", tmp_path / "raw" / "cmapss")
    monkeypatch.setattr(data, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(data, "SUBSETS", ["FD001"])
    return cm


def test_trailing_spaces_still_give_26_columns(raw_dir):
    # the raw lines end in two spaces: sep=" " would read 28 columns
    assert pd.read_csv(raw_dir / "train_FD001.txt", sep=" ", header=None).shape[1] == 28
    assert list(read_raw(raw_dir / "train_FD001.txt").columns) == COLS


def test_an_extra_column_fails_loudly(tmp_path):
    # with names=COLS, pandas would silently make the 27th column the index; read_raw refuses
    p = tmp_path / "bad.txt"
    p.write_text(" ".join(["1"] * 27) + "\n")
    with pytest.raises(ValueError, match="27 columns"):
        read_raw(p)


def test_test_trajectories_are_not_labelled_as_failures(raw_dir):
    # Trap 3: a test engine's last cycle is not its failure; its RUL there comes from RUL_FD001.txt
    df = data.load_cmapss()
    test = df[df.part == "test"]
    last = test[test.groupby("unit").cycle.transform("max") == test.cycle].set_index("unit").rul
    assert last.to_dict() == {1: 112, 2: 98}
    train = df[df.part == "train"]
    assert train.groupby("unit").rul.min().eq(0).all()          # training engines do run to failure


def test_truncated_rul_file_is_caught(raw_dir):
    # x.txt is the first 218 lines of RUL_FD002.txt (259 engines): a short file leaves engines unlabelled
    test = read_raw(raw_dir / "test_FD001.txt")
    assert check_rul_file(test, pd.Series([112, 98]), "RUL_FD001") == []
    assert check_rul_file(test, pd.Series([112]), "RUL_FD001") == ["RUL_FD001: 1 RUL lines for 2 test engines"]


def test_cycle_gaps_and_late_starts_are_caught():
    ok = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    assert check_cycles(ok, "x") == []
    gap = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 3, 2, 3]})
    assert len(check_cycles(gap, "x")) == 2


def test_constant_sensors_cannot_be_z_scored():
    # Trap 4: a constant sensor has std 0 in the baseline, so its z-score is 0/0 = NaN
    from engine_rul.baseline import healthy_baseline, normalise
    d = pd.DataFrame({"engine": "E1", "cycle": range(1, 41), "regime": 0, "s1": 518.67, "s2": np.linspace(640, 645, 40)})
    Z = normalise(d, healthy_baseline(d, ["s1", "s2"]))
    assert Z.s1.isna().all() and Z.s2.notna().all()
