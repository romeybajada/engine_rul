import numpy as np
import pandas as pd
import pytest

from engine_rul.baseline import BASELINE_CYCLES, health_sensors, healthy_baseline, normalise
from engine_rul.features import features
from engine_rul.model import engine_folds
from engine_rul.scoring import phm_score


def test_phm_score_punishes_late_more_than_early():
    # Figure 9 of the paper: 13 for early, 10 for late (Eq. 11 prints them swapped)
    assert phm_score([10])[0] == pytest.approx(np.e - 1)
    assert phm_score([-13])[0] == pytest.approx(np.e - 1)
    assert phm_score([20])[0] > phm_score([-20])[0]


def test_features_are_as_of(fleet):
    # deleting cycles after t must not change the features at t
    t = 25
    base = healthy_baseline(fleet, ["s2", "s4"])
    full = features(fleet, normalise(fleet, base))
    cut = fleet[fleet.cycle <= t]
    short = features(cut, normalise(cut, base))
    pd.testing.assert_frame_equal(full.loc[cut.index], short)


def test_engine_folds_never_split_an_engine(fleet):
    groups = fleet.engine.values
    for a, b in engine_folds(fleet, fleet.cycle, groups, n_folds=3, seed=3217):
        assert set(groups[a]).isdisjoint(groups[b])


def test_health_sensors_drop_constant_and_two_valued(fleet):
    # s1 is constant and s6 takes two values: both are dropped
    assert health_sensors(fleet, ["s1", "s2", "s4", "s6"]) == ["s2", "s4"]


def test_baseline_rows_normalise_to_zero_mean(fleet):
    base = healthy_baseline(fleet, ["s2", "s4"])
    Z = normalise(fleet, base)
    early = Z[fleet.cycle <= BASELINE_CYCLES]
    assert early.mean().abs().max() < 1e-9
    assert early.std().sub(1).abs().max() < 1e-9
