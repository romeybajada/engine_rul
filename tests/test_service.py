"""POST /rul (D-04 Option A). The first tests need no model; the last ones use the trained bundle and the data, if present."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import engine_rul.data as data
from engine_rul.baseline import healthy_baseline, normalise
from engine_rul.config import load_config
from engine_rul.features import features
from engine_rul.service import WINDOW, RulRequest

FLIGHT_COLS = ["cycle", "setting1", "setting2", "setting3"] + data.SENSORS


def flights(cycles):
    """Valid-looking FD001 rows (regime 0) for the given cycle numbers."""
    return [{"cycle": c, "setting1": 0.0, "setting2": 0.0, "setting3": 100.0, **{s: 1.0 for s in data.SENSORS}} for c in cycles]


def test_last_window_gives_the_full_history_features(fleet):
    # Option A rests on this: the last row's features only need the last WINDOW cycles (equal up to rounding)
    base = healthy_baseline(fleet, ["s2", "s4"])
    e1 = fleet[fleet.engine == "E1"]
    full = features(e1, normalise(e1, base)).iloc[-1]
    w = e1.tail(WINDOW)
    short = features(w, normalise(w, base)).iloc[-1]
    np.testing.assert_allclose(short.values, full.values, rtol=0, atol=1e-12)


@pytest.mark.parametrize("cycles", [range(1, 11), range(31, 61), [1]])
def test_complete_windows_are_accepted(cycles):
    assert RulRequest(engine_id="E1", flights=flights(cycles))


@pytest.mark.parametrize("cycles, why", [
    ([3, 2, 1], "order"),
    ([1, 2, 4], "gaps"),
    ([1, 2, 2, 3], "repeats"),
    (range(40, 61), "needs the last 30"),       # cycle 60 with only 21 flights
    (range(2, 11), "needs the last 10"),        # cycle 10 without cycle 1
    (range(1, 32), "at most 30"),               # more than the window
    ([], "at least 1"),
])
def test_incomplete_windows_are_rejected(cycles, why):
    with pytest.raises(ValidationError, match=why):
        RulRequest(engine_id="E1", flights=flights(cycles))


def test_missing_or_non_finite_values_are_rejected():
    rows = flights([1])
    del rows[0]["s4"]
    with pytest.raises(ValidationError, match="s4"):
        RulRequest(engine_id="E1", flights=rows)
    rows = flights([1])
    rows[0]["s4"] = float("nan")
    with pytest.raises(ValidationError, match="finite"):
        RulRequest(engine_id="E1", flights=rows)


# ---- with the trained bundle and the data ----
MODEL = Path(load_config()["model_path"])
CACHE = data.PROCESSED / "22_cmapss.parquet"
needs_model = pytest.mark.skipif(not (MODEL.exists() and CACHE.exists()), reason="needs rul-train and rul-fetch first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from engine_rul.service import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def cmapss():
    cm = data.load_cmapss()
    cm["regime"] = data.assign_regime(cm)
    return cm


def request_for(engine_rows):
    return {"engine_id": engine_rows.engine.iloc[-1], "flights": engine_rows.tail(WINDOW)[FLIGHT_COLS].to_dict("records")}


@needs_model
def test_rul_returns_a_prediction_and_the_model_version(client, cmapss):
    e = cmapss[cmapss.engine == "FD001-test-001"]
    r = client.post("/rul", json=request_for(e))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cycle"] == e.cycle.max() and np.isfinite(body["rul"])
    assert body["model_version"] == client.get("/health").json()["model_version"]


@needs_model
def test_a_regime_outside_the_training_fleet_is_rejected(client, cmapss):
    # FD002 flies six regimes; the FD001 model only has a baseline for regime 0 (the Section 6 silent failure)
    e = cmapss[(cmapss.engine == "FD002-test-001")]
    r = client.post("/rul", json=request_for(e))
    assert r.status_code == 422 and "outside the training fleet" in r.text


@needs_model
def test_window_scores_stay_within_the_reproduction_gate(cmapss):
    # The last-30 window and the full history differ only by rounding (<1e-14), but that can flip a near-tied split.
    # On the official test set the score must stay inside ADR 0002's gate: ±0.01 RMSE, ±0.5 PHM.
    from engine_rul.model import load_bundle
    from engine_rul.scoring import scores
    from engine_rul.service import predict_window
    b = load_bundle(MODEL)
    fd = cmapss[(cmapss.subset == "FD001") & (cmapss.part == "test")]
    full = b["model"].predict(features(fd, normalise(fd, b["baseline"]))[b["feature_names"]])
    is_last = (fd.groupby("engine").cycle.transform("max") == fd.cycle).values
    window = [predict_window(b, g.tail(WINDOW)[FLIGHT_COLS]) for _, g in fd.groupby("engine", sort=False)]
    true = fd.rul.values[is_last]
    a, w = scores(full[is_last], true, "full"), scores(np.array(window), true, "window")
    assert abs(a["RMSE"] - w["RMSE"]) <= 0.01
    assert abs(a["PHM score (sum)"] - w["PHM score (sum)"]) <= 0.5
    assert pd.Series(fd.engine.unique()).equals(pd.Series(fd.engine[is_last].values))   # same engine order on both sides


def test_a_broken_database_never_stops_a_prediction(monkeypatch, caplog):
    # D-06: logging must not break serving. Nothing listens on port 1, so the connection fails at once.
    from engine_rul.db import save_prediction
    monkeypatch.setenv("DATABASE_URL", "postgresql://rul:x@127.0.0.1:1/rul?connect_timeout=1")
    save_prediction("E1", 10, 42.0, "abc")
    assert "Could not store the prediction" in caplog.text
    monkeypatch.delenv("DATABASE_URL")
    save_prediction("E1", 10, 42.0, "abc")     # not configured: silently skipped
