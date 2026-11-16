"""POST /rul: stateless scoring (D-04 Option A). The client sends the engine's last flights; the service keeps no history."""
import hashlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine_rul.baseline import normalise
from engine_rul.config import load_config
from engine_rul.data import assign_regime
from engine_rul.features import features
from engine_rul.model import load_bundle

logger = logging.getLogger(__name__)

WINDOW = load_config()["window"]


class Flight(BaseModel):
    """One row of the health feed: the C-MAPSS columns after `unit`. NaN and infinity are rejected."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cycle: int = Field(ge=1)
    setting1: float
    setting2: float
    setting3: float
    s1: float; s2: float; s3: float; s4: float; s5: float; s6: float; s7: float
    s8: float; s9: float; s10: float; s11: float; s12: float; s13: float; s14: float
    s15: float; s16: float; s17: float; s18: float; s19: float; s20: float; s21: float


class RulRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine_id: str = Field(min_length=1)
    flights: list[Flight] = Field(min_length=1, max_length=WINDOW)

    @model_validator(mode="after")
    def window_is_complete(self):
        """The rolling features count rows, not cycles: a gap or a short window silently changes them."""
        cycles = [f.cycle for f in self.flights]
        if any(b != a + 1 for a, b in zip(cycles, cycles[1:])):
            raise ValueError("flights must be in cycle order with no gaps or repeats")
        need = min(WINDOW, cycles[-1])
        if len(cycles) < need:
            raise ValueError(f"cycle {cycles[-1]} needs the last {need} flights, got {len(cycles)}")
        return self


class RulResponse(BaseModel):
    engine_id: str
    cycle: int
    rul: float
    rul_cap: int
    model_version: str


def model_version(path) -> str:
    """First 12 hex digits of the bundle file's SHA-256: changes whenever the model file changes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def predict_window(bundle, flights: pd.DataFrame) -> float:
    """RUL at the last row of one engine's window. Same steps as rul-train: regime, z-score, as-of features."""
    d = flights.assign(engine="request").reset_index(drop=True)
    d["regime"] = assign_regime(d)
    known = set(bundle["baseline"].index)
    if not set(d.regime) <= known:
        raise ValueError(f"flight regime {sorted(set(d.regime) - known)} is outside the training fleet's {sorted(known)}")
    X = features(d, normalise(d, bundle["baseline"]))[bundle["feature_names"]]
    return float(bundle["model"].predict(X.iloc[[-1]])[0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("MODEL_PATH", load_config()["model_path"])
    app.state.bundle = load_bundle(path)
    app.state.version = model_version(path)
    logger.info("loaded model %s from %s", app.state.version, path)
    yield


app = FastAPI(title="engine-rul", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.version}


@app.post("/rul", response_model=RulResponse)
def rul(req: RulRequest):
    flights = pd.DataFrame([f.model_dump() for f in req.flights])
    try:
        pred = predict_window(app.state.bundle, flights)
    except (ValueError, AssertionError) as e:     # assign_regime asserts when a row matches no known regime
        raise HTTPException(422, str(e) or "a flight matches no known flight regime")
    if not np.isfinite(pred):
        raise HTTPException(500, "the model returned a non-finite prediction")
    return RulResponse(engine_id=req.engine_id, cycle=req.flights[-1].cycle, rul=pred,
                       rul_cap=app.state.bundle["rul_cap"], model_version=app.state.version)
