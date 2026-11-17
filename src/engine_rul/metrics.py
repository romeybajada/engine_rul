"""Prometheus metrics (D-07). Scraped from GET /metrics."""
import pandas as pd
from prometheus_client import Counter, Histogram

from engine_rul.data import assign_regime

REQUESTS = Counter(
    "rul_requests",
    "POST /rul requests, by HTTP status (422 = rejected input)",
    ["status"],
)

LATENCY = Histogram(
    "rul_request_latency_seconds",
    "Time taken to answer POST /rul, validation and logging included",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5],
)

INPUT_REGIME = Counter(
    "rul_input_flights",
    "Newest flight of each request, by flight regime (0 is the only one the FD001 model was trained on)",
    ["regime"],
)


def count_regime(newest: pd.DataFrame):
    """Count the newest flight's regime before any rejection, so flights the model can't score still show up."""
    try:
        regime = str(assign_regime(newest)[0])
    except AssertionError:                          # matches no known regime
        regime = "unknown"
    INPUT_REGIME.labels(regime).inc()
