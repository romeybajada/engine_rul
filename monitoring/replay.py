"""Replay real test engines against POST /rul, one flight at a time.

Adapted from Session 4's traffic.py: the RUL API needs each engine's last flights, not one random number,
so this reads the raw C-MAPSS test file and replays its engines in parallel, each sending its newest window
of up to 30 flights per request. Uses only the Python standard library, so it runs without a virtual environment.

Examples:
    python3 monitoring/replay.py --fleet FD001                  # the model's own fleet
    python3 monitoring/replay.py --fleet FD002                  # six regimes: the Section 6 shift
    python3 monitoring/replay.py --fleet FD001 --rate 20 --duration 300
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
WINDOW = 30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000/rul", help="prediction endpoint")
    parser.add_argument("--fleet", default="FD001", help="which test file to replay (FD001 to FD004)")
    parser.add_argument("--data", default="data/raw/cmapss/CMaps", help="folder with the raw test files")
    parser.add_argument("--rate", type=float, default=5.0, help="requests per second")
    parser.add_argument("--duration", type=float, default=120.0, help="how long to run, in seconds")
    parser.add_argument("--seed", type=int, default=None, help="random seed for the engine order, for a repeatable stream")
    return parser.parse_args()


def load_engines(path):
    """Each test engine's flights, in cycle order, as dicts of the API's field names."""
    engines = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = dict(zip(COLS, line.split()))
            unit = int(row.pop("unit"))
            flight = {k: (int(v) if k == "cycle" else float(v)) for k, v in row.items()}
            engines.setdefault(unit, []).append(flight)
    return engines


def send(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:        # a 422 is an answer, not a failure to connect
        return error.code


def main():
    args = parse_args()
    engines = load_engines(Path(args.data) / f"test_{args.fleet}.txt")
    order = list(engines)
    random.Random(args.seed).shuffle(order)
    position = {unit: 0 for unit in order}          # how many flights each engine has flown so far
    pause = 1.0 / args.rate
    end = time.monotonic() + args.duration
    counts, failed, turn = {}, 0, 0
    next_report = time.monotonic() + 10
    print(f"Replaying {len(engines)} {args.fleet} test engines at {args.rate}/s for {args.duration:.0f}s to {args.url}")
    while time.monotonic() < end:
        started = time.monotonic()
        unit = order[turn % len(order)]
        turn += 1
        flights = engines[unit]
        n = position[unit] = position[unit] % len(flights) + 1   # next flight; an engine that has finished starts again
        body = {"engine_id": f"{args.fleet}-test-{unit:03d}", "flights": flights[max(0, n - WINDOW):n]}
        try:
            status = send(args.url, body)
            counts[status] = counts.get(status, 0) + 1
        except (urllib.error.URLError, OSError) as error:
            failed += 1
            if failed == 1:
                print(f"Request failed: {error}")
        if time.monotonic() >= next_report:
            print(f"answers by status {counts}, {failed} failed")
            next_report += 10
        time.sleep(max(0.0, pause - (time.monotonic() - started)))
    print(f"Done: answers by status {counts}, {failed} failed")


if __name__ == "__main__":
    main()
