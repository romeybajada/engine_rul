"""Load test for POST /rul (D-08): p50 / p95 latency under stated loads, measured at the client.

Three profiles, each reported with its load:
  typical  one request at a time (a fleet of 100 engines x ~5 flights/day is about one request every 3 minutes)
  burst    the whole fleet at once: --engines concurrent requests, repeated --rounds times
  window   the cost of D-04 Option A: 30-flight requests against 1-flight requests, one at a time

    python scripts/load_test.py                          # -> reports/load_test.json
    python scripts/load_test.py --url http://localhost:8001/rul --label native
Standard library only.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
WINDOW = 30


def load_engines(path):
    engines = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = dict(zip(COLS, line.split()))
            unit = int(row.pop("unit"))
            engines.setdefault(unit, []).append({k: (int(v) if k == "cycle" else float(v)) for k, v in row.items()})
    return engines


def timed_post(url, body):
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    return time.perf_counter() - t0, status


def summary(timings, load):
    ms = sorted(1000 * t for t, _ in timings)
    return {"load": load, "requests": len(ms), "non_200": sum(s != 200 for _, s in timings),
            "p50_ms": round(statistics.median(ms), 1), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1),
            "max_ms": round(ms[-1], 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/rul")
    p.add_argument("--data", default="data/raw/cmapss/CMaps/test_FD001.txt")
    p.add_argument("--engines", type=int, default=100, help="concurrent requests in a burst (the fleet size)")
    p.add_argument("--rounds", type=int, default=5, help="bursts to send")
    p.add_argument("--label", default="docker linux/amd64 (emulated on Apple M3)")
    args = p.parse_args()

    engines = load_engines(args.data)
    full = [{"engine_id": f"FD001-test-{u:03d}", "flights": f[-WINDOW:]} for u, f in engines.items()]   # 30 flights each
    one = [{"engine_id": f"FD001-test-{u:03d}", "flights": f[:1]} for u, f in engines.items()]          # cycle 1: 1 flight

    for body in full[:5]:                            # warm-up, not measured
        timed_post(args.url, body)

    results = {"where": args.label}
    results["typical"] = summary([timed_post(args.url, b) for b in full], "1 at a time, 30-flight windows")
    with ThreadPoolExecutor(max_workers=args.engines) as pool:
        burst = []
        for _ in range(args.rounds):
            burst += list(pool.map(lambda b: timed_post(args.url, b), full[:args.engines]))
    results["burst"] = summary(burst, f"{args.engines} at once (whole fleet), {args.rounds} rounds")
    results["window_1"] = summary([timed_post(args.url, b) for b in one], "1 at a time, 1-flight windows")
    results["request_kb"] = {"30 flights": round(len(json.dumps(full[0])) / 1024, 1), "1 flight": round(len(json.dumps(one[0])) / 1024, 1)}

    out = Path("reports/load_test.json")
    out.parent.mkdir(exist_ok=True)
    previous = json.loads(out.read_text()) if out.exists() else {}
    previous[args.label] = results
    out.write_text(json.dumps(previous, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
