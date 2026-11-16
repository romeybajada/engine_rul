"""Training-serving skew test: the running container must give exactly the package's prediction for the same rows.

Both sides run in the same image (same platform, same model), so the only thing that can differ is the serving path:
JSON, validation, building the DataFrame, column order. Any difference at all is a failure.

    docker run -d --name rul-api --platform linux/amd64 -p 8000:8000 engine-rul:0.1
    python scripts/skew_test.py                  # -> reports/skew_test.json, exit 1 on any mismatch

With --reference, it runs inside the image and prints the expected predictions as JSON (the host half calls this).
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

CYCLES_PER_ENGINE = 5


def reference():
    """Inside the image: build the windows from the FD001 test files and predict them with the package directly."""
    import numpy as np

    from engine_rul.config import load_config
    from engine_rul.data import SENSORS, load_cmapss
    from engine_rul.model import load_bundle
    from engine_rul.service import WINDOW, model_version, predict_window

    path = load_config()["model_path"]
    bundle = load_bundle(path)
    cm = load_cmapss()
    fd = cm[(cm.subset == "FD001") & (cm.part == "test")]
    cols = ["cycle", "setting1", "setting2", "setting3"] + SENSORS
    rng = np.random.default_rng(load_config()["seed"])
    cases = []
    for engine, g in fd.groupby("engine", sort=False):
        ends = {len(g)} | set(rng.integers(1, len(g) + 1, CYCLES_PER_ENGINE - 1).tolist())   # last cycle + random ones
        for end in sorted(ends):
            w = g.iloc[max(0, end - WINDOW):end][cols]
            cases.append({"request": {"engine_id": engine, "flights": w.to_dict("records")},
                          "expected": predict_window(bundle, w)})
    json.dump({"model_version": model_version(path), "cases": cases}, sys.stdout)


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", action="store_true", help="run inside the image: print the expected predictions")
    p.add_argument("--image", default="engine-rul:0.1")
    p.add_argument("--url", default="http://localhost:8000")
    args = p.parse_args()
    if args.reference:
        return reference()

    here = Path(__file__).resolve().parent
    out = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{here}:/scripts:ro", args.image,
                          "python", "/scripts/skew_test.py", "--reference"], capture_output=True, text=True, check=True)
    ref = json.loads(out.stdout)

    with urllib.request.urlopen(args.url + "/health") as r:
        served = json.load(r)["model_version"]
    if served != ref["model_version"]:
        sys.exit(f"the container serves model {served}, but the image holds {ref['model_version']}: not the same model")

    diffs = [abs(post(args.url + "/rul", c["request"])["rul"] - c["expected"]) for c in ref["cases"]]
    report = {"image": args.image, "model_version": served, "cases": len(diffs),
              "mismatches": sum(d != 0 for d in diffs), "max_abs_diff": max(diffs)}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/skew_test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    sys.exit(1 if report["mismatches"] else 0)


if __name__ == "__main__":
    main()
