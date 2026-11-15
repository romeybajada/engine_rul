# ADR 0001: Package the given model unchanged

**Status:** accepted · **Date:** 2026-11-14 · **Entry:** 1, Step 1

## Context
The brief says to package Part B's code *unchanged*. Some changes are unavoidable (globals, paths, plotting), and if
the numbers then drift, I must be able to tell a plumbing change from a method change.

## Decision
Function bodies are copied byte-for-byte from the notebook. Only the plumbing around them changes:

| Change | Why | Can it change a number? |
|---|---|---|
| CAPITALS move to `config/config.yaml`, read once at import into module constants (`WINDOW`, `BASELINE_CYCLES`, …) | Lets function bodies stay identical | No: same values |
| `HEALTH` is computed by `health_sensors(train, SENSORS)` and passed to `healthy_baseline(d, cols)`; the `cols or HEALTH` default is removed | No hidden global computed from data in another cell | No: same rule, same 14 sensors |
| `print` in `load_cmapss` becomes `logger.info`; `PROCESSED.mkdir` moves inside it | Logging, no side effect at import | No |
| The `GroupKFold` loop becomes `engine_folds` / `out_of_fold` | Testable (no engine on both sides) | No: same splitter, same seed |
| `random_state` is appended to the LightGBM parameters last | Seed lives in one place | No: parameter order is irrelevant |
| **New:** the trained model is saved as a bundle with its healthy baseline, health sensors and feature order | Serving must rebuild the same features (training-serving skew) | No |

Not packaged yet: `simulate`, `policy_cost`, `fresh_fleet` (Section 4), `psi`, `weekly_rows` and Section 7's knowledge
layer. They come with the tasks that need them (C3, C5, E1).

## Evidence
- Package on this machine: test RMSE **12.926184**, PHM **225.7707**, OOF RMSE **12.306861**.
- The notebook run on the same machine with the same versions gives the same values to 6 decimals, so packaging changed nothing.
- Two runs in a row give the same result to 4 decimals: the pipeline is deterministic on one machine.

## Open issue
The notebook's published numbers (13.2 / 238, build machine) differ from both runs here (12.9 / 226, Apple M3, arm64),
with the same library versions, same seed and byte-identical data (holdout SHA-256 `5c2b2cf46966…` matches).
Hypothesis: CPU architecture (arm64 against x86_64). Test in Step 3: train inside a `linux/amd64` container.
Until then, the reproduction tolerance (D-02) must absorb at least 0.3 RMSE and 12 PHM across machines.
