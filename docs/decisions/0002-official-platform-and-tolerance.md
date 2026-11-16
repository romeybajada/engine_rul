# ADR 0002: One official platform (`linux/amd64`) and a two-tier reproduction tolerance

**Status:** accepted · **Date:** 2026-11-16 · **Entry:** 1, Step 3 · **Decision log:** D-02, D-05

## Context
The fresh copy must reproduce the notebook's 13.2 / 238. The same code, lock file, seed and byte-identical data
(holdout SHA-256 `5c2b2cf46966…`) give different numbers on different platforms, but identical numbers run after run
on any one platform:

| Platform | Test RMSE | PHM | OOF RMSE |
|---|---|---|---|
| macOS, ARM (Apple M3) | 12.9262 | 225.77 | 12.3069 |
| Linux, ARM (Docker `linux/arm64`) | 13.0039 | 230.66 | 12.3141 |
| Linux, x86 (Docker `linux/amd64`, emulated) | **13.1659** | **238.05** | 12.3339 |
| Notebook, build machine (published) | 13.2 | 238 | 12.3 |

Each row was run twice, with identical results both times. Spread on one platform: **0**. Spread across platforms:
**0.24 RMSE, 12.3 PHM**.

Cause: floating-point rounding depends on the order of operations, and each platform's LightGBM build (chip
instructions such as fused multiply-add, the OS maths library, thread summation order) rounds slightly differently.
Gradient boosting amplifies this: a last-digit difference can flip a near-tied split, and every later tree builds on
it. This is the likely mechanism; it was not proven at the compiler level.

So "the same model" has no single number unless the platform is fixed. A tolerance wide enough to cover all platforms
(≥ 0.25 RMSE) would also hide real regressions of that size.

## Decision
1. **Official platform: `linux/amd64`.** Training, the serving container and the CI gate all run there.
   - It is where the model is served: cloud hosts and CI runners are Linux x86.
   - It reproduces the notebook (13.17 / 238 against 13.2 / 238), which shows that packaging changed nothing.
   - One platform for training and serving removes the platform gap as a hidden source of training-serving skew.
2. **Official reference:** test RMSE **13.1659**, PHM **238.05**, from the `linux/amd64` run of the platform
   experiment above. (`reports/` is git-ignored, and the local `train_metrics.json` is from the arm64 Mac, so the
   reference is recorded here.)
3. **Two-tier tolerance:**

   | Tier | Where | Band around the reference | On breach |
   |---|---|---|---|
   | Gate | `linux/amd64` (CI, container) | ±0.01 RMSE, ±0.5 PHM | **Fail** |
   | Info | any other platform (e.g. a dev Mac) | ±0.25 RMSE, ±13 PHM | Warn |

   - The gate band is not zero, which leaves room for rounding in how the numbers are printed and stored. It is still
     about 25× smaller than the cross-platform spread, so running on the wrong platform also fails it.
   - The info band is the measured cross-platform spread, rounded up. It only tells a developer "plausibly the same
     model"; it never decides anything.

## Consequences
- **Good:** a code, data or library change that moves the result by more than 0.01 RMSE is caught. The official
  numbers match both production and the notebook.
- **Cost:** on an ARM Mac, `linux/amd64` runs under emulation, so local official runs and image builds are slower.
- **Accepted risk:** a deliberate library upgrade (for example LightGBM) may legitimately move the numbers past the
  gate. That is handled as a deliberate re-baseline: re-run on `linux/amd64`, record the new reference here, and
  commit the change. The gate is never silently widened.
- **Comparing models:** an improvement only counts if it is measured on `linux/amd64` and beats the reference by more
  than the gate band, across more than one seed. A single-seed gain of about 0.1 is within seed-to-seed variation.

## Alternatives rejected
- **Exact match (0 tolerance):** brittle against harmless storage/print rounding, with no gain over ±0.01.
- **One wide band (±0.25 / ±13) everywhere:** works on every machine but lets a real regression of 0.2 RMSE pass.
- **`linux/arm64` as official:** cheaper servers (for example Graviton), but it does not reproduce the notebook
  baseline, and x86 is still the CI default.
