# engine-rul

Remaining useful life (RUL) of turbofan engines, from NASA's C-MAPSS simulation (Task 22, ARI3217).
The model is the one given in `notebooks/22_turbofan_rul.ipynb`.

- **Data:** Kaggle `behrad3d/nasa-cmaps`, CC0. Simulated engines, no personal data.
- **Training fleet:** FD001 (one regime, one fault mode). **Frozen holdout:** the official FD001 test files, SHA-256 `5c2b2cf46966…`.
- **Target:** RUL capped at 125 cycles. **Window:** 30 cycles. Cost and capacity assumptions: `config/config.yaml`.

## Install

Python 3.12. From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install
```

## Run

```bash
rul-fetch               # downloads the whole dataset once into data/raw/cmapss (run this first)
rul-validate            # data contract on the raw files: exit 0 if valid
rul-train               # trains, scores the FD001 test files, writes models/model.joblib and reports/train_metrics.json
python scripts/data_check.py   # the counts behind the diary's data-check table -> reports/data_check.json
make test               # unit tests (the service tests that need the model skip until rul-train has run)
make serve              # POST /rul on http://localhost:8000 (docs at /docs)
```

`DATA_DIR` in the environment overrides where the data lives (default `data/`).

## Serve

`POST /rul` is stateless (D-04 Option A): the client sends the engine's last flights, the service keeps no history.

- **Request:** `engine_id` and `flights`, 1–30 rows of `cycle`, `setting1..3`, `s1..s21`.
  - Cycles must be in order, with no gaps or repeats.
  - At cycle *c*, the request must hold the last min(30, *c*) flights. The rolling features count rows, not cycles.
  - Values must be finite numbers, with no unknown fields.
- **Response:** `rul` (in cycles, capped at 125 in training), the `cycle` it refers to, and `model_version` (the first 12 hex digits of the bundle's SHA-256).
- **422** is returned for an incomplete window, or for a flight regime that the training fleet never flew (for example FD002's).
- **Window vs full history:** the features agree to within 1e-14, but that rounding can flip a near-tied tree split.
  - On the FD001 test rows, 1.3% of predictions differ, by at most 1.6 cycles.
  - The official score moves from 12.9262 to 12.9256 RMSE, inside ADR 0002's gate.

## Results (Apple M3, arm64)

| | Notebook (build machine) | This package |
|---|---|---|
| Test RMSE (last cycle, 100 engines) | 13.2 | 12.9 |
| Test PHM score (sum) | 238 | 226 |
| OOF RMSE (capped, all cycles) | 12.3 | 12.3 |
| Training time (OOF + final fit) | — | 6.8 s |
