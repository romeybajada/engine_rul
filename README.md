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
make test               # unit tests (no data needed)
```

`DATA_DIR` in the environment overrides where the data lives (default `data/`).

## Results (Apple M3, arm64)

| | Notebook (build machine) | This package |
|---|---|---|
| Test RMSE (last cycle, 100 engines) | 13.2 | 12.9 |
| Test PHM score (sum) | 238 | 226 |
| OOF RMSE (capped, all cycles) | 12.3 | 12.3 |
| Training time (OOF + final fit) | — | 6.8 s |
