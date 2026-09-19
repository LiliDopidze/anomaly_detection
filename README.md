# PON anomaly detection — Branch 2 synthetic restart

A small, reproducible workflow for generating PON optical-loss scenarios,
auditing the data and comparing two causal detection baselines. No external
data download or Google Drive access is required for this stage.

**Start with [SYNTHETIC_REVIEW.md](SYNTHETIC_REVIEW.md).** It explains the original
generator review, evidence, assumptions, mathematical choices and actual results.
Synthetic data is useful for development; it does not establish field accuracy.

## Run

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m jupyter notebook
```

Open the five notebooks in `notebooks/` in order. Notebook 1 generates and checks
the dataset. Notebook 2 performs EDA, including seasonality. Notebook 3 explains
the features. Notebook 4 compares baselines and either selects a qualified model
or reports STOP. Notebook 5 leaves the final test period sealed by default.

For generation without Jupyter:

```bash
python -m telco_anomaly.synthetic --config configs/synthetic.yml
python -m pytest
```

The defaults generate 96 ONTs over 90 days. Change `configs/synthetic.yml` before
generating another scenario, and choose a new output directory. Keep its path
aligned with `dataset` in `configs/synthetic_experiment.yml`. Generation and
model development refuse to overwrite existing runs. Existing notebooks reuse
saved outputs; change paths to run a revised experiment.

`fec_count` means **corrected codewords within the sampling interval** in v5.
It is not a cumulative counter and is not interchangeable with the old v4 field
without confirming semantics. Model input excludes truth and latent parameters.

## Small active workflow

- `notebooks/`: five readable entry points; calculations live in Python modules.
- `src/telco_anomaly/synthetic*.py`: generation, validation and baseline workflow.
- `configs/synthetic*.yml`: scenario and experiment settings.
- `tests/test_synthetic_stage.py`: statistical and temporal correctness checks.
- `reference/`: the original submitted generator, preserved unchanged.
- `data/` and `outputs/`: generated locally, never committed.

The earlier code/configuration and `notebooks/legacy/` remain for reproducibility.
They are not prerequisites for the new workflow. `METHODOLOGY.md` is the broader
proposed roadmap; `SYNTHETIC_REVIEW.md` identifies what this stage implements.
Public-data integration and advanced localisation are deferred.

## Current result

All 18 default dataset invariants and 130 project tests pass. On the development
period, robust detection finds 27/40 eligible events and Isolation Forest 30/40.
Both fail the configured alert-burden gates. No model is selected; final test
performance has not been examined. These are assumption-dependent development
results, not a production recommendation or a comparison on the old dataset.
