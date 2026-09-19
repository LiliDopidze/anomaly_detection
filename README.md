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

## Repository structure

```text
configs/                    # Generator and experiment settings (two files)
notebooks/                  # Five numbered notebooks
src/telco_anomaly/
    synthetic.py            # Generate measurements and separate truth
    synthetic_validation.py # Validate and audit generated data
    synthetic_pipeline.py   # Features, baselines and evaluation workflow
    evaluation.py           # Required matching and uncertainty helpers
    __init__.py
tests/                      # Tests for this workflow only
README.md                   # Setup and running instructions
SYNTHETIC_REVIEW.md          # Methodology, evidence, assumptions and results
pyproject.toml              # Installation and required dependencies
```

`data/` and `outputs/` are created locally and excluded from Git. They contain
simulation data and run results, not source code. Tests remain because they
protect causality, physical invariants and final-evaluation safeguards.

The earlier implementation remains on `main`. The original submitted generator
and broader design are preserved in the [pre-cleanup commit](https://github.com/LiliDopidze/anomaly_detection/tree/a4f37dd0e5d5702e863ca6ae9883c4173a72bcaa).
They are not needed to run this pipeline. Public-data integration and advanced
localisation are deferred.

## Current result

All 18 default dataset invariants and 15 focused tests pass. On the development
period, robust detection finds 27/40 eligible events and Isolation Forest 30/40.
Both fail the configured alert-burden gates. No model is selected; final test
performance has not been examined. These are assumption-dependent development
results, not a production recommendation or a comparison on the old dataset.
