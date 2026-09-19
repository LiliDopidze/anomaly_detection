# PON anomaly detection

A synthetic-first pipeline with an explicit source adapter, data-quality checks,
causal features, calibrated evidence channels, incident queues and locked
assessment. [SYNTHETIC_REVIEW.md](SYNTHETIC_REVIEW.md) describes the evidence,
assumptions and which parts of the approach document are implemented.

## Run the complete workflow

Python 3.10 or later, from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m telco_anomaly.pipeline all
```

This generates synthetic data, validates it, creates the canonical model pack,
fits the models, freezes thresholds before opening development labels, and writes
comparisons and incident queues. It **does not open the final test**. A failed
selection writes a STOP status, not a deployable model.

Alternatively, open Jupyter and run the five notebooks in order:

```bash
python -m jupyter notebook
```

1. `01_generate_and_validate.ipynb`: generate, validate, adapt and separate truth.
2. `02_eda_and_seasonality.ipynb`: training-only EDA and seasonal evidence.
3. `03_features.ipynb`: inspect the same features used by scoring.
4. `04_develop_baselines.ipynb`: compare channels, build queues and select or STOP.
5. `05_final_evaluation.ipynb`: explicit final assessment and future inference;
   both are disabled by default.

## Where is the adapter?

**`src/telco_anomaly/adapter.py`**, configured by **`configs/adapter.yml`**.
It maps native column names, units and interval/cumulative counter semantics;
handles reviewed vendor overrides; validates identifiers, timestamps and dated
topology; and maps alarm codes to canonical event families. It does not infer
what a counter means or search folders for data. Unknown fields and codes stop
conversion. Invalid numeric measurements become missing with quality codes.

The synthetic workflow calls `write_pack()` with explicit telemetry and inventory
tables. A different source uses the same function with a reviewed mapping:

```python
from telco_anomaly.adapter import load_mapping, write_pack

pack = write_pack(
    telemetry=native_telemetry,   # pandas DataFrame
    inventory=native_inventory,
    events=native_events,        # DataFrame, or None when unavailable
    output="data/operator_pack",
    mapping=load_mapping("configs/adapter.yml"),  # review for the actual source
    metadata={
        "start": "2025-01-01T00:00:00Z",
        "days": 90,
        "sample_minutes": 15,
        "n_onts": 96,
    },
)
```

For example, set a received-power field's source unit to `mW` to convert to dBm.
Set FEC's kind to `cumulative` only when the source counts corrected codewords
cumulatively; the adapter differences adjacent samples and marks gaps/restarts
as unavailable. Corrected bits or bytes are **not** interchangeable codewords.
Naive timestamps require the declared time zone; ambiguous DST times are rejected.
The supplied mapping is for this synthetic source, not a universal vendor mapping.

## Files you use

```text
configs/
    pipeline.yml              # Shared paths and operational policy
    adapter.yml               # Reviewed source semantics
    synthetic.yml             # Generator assumptions
    synthetic_experiment.yml  # Calibration and qualification gates
notebooks/                    # Five entry points
src/telco_anomaly/
    adapter.py                # Source -> canonical model pack
    synthetic.py              # Synthetic physical/measurement processes
    synthetic_validation.py   # Generator checks and sampling audits
    features.py               # Grid, gaps, seasonality and evidence channels
    synthetic_pipeline.py     # Baseline features, detectors and event metrics
    operations.py             # Incident consolidation, disposition and scope ranking
    evaluation.py             # Matching and uncertainty helpers
    pipeline.py               # One CLI, frozen artifacts, ledgers and inference
    __init__.py
tests/                        # Correctness, causality and end-to-end checks
```

Generated `data/`, evaluation-only `evaluation/`, and `outputs/` are excluded
from Git. No legacy workflow, public-data downloader or source dataset is needed.
The original work remains on `main` and in Git history.

## Reuse, qualification and inference

All paths come from `configs/pipeline.yml`. For a revised experiment, choose new
pack, truth and run paths. Existing prepared data is reused only when its hashes
match; model runs are never overwritten. The notebooks can display a verified
existing run. `pipeline all` expects a fresh run directory.

Each run includes label-free thresholds, seasonal decisions, saved scores,
per-channel and combined incident queues, per-mechanism recall/support,
a model card, and either a selected configuration or a STOP status.
Selection attempts and final-test openings are recorded beside evaluation truth.
The default development-attempt limit is three per truth root.

For an explicitly approved final assessment:

```bash
python -m telco_anomaly.pipeline holdout
```

For subsequent observations, after a model qualifies:

```bash
python -m telco_anomaly.pipeline infer \
  --input-pack data/new_observations \
  --output outputs/new_observations
```

Inference uses frozen preprocessing, models and thresholds, requires observations
after the original experiment window, and does not read truth. It cannot be used
to bypass the final-test opening. Supply history for rolling-window warm-up;
a new operator needs local calibration, and a changed cadence is refused.

```bash
python -m pytest
```

This is a complete executable synthetic development path, not an industry-ready
product. Scope probabilities remain uncalibrated. Independent generator testing,
operator-transfer evidence, public-data qualification and shadow deployment are
not claimed.
