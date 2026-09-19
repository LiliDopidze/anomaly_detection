# PON optical-loss anomaly detection

A small data-science workflow for detecting sustained deterioration in optical
power. Start with synthetic data, inspect errors, and adapt the same calculations
to an operator's measurements. This is an experimental baseline, not a validated
production alarm system or a detector for every telecom fault.

## Run the workflow

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e ".[dev]"
jupyter notebook
```

Run these notebooks in order:

1. `notebooks/01_data_and_eda.ipynb`: generate data, validate it, inspect missingness,
   optical traces and daily profiles.
2. `notebooks/02_fit_and_compare.ipynb`: fit the detector and compare it with
   Isolation Forest and an illustrative static power threshold.
3. `notebooks/03_error_analysis.ipynb`: inspect misses and early warnings; optionally
   run stress tests, then explicitly open the final assessment when ready.

There is no dataset to download. Notebook 1 creates `data/synthetic_pon_v5` from
`configs/synthetic.yml`. It can take a few minutes. Results go to
`outputs/simple_model`. Generated data and results are ignored by Git. Change
output paths for new experiments; existing runs are not overwritten.

## What the model does

It learns each device's normal optical power and variability, then smooths
standardised drops with an exponentially weighted moving average (EWMA). Two
successive high scores open a warning; two low scores confirm recovery. Missing
telemetry is unknown, not normal. New devices need a reference period.

The two main experiment settings are smoothing duration and threshold sensitivity,
visible in notebook 2. Higher sensitivity values mean a higher threshold.
References use the first 40% of time, thresholds the next 30%, development the next
15%. Final performance assessment uses the remaining 15% only when explicitly
requested in notebook 3. Simulator structural validation is separate from that
performance assessment.

## Adapt your company's data

`src/telco_anomaly/adapter.py` is the adapter. It accepts a DataFrame and explicitly
maps names, timezone and optical-power units. It does not infer vendors or units.
Downstream receive power is required; upstream receive power is optional.

```python
from telco_anomaly.adapter import adapt
from telco_anomaly.model import fit_reference, score, calibrate, warnings

# native is your existing DataFrame; timestamps here are local London time.
data = adapt(
    native,
    columns={
        "timestamp_utc": "sample_time",
        "ont_id": "device_id",
        "rx_power_dbm": "downstream_rx",
        "olt_rx_power_dbm": "upstream_rx",
    },
    units={"rx_power_dbm": "dBm", "olt_rx_power_dbm": "dBm"},
    timezone="Europe/London",
)

# Choose chronological periods using known operating history.
training = data.loc[data.timestamp_utc < fit_end]
model = fit_reference(training, cadence_minutes=15, smoothing_hours=1)
scored = score(data, model)
calibration = scored.loc[
    (scored.timestamp_utc >= fit_end)
    & (scored.timestamp_utc < calibration_end), "score"
]
threshold = calibrate(calibration, sensitivity=6)
recovery = min(threshold, (threshold + calibration.median()) / 2)
future = scored.loc[scored.timestamp_utc >= calibration_end]
alerts = warnings(
    future, threshold, cadence_minutes=15,
    recovery_fraction=recovery / threshold,
)
```

`fit_end` and `calibration_end` are timezone-aware timestamps chosen for your data.
Most reference/calibration readings must represent the intended healthy regime.
Use the real poll cadence, verify measurement resolution and noise, and review
power-class differences. The default 0.15 dB spread floor is an assumption.
`score` is a chronological batch replay: include recent history for EWMA warm-up.
`warnings` processes a complete evaluation window; separate calls do not preserve
warning state. It is not a live monitoring service.

Company agnostic means reusable calculations with explicit local calibration.
It does not mean a universal threshold, no onboarding, or validated transfer to an
unseen operator. Sparse devices abstain: at least 100 valid reference readings per
signal are required, but that minimum alone does not establish a good reference.

## Files worth keeping

| File | Purpose |
|---|---|
| `configs/synthetic.yml` | Simulation assumptions; the only YAML configuration |
| `src/telco_anomaly/synthetic.py` | Reproducible generator and separate fault truth |
| `src/telco_anomaly/synthetic_validation.py` | Mathematical and data-quality checks |
| `src/telco_anomaly/adapter.py` | Explicit field/unit mapping and gap report |
| `src/telco_anomaly/model.py` | Reference, features, scoring, threshold and warnings |
| `src/telco_anomaly/experiment.py` | Chronological comparison and final assessment |
| `src/telco_anomaly/evaluation.py` | Fault matching, misses, lead time and false alarms |
| `tests/` | Focused checks for leakage, units, missingness and evaluation errors |
| `SYNTHETIC_REVIEW.md` | Modelling rationale, evidence, assumptions and results |

No deployment framework, channel registry, topology localisation, operational
queue engine or layered adapter configuration is needed for this experiment.
Previous work remains in Git history. Branch `main` is unchanged.

Run checks with `python -m pytest`. See [the methodology](SYNTHETIC_REVIEW.md) for
results and their limitations. Public-data integration remains a later step.
