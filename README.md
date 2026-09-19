# Optical anomaly detector

Six explicit stages for sustained optical-degradation detection:
**adapter → validation → features → detectors → incidents → evaluation**.

This is a tested research implementation with production-oriented safeguards,
not a field-qualified production detector. The current synthetic validation
policy fails its nuisance-workload target. See [METHOD.md](METHOD.md) for the
mathematics, critical design decisions, evidence and limitations.

## Start here

Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e ".[dev]"
jupyter notebook
```

Run the notebooks in order:

1. `01_generator_eda.ipynb`: generate and inspect baseline telemetry.
2. `02_feature_distributions.ipynb`: inspect causal features and missingness.
3. `03_detector_tuning.ipynb`: fit both tiers and tune incident persistence.
4. `04_evaluation.ipynb`: inspect misses and workload; final assessment defaults off.
5. `05_end_to_end_demo.ipynb`: reproduce saved scores and see company adaptation.

Or run development from Python:

```python
from optical_anomaly.pipeline import develop

run = develop("configs/config.yaml")
```

No data download is needed. The generator creates the source measurements and a
separate truth file. A new experiment needs a new `output` path in the configuration.
`prepare` reuses an existing dataset only when its saved settings match; `develop`
refuses to overwrite a fitted model. Final assessment is a separate explicit call:

```python
from optical_anomaly.pipeline import final_evaluation

# Only after fixing the model and completing validation error analysis:
metrics = final_evaluation(run)
```

Once final data has been inspected, it is no longer an untouched test for further
tuning. Checksums and an opening marker prevent accidental reuse, not deliberate
filesystem changes. Load joblib model files only from trusted sources.

## Repository structure

```text
configs/config.yaml                 # One readable experiment configuration
notebooks/                          # Five ordered data-science notebooks
src/optical_anomaly/
    generator.py                    # Physical state, measurement and fault truth
    adapter.py                      # TelemetryAdapter: names, units, timezone
    validation.py                   # DataValidator: causal resampling, explicit gaps
    splitting.py                    # TemporalSplit: train/calibration/validation/test
    mathematics.py                  # Small independently tested formulas
    features.py                     # FeatureEngineer: normalisation and shape features
    detectors.py                    # StatisticalDetector and IsolationForestDetector
    incidents.py                    # IncidentManager: persistent hysteresis state
    evaluation.py                   # Evaluator: one-to-one matching and metrics
    pipeline.py                     # Fit/calibrate/tune/save/final orchestration
    __init__.py
tests/                              # Mathematics, causality, state, matching, integration
METHOD.md                           # Scientific rationale and limitations
pyproject.toml
README.md
```

One module per responsibility is enough here. Single-file subpackages and an
abstract detector superclass would add navigation without helping the current
implementation. Split a module when it develops genuinely different responsibilities.

There are no copied train/validation/test folders. Timestamp boundaries define the
splits; past-only rolling history can cross a boundary without future leakage.
Generated artifacts live under the configured output directory, ignored by Git:

- `telemetry.parquet`: source measurements; `ground_truth.parquet`: separate labels.
- `settings.json`, `manifest.json`: settings, boundaries and frozen fingerprints.
- `development_features.parquet`, `validation_scores.parquet`: development evidence.
- `validation_comparison.csv`, `validation_incidents.csv`, `validation_faults.csv`.
- `model.joblib`: feature references, both detectors and the chosen incident policy.

Test outputs appear only after explicitly opening final assessment. Previous
implementations remain in Git history; Branch `main` is unchanged.

## Company data

Map your native DataFrame explicitly; BER is optional:

```python
from optical_anomaly.adapter import TelemetryAdapter
from optical_anomaly.validation import DataValidator

adapter = TelemetryAdapter(
    timestamp_column="sample_time",
    entity_column="device_id",
    metrics={"received_power": "rx_power_dbm"},
    units={"rx_power_dbm": "dBm"},
    timezone="Europe/London",
)
telemetry = DataValidator("5min").transform(adapter.transform(native))
```

Then use the stage classes with reviewed local chronological periods: fit
`FeatureEngineer` and `IsolationForestDetector` on training data; calibrate both
detectors on a later mostly normal period; tune incident rules on validation.
The `develop` convenience function is specifically for the synthetic experiment.

The canonical schema is `timestamp, entity_id, metric_name, value`. Validation
adds `observed`; the feature stage uses Rx power. Company-specific assumptions
include units, timezone, cadence, representative healthy history, daily seasonality,
measurement precision, impact definition and acceptable workload. Company agnostic
means portable mathematics plus local calibration, not universal thresholds.

Feature extraction performs causal batch replay with historical context. It is not
a bounded-memory streaming feature service. `IncidentManager` does retain state
across consecutive chunks; its output includes new closures and active snapshots.
Upsert incidents by `incident_id`, and persist the manager if restarting a process.

Run the focused checks with `python -m pytest`.
