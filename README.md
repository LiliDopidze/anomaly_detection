# Optical anomaly detector

Six explicit stages for sustained optical-degradation detection:
**adapter → validation → features → detectors → incidents → evaluation**.

This is a tested research implementation with production-oriented safeguards,
not a field-qualified production detector. Synthetic validation is development evidence, not proof of operator performance. See [METHOD.md](METHOD.md) for the
mathematics, critical design decisions, evidence and limitations.

## Start here

Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e ".[dev]"
jupyter notebook
```

For **Google Colab**, open [00_colab_start.ipynb](notebooks/00_colab_start.ipynb)
and follow its setup and output-location cells. It runs the development pipeline
and defaults to saving results in your Google Drive. Use a CPU runtime.

For local analysis, run the notebooks in order:

1. `01_generator_eda.ipynb`: inspect and qualify native synthetic data, then preview canonical adaptation.
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
    generator.py                    # Expanded telemetry, static topology and fault truth
    optics.py                       # GPON-inspired directional FEC and invariants
    adapter.py                      # Canonical definitions and explicit adaptation
    sources.py                      # Synthetic source mapping
    diagnostics.py                  # Native-data EDA and qualification checks
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

- `telemetry.parquet`: 96 ONTs × 90 days × five-minute samples by default
  (2,488,320 rows; 14 measurement columns plus time/device).
- `ground_truth.parquet`: separate onset, visibility, impact and repair labels.
- `topology.parquet`: static `entity_id`, `olt_id`, `pon_port_id`, `splitter_id` mapping.
- `generation_checks.json`: data consistency and FEC count checks.
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
from optical_anomaly.validation import DataValidator, require_downstream_rx

adapter = TelemetryAdapter(
    timestamp_column="sample_time",
    entity_column="device_id",
    metrics={"received_power": "rx_power_dbm"},
    units={"rx_power_dbm": "dBm"},
    kinds={"rx_power_dbm": "gauge"},
    timezone="Europe/London",
)
telemetry = DataValidator("5min").transform(adapter.transform(native))
require_downstream_rx(telemetry)
```

Then use the stage classes with reviewed local chronological periods: fit
`FeatureEngineer` and `IsolationForestDetector` on training data; calibrate both
detectors on a later mostly normal period; tune incident rules on validation.
The `develop` convenience function is specifically for the synthetic experiment.

The generator includes downstream/upstream Rx, ONT/OLT Tx, ONT/OLT temperatures,
two pre-FEC BER proxies, and corrected/uncorrectable/total FEC interval counts in
each direction. The adapter recognises all 14 measurements. The baseline model
still uses downstream Rx; extra measurements are available in EDA for future
feature comparisons. Topology is context only. Operational events, voltage,
bias current, traffic and topology-based incident logic are not added.

The canonical schema is `timestamp, entity_id, metric_name, value`. Validation
adds `observed`; the feature stage uses Rx power. Company-specific assumptions
include units, timezone, cadence, representative healthy history, daily seasonality,
measurement precision, impact definition and acceptable workload. Company agnostic
means portable mathematics plus local calibration, not universal thresholds.

Feature extraction performs causal batch replay with historical context. It is not
a bounded-memory streaming feature service. `IncidentManager` does retain state
across consecutive chunks; its output includes new closures and active snapshots.
Upsert incidents by `incident_id`, and persist the manager if restarting a process.

Run the focused checks with `python -m pytest` after the editable installation.
Tests own their configuration and do not require your current working directory
to be the repository root. See [RUN_GUIDE.md](RUN_GUIDE.md) for Windows/macOS setup,
notebook order, output interpretation and rerunning an experiment.

Default local outputs are under `<repository>/outputs/optical_v7/`. The Colab
starter saves to `/content/drive/MyDrive/anomaly_detection/<RUN_NAME>/` with
`SAVE_TO_DRIVE=True`, or `/content/anomaly_detection/outputs/<RUN_NAME>/` otherwise.
All outputs stay outside Git. See [RUN_GUIDE.md](RUN_GUIDE.md) for exact commands.

### Inspect first, then adapt

Notebook 01 reports dataset size, the measurement dictionary, topology, per-ONT
missingness and gap lengths, distributions, daily patterns, residual statistics,
and development fault durations, warning opportunities and effect sizes. Full-data
checks are structural only; detailed final-test inspection is excluded. CSV reports
and structural checks are saved under `<configured output>/eda/`.

`adapter.py` defines canonical semantics without synthetic column defaults.
`sources.py` supplies the synthetic mapping; for a subset use
`synthetic_adapter(["rx_dbm"])`. Every explicitly mapped column must exist.
Company mappings declare names, units and gauge/interval-count semantics. Cumulative
counters must be converted with reset and gap handling before adaptation. The
adapter accepts temperature-only data; the downstream baseline separately requires
usable Rx data, and fitting checks the amount of baseline history.

After updating an existing checkout, choose a **new output folder** before fitting
again. Existing models and final-test manifests remain tied to their original code;
do not overwrite their hashes or reuse them as a new experiment.
