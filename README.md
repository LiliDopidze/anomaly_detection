# Optical anomaly detector

Six explicit stages for sustained optical-degradation detection:
**adapter → validation → features → detectors → incidents → evaluation**.

This is a tested research implementation with production-oriented safeguards,
not a field-qualified production detector. Synthetic validation is development evidence, not proof of operator performance. See [METHOD.md](METHOD.md) for the
mathematics, critical design decisions, evidence and limitations.

## Start here

Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e ".[dev,explain]"
jupyter notebook
```

For local analysis, run the notebooks in order:

0. `00_generate_and_diagnose.ipynb`: generate and inspect native data; check structure, missingness and development faults.
1. `01_canonical_eda.ipynb`: adapt, validate and inspect distributions, seasonality and relationships.
2. `02_feature_distributions.ipynb`: inspect causal features and missingness.
3. `03_detector_tuning.ipynb`: fit both tiers and tune incident persistence.
4. `04_evaluation.ipynb`: inspect misses and workload; final assessment defaults off.
5. `05_end_to_end_demo.ipynb`: reproduce saved scores and see company adaptation.
6. `06_feature_importance.ipynb`: inspect global/local SHAP and feature correlations.

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
notebooks/                          # Seven ordered local data-science notebooks
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
    multivariate.py                 # Five nested telemetry feature sets
    workflow.py                     # Canonical storage and per-ONT feature replay
    detectors.py                    # StatisticalDetector and IsolationForestDetector
    incidents.py                    # IncidentManager: persistent hysteresis state
    evaluation.py                   # Evaluator: one-to-one matching and metrics
    explanations.py                 # Validation-only SHAP; no feature removal
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
  (2,488,320 rows; 12 measurement columns plus time/device).
- `canonical_development.parquet`: adapted/validated long telemetry before the final period.
- `feature_set_comparison.csv`: best validation policy for each telemetry set.
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
corrected/uncorrectable/total FEC interval counts in
each direction. The adapter recognises all 12 measurements. The pipeline compares
Rx-only, both Rx directions, Tx/Rx relationships, FEC, and temperature cumulatively.
BER is internal to the simulation and is not exported or used as a feature. Topology is context only. Operational events, voltage,
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

Default local outputs are under `<repository>/outputs/optical_v11/`.
All outputs stay outside Git. See [RUN_GUIDE.md](RUN_GUIDE.md) for exact commands.

### Inspect first, then adapt

Notebook 00 reports dataset size, the measurement dictionary, topology, per-ONT
missingness and gap lengths, distributions, and development fault durations,
warning opportunities and effect sizes, plus generator-specific statistical
qualification. Notebook 01 adds canonical seasonality,
residual statistics and cross-measurement EDA. Full-data
checks are structural only; detailed final-test inspection is excluded. CSV reports
and structural checks are saved under `<configured output>/eda/`.

`adapter.py` defines canonical semantics without synthetic column defaults.
`sources.py` supplies the synthetic mapping; for a subset use
`synthetic_adapter(["rx_dbm"])` for the standalone Rx baseline. The five-set
synthetic experiment uses the full mapping. Every explicitly mapped column must exist.
Company mappings declare names, units and gauge/interval-count semantics. Cumulative
counters must be converted with reset and gap handling before adaptation. The
adapter accepts temperature-only data; the downstream baseline separately requires
usable Rx data, and fitting checks the amount of baseline history.

After updating an existing checkout, choose a **new output folder** before fitting
again. Existing models and final-test manifests remain tied to their original code;
do not overwrite their hashes or reuse them as a new experiment.

## Five controlled telemetry comparisons

| Feature set | Cumulative inputs | Feature count |
|---|---|---|
| `rx_only` | Downstream Rx shape, level and multi-window features | 12 |
| `both_rx` | Add upstream Rx | 18 |
| `tx_rx` | Add downstream and upstream Tx minus Rx loss proxies | 30 |
| `fec` | Add directional corrected and uncorrectable FEC fractions and error-interval frequency | 46 |
| `temperature` | Add ONT and OLT optical-module temperatures | 52 |

Every added channel contributes rolling standardised level, EWMA slope and rolling
variability. References are fitted per ONT on training data only. Fixed FEC log
transforms and precision floors are explicit modelling assumptions in METHOD.md.
Unseen entities abstain in the feature classes; missing required channels produce
missing scores, never healthy imputation. The synthetic fitting workflow requires
all measurements needed to compare all five sets.

The statistical comparator uses the downstream EWMA slope alone. Each feature set has its own Isolation
Forest and score calibration; combined scores require both tiers to be available.
Validation compares early recall, nuisance workload, delay and coverage. The configured `model.feature_set` is retained (default: all 52 features).
Comparisons do not automatically choose a smaller set or remove features. A failed nuisance budget is
flagged, even when an experimental best candidate is saved. Only that frozen
candidate is eligible for the final assessment; final results are not used to
choose among telemetry sets.

Notebook 01 saves canonical training statistics under `eda/`, including daily/weekly
lag correlations and chronological seasonality comparisons. Notebook 02 illustrates
all feature groups on one training ONT; notebook 03 runs the full fleet comparison.

## Feature explanations

Notebook 06 saves `explanations/<feature_set>/` inside the configured output:
`importance.csv`, global and high-score SHAP tables, feature samples, score
reconstruction checks, correlations and sampling metadata. All five fitted forests
remain in `model.joblib` for review. SHAP explains raw Isolation Forest scores,
not incident decisions or physical causes; it never removes features.
The full feature catalogue and source/assumption mapping are in METHOD.md.

Six-hour optical features require uninterrupted history. Missing readings restart
that history, which can substantially reduce score coverage. Notebook 02 reports
coverage, and notebook 04 reports missed faults as well as nuisance alerts.

Notebook 01 includes per-ONT Pearson/Spearman correlations and selected-lag
autocorrelations, both raw and after daily-pattern removal, alongside seasonality.
Reports include paired counts and are saved under `eda/`. No cutoff drops features.
FEC display labels use received/corrected/uncorrectable blocks, with upstream and
downstream explicit; a block means one FEC codeword per reporting interval.

The generator exports power, temperature and FEC measurements. Receiver
parameters are independent of impact-label thresholds. Impact requires three consecutive
low-power readings and is dated at the confirming reading; it remains a proxy,
not verified customer-service loss. Receiver offsets and FEC dispersion are explicit
simulation assumptions. Variance-shift delay is reported from physical onset,
separately from observable-onset delay. Use a fresh output folder; do not compare
old/new results as the same benchmark or reuse old fitted models with new code.
