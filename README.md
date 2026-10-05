# Optical anomaly detector

Six explicit stages for sustained optical-degradation detection:
**adapter → validation → features → detectors → incidents → evaluation**.

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
METHOD.md                           # rationale and limitations
pyproject.toml
README.md
```
