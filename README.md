# Time-series anomaly detection blueprint — NAB pilot

This branch starts a company-agnostic, explicitly configured workflow. The active
entry point is **[01_data_assessment.ipynb](notebooks/01_data_assessment.ipynb)**.
It loads two pinned NAB development series, audits the observations and examines
temporal behaviour. It does not fit models or select algorithms automatically.
Expected-behaviour modelling will be a separate Notebook 02, after its design is agreed.

## Run Notebook 01

From this repository root in WSL or another Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name nab-blueprint --display-name "NAB blueprint"
code .
```

Open `notebooks/01_data_assessment.ipynb`, select the **NAB blueprint** kernel and
run the cells in order. The first load downloads about 0.9 MB of pinned public
source files into ignored `data/nab/`; later loads verify hashes and use the cache.
Raw data and generated outputs are not committed. No NAB detector package is installed.

To validate or execute without VS Code:

```bash
python -m pytest tests/test_nab_blueprint.py
python tools/execute_nab_assessment.py
```

The runner uses the calling Python interpreter and saves successful outputs.
Jupyter requires permission to open local kernel sockets. For Python.org macOS
installations, an HTTPS certificate error on first download requires completing
that Python installation's certificate setup; do not disable verification.

## The small active pipeline

| Component | Responsibility |
|---|---|
| `configs/nab_pilot.json` | Pinned source files and hashes; metric semantics, cadence assumptions and experiment scope |
| `src/ts_anomaly/nab.py` | Download and verify NAB files; retain source order and keep annotations separate |
| `src/ts_anomaly/contracts.py` | Explicit metric definitions and observation schema |
| `src/ts_anomaly/assessment.py` | Generic quality audit, reversible diagnostic views and elapsed-time lag correlations |
| `notebooks/01_data_assessment.ipynb` | Questions, plots, interpretations and reproducible hand-off |

A new source needs its own small loader and definitions. The assessment functions
contain no NAB file names, telecom defaults or model selection rules. Both NAB
pilot signals are gauges; count/counter modelling and multivariate monitoring are
not established by this pilot.

Read [NAB_REVIEW.md](NAB_REVIEW.md) for scientific limitations, validation evidence
and the development-only evaluation protocol. Temperature has conflicting repeated
timestamps, so the raw export is explicitly **unresolved**, not model-ready.

## Legacy optical implementation

The existing `optical_anomaly` package and notebooks 00–06 below are retained from
`main` as legacy material; **do not run them as the next stages of the NAB blueprint**.
The Python distribution retains its existing installation name for compatibility;
the active source namespace is `ts_anomaly`. No scientific-foundation branch changes
were merged. Original optical source and notebooks are unchanged by this pilot.

### Original optical workflow

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
