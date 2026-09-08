# Sector-agnostic anomaly detection research

This repository is intentionally notebook-first. It is the working environment
for one data scientist testing a sector-agnostic anomaly-detection design, not a
production platform.

## Current Milestone 1 workflow

```text
01A_TELECOM_PACK.ipynb ───────┐
                              ├──> 01B_COMMON_CANONICAL_ADAPTER.ipynb
01A_PETROBRAS_3W_PACK.ipynb ──┘                 │
                                                ├──> SPEC-CORE
                                                └──> SPEC-EVAL

SPEC-CORE ──> 02_CANONICAL_EDA.ipynb ──> 03_EVALUATION_HARNESS.ipynb
          ──> 04_SIMPLE_ANOMALY_MODELS.ipynb ──> ranked cases and demo
```

- Each `01A` notebook owns its native filenames, metric meanings, entity
  vocabulary and label translation. The active interface is telemetry-only.
  Telecom topology is preserved separately in `SPLITS` for holdout design,
  grouped-fault evaluation and incident aggregation; it is not a model input.
- `01B_COMMON_CANONICAL_ADAPTER.ipynb` has no sector-specific translation
  branch. It accepts either standardized pack and preserves the same canonical
  entity, episode and metric structure. Pack observations are metric-level, so
  different sensors may use different timestamp grids and cadences.
- `02_CANONICAL_EDA.ipynb` is shared unchanged across sectors and reads
  `SPEC-CORE` only. It produces reproducible statistical profiles and figures,
  not anomaly labels or model scores.
- `03_EVALUATION_HARNESS.ipynb` freezes event matching, class-specific 3W
  decision horizons, case workload and statistical-resolution reporting before
  model comparison.
- `04_SIMPLE_ANOMALY_MODELS.ipynb` compares a deliberately small nested set of
  rapid, drift and dispersion residual detectors. More complex models are
  challengers, not assumed improvements.
- `milestone1_core.py` contains only settled, shared mechanics: interface
  validation, immutable materialisation, one content fingerprint, quality-code checks, gaps,
  episode boundaries and isolation-test support.
- Datasets and outputs stay outside Git. Local WSL runs use
  `~/anomaly_detection_data/` by default; Colab runs use Google Drive.

The research path is:

```text
SPEC-CORE ──> calibration EDA ──> frozen residuals ──> case-level evaluation
          ──> ranked incidents
```

See the
[Milestone 1 run guide](notebooks/drive_research/README.md)
for paths, execution order, and the process for adding another sector.

The previous combined notebooks, EDA/model drafts, and package/CLI
implementation remain available in Git history. The frozen legacy baseline
descriptor remains in `legacy/baseline-freeze.json`.

Code is licensed under [Apache License 2.0](LICENSE). Datasets retain their
original licences and are not redistributed here.
