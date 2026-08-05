# Sector-agnostic anomaly detection research

This repository is intentionally notebook-first. It is the working environment
for one data scientist testing a sector-agnostic anomaly-detection design, not a
production platform.

## Current Week 1 workflow

```text
01A_TELECOM_PACK.ipynb ───────┐
                              ├──> 01B_COMMON_CANONICAL_ADAPTER.ipynb
01A_PETROBRAS_3W_PACK.ipynb ──┘                 │
                                                ├──> SPEC-CORE
                                                └──> SPEC-EVAL

SPEC-CORE ──> 02_CANONICAL_EDA.ipynb ──> frozen EDA evidence
```

- Each `01A` notebook owns its native filenames, metric meanings, entity
  vocabulary and label translation. The active interface is telemetry-only;
  topology and operational context are not model inputs.
- `01B_COMMON_CANONICAL_ADAPTER.ipynb` has no sector-specific translation
  branch. It accepts either standardized pack and writes the same canonical
  tables.
- `02_CANONICAL_EDA.ipynb` is shared unchanged across sectors and reads
  `SPEC-CORE` only. It produces reproducible statistical profiles and figures,
  not anomaly labels or model scores.
- `week1_core.py` contains only settled, shared mechanics: interface
  validation, immutable materialisation, logical hashes, quality codes, gaps,
  and isolation-test support.
- Datasets and outputs stay in Google Drive. They are not committed to Git.

The research path is:

```text
SPEC-CORE ──> canonical EDA ──> feature engineering ──> modelling and ranking
```

See the
[Drive run guide](notebooks/drive_research/README.md)
for paths, execution order, and the process for adding another sector.

The previous combined notebooks, EDA/model drafts, and package/CLI
implementation remain available in Git history. The frozen legacy baseline
descriptor remains in `legacy/baseline-freeze.json`.

Code is licensed under [Apache License 2.0](LICENSE). Datasets retain their
original licences and are not redistributed here.
