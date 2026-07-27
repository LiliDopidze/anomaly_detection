# Sector-agnostic anomaly detection research

This repository is intentionally notebook-first. It is the working environment for
one data scientist validating the product idea—not a premature production package.

There are three notebooks and one shared Python file:

1. `01_TELECOM_WEEK1_END_TO_END.ipynb` — defines the neutral contract and Telecom
   Pack, translates the synthetic source, and proves evaluation truth is isolated.
2. `02_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb` — tests the same contract with a
   second sector and records what the public source cannot express.
3. `03_SECTOR_AGNOSTIC_MODELLING_AND_RANKING.ipynb` — calibrates a truth-free
   statistical baseline, forms persistent cross-signal episodes, applies an
   explicit alert budget, and creates the operator-facing
   `ranked_incidents.csv`.
4. `week1_core.py` — only the settled mechanics where a silent copy error would
   invalidate the experiment: exposure, clipping, batching, canonical hashing,
   truth routing, and the verified 3W selector.

The contract, sector mappings, assertions, modelling choices, and ranking formula
remain visible in notebook execution order.

```text
native data ──> Notebook 01 or 02 ──> SPEC-CORE ──> Notebook 03 ──> ranked incidents
                         │                                  │
                         └────────> SPEC-EVAL ──────────────┘
                                      read only after outputs are frozen
```

Datasets and outputs stay in Google Drive and are not committed to Git. See the
[Drive run guide](notebooks/drive_research/README.md) for the exact folder layout,
run order, controls, output locations, and the next modelling steps.

The previous package/CLI/release implementation is preserved in Git history and its
separate engineering branch. The legacy baseline descriptor remains in
`legacy/baseline-freeze.json`.

Code is licensed under [Apache License 2.0](LICENSE). Datasets retain their original
licences and are not redistributed here.
