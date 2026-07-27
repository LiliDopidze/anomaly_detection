# Sector-agnostic telemetry anomaly platform

Milestone 1 establishes a neutral telemetry contract, sector packs, source
translators, and a tested boundary between production observables and evaluation
truth. Modelling starts only after that boundary is proven.

```text
Native source ──> sector translator ──> SPEC-CORE ──> detector
                         │
                         └────────────> SPEC-EVAL ──> offline evaluation only
```

`SPEC-CORE` contains only information available to a production detector.
`SPEC-EVAL` contains fault truth, condition truth, true causes, intervals, and ticket
links. The translator is the only component allowed to see both.

## Clean source layout

There is one public package, `anomaly_detection`, with eight meaningful modules:

| Module | Responsibility |
|---|---|
| `core.py` | Neutral SPEC-CORE vocabulary, interfaces, validation, and hashing |
| `evaluation.py` | Physically separate SPEC-EVAL vocabulary and validation |
| `packs.py` | Generic sector-pack loader plus telecom and oil-well phrasebooks |
| `telecom.py` | Synthetic telecom translator |
| `oil_well.py` | Petrobras 3W translator and contract challenge |
| `runtime.py` | Runtime-safe baseline that reads only SPEC-CORE |
| `workflows.py` | Named end-to-end workflows called by notebooks |
| `cli.py` | Small automation/child-process command line |

The telemetry generator is frozen under `legacy/telemetry_synth_v4/`. It is not part
of the installed product package.

## Install and test

Python 3.10 or later is required.

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Three Milestone 1 notebooks

The maintained Colab notebooks are in
`notebooks/google_drive/milestone1_v0_3/`:

1. `02_M1_V03_TELECOM_MATERIALISATION.ipynb`
2. `03_M1_V03_WEEK1_ACCEPTANCE.ipynb`
3. `04_M1_V03_PETROBRAS_3W_CHALLENGE.ipynb`

They contain paths and workflow calls—not duplicate implementations. In Colab they
install the package from GitHub and record the resolved Git commit in each workflow
report. See the [step-by-step notebook guide](docs/RUN_NOTEBOOKS.md).

## Important contract decisions

- `entity_service_windows.csv` remains a native input; canonical validity is stored
  once in `entity_registry.valid_from` and `valid_to`.
- Null telemetry rows are retained with `quality_code = invalid`.
- FEC values at or above the generator ceiling use `quality_code = clipped`.
- FEC exposure follows the frozen generator mechanism and is expected to be constant.
- CRC exposure is derived from each row’s throughput and the pack’s 1,500-byte frame
  parameter, so it must vary when throughput varies.
- `tickets.csv` and all ground-truth fields are evaluation-only.
- Petrobras 3W adds condition-state truth without inventing severity, topology,
  shared manifolds, cause groups, or tickets.

## Data policy

Datasets and outputs stay in Google Drive and are ignored by Git. Only code,
contracts, pack metadata, tests, notebooks, and small licence-compatible fixtures
belong in this repository.

See [architecture](docs/ARCHITECTURE.md) and the
[Week 1 acceptance map](docs/WEEK1_ACCEPTANCE.md).

## Licence

Code is available under the [Apache License 2.0](LICENSE). Datasets retain their
original licences and are not redistributed here.
