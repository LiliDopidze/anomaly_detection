# Sector-agnostic telemetry anomaly platform

Milestone 1 establishes a sector-neutral telemetry contract, sector-pack interface,
source adapters and a tested boundary between production observables and evaluation
truth.

The repository currently includes a synthetic telecom fixture adapter and a Petrobras
3W 2.0.0 contract challenge. Modelling is deliberately deferred until the contract
and leakage boundary are stable.

## Architecture

```text
Native source ──> sector adapter ──> SPEC-CORE ──> detector runtime
                         │
                         └────────> SPEC-EVAL ──> offline evaluation only
```

`SPEC-CORE` contains telemetry, metric semantics, entities, relationships, observable
operational events and collection gaps. `SPEC-EVAL` contains fault IDs, true
intervals, cause groups, condition states, ticket linkage and evaluation-only gap
reasons.

The translator is the only component allowed to see both sides. Runtime code imports
no evaluation package.

## Packages

- `telemetry_contract`: sector-neutral SPEC-CORE models, schemas and interfaces.
- `telemetry_eval_contract`: physically separate evaluation schemas.
- `telemetry_packs.telecom`: telecom metric and relationship phrasebook.
- `telemetry_packs.oil_well`: minimal Petrobras 3W phrasebook.
- `telemetry_adapters`: telecom and Petrobras 3W translators.
- `telemetry_runtime`: deterministic Week 1 baseline used for isolation evidence.
- `telemetry_synth`: frozen synthetic telecom generator source.

Tickets from the synthetic telecom fixture are evaluation-only. Entity validity is
stored once in `entity_registry.valid_from` and `valid_to`; there is no duplicate
canonical service-window table.

## Install and test

Python 3.10 or later is required.

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Build the wheel and source distribution:

```bash
python -m build
python -m twine check dist/*
```

## Milestone 1 notebooks

The maintained Google Drive/Colab notebooks are in
`notebooks/google_drive/milestone1_v0_3/`:

1. `02_M1_V03_CONTRACT_AND_PACK.ipynb`
2. `03_M1_V03_TELECOM_TRANSLATOR.ipynb`
3. `04_M1_V03_LOCK_AND_ACCEPTANCE_TESTS.ipynb`
4. `05_M1_V03_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb`

The notebooks are intentionally thin. Reusable implementation belongs in the Python
packages, not in notebook source cells.

## Data policy

Datasets and materialised outputs are not stored in Git:

- native telecom files remain in Google Drive;
- Petrobras 3W remains under
  `sources/petrobras_3w/2.0.0/raw/3w_dataset_2.0.0`;
- canonical outputs remain under `outputs/milestone_1/v0.3/`;
- Parquet, CSV, credentials and local environment files are ignored.

Only small, licence-compatible test fixtures should ever be committed.

## Verified Milestone 1 evidence

- 19 package tests pass.
- Translator output is invariant after truth columns, evaluation files and
  `tickets.csv` are removed.
- Detector output is identical with SPEC-EVAL mounted, renamed, empty or removed.
- The internal package graph is acyclic.
- The full telecom panel translated 6,288,215 native rows into 69,170,365 canonical
  observations with a 3.72 GiB process high-water mark under an 8 GiB budget.
- The Petrobras 3W challenge produced 8,034,930 observations from the smallest
  qualifying three-well subset and required `gt_condition_states` without inventing
  topology or severity.

See [the acceptance record](docs/WEEK1_ACCEPTANCE.md) and
[architecture notes](docs/ARCHITECTURE.md).

## Releases

CI runs tests, rebuilds notebooks and checks the distribution on every pull request.
A `v*` tag builds a wheel, source distribution, notebook archive and SHA-256 manifest
as GitHub Release assets.

## Licence

The source code is available under the
[Apache License 2.0](LICENSE). Dataset files retain their original licences and are
not redistributed by this repository.
