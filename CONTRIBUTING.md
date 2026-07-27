# Contributing

## Local setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Design rules

1. `anomaly_detection.core` stays sector-neutral.
2. `anomaly_detection.runtime` must not import evaluation or sector modules.
3. Translators use explicit operational allowlists.
4. Truth-bearing native fields and `tickets.csv` never enter SPEC-CORE.
5. Schema changes require a contract version change and migration note.
6. Notebooks explain and orchestrate; reusable logic belongs under `src/`.
7. Do not commit datasets, outputs, credentials, personal paths, or unlicensed assets.
8. Every new sector records what the contract could not express and what would have
   required distortion or invention.

## Pull requests

Include the reason for the change, affected contract/pack/adapter versions,
leakage-boundary impact, tests, regenerated notebooks, and any new contract-fit
limitation.
