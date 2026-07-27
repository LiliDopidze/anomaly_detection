# Contributing

## Local setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Design rules

1. Generic contracts must not contain sector-specific branches.
2. Runtime code must not import `telemetry_eval_contract`.
3. Adapters use explicit operational allowlists.
4. Truth-bearing native fields and `tickets.csv` never enter SPEC-CORE.
5. Schema changes require a contract version change and migration note.
6. Notebooks orchestrate and explain; reusable logic belongs under `src/`.
7. Do not commit datasets, outputs, credentials, personal paths or unlicensed assets.

## Pull requests

Include:

- the reason for the change;
- affected contract, pack or adapter versions;
- leakage-boundary impact;
- tests and regenerated notebook artifacts;
- any contract-fit distortion discovered in a new sector.
