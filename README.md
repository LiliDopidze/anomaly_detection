# Time-series anomaly detection blueprint

Start with **[01_data_assessment.ipynb](notebooks/01_data_assessment.ipynb)**.
This branch contains only the NAB data-exploration stage and its supporting files.
Model selection, expected-behaviour modelling and anomaly detection are later,
separately agreed steps.

## What we do first

Work through Notebook 01 in this order:

1. Understand the two selected measurements and their source metadata.
2. Load the pinned NAB files and inspect their rows and separate annotations.
3. Check timestamps, duplicates, gaps and invalid values.
4. Explore distributions, changes over time and temporal dependence.
5. Record findings and the decisions needed before modelling.

The pilot uses machine temperature and CPU utilisation. They are unrelated
development series, assessed separately. Both complete files have been inspected;
neither is an untouched final test set. Temperature has conflicting duplicate
timestamps. Diagnostic calculations exclude those timestamps, while the raw
observations remain unchanged. Units and timezone are left unspecified where the
source metadata does not establish them. Anomaly windows are benchmark annotations,
not verified fault onset and recovery records.

## Files and responsibilities

```text
notebooks/01_data_assessment.ipynb   # Exploration, plots and interpretation
configs/nab_pilot.json              # Selected files, hashes and measurement metadata
src/ts_anomaly/
    __init__.py
    nab.py                         # Download and read NAB; keep labels separate
    contracts.py                   # Measurement definitions and observation schema
    assessment.py                  # Data-quality and temporal diagnostics
tests/test_nab_blueprint.py         # Tests for those supporting functions
tools/execute_nab_assessment.py     # Reproduce the notebook using the active Python
pyproject.toml                     # Installation and dependencies
README.md
LICENSE
.gitignore
.gitattributes
```

The source loader and configuration are NAB-specific. Contracts and assessment
accept explicit measurement definitions; they contain no telecom defaults.

## Run in WSL and VS Code

From this checkout's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name nab-blueprint --display-name "NAB blueprint"
code .
```

Open Notebook 01 and select the **NAB blueprint** kernel. Start with sections 1–2
to understand the data and inspect the loaded rows before continuing to diagnostics.
The first load downloads the selected files and upstream license into ignored
`data/nab/`. Subsequent loads verify the hashes and use the cache. Generated
assessment products go into ignored `outputs/`.

To check the implementation or reproduce all notebook outputs:

```bash
python -m pytest
python tools/execute_nab_assessment.py
```

The runner uses the calling Python interpreter and saves successful outputs.
Jupyter needs local kernel sockets. Python.org macOS installations may also need
their certificate-setup script completed before HTTPS downloads work.

Sources, statistical interpretation and limitations are included in the notebook.
No subsequent notebooks, detectors, synthetic generators or legacy pipeline code
are included in this branch's current files.
