# Milestone 1 v0.3 notebooks

Run in this order:

1. `02_M1_V03_TELECOM_MATERIALISATION.ipynb`
2. `03_M1_V03_WEEK1_ACCEPTANCE.ipynb`
3. `04_M1_V03_PETROBRAS_3W_CHALLENGE.ipynb`

Notebook 03 reads the immutable telecom run created by Notebook 02, so both must use
the same `RUN_ID`. Notebook 04 is independent and may run concurrently.

The notebooks install the implementation from GitHub. There is no source-code bundle
to upload to Drive. For a signed-off run, set `ANOMALY_RUNTIME_REF` to a release tag
or exact Git commit; the resolved commit is recorded in the workflow report.

See [the complete run guide](../../../docs/RUN_NOTEBOOKS.md) for the Drive tree,
configuration, output descriptions, and troubleshooting.
