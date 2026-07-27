# Architecture and trust boundary

## Logical flow

```text
telecom source ──> telecom.py ──┐
                               ├──> core.py ──> runtime.py
3W source ───────> oil_well.py ─┘
                         │
                         └────────> evaluation.py (offline only)

packs.py ──> translator configuration
workflows.py ──> orchestration and evidence
cli.py ──> clean child-process materialisation
```

`core.py` and `evaluation.py` are dependency roots. `runtime.py` has no dependency on
evaluation, either sector translator, or notebook workflows. Tests assert the
internal graph is acyclic.

## Why there are modules instead of notebook implementations

The eight modules separate responsibilities that change for different reasons. The
notebooks stay readable because they call `materialise_telecom`,
`run_week1_acceptance`, or `challenge_threew`; a bug fix is made and tested once in
the package.

This is not a runtime bundle embedded in a notebook. Colab installs the GitHub
package. A signed-off run records the resolved commit so the exact implementation can
be recovered.

## Three distinct absence states

The canonical representation distinguishes:

1. a row exists but its value is null (`quality_code = invalid`);
2. an expected observation is absent inside an entity validity window
   (`collection_gaps`);
3. the entity is outside its validity window, so no observation is expected.

The third case must never be labelled as a collection gap.

## Evaluation shapes

SPEC-EVAL supports event truth and condition-state truth. Petrobras 3W required
`gt_condition_states`; it did not justify `severity_ordinal`, because the source
provides state codes rather than graded severity.

## Physical data interface

The logical telemetry schema is long. Physical materialisation is partitioned
Parquet, and consumers must read batches rather than loading the entire long table.
Each telecom run records row expansion, stored bytes, and storage extrapolations in
`representation_report.json`.

## Deployment boundary

The alpha repository builds one Python distribution while enforcing the runtime
boundary through imports and tests. A production release can split runtime-safe code
from offline translation/evaluation tooling into separate distributions or images
without changing the contract.
