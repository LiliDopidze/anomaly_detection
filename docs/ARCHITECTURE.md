# Architecture and trust boundary

## Dependency direction

```text
telemetry_contract
        ↑
telemetry_packs
        ↑
telemetry_adapters ──> telemetry_eval_contract

telemetry_runtime     # no internal package dependency
```

`telemetry_contract` is sector-neutral and has no internal dependency. Packs depend
only on the generic contract. Adapters are offline translation components and may
depend on both contracts. Runtime scoring accepts only a SPEC-CORE directory.

## Three distinct absence states

The canonical representation distinguishes:

1. an observation row exists but its value is null;
2. an expected observation is missing inside an entity validity window;
3. the entity is outside its validity window and no observation is expected.

These map respectively to an `invalid` telemetry quality code, a collection-gap
interval, and no gap.

## Evaluation shapes

SPEC-EVAL contains both event truth and condition-state truth. Petrobras 3W forced the
addition of `gt_condition_states`; it did not justify a severity field because the
source provides state codes rather than graded severity.

## Artifact boundaries

The current alpha release builds one Python distribution while enforcing the boundary
through package imports and tests. A production deployment should split runtime-safe
core from offline translation/evaluation tooling into separate distributions or
container images.
