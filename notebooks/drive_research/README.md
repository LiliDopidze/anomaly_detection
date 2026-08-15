# Milestone 1 research workflow

The maintained code lives in the GitHub repository:

```text
notebooks/drive_research/
├── 01A_TELECOM_PACK.ipynb
├── 01A_PETROBRAS_3W_PACK.ipynb
├── 01B_COMMON_CANONICAL_ADAPTER.ipynb
├── 02_CANONICAL_EDA.ipynb
├── 03_EVALUATION_HARNESS.ipynb
├── milestone1_core.py
└── evaluation_core.py
```

The notebooks hold the visible research decisions. `milestone1_core.py` holds
the settled logic that must not be copied between them: schemas, validation,
pack writing, canonical materialisation, one content fingerprint, and the
isolation helpers. It contains no sector logic.

`evaluation_core.py` is the same deliberate pattern for Notebook 03: one flat
file containing tested truth partitioning, score-to-alert conversion, matching
and metric calculations. Evaluation policy and adversarial controls remain
visible in the notebook. It is not a package, CLI or sector-specific layer.

```text
native source  ->  PACK          (sector notebook 01A translates)
PACK           ->  SPEC-CORE     (common adapter 01B canonicalises)
                 + SPEC-EVAL + SPLITS
```

A detector reads `SPEC-CORE` only and must run with `SPEC-EVAL` absent.

## Storage and runtime

Code and data stay separate:

```text
~/projects/anomaly_detection/       # Git repository
~/anomaly_detection_data/           # local data and outputs (WSL/Linux default)
```

Colab uses `MyDrive/anomaly_detection/` by default. To use another location,
set `ANOMALY_DATA_ROOT`; the earlier `ANOMALY_DRIVE_ROOT` name remains
accepted. Each notebook prints the resolved runtime, data root and code root
before reading data.

## Contracts

| Contract       | Version |
|----------------|---------|
| Pack interface | `0.7.1` |
| SPEC-CORE      | `0.10.1`|
| SPEC-EVAL      | `0.8.0` |
| Canonical EDA  | `0.6.0` |

`SPEC-CORE v0.10.1` is the telemetry-only modelling floor:

```text
telemetry
metric_catalogue
entity_registry
observation_episodes
collection_gaps
```

Pack tables use the same names and the same columns, minus the four the
adapter derives (`observed_from`, `observed_to`, `validity_basis`) and the one
table it computes (`collection_gaps`). There is one schema definition, not two.

### The three rules that make a row mean the same thing in every sector

**1. Presence.** Telemetry is long and metric-level: one row means one metric
was observed or attempted at that timestamp.

- an **absent row** is not an observation;
- a **null value with `quality_code='invalid'`** is an attempted observation
  that failed;
- an **absent `(episode, metric)` pair** means only that the source supplied no
  observation attempt; capability metadata is needed to distinguish "not
  installed" from "installed but never reported".

Both packs now apply this identically. Previously Telecom emitted invalid rows
for a sensor that was never fitted while 3W omitted it, so `valid_rate` and
`coverage` meant different things in the two sectors.

**2. Episodes.** Every row carries an `episode_id` — one source-declared
observation run across which differences may be computed. It is one named
synthetic generator-run episode per Telecom ONT, and one source recording per
3W file. Episode boundaries are never inferred from telemetry gaps.

**3. Gaps.** A gap is found inside one `(entity, episode, metric)` series
wherever a metric declaring a cadence skipped more than
`GAP_TOLERANCE_FACTOR × cadence`. This applies to `periodic` and `recording`
metrics alike, so a hole inside a 3W recording is reported while the interval
*between* two recordings stays correctly undefined.

### Truth isolation

Faults and condition states live in `SPEC-EVAL`, physically separate. There is
no `SPEC-CONTEXT` layer.

`truth_like_columns()` rejects metric IDs that look like labels. It is anchored
to exact names (`class`, `state`, `label`, `target`, `fault`, `anomaly`,
`condition_code`), the prefixes `gt_`, `truth_`, `anomaly_`, and the suffixes
`_label`, `_labels`, `_anomaly`, `_ground_truth`. Free substring matching was
removed: it rejected legitimate measurements such as `ground_fault_current`,
`fault_passage_indicator` and `distance_to_fault`, all of which a power-sector
pack would need.

## Native data locations

Telecom:

```text
<data_root>/telco_syntetic_data/
├── reference_dataset.parquet   # observable-only; no gt_*, class, state, fault, label
├── gt_fault_registry.csv
├── fault_entity_intervals.csv
├── tickets.csv                 # optional; not read
└── topology.csv                # required grouping metadata
```

The notebook reads only `ont_id`, `olt_id`, `pon_port`, `splitter_l1`,
`splitter_l2` and `geo_cluster` from topology, and writes those memberships to
`SPLITS/entity_groups.parquet`. Topology never enters `SPEC-CORE` or the
detector. Any `gt_*` topology columns remain evaluation truth, and the
isolation test proves they cannot affect the approved groups.

Petrobras 3W:

```text
<data_root>/sources/petrobras_3w/2.0.0/raw/3w_dataset_2.0.0/
├── dataset.ini
└── 0/ ... 9/
```

The notebook verifies the official inventory count
(`THREEW_EXPECTED_FILES`, default 2228), excludes simulated, drawn and
duplicate download variants, and selects one deterministic real recording per
`(well, event-directory)` pair. It then reads the actual `class` values from
every selected file. Development and holdout coverage is constrained by those
values, not by the directory name; folder/class mismatches are reported.

Finite-value validation uses only constraints documented in `dataset.ini`:
choke openings are percentages and valve states are in `{0, 0.5, 1}`.
Unexplained finite pressure, temperature and flow extremes remain measured and
are surfaced in the EDA tail audit rather than silently removed.

## Run order

Set the sector, then run top to bottom. In VS Code select the repository
`.venv` kernel and use **Run All**; in Colab use **Runtime → Run all**.

| Step | Notebook | Setting |
|------|----------|---------|
| 1 | `01A_<SECTOR>_PACK.ipynb` | — |
| 2 | `01B_COMMON_CANONICAL_ADAPTER.ipynb` | `SECTOR = "telecom"` or `"petrobras_3w"` |
| 3 | `02_CANONICAL_EDA.ipynb` | same `SECTOR` |
| 4 | `03_EVALUATION_HARNESS.ipynb` | same `SECTOR` |

Notebooks 01B and 02 run unchanged across sectors — that is the claim they
exist to demonstrate. Output directories are immutable; change the relevant
run ID before rebuilding a completed stage. Every setting is also readable
from an environment variable, so the whole pipeline can be executed headlessly
for regression testing.

## Splits

`SPLITS` is orchestration metadata, never model input.

| Table | Telecom | 3W | Tests |
|-------|---------|-----|-------|
| `time_partitions` | yes | — | temporal drift |
| `entity_partitions` | yes (whole `geo_cluster`) | yes (whole well) | unseen entity |
| `entity_groups` | yes (OLT/PON/splitter/geo) | — | grouped-fault evaluation |

Both sectors now carry an `entity_partitions` table, so Notebook 03 can pose
the *same* generalisation question in both. Previously Telecom split on time
and 3W on entity, which meant the two sectors were answering different
questions and no cross-sector number was comparable.

## Outputs

```text
outputs/packs/<sector>/<pack_run_id>/
├── PACK-CORE/{telemetry/, metric_catalogue, entity_registry, observation_episodes}
├── PACK-EVAL/      # optional, never read by detector code
├── SPLITS/
└── pack_manifest.json

outputs/canonical/v0.10.1/<sector>/<canonical_run_id>/
├── SPEC-CORE/      # + collection_gaps, derived bounds, manifest with fingerprint
├── SPEC-EVAL/      # optional
├── SPLITS/
└── run_manifest.json

outputs/eda/v0.6.0/<sector>/<eda_run_id>/
├── *.parquet       # compact evidence tables
├── figures/
└── eda_summary.json

outputs/evaluation/v0.1.0/<sector>/<evaluation_run_id>/
├── TRUTH/calibration/
├── TRUTH/development/
├── TRUTH/holdout_sealed/
├── CONTROLS/
├── evaluation_policy.json
└── evaluation_manifest.json
```

EDA outputs are immutable. The notebook uses a temporary figure directory and
publishes all tables and figures together only after successful completion.
It profiles a balanced metric set, uses gap-safe transformations, Spearman
correlations, cadence-aware autocorrelation lags, bounded stationarity tests,
and explicit test statuses. No EDA value is imputed or deleted.

## Adding another sector

Write one `01A_<SECTOR>_PACK.ipynb` containing three things:

1. a **phrasebook** — the native-field to `metric_id` map, with measurement
   kind, unit, sampling mode and cadence;
2. a **telemetry generator** — any iterable yielding long DataFrames with the
   telemetry columns, emitting a `(episode, metric)` pair only where the source
   attempted to observe it;
3. a **truth translator** — native labels routed to `PACK-EVAL` only.

Then call `save_pack(...)`. It owns directory layout, part numbering,
validation, the fingerprint and the manifest, so a sector notebook never
handles any of them.

The pack must pass the original-versus-redacted isolation test. Topology is an
optional sector capability: when a sector has reliable relationship data, store
memberships in `SPLITS`; do not add topology columns to telemetry or let a
detector depend on them.

Do not add a sector branch to `milestone1_core.py`, Notebook 01B or Notebook
02. If a genuine source concept cannot be represented without distortion,
record the failure in the contract-fit report before changing the versioned
interface. That report is the actual research output of Milestone 1.

## Evaluation harness

Notebook 03 freezes the common score and alert schemas, physically separates
development from sealed holdout truth, and tests event matching before a real
model exists. Telecom uses chronological partitions as its primary evaluation;
3W uses whole-well partitions. The choice is made from split capabilities and
can be overridden explicitly with `EVALUATION_PRIMARY_SPLIT`.

An alert is matched from the later of `observable_ts` and the affected-entity
interval start until the earlier of their two end times. One alert matches at
most one fault, one fault receives one event-level credit, and later eligible
alerts are duplicates. Event recall, pre-impact recall, precision, false-alert
rate, latency, shared-fault recall and entity coverage are reported separately.
Ratios include Wilson 95% intervals; fewer than five events of a type is marked
descriptive only.

Constant, random, perfect-event, late, duplicate and deliberately leaky
controls must pass before outputs are published. Latency remains sector-specific
because Telecom and 3W derive `observable_ts` differently.

## Next stage

Build `04_GENERIC_BASELINE_MODELS.ipynb` using calibration and development
SPEC-CORE only. Notebook 04 may pass development scores to the evaluator, but
must not open `TRUTH/holdout_sealed`; that directory is first used by Notebook
05 after the feature definition, model and threshold are frozen.
