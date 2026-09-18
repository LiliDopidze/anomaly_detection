# Development diagnostic implementation

Open `notebooks/10A_DEVELOPMENT_DIAGNOSTICS.ipynb` from this repository and run
all cells. Colab mounts Drive. The default data root resolver supports
`/content/drive/MyDrive/anomaly_detection`. This is a diagnostic stage after
the development evaluation; it is not a prerequisite for model training.

The notebook defaults to the new v14 results, v12 model (resolved from
the evaluation manifest), v2 core, v6 features and v3 truth. Change the run
IDs only to the inputs that actually produced your results. It stops on
lineage or replay mismatch rather than blending incompatible runs.

## Command-line usage

From the repository root, after installing project dependencies:

```bash
PYTHONPATH=src python -m telco_anomaly.diagnostics \
  --results /content/drive/MyDrive/anomaly_detection/results/synthetic_pon/synthetic_pon_development_v14 \
  --data-root /content/drive/MyDrive/anomaly_detection \
  --output /content/drive/MyDrive/anomaly_detection/diagnostics/synthetic_pon/development_diagnostics_v1 \
  --deep-data \
  --trace-fault F-00023
```

For the extracted results ZIP only, omit `--data-root`, `--deep-data`, and
`--trace-fault`. Point `--results` to the directory containing
`evaluation_manifest.json`, not to the ZIP or its outer results directory.

The output path must not already exist. Use a new output name on each run.
The module writes only its new output directory, never its inputs. It does
not download data, refit models, change thresholds, select candidates, or
open a locked holdout. Git changes are not pushed by the module.

For a nuisance incident, replace `--trace-fault F-00023` with
`--trace-case C-000001`, using an actual ID from `incident_diagnostics.csv`.
Choose one trace target per run. `--trace-limit` defaults to 20,000 rows per
table. Traces include 24 hours of context and are clipped to development;
long traces are truncated deterministically and marked in the trace manifest.

## Outputs

Each table is saved as both CSV and Parquet.

| Artifact | Purpose |
|---|---|
| `report.md` | Recorded metrics and important interpretation limits |
| `diagnostic_manifest.json` | Input hashes, replay status, selected channels and audit scope |
| `fault_diagnostics` | Every fault including misses, delay, scope errors and evidence category |
| `review_queue` | Misses and detections without demonstrated pre-impact credit |
| `by_fault_type`, `by_domain_type`, `by_outcome`, `by_localisation_outcome` | Fault slices; delays condition on detected faults |
| `score_evidence` | Each observed fault/entity/episode/channel window: finite scores, threshold breaches, first breach, peak margin, pre-impact breaches, window-local longest run |
| `score_availability` | Per entity/channel/day availability among present rows; missing timestamps are separate |
| `alerts`, `incident_members` | Frozen replay output for tracing incident construction |
| `incident_diagnostics`, `incident_status_counts` | Exactly credited, duplicate and unmatched cases using the existing evaluator |
| `latency_recall_curve` | Fresh event matching at 1, 6, 24, 48 and 168 hours |
| `candidate_frontier` | Existing candidate outcomes with a descriptive prompt-recall/workload frontier; original eligibility remains intact |
| `feature_quality_and_shift` | Optional finite coverage, constants, ranges and mean shifts versus calibration fit |
| `raw_quality_by_metric` | Optional development raw quality-code counts |
| `collection_gap_summary` | Optional development-clipped gap hours per entity/metric when the gap artifact exists |
| `operational_event_counts` | Optional observable event counts when the event artifact exists |
| `trace_raw`, `trace_features`, `trace_scores`, `trace_manifest` | Bounded context for one missed/late fault or nuisance incident |

Full mode replays the frozen score-to-alert and incident functions and checks
the result against saved metric values, numerators, denominators, and
per-fault matching/localisation outcomes. Prompt recall is independently
matched rather than inferred by filtering the active-window assignment.
Feature traces show the persisted feature values, not fitted-model feature
attribution. No joblib model is deserialised.

## How to use the evidence

1. Start with `fault_diagnostics`: no rows, no finite scores, no threshold
   crossing, crossing without alert, or alert without distinct event credit.
2. Review the raw and feature traces to distinguish absent/weak signal from
   poor feature representation or poor score discrimination. The automated
   category alone cannot make that distinction.
3. Inspect duplicates separately from unmatched incidents. The latter may
   represent benign changes, quality issues, or unlabelled anomalies; the
   script does not invent a cause.
4. Compare candidate timely recall at comparable workload, preserving the
   calibration-admissible and selection-eligible columns. A Pareto flag is
   not permission to adopt a candidate.
5. Design a small new development experiment and retain this run as its
   baseline. Do not repeatedly optimise against a sealed holdout.

## Limits and runtime

Results-only mode cannot attribute a miss to thresholding, persistence,
features, missing telemetry or incident formation. These are explicitly
unavailable until full mode is run.

Full mode reads the selected development score file several times for replay,
exposure, boundary checks and evidence. Scores stream by complete episode;
very large individual episodes and the resulting alerts still consume memory.
Deep mode scans all numeric feature columns and raw development telemetry,
so Drive I/O can dominate runtime. DuckDB uses the existing bounded-memory
helper; configure `TELCO_MODEL_DUCKDB_MEMORY_LIMIT` and `TELCO_WORK_ROOT` to
control memory and local spill storage. Avoid Drive for spill storage.

Manifest hashes establish the intended lineage. Large scores/features were
not all content-hashed by the original pipeline: replay agreement does not
prove byte-for-byte provenance of every telemetry row. The diagnostic records
the current score hash. Raw/feature profiles and trace review are descriptive,
not a new leakage proof or physical root-cause identification.

Window-local high runs deliberately do not carry alert state from before the
fault. Actual alert openings come from the existing state machine, including
pre-existing alerts, recovery, episode changes and gap resets. The automated
evidence categories remain conservative when these interact.

Synthetic results, sparse fault types, and correlated faults still require
independent scenarios and real PON validation before operational claims.
