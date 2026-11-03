# NAB pilot: scope, evidence and decisions

The active workflow starts at `notebooks/01_data_assessment.ipynb`. It is a
descriptive assessment, not a detector or an automatic model-selection system.
The new `ts_anomaly` functions depend on explicit measurement definitions rather
than PON defaults. The legacy optical implementation is untouched and is not the
next stage of the new workflow.

## Source and experiment

Upstream: [Numenta/NAB](https://github.com/numenta/NAB), commit
`ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`. The manifest records SHA-256 hashes
for the two selected CSVs, both combined annotation files and the upstream license.
The loader checks the hashes on every use. The upstream repository distributes
these resources under its MIT license, which is retained alongside the cache.

Both **complete files are development data**. We have inspected observations and
labels, so a later temporal split cannot retroactively make an untouched final test.
Notebook 02 should use chronological development validation, with a separately
specified final evaluation set reserved before tuning. Full NAB scoring has not
been implemented, and no detector-performance claim is made.

## Findings calculated from the pinned files

| Property | Machine temperature | EC2 CPU utilisation `24ae8d` |
|---|---:|---:|
| Original rows | 22,695 | 4,032 |
| Unique timestamps | 22,683 | 4,032 |
| Observed nominal cadence | 5 minutes | 5 minutes |
| Conflicting timestamps | 12, affecting 24 rows | 0 |
| Backward steps in file order | 1 | 0 |
| Missing values in original rows | 0 | 0 |
| Absent grid slots within file span | 0 | 0 |
| Usable file-span fraction after duplicate exclusion | 99.9471% | 100% |
| Longest diagnostic gap | 60 minutes | 0 |
| Five-minute level correlation | 0.9969 | -0.0401 |
| Published annotation windows | 4 | 2 |

The repeated temperature hour is 7 January 2014, 02:00–02:55. Values differ
between repetitions. No evidence identifies which reading is correct; averaging,
keeping the first or treating the repeated segment as a new hour would invent a
resolution. All readings at ambiguous timestamps are therefore excluded from
diagnostic calculations, but kept in the raw observation export. Exclusion is a
documented assessment policy, not a verified repair of the source.

Temperature changes level and has strong persistence. CPU has a narrow central
distribution and intermittent large readings; adjacent correlation is close to
zero. These findings motivate separate baseline comparisons in Notebook 02, not
an algorithm choice in Notebook 01. Correlation need not reflect seasonality:
regimes and excursions affect it too. First differences are inspected only as a
diagnostic; they can themselves induce negative correlation.

The source README describes three machine anomalies, including planned shutdown,
while the combined annotations specify four windows. We retain the four published
windows without assigning individual causes. Labels are neither operational alarm
requirements nor verified failure/onset/repair records.

## Methodological decisions

1. **Explicit semantics and unknowns.** Keep naive recorded timestamps because the
   source timezone is undocumented. Do not infer UTC, a local calendar or data
   availability times. Temperature units are unspecified. Standard CloudWatch
   CPU units are percent, but export scaling/statistic are unconfirmed; retain
   values without conversion and mark units unknown. Physical thresholds and
   cross-source alignment require more metadata.
2. **Audit before sorting or cleaning.** The four-column representation separates
   observations from annotation metadata. Structural validation permits duplicates
   so the audit can reveal them. Model-readiness requires unique keys or an explicit
   policy; `observations_unresolved.parquet` intentionally makes this limitation clear.
3. **Conditional coverage.** The regular grid is anchored at each file's first
   timestamp, not the Unix epoch. File-span coverage cannot reveal missing records
   outside that span. Off-grid timestamps are reported and rejected by the regular
   diagnostic view rather than rounded. Irregular-source modelling remains future work.
4. **Preserve missing time.** Pearson lag correlations use finite pairs at exact
   elapsed lags. No interpolation, zero filling, compression of gaps or IID
   significance bands are used. Daily summaries are elapsed 24-hour blocks with
   at least 90% usable slots, a display rule rather than an alert threshold.
5. **Small reusable foundation.** Contracts and assessment are source-independent;
   `nab.py` and the manifest are source-specific. No inheritance, plugin discovery,
   topology abstraction, model fitting or alert policy is introduced. Simple
   interval-count validity is supported, but this two-gauge pilot does not validate
   count models, counter resets, exposure modelling or multivariate detection.

These choices follow the emphasis on temporal graphics and dependence in
[Forecasting: Principles and Practice](https://otexts.com/fpp3/graphics.html)
(statistical textbook). Annotation interpretation is grounded in NAB's
[label-construction code](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/nab/labeler.py)
and [Lavin & Ahmad, IEEE ICMLA 2015](https://arxiv.org/abs/1510.03336)
(peer-reviewed conference paper). Source descriptions are
[maintainer documentation](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/README.md),
not independent verification of collection or completeness. Standard CPU semantics
come from [AWS documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/viewing_metrics_with_cloudwatch.html).

## Validation and limitations

Notebook 01 was executed using Python 3.13.2, NumPy 2.5.4, Pandas 3.1.0,
Matplotlib 3.11.2, PyArrow 26.0.0, nbformat 5.11.1 and nbclient 0.11.0.
Its 10 code cells completed, with 3 figures visually reviewed. Actual source hashes
and exact runtime versions are recorded in the generated assessment manifest.
WSL execution has not been performed here.

The new tests cover duplicate conflicts, non-epoch sampling phase, off-grid rows,
missing/invalid values, count constraints, missing-lag pairs, constant series,
schema separation, pinned downloads, corruption and annotation endpoints. They use
small deterministic fixtures and do not require downloaded NAB files.

A full-suite run before the final immutable-version test was added produced
**132 passed, 1 failed**. The failure is the legacy
`tests/test_pipeline.py::test_frozen_end_to_end` score-range assertion. It was
reproduced on a separate clean export of unchanged main commit
`3333e789590ded14dd313ef42ffd32fd073b68a7` using the same environment. No legacy
source or tests were changed to hide or fix this unrelated failure. The suite must
not be described as entirely passing. The final NAB-only test result is recorded
in `NAB_VALIDATION.json`.

Fixed files and checksums establish reproducibility, not field authenticity or
complete anomaly labels. The two short, unrelated examples do not establish
company-wide generalisation, realistic false-alert budgets or predictive lead time.
Before Notebook 02, agree on input quarantine, adaptation and chronological
fit/calibration/validation policies; confirm what an actionable anomaly means.

## Git lineage and recovery

This branch starts from main commit `3333e789590ded14dd313ef42ffd32fd073b68a7`.
The retired scientific-foundation work was published at
`a57f33e0e883dd55d1e0ec555aec4635230658c9`; its local equivalent final tree was
also committed as `31f7eaf`. No changes from that branch were merged. The old
remote work is preserved under archive tag `archive/scientific-foundation-00-01`
before its development branch is removed.
