# Pilot and model maintenance

The software experiment is complete only when its final artifacts and correctness gates pass. It does not run an operator pilot. A no-go synthetic result keeps deployment disabled; the work below is a preparation procedure, not evidence that a pilot occurred.

## Before a pilot

Agree the monitored population, metric units and cadence, supported operating states, latency deadlines, quality expectations, alert budget, observable onset definition, required lead and independent ascertainment process. Confirm the proposed 95% recall and 5 unmatched investigations per 1,000 entity-days targets with operators; they are project hypotheses, not standards. Use several independent sites and sufficient ascertainable faults. Do not infer sample size or precision from overlapping windows.

Verify a normal-history eligibility ledger, channel groups, scale floors, coverage by condition and the absence of unresolved incidents in refit data. Freeze a version with source/dependency/configuration/data hashes, training identities and bank hashes. Replay all recursive history and lifecycle state before activating a shadow run.

## Operator review and fault ascertainment

Record each investigation's immutable ID, first notification, model/policy version, scope, contributing alerts and later membership updates. Review alerts as confirmed relevant, benign behaviour, duplicate, data quality, or unresolved; record review time, reviewer and evidence. Do not turn unresolved alerts into verified false positives.

Independently review maintenance records, fault tickets and a prespecified random sample of non-alert monitoring intervals. Record observable onset, impact, end, timing uncertainty, whether a precursor was measurable, and the time each label became available. This catches missed faults that alert-only review cannot ascertain. Keep diagnostic alarms and simulator latent states outside predictors. Unlabelled intervals cannot establish recall.

Use the templates under `pilot/`. They are empty operational forms. Keep real review data outside version control. Join them by stable investigation/fault IDs, never by a guessed match or an inferred cause.

## Monitoring and replacement

Review quality reasons and longest gaps, any-route and full-capability coverage, residual centre/scale and seasonal dependence, normal-bank support, channel availability, workload, time in alarm, recovery delay, duplicates, harmful merges and independently ascertained outcomes. Compare to the frozen contract and show raw event/exposure counts. Drift triggers investigation; it does not make recent anomalies eligible normal history.

For a proposed replacement: record the reason and eligible interval; fit only information available at that time; replay and compare against the active version under the same schedule; shadow-test without altering the active policy; obtain operational approval using the recorded workload and fault evidence. Preserve the full old model, bank, policy and activation ledger. Activate at a recorded availability time. Replay sufficient contiguous context under the new model to initialise recursive state. Link new constituent alerts to ongoing external investigations explicitly; do not clear an unresolved investigation at a version boundary.

Rollback restores the prior frozen artifacts and policy at a logged time. Preserve every original notification and membership update. Reconcile still-active investigations by stable ID, replay the necessary context and explain any administrative closure. Evaluate any future automated update rule as a complete policy; frozen-model evidence cannot validate adaptation.
