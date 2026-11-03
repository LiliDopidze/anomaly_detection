# Independent review and delivery — notebooks 00 and 01

## A. Assessment of the original approach

The original implementation had useful foundations: stationary AR initialization,
physical loss accounting in logarithmic units, separate ground truth, causal FEC
intervals, chronological splits, and tests of sensor/collection independence.
These were retained. A more complicated detector would not repair the weaknesses
below.

| Priority | Original finding | Disposition |
|---|---|---|
| High | `validation.py` summed any available count with `min_count=1`. A missing numerator interval could be divided by a more complete denominator. | **Replace:** complete source-interval aggregation and joint availability of declared count groups. |
| High | Coverage derived from available rows could hide missing boundary rows or entirely absent series. | **Add:** explicit schedule-based denominators and source sampling audits. |
| High | `generator.py` allocated faults to every entity, placed starts in narrow windows, imposed fixed initial jumps on gradual faults, and restored mean faults abruptly. | **Modify:** healthy controls, broader controlled windows, gradual onset/repair and weak-effect experiments. |
| High | A stationary daily reference and visually convincing faults could be mistaken for a realistic operating environment. | **Keep as reference; challenge with** weekly, drift, changing-variance and contiguous-gap controls across seeds. |
| Medium | The adapter's metric registry was optical-only, despite a generic long-table representation. | **Separate:** generic definitions and supplied registries from `pon.py`; retain the PON default for existing callers. |
| Medium | Topology did not generate shared physical interventions, and independent counts of affected ONTs could inflate event totals. | **Add:** port-scoped causes with entity-level labels and a shared `cause_id`. Later evaluation still needs cause-aware accounting. |
| Medium | Dataset reuse compared configuration but not generator code, resolved defaults or runtime. | **Modify:** generation provenance and explicit stale-run rejection. |
| Medium | Seasonal comparisons included signal types without sufficiently explicit measurement semantics; the methodology file was empty. | **Modify:** gauge-only baseline comparisons, separate count analysis, a research decision record and interpreted outputs. |

Evidence was reviewed against original revision
[`3333e78`](https://github.com/LiliDopidze/anomaly_detection/tree/3333e789590ded14dd313ef42ffd32fd073b68a7),
especially `generator.py`, `adapter.py`, `validation.py`, `optics.py`,
`diagnostics.py`, `pipeline.py`, and both notebooks. Downstream interfaces and tests
were inspected for compatibility. No direct use of future truth as a model feature
was identified in the reviewed paths; that is narrower than a proof that every
possible workflow is leakage-free.

## B. Research-backed decisions

The full assumptions, alternatives and limitations are in [METHOD.md](METHOD.md).

* **Start with measurement semantics.** A gauge, exposure-normalized fraction and
  cumulative counter are different observations. [ETSI's telemetry definitions](https://www.etsi.org/deliver/etsi_gs/F5G/001_099/011/01.01.01_60/gs_F5G011v010101p.pdf)
  support this distinction; standards do not establish synthetic distributions.
  Rejecting incomplete count bins is conservative. A partial-exposure model is an
  alternative, but would need explicit interval/exposure metadata.
* **Keep an interpretable stochastic reference.** For stationary AR(1), use
  innovation variance `sigma²(1-phi²)` and a stationary initial state. Measured
  residual variance includes independent sensor variance. A state-space model or
  richer count distribution is justified only after real data challenges this
  reference; additional simulator parameters alone do not establish realism.
* **Compare baselines chronologically.** Constant versus daily versus daily+weekly
  regressions are evaluated on an inner training holdout, following established
  [forecast evaluation principles](https://otexts.com/fpp3/toolbox.html).
  Fixed harmonics fail under changing phase/amplitude; one holdout does not establish
  production calibration. The original function's `*_holdout_r2` column names remain
  for compatibility, but notebooks correctly label them relative MSE skill.
* **Preserve time gaps.** Pairwise lag correlations retain actual elapsed offsets;
  dropping missing observations can change the meaning of a lag, as documented by
  [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html).
  Pair counts are displayed; no iid confidence bands or significance claims are made.
* **Use controls and interventions.** Separate random streams enable paired
  experiments. [Benchmark research](https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ICDE2022_TSAD.pdf)
  motivates retaining weak, non-impacting and healthy cases. The three fault strengths
  are predeclared experimental settings, not fitted population severities.

Anodot's [vendor-authored workshop paper](https://proceedings.mlr.press/v71/toledano18a.html)
supports signal-specific treatment. Merlion's [technical report](https://arxiv.org/abs/2109.09265)
provides useful preprocessing/postprocessing separation. [MathWorks documentation](https://uk.mathworks.com/products/time-series-anomaly-detection.html)
illustrates synthetic-anomaly workflows. These sources informed the boundaries;
their automation/tooling claims were not treated as evidence of transferability.

## C. Implementation and interfaces

* `contracts.py`, `assessment.py`: small explicit generic definitions, sampling
  diagnostics and scheduled coverage; no factories, plugins or inheritance hierarchy.
* `pon.py`, `adapter.py`, `validation.py`: domain registry, configurable adaptation,
  UTC-aligned causal bins and complete paired-count exposure.
* `generator.py`, `optics.py`: independent fault allocation/shape streams, normal
  stress scenarios, event strength, repair, shared causes and stronger invariants.
* `diagnostics.py`: removes a dependency on model-feature code for descriptive daily fits.
* `pipeline.py`: only dataset preparation/provenance changes; detector orchestration
  and model-selection logic remain unchanged.
* `configs/foundation.yaml`, notebooks 00/01, `tools/execute_foundation.py`: small
  reproducible analyses, saved outputs and an explicit-environment notebook runner.
* Tests and documentation: additional scientific/contract checks and one Pandas
  compatibility correction that explicitly excludes missing scores from a range test.

The four-column adapted schema and five-column validated schema are preserved.
Ground truth gains `recovery_start_time`, `cause_id`, `scope` and
`duration_truncated`; existing required fields remain. Generated numerical data
intentionally changes. Old materialized runs must be regenerated in a fresh output
folder. Count coarsening now requires an explicit `source_interval`; partial bins
become missing. The simulator accepts day-dividing cadences with sufficient polls.

Notebooks 02–06 and model implementations were not modified. They still select
their original config/output path. Full integration tests establish interface
compatibility, not scientific readiness of their existing evaluation metrics.

## D. Validation evidence

[FOUNDATION_VALIDATION.json](FOUNDATION_VALIDATION.json) records the runtime,
code hashes and numerical results. Verification ran locally on macOS/Python 3.13;
WSL/Python 3.14 execution has not been claimed.

* Both notebooks executed without cell errors: **9 and 8 code cells**, respectively,
  with **four inspected figures each**. Results and narrative are saved in the notebooks.
* **133 tests passed**. They cover mathematical identities, fixed-seed reproduction, five independent
  reference seeds, incomplete count exposure, a non-telecom registry, boundary/absent
  series coverage, causal prefixes, repair timing, shared causes, physical/count
  invariants and the existing end-to-end pipeline. See the JSON for the final count.
* Reference experiment: **6 ONTs, 3 ports, 90 days, 5-minute cadence**. Each ONT has
  **10,368 scheduled training slots**; Rx coverage is **97.83–98.11%**.
* Fitted daily loss amplitude: **0.246–0.256 dB**, versus configured **0.25 dB**.
  Residual variance/theory: **0.957–1.060**. Observed lag-one correlation:
  **0.665–0.689**, versus theoretical **0.677**. These are consistency checks,
  not field calibration or confidence intervals.
* Median daily holdout skill: **0.783**. Median gain from weekly terms:
  **−0.0013**. The reference does not justify adding a weekly component.
* Across three stress seeds, daily-adjusted variance/reference rises to about
  **3.3–3.5** with a weekly component, **4.7–4.8** with expected drift, and
  **3.6–3.7** with changing variance. No faults were injected: a fixed baseline
  could confuse unmodelled normal structure with anomalies.
* Strengths **0.1 / 0.5 / 1.0** each include nine complete development events.
  **0 / 1 / 2** reach the impact proxy. These are oracle event characteristics,
  **not detector recall**. Non-impacting events are retained.

The second review corrected UTC alignment, partial FEC tuple checks, repair-duration
effects on earlier random draws, weak-fault coverage, a missing subplot date axis,
and an analysis path that should use validated values rather than native values.

## E. Architecture assessment

**Reusable across companies and industries:** explicit signal definitions, source
mapping, duplicate/time checks, fixed-cadence aggregation, paired-count availability,
scheduled coverage and chronological diagnostic principles. A manufacturing
temperature/reject-count test exercises this directly.

**Still domain-specific:** optical names/units, physical loss relationships, GPON
topology, FEC assumptions, fault mechanisms and impact thresholds. The downstream
optical detector is not made industry-agnostic by this change. A new domain needs
its own registry/source adapter and reviewed baseline/alert semantics.

**Experiment-specific:** the clean first 55%, controlled event windows, assumed
noise/seasonality, simplified repair and seed protocol. These must not become
unexamined production assumptions.

## F. Git delivery

Development branch: **`codex/scientific-foundation-00-01`**, based on the original
main revision above. Main is unchanged. The delivery message records the verified
GitHub commit. Use a new directory to keep your existing WSL checkout intact.

From a WSL terminal:

```bash
mkdir -p ~/projects
git clone --branch codex/scientific-foundation-00-01 --single-branch https://github.com/LiliDopidze/anomaly_detection.git ~/projects/anomaly_detection_foundation
cd ~/projects/anomaly_detection_foundation
git branch --show-current
git status
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
code .
```

Select `.venv/bin/python` as the notebook kernel in VS Code. Run notebook 00
before 01. The validated execution used Python 3.13; WSL and Python 3.14 have not
been tested in this review.

## G. Next-stage readiness

**Ready for controlled baseline experiments; not for production or cross-industry
performance claims.** The next steps are:

1. Inspect these notebooks together and agree the actual observation/alert contract.
2. Obtain real operational histories with known timestamps, exposure, resets,
   missingness, maintenance context and reviewed outcomes. Calibrate the simulator
   against measured temporal and cross-metric behaviour.
3. Evaluate a simple per-signal residual baseline on chronological blocks and held-out
   entities, including healthy controls and each normal-operation stress.
4. Before fleet metrics, revise later evaluation to use scheduled exposure, coverage,
   shared-cause grouping, event duration/detection delay and an agreed alert budget.
5. Add adaptive, multivariate or more complex models only when those comparisons
   show a material benefit under the operational objective.

The [Fraunhofer OLT-OTDR dataset](https://www.hhi.fraunhofer.de/en/pn-software/olt-otdr-dataset.html)
is relevant for physical fault classification/localisation: it contains experimentally
introduced faults and complementary OLT/OTDR measurements, including augmented data.
Do not assume its samples provide the longitudinal healthy exposure or genuine
pre-failure chronology needed for early-warning evaluation. Inspect those semantics
and keep augmented copies/testbeds separated before deciding its role.

Unresolved scientific limits include the uncalibrated power-to-BER curve, idealized
coding, strongly shared directional loss, simplified recovery, absence of overlapping
causes and outage-dependent missingness, and the distinction between optical threshold
impact and actual customer-service failure. Synthetic consistency cannot resolve them.
