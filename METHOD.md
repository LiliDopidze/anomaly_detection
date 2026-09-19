# Method and critical assessment

## What is sound in the proposed architecture

Separate input mapping, causal validation, features, scoring, incident decisions
and event evaluation. Keep mathematical primitives independently testable.
Preserve explicit onset and impact labels, chronological splits, hysteresis and
one-to-one event matching. These address real statistical and operational errors.

The proposed folder structure is more fragmented than this implementation needs.
Use one package, one short module per stage, one configuration with clearly named
sections, and five notebooks. Avoid three YAML files for parameters used together,
copied temporal datasets, and an abstract superclass with only two implementations.
The previous implementation is preserved at
[commit c999529](https://github.com/LiliDopidze/anomaly_detection/tree/c999529696dba66dd0060eeebc40612f7a04a624).
We retain its principles of separated truth, explicit units, causal calculations,
missing-data abstention, temporal evaluation and protected final assessment.

## Synthetic data: evidence versus assumptions

Normal latent Rx power is a device baseline plus a 24-hour sinusoid and stationary
Gaussian AR(1) residual. For sampling duration dt and correlation time tau:

`phi = exp(-dt/tau)`

`e[t] = phi*e[t-1] + sigma*sqrt(1-phi²)*epsilon[t]`, with `e[0] ~ N(0,sigma²)`.

This gives stationary variance sigma² and lag-k correlation phi^k. Tests check
both. Gaussian noise satisfies the requested stationary Gaussian-or-pink choice;
pink noise is not added without evidence that its extra structure is needed.
Device phases differ. Independent sensor noise is added after physical state.
Independent random streams ensure changing collection loss does not redraw the
faults or their impact times. Missing polls stay missing.

Three mechanisms alter latent power, with a separate repair/end time:

- Random walk: increments `Normal(-0.15*dt, 0.06²*dt)` dB plus an initial -0.2 dB
  change. Negative drift does not mean every individual increment is negative.
- Exponential attenuation: `-0.2 - 0.35*expm1(min(hours/12, 3))` dB. This models
  accelerating loss in dB followed by a capped loss, not exponential linear power.
  A pure exponential decay in watts would be linear in dBm.
- Variance shift: an additional correlated zero-mean Gaussian component with
  0.5 dB standard deviation. It is a regime anomaly, not necessarily a mean loss
  or a customer-impacting event.

These rates, amplitudes, caps, device baselines and durations are explicit scenario
assumptions, not values established by a telecom field study. The long severe
faults make many cases easier than weak real degradations. The simulator starts
with a known normal baseline and places separated validation/test events; it does
not estimate field prevalence, overlapping-fault attribution or shared topology
failures. All these are future realism tests, not demonstrated capabilities.

BER is an illustrative clipped monotone mapping of latent optical margin:
`10**(-9 - (rx_latent - impact_threshold))`, bounded to [1e-12, 0.1]. It is not a
measured receiver BER curve. It is emitted for inspection but excluded from the
model, avoiding a redundant simulator-derived confirmation signal. Real BER
integration needs reviewed counter/measurement semantics and empirical calibration.
[ITU-T G.984.2](https://www.itu.int/rec/T-REC-G.984.2) supports the importance of
optical interface budgets and receiver requirements; it does **not** validate this
BER equation, a universal -27 dBm threshold, or our fault distributions.

Ground truth is emitted separately:

- `onset_time`: the first sample governed by the changed distribution.
- `observable_onset_time`: first available reading after an injected effect exceeds
  twice the baseline physical-noise SD; variance faults use their added SD.
  This is a declared visibility proxy, not a statistical detectability theorem.
- `impact_time`: first latent Rx sample below the hypothetical -27 dBm threshold.
  It can be absent; sensor readout noise cannot manufacture impact.
- `end_time`: repair/end of the injected mechanism, not the last alert timestamp.

A real customer-impact definition requires operator evidence, not this proxy.

## Causal validation and feature mathematics

Resampling uses right-closed, right-labelled bins: a reading at 00:02 is available
at the 00:05 decision, not at 00:00. No interpolation or forward fill is performed.
The adapter rejects duplicate keys and ambiguous/nonexistent local timestamps.
Invalid physical values become missing. Units are explicit, never guessed.

Training-only least squares fits an intercept and daily sine/cosine per device.
Residuals are divided by a training MAD scale, floored at 0.05 dB. The baseline must
be representative and mostly healthy; ordinary least squares is not robust to
heavy contamination. `seasonal: false` enables a median-only reference for an
ablation. Weekly or changing seasonal patterns are not modelled automatically.

The feature engineer produces the requested five features plus the slope needed
by Tier 1. All windows end at the current observation; CUSUM's reference window
ends at the preceding observation. Missing data resets window/state history.

| Feature | Implemented meaning | Important qualification |
|---|---|---|
| CoV | Population SD / absolute mean of linear optical power | dBm is logarithmic, so its CoV is not physically scale invariant |
| Lag-1 correlation | Centred lag product sum / full centred squared sum | Exact requested estimator; constant finite windows return zero; invalid windows return NaN |
| Negative CUSUM | `max(0, previous + prior_rolling_mean - current - C)` on normalised residuals | C=0.25 is in residual units; rolling means can absorb slow loss |
| Shannon entropy | `-sum(p*log(p))` using fixed residual bins | Fixed bins preserve comparability; extreme loss can reduce entropy |
| EWMA acceleration | EWMA of second difference / dt² | Scale-normalised and cadence-aware; differentiation amplifies noise |
| EWMA slope | EWMA of first difference / dt | Tier 1 uses negative slope; plateaued loss need not maintain a high slope score |

Empty or non-finite windows return NaN for CoV, autocorrelation and entropy.
Zero-mean CoV is undefined (NaN); constant positive CoV and constant-window
entropy are zero. These cases are covered by parameterised tests.

Entropy edges are `[-inf,-3,-2,-1,0,1,2,3,inf]` in training-standardised residual
units. EWMA alpha is `1-exp(-dt/smoothing_hours)`. Both the one-hour memory and
12-observation rolling window are choices to test, not universal optical constants.
A change in poll cadence should prompt a review of window and debounce durations.

[NIST's CoV guidance](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/coefvari.htm)
supports using a ratio scale. An additive dBm offset becomes a multiplicative
linear-power factor, leaving CoV unchanged. Tests also verify invariance of the
other features after refitting local references. This does not prove transfer
across sensor quantisation, noise, hardware classes or new operating regimes.
[NIST's EWMA guidance](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc314.htm)
supports sensitivity to small sustained changes; independent-Gaussian chart
false-alarm guarantees do not automatically apply to this correlated telemetry.

## Detectors, calibration and temporal splits

Tier 1 scores negative EWMA slope. Tier 2 uses Isolation Forest on six complete
features, fitted on at most 20,000 training rows with a fixed seed. Missing feature
vectors abstain rather than receive imputed healthy values. IDs and truth are not
features. The forest's raw anomaly score is negated `score_samples`, following
[scikit-learn's convention](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

Each raw score is mapped to [0,1] using an interpolated empirical rank fitted on
normal calibration data. These are anomaly ranks, **not** posterior probabilities
of faults or calibrated p-values. Serial dependence and multiple entities still
require workload validation. The maximum of the two ranks is a third candidate;
it is not itself a calibrated probability and can increase false alarms.

Chronological fractions: 40% train, 15% score calibration, 20% validation, 25% test.
No future samples enter reference fitting, forest fitting or score calibration.
Causal rolling history is allowed across split boundaries; resetting every split
would introduce artificial cold starts. Incident evaluation starts with closed
state at each evaluation boundary; crossing faults are excluded and counted.
[Scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
supports separating fitting from later evaluation. Fractions are engineering choices.

Validation compares N in {2,3,6}, M in {3,6}, with high=0.99 and low=0.8, separately
for both tiers and their maximum. Selection prefers the declared workload budget,
then pre-impact recall, then lower workload with deterministic ties. If no policy
meets the budget, the saved experimental policy is explicitly flagged as failing.
No default setting constitutes an operator-approved alarm policy.

## Incidents and evaluation

N successive scores strictly above high open an incident at the actual Nth decision.
M successive scores strictly below low close it. Equality satisfies neither rule.
Missing scores reset confirmation counters but keep an open incident unknown.
A gap beyond six hours administratively closes at the last valid observation,
labelled `telemetry_gap`; it is not evidence of recovery. Active incidents retain
missing end times. State persists across chunks; repeated/overlapping chunks fail.
Outputs contain new closures and active snapshots, keyed by stable incident IDs.

Matching is deterministic maximum-cardinality one-to-one matching by entity and
alert start within the fault's physical interval. It never credits an entire
interval's individual points. Overlapping labels can make attribution ambiguous;
matching is association, not causal proof. Earlier warnings are not retroactively
excused by a later fault. Boundary events are reported separately.

A defensible warning opportunity is declared before scoring: an impacting fault
has an observable-onset proxy plus at least three consecutive observed Rx readings
by impact minus 30 minutes. It does not change with detector success or the tuned
N. This is an operational proxy, not a guarantee of sufficient statistical power.

- Pre-impact recall: matched warnings before impact / faults with that opportunity.
- Nuisance workload: unmatched plus duplicate incidents per 1,000 scheduled monitored
  entity-days. Missingness coverage is reported alongside exposure; telemetry loss
  must not be interpreted as good detector performance.
- Delay: actual alert emission minus observable-onset proxy, in minutes. Delays are
  reported for detected events; misses remain explicit and are not assigned zero.

## Results and remaining work

Default validation, 24 faults: the selected Isolation Forest policy (N=6, M=3)
detected 24/24 and warned before impact on 16/16 declared opportunities. Four
unmatched incidents correspond to 29.8 nuisance incidents per 1,000 entity-days,
above the configured budget of five. Combining tiers increased unmatched incidents
to six. These are tuned validation results, not final-test or operator accuracy.

Keeping that incident policy fixed while refitting local baselines/calibration:

| Scenario, 12 entities | Detected | Pre-impact | Unmatched | Duplicates | Nuisance / 1,000 days |
|---|---:|---:|---:|---:|---:|
| New seed | 12/12 | 8/8 | 3 | 3 | 89.3 |
| Double physical noise, 10% missing polls | 12/12 | 5/8 | 2 | 6 | 119.1 |
| No injected faults | N/A | N/A | 6 | 0 | 89.3 |

Complete-score coverage was 78.3% in the default validation run and 26.7% in
the noisier/10%-missing run. Requiring a full window after each missing reading
causes this loss of coverage; missingness handling is a priority before deployment.
Neither absence of a score nor a long telemetry gap means a healthy network.

The stress results weaken the clean-scenario claim. No candidate should be labelled
production ready. Final performance assessment remains unopened. Tiny event counts,
known-normal calibration, strong long faults, independent devices and simplified
seasonality all limit generalisation.

Next work should challenge weak/slow faults, baseline contamination, quantisation,
shared faults, topology changes and seasonal shifts; measure uncertainty across
independent runs; compare feature ablations and a fixed-reference level baseline;
and validate against representative operator measurements and incidents. A bounded
streaming feature implementation, durable operational state, monitoring and shadow
operation are still needed for deployment. Add these only against concrete operating
requirements; the current library does not pretend they already exist.
