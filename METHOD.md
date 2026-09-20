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

## Expanded generator (v7): evidence versus assumptions

The default is **96 ONTs over 90 days at five-minute cadence**: 2,488,320 rows.
`make_topology` produces one static row per ONT, with OLT, globally unique PON-port
and splitter IDs. Defaults give 12 splitters, six ports and two OLTs. Membership
is an illustrative allocation (8 ONTs/splitter, 2 splitters/port, 4 ports/OLT),
not a prescribed operator design. No operational events, peer detector, historical
inventory engine or incident grouping is introduced.

The 14 measurements are:

| Native field | Meaning / unit |
|---|---|
| `rx_dbm` | Downstream receive power at the ONT, dBm |
| `upstream_rx_dbm` | Upstream receive power at the OLT for that ONT, dBm |
| `ont_tx_dbm` | ONT transmit power, dBm |
| `olt_tx_dbm` | Shared PON-port transmit power, dBm |
| `ont_temperature_c`, `olt_temperature_c` | Optical module temperatures, Celsius |
| `ber`, `upstream_ber` | Illustrative instantaneous pre-FEC bit error ratios |
| `{downstream,upstream}_fec_corrected_codewords` | Corrected codewords in the preceding interval |
| `{downstream,upstream}_fec_uncorrectable_codewords` | Uncorrectable codewords in the preceding interval |
| `{downstream,upstream}_fec_total_codewords` | Received codeword opportunities in the preceding interval |

These measurement categories follow [ETSI GS F5G 011, sections 8.3–8.4](https://www.etsi.org/deliver/etsi_gs/F5G/001_099/011/01.01.01_60/gs_F5G011v010101p.pdf)
and [ITU-T G.988](https://www.itu.int/rec/T-REC-G.988). Actual device availability,
accuracy, aggregation and counter semantics must still be mapped explicitly.
The generator is not a standards-compliance implementation or a calibrated digital twin.

Received power follows directional link budgets: OLT Tx minus downstream path
loss, and ONT Tx minus upstream path loss. The path components are related but
have an assumed wavelength-dependent difference. Each optical-loss fault changes
both paths, with an assumed upstream multiplier of 1.1. Tx power stays independent
of the injected path fault. OLT Tx/temperature are shared per port and repeated in
ONT rows for convenience: they are not independent observations of the OLT.

Module temperatures have daily patterns and correlated residuals. The OLT also has
an assumed slow annual component; 90 days cannot validate an annual cycle. Small
explicit thermal coefficients link temperatures to transmitter power. Missed polls are independent of severity in this revision; loss of remote
telemetry during severe optical failure is not yet modelled. Stationary
Gaussian residuals use `phi=exp(-dt/tau)` and innovation SD `sigma*sqrt(1-phi²)`,
including a stationary initial value. Sensor noise is added after physical state;
changing `sensor_noise_db` cannot manufacture FEC errors or impact labels.

Fault signatures remain negative-drift random walks, accelerating dB attenuation
and variance shifts. Their duration is now lognormal with a 36-hour median and
log-SD 0.5, truncated by the scenario boundary. Severity is multiplied by a
uniform 0.4–1.4 factor. Longer observation windows therefore do not automatically
create proportionally longer, more severe faults. Faults remain individual-ONT
scenarios; shared topology faults are deferred. The first 55% is deliberately
fault-free and two faults per ONT are scheduled into development/final periods.
These are enriched scenarios, not estimates of real arrival rates or prevalence.

The FEC approximation uses optional GPON RS(255,239) coding: an ideal decoder can
correct up to eight erroneous byte symbols. GPON rates here are 2.48832 Gbit/s
downstream and 1.24416 Gbit/s upstream, consistent with
[ITU-T G.984.3](https://www.itu.int/rec/T-REC-G.984.3). Always-on FEC, full downstream
coding, equal upstream allocations, omitted framing/burst overhead and independent
bit errors are simplifying assumptions. This is not an XGS-PON/LDPC counter model.

For pre-FEC bit error probability p, symbol error probability is `q=1-(1-p)^8`.
For `K~Binomial(255,q)`, corrected probability is `P(1<=K<=8)` and uncorrectable
probability is `P(K>8)`. Joint categorical sampling guarantees corrected plus
uncorrectable never exceeds total. Corrected counts can fall at severe corruption
as uncorrectable counts rise. Upstream allocation sums to at most the port line
rate across its ONTs; downstream broadcast reception is counted separately per ONT.

The assumed BER response is `10**clip(-5-(latent_rx-impact_threshold), -12, -1)`.
Standards do **not** establish this receiver curve or these scenario thresholds.
The counters use piecewise-constant physical state over the preceding interval;
a state change at t first affects counts reported at t+dt. The first row has no
preceding interval and its counters are missing. Counts are interval totals, not
cumulative counters, so there are no synthetic resets or reboot events. Coarser
resampling sums observed interval counts; missing portions remain unobserved and
must not be interpreted as a fully measured interval. Ratios should use matching
corrected/uncorrectable and total observations.

The separate truth log retains distribution-change onset, an observable-onset
proxy, hypothetical impact and repair time. Impact is the first latent crossing
of either -27 dBm downstream or -28 dBm upstream, configurable assumptions rather
than universal receiver limits or customer SLAs. Variance shifts need not cause
impact. The visibility proxy uses the known injected effect, never model scores.

The adapter accepts the expanded measurements. For this revision the detector
remains the downstream-Rx baseline: no performance benefit from the added signals
is claimed until feature ablations demonstrate it. Topology never enters the
feature matrix. `generation_checks.json` records structural/count invariants;
passing them demonstrates consistency, not empirical realism.

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

## Current v7 development check

The full default run completed locally: 2,488,320 telemetry rows, 96 topology rows,
192 injected faults across development/final scenarios. The 11 generation checks
passed. Final detector performance assessment remains unopened.

The selected validation policy was Isolation Forest with N=6, M=3. It detected
95/96 validation faults and warned before impact on 46/46 declared opportunities.
However, 112 unmatched plus six duplicate incidents produced 68.3 nuisance
incidents per 1,000 entity-days, above the configured budget of five. These are
tuned synthetic validation results, not evidence of production readiness or of
benefit from the additional measurements (which are not yet detector features).

The final full development run took about 197 seconds on this machine and wrote
about 190 MB of run artifacts. These are observations, not Colab resource or timing
guarantees. The synthetic family, known-normal fit/calibration periods, hypothetical
impact thresholds and individual fault mechanisms still limit generalisation.

## Historical v6 results and remaining work

The following figures describe the earlier 24-ONT/28-day generator, **not v7**.
They are retained for context and must not be compared as a like-for-like dataset.

Earlier validation, 24 faults: the selected Isolation Forest policy (N=6, M=3)
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

## Native-data qualification before adaptation

Notebook 01 checks native telemetry before any unit conversion or resampling. The
full dataset receives structural checks only. Development data supplies missingness,
gaps, distributions, fault contrasts and warning-opportunity screens; the final
period is excluded from those reports. Measurements are distinguished from engineered
features. Qualification is a review step, not a claim of operator-level validity.

For healthy downstream telemetry, subtract Rx from measured OLT Tx and fit a daily
harmonic per ONT. The remaining path-loss residual has expected variance
`noise_db**2 + sensor_noise_db**2` and lag-one correlation
`exp(-dt / correlation_hours) * noise_db**2 / expected_variance`, apart from rounding
and finite-sample harmonic estimation. Diagnostic bands are 25% for variance,
0.1 absolute for correlation, and max(0.05 dB, 20%) for daily amplitude. These are
explicit screening assumptions, not standards or formal confidence intervals.
Missing samples are not compressed when estimating adjacent-sample correlation.
Fault-versus-preceding-day contrasts expose very easy faults and potential
missingness shortcuts; they do not prove independence or causal effects.

Canonical definitions describe units and gauge/interval-count semantics. The
synthetic generator is one explicit source mapping, separate from the generic
adapter. All mapped columns are required; unknown units or counter semantics fail.
Invalid numeric observations become missing. Cumulative counters require an explicit
upstream conversion; no reset behaviour is inferred. Detector requirements are
checked after adaptation. This separates input portability from demonstrated
cross-company detection performance.
