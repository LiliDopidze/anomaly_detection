"""Explicit metric contracts and fixed decision schedules; never backfill decisions."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

KEYS = ["entity_id", "metric_name", "event_time"]
CANONICAL = KEYS + ["available_time", "value", "quality"]


@dataclass(frozen=True)
class Metric:
    unit: str
    cadence_minutes: int = 5
    kind: str = "gauge"
    aggregation: str = "mean"
    transform: str = "identity"
    direction: str = "two-sided"
    floor: float = 0.05
    minimum: float = -np.inf
    maximum: float = np.inf
    supported: bool = True
    min_observations: int = 100
    min_hours: float = 48
    min_cycles: int = 3

    def __post_init__(self):
        if self.kind not in {"gauge", "interval_count", "counter"}:
            raise ValueError("Unknown measurement kind")
        if self.aggregation not in {"mean", "sum"}:
            raise ValueError("Declare aggregation")
        if self.transform not in {"identity", "log1p", "log"}:
            raise ValueError("Unsupported transform")
        if self.direction not in {"high", "low", "two-sided"}:
            raise ValueError("Declare adverse direction")
        if not np.isfinite(self.floor) or self.floor <= 0 or self.cadence_minutes <= 0:
            raise ValueError("Positive cadence and scale floor required")
        if self.minimum >= self.maximum or self.min_observations < 3 or self.min_hours <= 0:
            raise ValueError("Invalid support or range")

    def to_dict(self):
        return asdict(self)


def transform(values, metric):
    x = np.asarray(values, dtype=float).copy()
    x[~np.isfinite(x) | (x < metric.minimum) | (x > metric.maximum)] = np.nan
    if metric.transform == "log":
        x[x <= 0] = np.nan
        x = np.log(x)
    elif metric.transform == "log1p":
        x[x < 0] = np.nan
        x = np.log1p(x)
    return x


def schedule(entities, registry, start, end, latency_minutes=0):
    """Rows describe [event_time, interval_end); exposure does not depend on data."""
    if latency_minutes < 0 or pd.Timestamp(end) <= pd.Timestamp(start):
        raise ValueError("Invalid monitoring interval")
    frames = []
    for entity in sorted(set(entities)):
        for name, spec in registry.items():
            times = pd.date_range(start, end, freq=f"{spec.cadence_minutes}min", inclusive="left")
            if times.tz is None:
                raise ValueError("Use timezone-aware schedules")
            frames.append(pd.DataFrame({"entity_id": entity, "metric_name": name,
                "event_time": times, "decision_time": times + pd.Timedelta(minutes=latency_minutes),
                "interval_end": np.minimum(times + pd.Timedelta(minutes=spec.cadence_minutes), pd.Timestamp(end)),
                "grid_index": np.arange(len(times))}))
    return pd.concat(frames, ignore_index=True)


def validate(observations, expected, registry):
    """First immutable decision at each deadline. Late observations remain quality evidence."""
    if set(observations.columns) != set(CANONICAL):
        raise ValueError(f"Expected only {CANONICAL}; truth/context are separate")
    if observations.duplicated(KEYS).any() or expected.duplicated(KEYS).any():
        raise ValueError("Ambiguous duplicate observation or schedule")
    obs = observations.copy()
    for field in ["event_time", "available_time"]:
        if not isinstance(obs[field].dtype, pd.DatetimeTZDtype) or obs[field].isna().any():
            raise ValueError("Use explicit timezone-aware nonmissing times")
        obs[field] = obs[field].dt.tz_convert("UTC")
    if (obs.available_time < obs.event_time).any():
        raise ValueError("Availability precedes event time")
    if obs.entity_id.isna().any() or obs.entity_id.astype(str).str.strip().eq("").any():
        raise ValueError("Missing entity")
    if not set(obs.metric_name).issubset(registry):
        raise ValueError("Unregistered metric")
    check = obs.merge(expected[KEYS], on=KEYS, how="left", indicator=True)
    if check._merge.ne("both").any():
        raise ValueError("Observation outside declared schedule; aggregate explicitly first")
    frame = expected.merge(obs, on=KEYS, how="left", validate="one_to_one")
    frame["quality"] = frame.quality.fillna("missing")
    frame.loc[frame.available_time.gt(frame.decision_time), "quality"] = "late"
    frames = []
    for (entity, name), part in frame.groupby(KEYS[:2], sort=True):
        part = part.sort_values("event_time").copy()
        spec = registry[name]
        if spec.kind == "counter":
            raise ValueError("Convert counter with explicit reset/wrap handling first")
        bad = ~np.isfinite(part.value) | ~part.value.between(spec.minimum, spec.maximum)
        if spec.kind == "interval_count":
            bad |= part.value.lt(0) | part.value.ne(np.floor(part.value))
        part.loc[bad & part.quality.eq("ok"), "quality"] = "invalid_value"
        part["value"] = part.value.where(part.quality.eq("ok"))
        valid = part.value.notna()
        new = ~valid.shift(fill_value=False) | part.event_time.diff().ne(pd.Timedelta(minutes=spec.cadence_minutes))
        part["segment"] = new.cumsum().where(valid, -1).astype(int)
        part["row_id"] = [f"{entity}|{name}|{t.isoformat()}" for t in part.event_time]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def counter_rate(values, times, cadence_minutes, modulus=None, reset=None, wrapped=None):
    """A negative delta is unknown unless a wrap flag and modulus are supplied."""
    x = np.asarray(values, float)
    t = pd.DatetimeIndex(times)
    dt = np.diff(t.asi8) / 1e9
    if np.any(dt <= 0):
        raise ValueError("Counter times must increase")
    change = np.diff(x)
    reset = np.zeros(len(x), bool) if reset is None else np.asarray(reset, bool)
    wrapped = np.zeros(len(x), bool) if wrapped is None else np.asarray(wrapped, bool)
    wrap = wrapped[1:] & (change < 0) & ~reset[1:]
    if wrap.any():
        if modulus is None or modulus <= 0 or np.any((x < 0) | (x >= modulus)):
            raise ValueError("Valid modulus and range required for declared wraps")
        change[wrap] += modulus
    valid = np.isfinite(change) & (change >= 0) & ~reset[1:] & (dt == cadence_minutes * 60)
    result = np.full(len(x), np.nan)
    result[1:] = np.where(valid, change / dt, np.nan)
    return result


def interval_ratio(numerator, denominator, numerator_start, denominator_start,
                   numerator_end, denominator_end):
    if not np.array_equal(numerator_start, denominator_start) or not np.array_equal(numerator_end, denominator_end):
        raise ValueError("Incompatible numerator/denominator intervals")
    n, d = np.asarray(numerator, float), np.asarray(denominator, float)
    valid = np.isfinite(n) & np.isfinite(d) & (d > 0) & (n >= 0) & (n <= d)
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=valid)


def aggregate_complete(observations, source_minutes, target_minutes, aggregation):
    """Explicit right-closed complete-bin aggregation, retaining latest availability."""
    if target_minutes % source_minutes or target_minutes < source_minutes or aggregation not in {"sum", "mean"}:
        raise ValueError("Require compatible intervals and sum/mean aggregation")
    if observations.duplicated(KEYS).any():
        raise ValueError("Duplicate source intervals")
    records = []
    for (entity, metric), part in observations.groupby(KEYS[:2]):
        for end, group in part.set_index("event_time").resample(f"{target_minutes}min", closed="right", label="right"):
            required = pd.date_range(end-pd.Timedelta(minutes=target_minutes-source_minutes), end, freq=f"{source_minutes}min")
            complete = group.index.sort_values().equals(required) and group.quality.eq("ok").all() and np.isfinite(group.value).all()
            records.append(dict(entity_id=entity, metric_name=metric, event_time=end,
                available_time=max(end, group.available_time.max()) if len(group) else end,
                value=getattr(group.value, aggregation)() if complete else np.nan,
                quality="ok" if complete else "incomplete_interval"))
    return pd.DataFrame(records, columns=CANONICAL)
