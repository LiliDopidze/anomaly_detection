"""Local frozen median or bounded training-qualified Fourier references."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .validation import transform


def design(times, origin, period_hours=24):
    h = (pd.DatetimeIndex(times)-origin).total_seconds().to_numpy()/3600
    return np.column_stack([np.ones(len(h)), np.sin(2*np.pi*h/period_hours), np.cos(2*np.pi*h/period_hours)])


@dataclass
class References:
    registry: dict
    models: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)

    def fit(self, data, training_end, allow_seasonal=True):
        if self.models:
            raise ValueError("Reference already fitted")
        self.training_end = pd.Timestamp(training_end)
        train = data.loc[(data.event_time < self.training_end) & (data.decision_time < self.training_end)]
        for key, part in train.groupby(["entity_id", "metric_name"], sort=True):
            spec = self.registry[key[1]]
            y = transform(part.value, spec)
            ok = np.isfinite(y) & part.quality.eq("ok").to_numpy()
            good, y = part.loc[ok], y[ok]
            duration = (good.event_time.max()-good.event_time.min()).total_seconds()/3600 if len(good) else 0
            diag = dict(entity_id=key[0], metric_name=key[1], observations=len(good), hours=duration)
            if not spec.supported or len(y) < spec.min_observations or duration < spec.min_hours:
                self.diagnostics.append({**diag, "status": "unsupported"})
                continue
            origin = good.event_time.iloc[0]
            coefficient = np.array([np.median(y), 0., 0.])
            kind, improvement = "median", 0.
            # One blocked training holdout, not random rows or development/test fitting.
            cut = origin + (good.event_time.iloc[-1]-origin)*.75
            before = (good.event_time < cut).to_numpy()
            phase = ((good.event_time.dt.hour*60+good.event_time.dt.minute)//180)
            if allow_seasonal and duration >= 24*spec.min_cycles and phase.nunique()==8 and before.sum()>=100:
                X = design(good.event_time, origin)
                beta = np.linalg.lstsq(X[before], y[before], rcond=None)[0]
                base_error = np.median(np.abs(y[~before]-np.median(y[before])))
                seasonal_error = np.median(np.abs(y[~before]-X[~before]@beta))
                improvement = 1-seasonal_error/max(base_error, spec.floor)
                if improvement >= .1:
                    coefficient, kind = np.linalg.lstsq(X, y, rcond=None)[0], "daily_harmonic"
            r = y-design(good.event_time, origin)@coefficient
            centre = float(np.median(r))
            scale = max(float(1.4826*np.median(np.abs(r-centre))), spec.floor)
            self.models[key] = dict(origin=origin, coefficient=coefficient, centre=centre,
                scale=scale, kind=kind, rows=good.row_id.tolist())
            residual = pd.Series((r-centre)/scale,index=good.event_time).asfreq(f'{spec.cadence_minutes}min')
            lag1=residual.autocorr(1) if residual.iloc[:-1].std()>0 and residual.iloc[1:].std()>0 else np.nan
            daily_lag=int(24*60/spec.cadence_minutes)
            daily=residual.autocorr(daily_lag) if len(residual)>daily_lag+3 and residual.std()>1e-12 else np.nan
            self.diagnostics.append({**diag, "status": "fitted", "kind": kind,
                "blocked_improvement": improvement, "residual_centre": centre, "scale": scale,
                "lag1": lag1, "lag_daily":daily, "q01": residual.quantile(.01), "q99": residual.quantile(.99),
                "skew":residual.skew(),"excess_kurtosis":residual.kurtosis(),
                "first_half_sd":residual.iloc[:len(residual)//2].std(),"last_half_sd":residual.iloc[len(residual)//2:].std(),
                "ols_contamination_robust": False})
        return self

    def score(self, data):
        result = data.copy()
        result["y"], result["expected"], result["z"] = np.nan, np.nan, np.nan
        result["reference_reason"] = "no_local_reference"
        for key, index in result.groupby(["entity_id", "metric_name"]).groups.items():
            model = self.models.get(key)
            if model is None:
                continue
            rows = result.loc[index]
            y = transform(rows.value, self.registry[key[1]])
            expected = design(rows.event_time, model["origin"])@model["coefficient"]+model["centre"]
            result.loc[index, "y"] = y
            result.loc[index, "expected"] = expected
            result.loc[index, "z"] = (y-expected)/model["scale"]
            result.loc[index, "reference_reason"] = np.where(np.isfinite(y), "ok", rows.quality)
        return result
