"""Complete right-closed windows with full contiguous recursive replay."""
import numpy as np
import pandas as pd

COMPACT = ["mean", "std", "slope", "short_minus_long"]
HISTORICAL = ["mean", "std", "ewma_difference", "slope", "long_mean", "short_minus_long", "persistence", "cusum"]


def directional(z, direction):
    if direction == "high": return np.maximum(z, 0)
    if direction == "low": return np.maximum(-z, 0)
    if direction == "two-sided": return np.abs(z)
    raise ValueError("Unknown direction")


def features(residuals, registry, short=12, long=72, half_life_hours=1., allowance=.25, boundary=2.):
    if not isinstance(short, int) or not isinstance(long, int) or short<3 or long<=short or min(half_life_hours, allowance, boundary)<=0:
        raise ValueError("Invalid windows, half-life, allowance or boundary")
    frames = []
    for (_, metric), part in residuals.groupby(["entity_id", "metric_name"], sort=True):
        part = part.sort_values("event_time").copy()
        spec = registry[metric]
        z = part.z.to_numpy()
        out = np.full((len(z), 8), np.nan)
        level, shift = directional(z, spec.direction), np.full(len(z), np.nan)
        long_count = np.zeros(len(z), int)
        cp=cm=0.; ew=np.nan; start=0
        dt = spec.cadence_minutes/60
        alpha=1-2**(-dt/half_life_hours)
        times=np.arange(short)*dt; times-=times.mean()
        for i, value in enumerate(z):
            if not np.isfinite(value):
                cp=cm=0.; ew=np.nan; start=i+1
                continue
            if i and (part.segment.iloc[i]!=part.segment.iloc[i-1] or not np.isfinite(z[i-1])):
                cp=cm=0.; ew=np.nan; start=i
            cp=max(0.,cp+value-allowance);cm=max(0.,cm-value-allowance)
            shift[i]=cp if spec.direction=="high" else cm if spec.direction=="low" else max(cp,cm)
            if i>start:
                difference=(value-z[i-1])/dt
                ew=difference if np.isnan(ew) else alpha*difference+(1-alpha)*ew
            if i-start+1<short: continue
            q=z[i-short+1:i+1];history=z[max(start,i-long+1):i+1]
            mu=q.mean();long_mu=history.mean()
            exceed=directional(q,spec.direction)>boundary
            out[i]=[mu,q.std(ddof=0),ew,np.dot(times,q-q.mean())/np.dot(times,times),long_mu,mu-long_mu,exceed.mean(),shift[i]]
            long_count[i]=len(history)
        part[HISTORICAL]=out
        part["u_level"],part["u_shift"],part["long_count"]=level,shift,long_count
        part["window_start"]=part.event_time-pd.Timedelta(minutes=(short-1)*spec.cadence_minutes)
        part["effective_long_hours"]=long_count*dt
        frames.append(part)
    return pd.concat(frames,ignore_index=True)


def joint_vectors(frame, channels, profile="compact"):
    names=COMPACT if profile=="compact" else HISTORICAL if profile=="historical8" else None
    if names is None or not channels or len(set(channels))!=len(channels):
        raise ValueError("Declare profile and unique ordered channels")
    index=["entity_id","event_time","decision_time","grid_index"]
    base=frame[index].drop_duplicates().set_index(index).sort_index()
    columns=[]
    for channel in channels:
        part=frame.loc[frame.metric_name.eq(channel)].set_index(index)
        for name in names:
            column=f"{channel}:{name}";columns.append(column)
            base[column]=part[name]
    base=base.reset_index()
    base["row_id"]=[f"{e}|{t.isoformat()}" for e,t in zip(base.entity_id,base.event_time)]
    return base,columns
