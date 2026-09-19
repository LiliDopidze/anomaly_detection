"""Incident grouping, event disposition and explicitly uncalibrated scope ranking."""

import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp


def active_inventory(inventory, timestamp):
    return inventory.loc[
        (inventory.valid_from <= timestamp) & (inventory.valid_to > timestamp)
    ]


def incident_queue(alerts, inventory, events, policy):
    """Consolidate within port and quiet period; rank at the last member decision.

    Ranked probabilities are conditional on this simplified single-fault model,
    not calibrated probabilities of a real fault.
    """
    columns = [
        "incident_id",
        "start_ts",
        "end_ts",
        "decision_ts",
        "scope",
        "affected_entities",
        "member_count",
        "disposition",
        "power_event_share",
        "top_scope",
        "scope_probability",
        "ambiguous",
        "candidates",
    ]
    if alerts.empty:
        return pd.DataFrame(columns=columns)
    groups = []
    quiet = pd.Timedelta(minutes=policy["quiet_minutes"])
    for alert in alerts.sort_values("start_ts").itertuples(index=False):
        membership = active_inventory(inventory, alert.start_ts)
        member = membership.loc[membership.ont_id.eq(alert.entity_id)]
        if member.empty:
            raise ValueError("Alert has no effective topology membership")
        port = member.pon_port.iloc[0]
        group = next(
            (
                g
                for g in reversed(groups)
                if g["port"] == port and alert.start_ts <= g["end"] + quiet
            ),
            None,
        )
        if group is None:
            group = {
                "port": port,
                "start": alert.start_ts,
                "end": alert.end_ts,
                "decision": alert.start_ts,
                "entities": set(),
            }
            groups.append(group)
        group["end"] = max(group["end"], alert.end_ts)
        group["decision"] = max(group["decision"], alert.start_ts)
        group["entities"].add(alert.entity_id)
    rows = []
    for number, group in enumerate(groups):
        membership = active_inventory(inventory, group["decision"])
        neighborhood = membership.loc[membership.pon_port.eq(group["port"])]
        scope_entities = set(neighborhood.ont_id)
        affected = group["entities"] & scope_entities
        hypotheses = []
        # Equal prior per unique observable footprint avoids duplicated topology votes.
        footprints = {}
        for level in ("ont_id", "splitter_l2", "splitter_l1", "pon_port"):
            for name, members in neighborhood.groupby(level):
                key = tuple(sorted(members.ont_id))
                footprints.setdefault(key, []).append(f"{level}:{name}")
        hit = policy["scope_hit_probability"]
        background = policy["scope_background_probability"]
        for footprint, names in footprints.items():
            inside = set(footprint)
            log_likelihood = 0.0
            for entity in scope_entities:
                probability = hit if entity in inside else background
                log_likelihood += np.log(
                    probability if entity in affected else 1 - probability
                )
            hypotheses.append((names, log_likelihood))
        weights = np.array([value for _, value in hypotheses])
        probabilities = np.exp(weights - logsumexp(weights))
        ranked = sorted(zip(hypotheses, probabilities), key=lambda row: -row[1])
        candidates = [
            {"equivalent_scopes": names, "probability": float(probability)}
            for (names, _), probability in ranked
        ]
        window = pd.Timedelta(minutes=policy["power_window_minutes"])
        power = events.loc[
            events.family.eq("onu_power_loss")
            & events.ont_id.isin(scope_entities)
            & events.timestamp_utc.between(
                group["decision"] - window, group["decision"]
            )
        ]
        share = power.ont_id.nunique() / len(scope_entities) if scope_entities else 0.0
        top = candidates[0]
        rows.append(
            {
                "incident_id": f"CASE-{number:06d}",
                "start_ts": group["start"],
                "end_ts": group["end"],
                "decision_ts": group["decision"],
                "scope": group["port"],
                "affected_entities": json.dumps(sorted(affected)),
                "member_count": len(affected),
                "power_event_share": share,
                "disposition": (
                    "probable_power_loss"
                    if share >= policy["power_share"]
                    else "network_candidate"
                ),
                "top_scope": " | ".join(top["equivalent_scopes"]),
                "scope_probability": top["probability"],
                "ambiguous": len(top["equivalent_scopes"]) > 1,
                "candidates": json.dumps(candidates[:3]),
            }
        )
    return pd.DataFrame(rows, columns=columns)
