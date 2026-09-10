"""Operational score-to-alert and alert-to-incident interfaces."""

from __future__ import annotations

from .evaluation import (
    ALERT_COLUMNS,
    CASE_COLUMNS,
    apply_alert_cooldown,
    form_cases,
    merge_nearby_alerts,
    scores_to_alerts,
)


def consolidate_incidents(
    alerts,
    topology=None,
    *,
    quiet_period_seconds,
    thresholds,
):
    """Consolidate alerts using time and justified topology containment."""

    return form_cases(
        alerts,
        topology,
        gap_seconds=quiet_period_seconds,
        thresholds=thresholds,
    )


__all__ = [
    "ALERT_COLUMNS",
    "CASE_COLUMNS",
    "scores_to_alerts",
    "merge_nearby_alerts",
    "apply_alert_cooldown",
    "consolidate_incidents",
]
