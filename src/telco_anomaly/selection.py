"""Transparent, fail-closed development model selection."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "candidate_key",
    "portfolio",
    "event_recall",
    "false_incidents_per_entity_day_ci_high",
    "missing_score_fraction",
}


def select_development_candidate(
    comparison: pd.DataFrame,
    *,
    development_faults: int,
    false_incident_budget: float,
    budget_safety_factor: float,
    minimum_faults: int,
    minimum_recall: float,
    maximum_missing_score_fraction: float,
    portfolio_preference: list[str],
    equivalence_margin: float = 0.02,
):
    """Select one declared candidate, or return a reason to stop.

    The function never substitutes a merely less-bad configuration when no
    candidate satisfies the gates.  That distinction prevents a diagnostic
    experiment from silently becoming a deployable model.
    """

    missing = REQUIRED_COLUMNS - set(comparison.columns)
    if missing:
        raise ValueError(f"Comparison is missing columns: {sorted(missing)}")
    if development_faults < int(minimum_faults):
        return None, {
            "status": "blocked_insufficient_faults",
            "development_faults": int(development_faults),
            "minimum_faults": int(minimum_faults),
        }

    budget_gate = float(false_incident_budget) * float(budget_safety_factor)
    eligible = comparison.loc[
        comparison["false_incidents_per_entity_day_ci_high"].le(budget_gate)
        & comparison["event_recall"].ge(float(minimum_recall))
        & comparison["missing_score_fraction"].le(
            float(maximum_missing_score_fraction)
        )
    ].copy()
    if eligible.empty:
        return None, {
            "status": "blocked_no_candidate_meets_gates",
            "development_faults": int(development_faults),
            "false_incident_budget_gate": budget_gate,
            "minimum_recall": float(minimum_recall),
            "maximum_missing_score_fraction": float(
                maximum_missing_score_fraction
            ),
        }

    best_recall = eligible["event_recall"].max()
    equivalent = eligible.loc[
        eligible["event_recall"].ge(best_recall - float(equivalence_margin))
    ].copy()
    preference = {name: rank for rank, name in enumerate(portfolio_preference)}
    equivalent["simplicity_rank"] = equivalent["portfolio"].map(
        preference
    ).fillna(len(preference))
    selected = equivalent.sort_values(
        [
            "simplicity_rank",
            "false_incidents_per_entity_day_ci_high",
            "event_recall",
            "candidate_key",
        ],
        ascending=[True, True, False, True],
        kind="stable",
    ).iloc[0].drop(labels="simplicity_rank")
    return selected, {
        "status": "selected_within_all_gates",
        "development_faults": int(development_faults),
        "false_incident_budget_gate": budget_gate,
        "candidate_key": str(selected["candidate_key"]),
    }
