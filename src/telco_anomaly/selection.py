"""Transparent, fail-closed development model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "candidate_key",
    "portfolio",
    "event_recall",
    "event_recall_ci_low",
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
    minimum_recall_ci_low: float,
    maximum_missing_score_fraction: float,
    portfolio_preference: list[str],
    equivalence_margin: float = 0.02,
    ranking_metric: str = "event_recall",
):
    """Select one declared candidate, or return a reason to stop.

    The function never substitutes a merely less-bad configuration when no
    candidate satisfies the gates. Recall is gated on its Wilson confidence
    lower bound, not on the optimistic point estimate. That distinction
    prevents a diagnostic experiment from silently becoming a deployable
    model.
    """

    if ranking_metric not in {"event_recall", "prompt_event_recall"}:
        raise ValueError("Unsupported selection ranking metric")
    missing = (REQUIRED_COLUMNS | {ranking_metric}) - set(comparison.columns)
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
        & comparison["event_recall_ci_low"].ge(float(minimum_recall_ci_low))
        & comparison["missing_score_fraction"].le(
            float(maximum_missing_score_fraction)
        )
        & np.isfinite(comparison[ranking_metric])
    ].copy()
    if eligible.empty:
        return None, {
            "status": "blocked_no_candidate_meets_gates",
            "development_faults": int(development_faults),
            "false_incident_budget_gate": budget_gate,
            "recall_gate_metric": "event_recall_ci_low",
            "minimum_recall_ci_low": float(minimum_recall_ci_low),
            "maximum_missing_score_fraction": float(
                maximum_missing_score_fraction
            ),
        }

    best_recall = eligible[ranking_metric].max()
    equivalent = eligible.loc[
        eligible[ranking_metric].ge(best_recall - float(equivalence_margin))
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
        "recall_gate_metric": "event_recall_ci_low",
        "minimum_recall_ci_low": float(minimum_recall_ci_low),
        "selected_recall_ci_low": float(selected["event_recall_ci_low"]),
        "candidate_key": str(selected["candidate_key"]),
        "ranking_metric": ranking_metric,
    }


def qualify_localisation(
    row,
    *,
    minimum_multi_entity_faults,
    minimum_joint_recall_ci_low=None,
):
    """Qualify localisation independently from anomaly detection selection."""

    fields = {
        "multi_entity_faults",
        "multi_entity_detected_and_localised",
        "multi_entity_joint_detection_and_localisation_recall",
        "multi_entity_joint_detection_and_localisation_recall_ci_low",
        "multi_entity_joint_detection_and_localisation_recall_ci_high",
    }
    missing = fields - set(row.index)
    if missing:
        return {
            "status": "not_established_metric_unavailable",
            "missing_fields": sorted(missing),
        }

    total = int(row["multi_entity_faults"])
    evidence = {
        "status": None,
        "multi_entity_faults": total,
        "detected_and_localised": int(
            row["multi_entity_detected_and_localised"]
        ),
        "joint_recall": float(
            row["multi_entity_joint_detection_and_localisation_recall"]
        ),
        "joint_recall_ci_low": float(
            row[
                "multi_entity_joint_detection_and_localisation_recall_ci_low"
            ]
        ),
        "joint_recall_ci_high": float(
            row[
                "multi_entity_joint_detection_and_localisation_recall_ci_high"
            ]
        ),
        "minimum_multi_entity_faults": int(minimum_multi_entity_faults),
        "gate_metric": (
            "multi_entity_joint_detection_and_localisation_recall_ci_low"
        ),
        "minimum_joint_recall_ci_low": minimum_joint_recall_ci_low,
    }
    if total < int(minimum_multi_entity_faults):
        evidence["status"] = "not_established_insufficient_multi_entity_faults"
    elif minimum_joint_recall_ci_low is None:
        evidence["status"] = "not_established_unregistered_performance_gate"
    elif not np.isfinite(evidence["joint_recall_ci_low"]):
        evidence["status"] = "not_established_metric_unavailable"
    elif evidence["joint_recall_ci_low"] >= float(minimum_joint_recall_ci_low):
        evidence["status"] = "qualified"
    else:
        evidence["status"] = "not_qualified_joint_recall_bound"
    return evidence


def qualify_early_warning(
    row,
    *,
    minimum_faults,
    minimum_preimpact_recall_ci_low,
    minimum_prompt_recall_ci_low,
):
    """Qualify early warning separately from active-fault detection.

    Early warning means detecting observable evidence before recorded impact.
    Prompt recall adds the registered maximum response horizon. Both are
    conservative Wilson lower-bound gates; neither changes detector fitting.
    """

    fields = {
        "preimpact_scoreable_faults",
        "preimpact_event_recall",
        "preimpact_event_recall_ci_low",
        "prompt_scoreable_faults",
        "prompt_event_recall",
        "prompt_event_recall_ci_low",
    }
    missing = fields - set(row.index)
    if missing:
        return {
            "status": "not_established_metric_unavailable",
            "missing_fields": sorted(missing),
        }

    evidence = {
        "status": None,
        "preimpact_scoreable_faults": int(row["preimpact_scoreable_faults"]),
        "preimpact_event_recall": float(row["preimpact_event_recall"]),
        "preimpact_event_recall_ci_low": float(
            row["preimpact_event_recall_ci_low"]
        ),
        "prompt_scoreable_faults": int(row["prompt_scoreable_faults"]),
        "prompt_event_recall": float(row["prompt_event_recall"]),
        "prompt_event_recall_ci_low": float(
            row["prompt_event_recall_ci_low"]
        ),
        "minimum_faults": int(minimum_faults),
        "minimum_preimpact_recall_ci_low": float(
            minimum_preimpact_recall_ci_low
        ),
        "minimum_prompt_recall_ci_low": float(
            minimum_prompt_recall_ci_low
        ),
    }
    if min(
        evidence["preimpact_scoreable_faults"],
        evidence["prompt_scoreable_faults"],
    ) < int(minimum_faults):
        evidence["status"] = "not_established_insufficient_faults"
    elif not np.isfinite([
        evidence["preimpact_event_recall_ci_low"],
        evidence["prompt_event_recall_ci_low"],
    ]).all():
        evidence["status"] = "not_established_metric_unavailable"
    elif (
        evidence["preimpact_event_recall_ci_low"]
        >= float(minimum_preimpact_recall_ci_low)
        and evidence["prompt_event_recall_ci_low"]
        >= float(minimum_prompt_recall_ci_low)
    ):
        evidence["status"] = "qualified"
    else:
        evidence["status"] = "not_qualified_recall_bound"
    return evidence
