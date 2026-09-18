import pandas as pd

from telco_anomaly.selection import (
    qualify_early_warning,
    qualify_localisation,
    select_development_candidate,
)


def _comparison():
    return pd.DataFrame([
        ("simple", "rapid_only", 0.30, 0.22, 0.008, 0.01),
        ("rich", "full_topology", 0.31, 0.21, 0.008, 0.01),
    ], columns=[
        "candidate_key", "portfolio", "event_recall", "event_recall_ci_low",
        "false_incidents_per_entity_day_ci_high", "missing_score_fraction",
    ])


def _select(frame, faults=50, ranking_metric="event_recall"):
    return select_development_candidate(
        frame,
        development_faults=faults,
        false_incident_budget=0.01,
        budget_safety_factor=0.90,
        minimum_faults=30,
        minimum_recall_ci_low=0.20,
        maximum_missing_score_fraction=0.20,
        portfolio_preference=["rapid_only", "full_topology"],
        ranking_metric=ranking_metric,
    )


def test_prefers_simpler_statistically_equivalent_candidate():
    selected, decision = _select(_comparison())
    assert selected["candidate_key"] == "simple"
    assert decision["status"] == "selected_within_all_gates"


def test_prompt_objective_prefers_timely_detection_without_relaxing_budget():
    frame = _comparison()
    frame["prompt_event_recall"] = [0.10, 0.25]
    selected, decision = _select(frame, ranking_metric="prompt_event_recall")
    assert selected.candidate_key == "rich"
    assert decision["ranking_metric"] == "prompt_event_recall"
    frame.loc[1, "false_incidents_per_entity_day_ci_high"] = 0.02
    selected, _ = _select(frame, ranking_metric="prompt_event_recall")
    assert selected.candidate_key == "simple"



def test_fails_closed_when_budget_is_missed():
    frame = _comparison()
    frame["false_incidents_per_entity_day_ci_high"] = 0.02
    selected, decision = _select(frame)
    assert selected is None
    assert decision["status"] == "blocked_no_candidate_meets_gates"


def test_fails_closed_when_development_denominator_is_too_small():
    selected, decision = _select(_comparison(), faults=10)
    assert selected is None
    assert decision["status"] == "blocked_insufficient_faults"


def test_recall_point_estimate_cannot_bypass_uncertainty_gate():
    frame = _comparison().iloc[[0]].copy()
    frame["event_recall"] = 0.50
    frame["event_recall_ci_low"] = 0.19

    selected, decision = _select(frame)

    assert selected is None
    assert decision["status"] == "blocked_no_candidate_meets_gates"
    assert decision["minimum_recall_ci_low"] == 0.20


def _localisation_row(faults=20, successes=8, ci_low=0.20):
    return pd.Series({
        "multi_entity_faults": faults,
        "multi_entity_detected_and_localised": successes,
        "multi_entity_joint_detection_and_localisation_recall": (
            successes / faults if faults else float("nan")
        ),
        "multi_entity_joint_detection_and_localisation_recall_ci_low": ci_low,
        "multi_entity_joint_detection_and_localisation_recall_ci_high": 0.62,
    })


def test_localisation_requires_a_registered_performance_gate():
    result = qualify_localisation(
        _localisation_row(), minimum_multi_entity_faults=20
    )

    assert result["status"] == "not_established_unregistered_performance_gate"


def test_localisation_is_separate_and_fail_closed():
    insufficient = qualify_localisation(
        _localisation_row(faults=19, successes=19, ci_low=0.83),
        minimum_multi_entity_faults=20,
        minimum_joint_recall_ci_low=0.20,
    )
    below = qualify_localisation(
        _localisation_row(ci_low=0.19),
        minimum_multi_entity_faults=20,
        minimum_joint_recall_ci_low=0.20,
    )
    boundary = qualify_localisation(
        _localisation_row(ci_low=0.20),
        minimum_multi_entity_faults=20,
        minimum_joint_recall_ci_low=0.20,
    )

    assert insufficient["status"] == "not_established_insufficient_multi_entity_faults"
    assert below["status"] == "not_qualified_joint_recall_bound"
    assert boundary["status"] == "qualified"


def _early_warning_row(preimpact_low=0.25, prompt_low=0.22):
    return pd.Series({
        "preimpact_scoreable_faults": 50,
        "preimpact_event_recall": 0.40,
        "preimpact_event_recall_ci_low": preimpact_low,
        "prompt_scoreable_faults": 50,
        "prompt_event_recall": 0.36,
        "prompt_event_recall_ci_low": prompt_low,
    })


def test_early_warning_requires_preimpact_and_prompt_evidence():
    qualified = qualify_early_warning(
        _early_warning_row(),
        minimum_faults=30,
        minimum_preimpact_recall_ci_low=0.20,
        minimum_prompt_recall_ci_low=0.20,
    )
    late = qualify_early_warning(
        _early_warning_row(preimpact_low=0.19),
        minimum_faults=30,
        minimum_preimpact_recall_ci_low=0.20,
        minimum_prompt_recall_ci_low=0.20,
    )

    assert qualified["status"] == "qualified"
    assert late["status"] == "not_qualified_recall_bound"
