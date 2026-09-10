import pandas as pd

from telco_anomaly.selection import select_development_candidate


def _comparison():
    return pd.DataFrame([
        ("simple", "rapid_only", 0.30, 0.008, 0.01),
        ("rich", "full_topology", 0.31, 0.008, 0.01),
    ], columns=[
        "candidate_key", "portfolio", "event_recall",
        "false_incidents_per_entity_day_ci_high", "missing_score_fraction",
    ])


def _select(frame, faults=50):
    return select_development_candidate(
        frame,
        development_faults=faults,
        false_incident_budget=0.01,
        budget_safety_factor=0.90,
        minimum_faults=30,
        minimum_recall=0.20,
        maximum_missing_score_fraction=0.20,
        portfolio_preference=["rapid_only", "full_topology"],
    )


def test_prefers_simpler_statistically_equivalent_candidate():
    selected, decision = _select(_comparison())
    assert selected["candidate_key"] == "simple"
    assert decision["status"] == "selected_within_all_gates"


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
