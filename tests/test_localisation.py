import pandas as pd

from telco_anomaly.localisation import localisation_view, topology_identifiability


def test_topology_identifiability_finds_equal_descendant_sets():
    topology = pd.DataFrame([
        ("a", "l2", "l2-1", 2, "physical_topology"),
        ("b", "l2", "l2-1", 2, "physical_topology"),
        ("a", "pon", "pon-1", 1, "physical_topology"),
        ("b", "pon", "pon-1", 1, "physical_topology"),
        ("c", "l2", "l2-2", 2, "physical_topology"),
    ], columns=[
        "entity_id", "group_type", "group_id", "hierarchy_level", "group_family"
    ])
    audit = topology_identifiability(topology)
    equivalent = audit.loc[audit["group_id"].isin(["l2-1", "pon-1"])]
    assert equivalent["equivalence_size"].eq(2).all()
    assert (~equivalent["identifiable"]).all()


def test_localisation_view_uses_product_field_names():
    cases = pd.DataFrame([{
        "case_id": "C-1",
        "scope_type": "l2",
        "scope_id": "L2-1",
        "scope_type_2": "pon",
        "scope_id_2": "PON-1",
        "affected_fraction_estimate": 0.75,
        "footprint_size": 8,
        "identifiability_status": "hierarchical_candidate",
        "location_explanation": "test",
    }])
    result = localisation_view(cases)
    assert result.loc[0, "predicted_scope_id"] == "L2-1"
    assert "scope_id" not in result
