import pandas as pd

from recommender_drift_monitor.data import generate_interactions, make_catalog
from recommender_drift_monitor.monitor import build_drift_report, calculate_psi


def test_psi_is_near_zero_for_same_distribution() -> None:
    values = ["Action", "Drama", "Comedy", "Action", "Drama"] * 20
    assert calculate_psi(values, values) < 1e-8


def test_psi_detects_a_strong_distribution_shift() -> None:
    reference = ["Comedy"] * 95 + ["Action"] * 5
    current = ["Comedy"] * 10 + ["Action"] * 90
    assert calculate_psi(reference, current) > 0.2


def test_report_flags_shifted_demo_data() -> None:
    catalog = make_catalog(n_items=42, seed=17)
    reference = generate_interactions(catalog, n_users=120, n_interactions=3000, seed=42)
    shifted = generate_interactions(catalog, n_users=120, n_interactions=3000, seed=123, shifted=True)
    report = build_drift_report(reference, shifted)
    assert any(row["drift_detected"] for row in report)


def test_report_rejects_missing_features() -> None:
    reference = pd.DataFrame({"genre": ["Action"], "rating": [4]})
    current = pd.DataFrame({"genre": ["Drama"]})
    try:
        build_drift_report(reference, current)
    except ValueError as error:
        assert "rating" in str(error)
    else:
        raise AssertionError("missing feature should have raised ValueError")
