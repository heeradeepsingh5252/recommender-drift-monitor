from recommender_drift_monitor.data import generate_interactions, make_catalog


def test_catalog_and_interactions_have_expected_schema() -> None:
    catalog = make_catalog(n_items=21, seed=3)
    interactions = generate_interactions(catalog, n_users=12, n_interactions=100, seed=4)

    assert len(catalog) == 21
    assert {"item_id", "title", "genre", "release_year"}.issubset(catalog.columns)
    assert len(interactions) == 100
    assert {"user_id", "item_id", "rating", "timestamp", "genre"}.issubset(interactions.columns)
    assert interactions["rating"].between(1, 5).all()
    assert interactions["item_id"].isin(catalog["item_id"]).all()


def test_data_generation_is_reproducible() -> None:
    catalog = make_catalog(n_items=14, seed=9)
    first = generate_interactions(catalog, n_users=8, n_interactions=30, seed=10)
    second = generate_interactions(catalog, n_users=8, n_interactions=30, seed=10)
    assert first.equals(second)
