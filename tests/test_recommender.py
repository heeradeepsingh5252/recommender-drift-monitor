from recommender_drift_monitor.data import generate_interactions, make_catalog
from recommender_drift_monitor.recommender import ItemItemRecommender


def test_recommender_returns_ranked_unseen_items() -> None:
    catalog = make_catalog(n_items=21, seed=3)
    interactions = generate_interactions(catalog, n_users=15, n_interactions=300, seed=5)
    model = ItemItemRecommender().fit(interactions, catalog)
    results = model.recommend(user_id=1, limit=5)

    rated_items = set(interactions.loc[interactions["user_id"] == 1, "item_id"])
    assert len(results) == 5
    assert not rated_items.intersection({result["item_id"] for result in results})
    assert all(1 <= float(result["score"]) <= 5 for result in results)
    assert [item["score"] for item in results] == sorted(
        [item["score"] for item in results], reverse=True
    )


def test_unknown_user_uses_cold_start_fallback() -> None:
    catalog = make_catalog(n_items=21, seed=3)
    interactions = generate_interactions(catalog, n_users=15, n_interactions=300, seed=5)
    model = ItemItemRecommender().fit(interactions, catalog)
    results = model.recommend(user_id=9999, limit=3)
    assert len(results) == 3
    assert all("title" in result and "genre" in result for result in results)
