"""Application service that owns demo datasets, recommender, and drift state."""

from __future__ import annotations

from recommender_drift_monitor.data import generate_interactions, make_catalog
from recommender_drift_monitor.monitor import DEFAULT_THRESHOLD, build_drift_report
from recommender_drift_monitor.recommender import ItemItemRecommender


class DemoService:
    """In-memory, deterministic service used by the API and tests."""

    def __init__(self) -> None:
        self.catalog = make_catalog(n_items=42, seed=17)
        self.reference = generate_interactions(
            self.catalog, n_users=120, n_interactions=3000, seed=42, shifted=False
        )
        self.current = generate_interactions(
            self.catalog, n_users=120, n_interactions=3000, seed=99, shifted=False
        )
        self.model = ItemItemRecommender().fit(self.reference, self.catalog)
        self.threshold = DEFAULT_THRESHOLD
        self.shifted = False

    def simulate_drift(self, shifted: bool = True) -> dict[str, object]:
        """Switch the current window between normal and shifted synthetic data."""
        self.shifted = bool(shifted)
        self.current = generate_interactions(
            self.catalog,
            n_users=120,
            n_interactions=3000,
            seed=123 if self.shifted else 99,
            shifted=self.shifted,
        )
        return self.drift_report()

    def drift_report(self) -> dict[str, object]:
        features = build_drift_report(self.reference, self.current, self.threshold)
        drifted = [row["feature"] for row in features if row["drift_detected"]]
        return {
            "status": "DRIFT DETECTED" if drifted else "STABLE",
            "drift_detected": bool(drifted),
            "drifted_features": drifted,
            "threshold": self.threshold,
            "features": features,
        }

    def overview(self) -> dict[str, object]:
        report = self.drift_report()
        genre_reference = self.reference["genre"].value_counts().to_dict()
        genre_current = self.current["genre"].value_counts().to_dict()
        all_genres = sorted(set(genre_reference) | set(genre_current))
        genre_distribution = [
            {
                "genre": genre,
                "reference": int(genre_reference.get(genre, 0)),
                "current": int(genre_current.get(genre, 0)),
            }
            for genre in all_genres
        ]
        rating_reference = self.reference["rating"].value_counts().to_dict()
        rating_current = self.current["rating"].value_counts().to_dict()
        rating_distribution = [
            {
                "rating": int(rating),
                "reference": int(rating_reference.get(rating, 0)),
                "current": int(rating_current.get(rating, 0)),
            }
            for rating in sorted(set(rating_reference) | set(rating_current))
        ]
        return {
            "project": "Recommender Drift Monitor",
            "mode": "synthetic_demo_data",
            "user_count": int(self.reference["user_id"].nunique()),
            "item_count": int(len(self.catalog)),
            "reference_interactions": int(len(self.reference)),
            "current_interactions": int(len(self.current)),
            "reference_mean_rating": round(float(self.reference["rating"].mean()), 3),
            "current_mean_rating": round(float(self.current["rating"].mean()), 3),
            "status": report["status"],
            "drift_detected": report["drift_detected"],
            "genre_distribution": genre_distribution,
            "rating_distribution": rating_distribution,
            "sample_user_ids": list(range(1, 16)),
            "synthetic_shift_enabled": self.shifted,
        }

    def recommendations(self, user_id: int, limit: int = 10) -> list[dict[str, object]]:
        return self.model.recommend(user_id=user_id, limit=limit)

    def health(self) -> dict[str, object]:
        return {"status": "ok", "service": "recommender-drift-monitor", "version": "1.0.0"}
