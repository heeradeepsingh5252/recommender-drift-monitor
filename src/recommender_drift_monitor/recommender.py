"""A compact item-item collaborative-filtering recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemRecommender:
    """Recommend unrated catalog items using similarity between item rating vectors."""

    def __init__(self) -> None:
        self.catalog = pd.DataFrame()
        self.interactions = pd.DataFrame()
        self.rating_matrix = pd.DataFrame()
        self.centered_matrix: np.ndarray | None = None
        self.similarity: np.ndarray | None = None
        self.item_ids: list[int] = []
        self.global_mean = 3.5

    def fit(self, interactions: pd.DataFrame, catalog: pd.DataFrame) -> ItemItemRecommender:
        required_interactions = {"user_id", "item_id", "rating"}
        if not required_interactions.issubset(interactions.columns):
            missing = sorted(required_interactions - set(interactions.columns))
            raise ValueError(f"interactions is missing required columns: {missing}")
        if not {"item_id", "title", "genre"}.issubset(catalog.columns):
            raise ValueError("catalog must contain item_id, title, and genre")
        if interactions.empty:
            raise ValueError("cannot train on an empty interactions dataset")

        self.catalog = catalog.copy().reset_index(drop=True)
        self.interactions = interactions.copy()
        self.global_mean = float(self.interactions["rating"].mean())
        matrix = self.interactions.pivot_table(
            index="user_id", columns="item_id", values="rating", aggfunc="mean"
        ).sort_index(axis=0).sort_index(axis=1)
        values = matrix.to_numpy(dtype=float)
        observed = ~np.isnan(values)
        safe_values = np.nan_to_num(values, nan=0.0)
        row_counts = observed.sum(axis=1)
        row_means = np.divide(
            (safe_values * observed).sum(axis=1),
            row_counts,
            out=np.full(len(values), self.global_mean, dtype=float),
            where=row_counts > 0,
        )
        centered = np.where(observed, safe_values - row_means[:, None], 0.0)
        similarity = cosine_similarity(centered.T)
        np.fill_diagonal(similarity, 0.0)

        self.rating_matrix = matrix
        self.centered_matrix = centered
        self.similarity = similarity
        self.item_ids = [int(item_id) for item_id in matrix.columns]
        return self

    def recommend(self, user_id: int, limit: int = 10) -> list[dict[str, object]]:
        """Return ranked unseen items, with a popularity fallback for unknown users."""
        if self.similarity is None or self.centered_matrix is None:
            raise RuntimeError("model must be fitted before requesting recommendations")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        user_id = int(user_id)
        known_user = user_id in self.rating_matrix.index
        if known_user:
            user_row = self.rating_matrix.loc[user_id]
            watched = user_row.dropna().index.astype(int).tolist()
            centered_row = self.centered_matrix[self.rating_matrix.index.get_loc(user_id)]
        else:
            watched = []
            centered_row = np.zeros(len(self.item_ids), dtype=float)

        catalog_by_id = self.catalog.set_index("item_id")
        item_position = {item_id: index for index, item_id in enumerate(self.item_ids)}
        aggregates = self.interactions.groupby("item_id")["rating"].agg(["mean", "count"])
        candidates = [int(item_id) for item_id in self.catalog["item_id"] if int(item_id) not in watched]
        scored: list[dict[str, object]] = []

        for candidate_id in candidates:
            score = self.global_mean
            if known_user and candidate_id in item_position:
                candidate_position = item_position[candidate_id]
                weights = self.similarity[candidate_position]
                denominator = float(np.abs(weights).sum())
                if denominator > 1e-12:
                    collaborative_score = float(np.dot(weights, centered_row) / denominator)
                    item_mean = float(aggregates.loc[candidate_id, "mean"]) if candidate_id in aggregates.index else self.global_mean
                    score = 0.75 * (self.global_mean + collaborative_score) + 0.25 * item_mean
                elif candidate_id in aggregates.index:
                    score = float(aggregates.loc[candidate_id, "mean"])
            elif candidate_id in aggregates.index:
                score = float(aggregates.loc[candidate_id, "mean"])

            item = catalog_by_id.loc[candidate_id]
            scored.append(
                {
                    "item_id": candidate_id,
                    "title": str(item["title"]),
                    "genre": str(item["genre"]),
                    "score": round(float(np.clip(score, 1.0, 5.0)), 3),
                    "rating_count": int(aggregates.loc[candidate_id, "count"])
                    if candidate_id in aggregates.index
                    else 0,
                }
            )

        scored.sort(key=lambda item: (float(item["score"]), int(item["rating_count"])), reverse=True)
        return scored[:limit]
