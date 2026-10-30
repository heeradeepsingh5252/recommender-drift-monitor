"""Deterministic synthetic data for a self-contained recommender demo.

This project intentionally uses synthetic data so it can be run without API keys,
external datasets, or paid services. Do not treat the generated records as real user data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = ["Action", "Adventure", "Comedy", "Drama", "Romance", "Sci-Fi", "Thriller"]

_TITLE_WORDS = [
    "Midnight", "Hidden", "Golden", "Silent", "Last", "Electric", "Crimson",
    "Distant", "Broken", "Infinite", "Northern", "Forgotten", "Parallel", "Blue",
    "Rising", "Wild", "Secret", "Starlight", "Paper", "Deep",
]
_TITLE_NOUNS = [
    "Signal", "Voyage", "Kingdom", "Promise", "Orbit", "Echo", "Frontier",
    "Equation", "Garden", "Protocol", "Horizons", "Memory", "Station", "Mirage",
    "Legacy", "River", "Paradox", "Witness", "Compass", "Discovery",
]


def make_catalog(n_items: int = 42, seed: int = 17) -> pd.DataFrame:
    """Create a stable movie-like catalog with a balanced genre mix."""
    if n_items < len(GENRES):
        raise ValueError(f"n_items must be at least {len(GENRES)} to cover all genres")

    rng = np.random.default_rng(seed)
    genres = [GENRES[i % len(GENRES)] for i in range(n_items)]
    rng.shuffle(genres)
    rows = []
    for index in range(n_items):
        first = _TITLE_WORDS[(index * 7 + seed) % len(_TITLE_WORDS)]
        second = _TITLE_NOUNS[(index * 11 + seed) % len(_TITLE_NOUNS)]
        rows.append(
            {
                "item_id": index + 1,
                "title": f"{first} {second}",
                "genre": genres[index],
                "release_year": int(rng.integers(1995, 2026)),
            }
        )
    return pd.DataFrame(rows)


def generate_interactions(
    catalog: pd.DataFrame,
    n_users: int = 120,
    n_interactions: int = 3000,
    seed: int = 42,
    shifted: bool = False,
) -> pd.DataFrame:
    """Generate ratings. ``shifted=True`` simulates a changed production data mix."""
    if n_users < 2:
        raise ValueError("n_users must be at least 2")
    if n_interactions < 1:
        raise ValueError("n_interactions must be positive")
    if catalog.empty or not {"item_id", "genre"}.issubset(catalog.columns):
        raise ValueError("catalog must contain item_id and genre columns")

    rng = np.random.default_rng(seed)
    genres = catalog["genre"].astype(str).to_numpy()
    item_ids = catalog["item_id"].to_numpy(dtype=int)
    unique_genres = sorted(set(genres))
    items_by_genre = {genre: item_ids[genres == genre] for genre in unique_genres}

    if shifted:
        # A clear, controllable change: more Action/Thriller/Sci-Fi traffic and lower ratings.
        weights = np.array(
            [5.5 if genre in {"Action", "Thriller", "Sci-Fi"} else 0.25 for genre in unique_genres],
            dtype=float,
        )
    else:
        weights = np.ones(len(unique_genres), dtype=float)
    weights /= weights.sum()

    user_ids = rng.integers(1, n_users + 1, size=n_interactions)
    sampled_genres = rng.choice(unique_genres, size=n_interactions, p=weights)
    sampled_items = np.empty(n_interactions, dtype=int)
    ratings = np.empty(n_interactions, dtype=int)

    for index, (user_id, genre) in enumerate(zip(user_ids, sampled_genres, strict=True)):
        chosen_items = items_by_genre[str(genre)]
        sampled_items[index] = int(rng.choice(chosen_items))
        preferred_genre = unique_genres[(int(user_id) - 1) % len(unique_genres)]
        if shifted:
            ratings[index] = int(rng.choice([1, 2, 2, 3, 3, 4, 4], p=None))
        elif str(genre) == preferred_genre:
            ratings[index] = int(rng.choice([3, 4, 4, 5, 5]))
        else:
            ratings[index] = int(rng.choice([2, 3, 3, 4, 4, 5]))

    timestamps = pd.Timestamp("2026-10-01", tz="UTC") - pd.to_timedelta(
        rng.integers(0, 90, size=n_interactions), unit="D"
    )
    item_genres = catalog.set_index("item_id")["genre"].to_dict()
    return pd.DataFrame(
        {
            "user_id": user_ids.astype(int),
            "item_id": sampled_items.astype(int),
            "rating": ratings.astype(int),
            "timestamp": timestamps,
            "genre": [item_genres[item_id] for item_id in sampled_items],
        }
    )
