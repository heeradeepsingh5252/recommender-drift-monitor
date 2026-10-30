"""Population Stability Index (PSI) based drift checks for demo data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 0.20
FEATURES = ("genre", "rating")


def calculate_psi(reference: Iterable[object], current: Iterable[object], epsilon: float = 1e-4) -> float:
    """Calculate categorical PSI with small smoothing to avoid zero-proportion errors.

    PSI is a heuristic drift indicator. Thresholds such as 0.10 or 0.20 are conventions,
    not universal statistical guarantees.
    """
    reference_series = pd.Series(list(reference)).dropna().astype(str)
    current_series = pd.Series(list(current)).dropna().astype(str)
    if reference_series.empty or current_series.empty:
        raise ValueError("reference and current samples must both contain data")

    categories = sorted(set(reference_series.unique()) | set(current_series.unique()))
    expected = reference_series.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
    actual = current_series.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, object]]:
    """Compare selected reference/current columns and return JSON-friendly results."""
    if not 0 < threshold < 10:
        raise ValueError("threshold must be between 0 and 10")
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            raise ValueError(f"both datasets must contain the {feature!r} column")
        psi = calculate_psi(reference[feature], current[feature])
        reference_mean = float(pd.to_numeric(reference[feature], errors="coerce").mean()) if feature == "rating" else None
        current_mean = float(pd.to_numeric(current[feature], errors="coerce").mean()) if feature == "rating" else None
        rows.append(
            {
                "feature": feature,
                "psi": round(psi, 4),
                "threshold": threshold,
                "drift_detected": psi >= threshold,
                "reference_mean": round(reference_mean, 3) if reference_mean is not None else None,
                "current_mean": round(current_mean, 3) if current_mean is not None else None,
            }
        )
    return rows
