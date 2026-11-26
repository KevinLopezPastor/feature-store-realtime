"""Shared post-retrieval step for zone features: used by build_training_set (offline) and service/app.py (online).

Contract, from docs/plan/PROGRESS.md: after this runs, the model sees the same values in training and in serving.
- trip_count: missing or stale -> 0.
- mean_duration: missing or stale stays null (LightGBM handles NaN).
- baseline_mean_speed: never missing; a null here is a bug, not something to impute.
"""

import pandas as pd

ZONE_TTL = pd.Timedelta(minutes=15)
ZONE_FEATURE_COLS = ["trip_count", "mean_duration"]


def expire_stale(
    features: pd.DataFrame, feature_ts: pd.Series, as_of: pd.Series
) -> pd.DataFrame:
    """ONLINE only: Redis ignores the TTL, so blank out zone features older than the TTL at `as_of`.

    Offline, `get_historical_features` already dropped expired rows, so training skips this step.
    """
    if as_of.isna().any():
        raise ValueError("as_of contains NaT; cannot evaluate feature freshness.")

    out = features.copy()
    age = as_of - feature_ts
    stale_mask = feature_ts.isna() | (age > ZONE_TTL)
    out.loc[stale_mask, ZONE_FEATURE_COLS] = float("nan")
    return out


def impute(features: pd.DataFrame) -> pd.DataFrame:
    """Apply the contract above and return the columns in the dtypes the model was trained on."""
    if features["baseline_mean_speed"].isna().any():
        raise ValueError(
            "baseline_mean_speed contains null values; expected complete 168-hour coverage."
        )

    out = features.copy()
    out["trip_count"] = pd.to_numeric(out["trip_count"]).fillna(0).astype("int64")
    out["mean_duration"] = out["mean_duration"].astype("float64")
    out["baseline_mean_speed"] = out["baseline_mean_speed"].astype("float64")
    return out
