"""Batch job: raw TLC trips -> feature Parquet files for Feast's offline store.

Run with `make features` (or `uv run python -m jobs.compute_aggregates`).
"""

import logging
import os
from pathlib import Path

import pandas as pd

from feature_store_realtime.entity_df import localize

logger = logging.getLogger(__name__)

RAW_DIR = Path(os.environ.get("RAW_DIR", "../mlops-reproducible-pipeline/data/raw"))
OUT_DIR = Path(os.environ.get("FEATURES_DIR", "data/features"))

# Decisions from docs/plan/PROGRESS.md ("Feature design decisions").
STEP = pd.Timedelta(minutes=15)
WINDOW = pd.Timedelta(hours=1)
BASELINE_MONTH = "2024-01"
BASELINE_STAMP = pd.Timestamp("2024-02-01", tz="UTC")

RAW_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "trip_distance",
]


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the pickup/dropoff times and pickup zone of every monthly file. Times are NYC wall-clock, no zone."""
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {raw_dir}; set RAW_DIR.")
    trips = pd.concat(
        (pd.read_parquet(f, columns=RAW_COLUMNS) for f in files), ignore_index=True
    )
    initial_count = len(trips)
    duration_min = (
        trips["tpep_dropoff_datetime"] - trips["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Bound dates to Q1 2024 to drop bogus years. Dropoffs get an extra day for late March 31st trips.
    valid_mask = (
        (duration_min >= 0)
        & (duration_min <= 1500)
        & (trips["PULocationID"] >= 1)
        & (trips["PULocationID"] <= 265)
        & (trips["trip_distance"] >= 0)
        & (trips["tpep_pickup_datetime"] >= pd.Timestamp("2024-01-01"))
        & (trips["tpep_pickup_datetime"] < pd.Timestamp("2024-04-01"))
        & (trips["tpep_dropoff_datetime"] < pd.Timestamp("2024-04-02"))
    )

    trips = trips[valid_mask].copy()
    logger.info(
        "Loaded %d valid trips from %d files (dropped %d invalid rows)",
        len(trips),
        len(files),
        initial_count - len(trips),
    )

    return trips


def localize_to_utc(naive_local: pd.Series) -> pd.Series:
    """Turn NYC wall-clock timestamps into timezone-aware UTC."""
    utc, _ = localize(naive_local)
    return utc


def rolling_zone_features(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per (pickup zone, stamp) with the trailing-WINDOW trip count and mean duration.

    Stamp = window END in UTC, every STEP. Trip count assigns trips by pickup time; mean duration assigns them by
    dropoff time. Zones with no trips in a window get no row (sparse), so the 15-minute TTL expires them to null.
    """
    # 1. Calculate duration in minutes before mutating the datetimes
    trips["duration_min"] = (
        trips["tpep_dropoff_datetime"] - trips["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Localize timestamps to UTC
    trips["pickup_utc"] = localize_to_utc(trips["tpep_pickup_datetime"])
    trips["dropoff_utc"] = localize_to_utc(trips["tpep_dropoff_datetime"])

    periods = int(WINDOW / STEP)

    # 2. Trip count (Trailing WINDOW, assigned by pickup time, steps of STEP)
    pickups = trips.set_index("pickup_utc")[["PULocationID"]]
    # label="right", closed="left" means a trip at 10:14 falls into the bucket ending at 10:15.
    pickup_buckets = (
        pickups.groupby("PULocationID")
        .resample(STEP, label="right", closed="left")
        .size()
    )

    rolling_counts = (
        pickup_buckets.groupby(level=0).rolling(periods, min_periods=1).sum()
    )
    # Drop the extra index level inserted by groupby.rolling()
    rolling_counts.index = rolling_counts.index.droplevel(0)
    rolling_counts.index.names = ["PULocationID", "window_end"]

    # 3. Mean duration (Trailing WINDOW, assigned by dropoff time, steps of STEP)
    dropoffs = trips.set_index("dropoff_utc")[["PULocationID", "duration_min"]]
    dropoff_buckets = (
        dropoffs.groupby("PULocationID")
        .resample(STEP, label="right", closed="left")
        .agg(
            sum_duration=("duration_min", "sum"),
            count_duration=("duration_min", "count"),
        )
    )

    rolling_durations = (
        dropoff_buckets.groupby(level=0).rolling(periods, min_periods=1).sum()
    )
    rolling_durations.index = rolling_durations.index.droplevel(0)
    rolling_durations.index.names = ["PULocationID", "window_end"]

    rolling_mean_duration = (
        rolling_durations["sum_duration"] / rolling_durations["count_duration"]
    ).rename("mean_duration")

    # 4. Combine and enforce sparsity
    features = pd.concat(
        [rolling_counts.rename("trip_count"), rolling_mean_duration], axis=1
    )
    # NaN trip counts happen when a zone had dropoffs but no pickups in the window.
    features["trip_count"] = features["trip_count"].fillna(0)

    # Sparse: Drop rows with no pickups in the trailing window (removes NaNs and 0s)
    features = features[features["trip_count"] > 0].reset_index()
    features["trip_count"] = features["trip_count"].astype(int)

    return features[["PULocationID", "window_end", "trip_count", "mean_duration"]]


def hour_of_week_baseline(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per hour of week (0 to 167, NYC local time) with the mean speed from BASELINE_MONTH."""
    is_baseline = trips["tpep_pickup_datetime"].dt.strftime("%Y-%m") == BASELINE_MONTH
    df = trips[is_baseline].copy()

    df["duration_hrs"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 3600

    # Filter absurd outliers before summing to protect the aggregate speed.
    # We cap distance at 100 miles (extreme for NYC) and per-trip speed at 80 mph.
    valid_mask = (
        (df["duration_hrs"] > 0)
        & (df["trip_distance"] >= 0)
        & (df["trip_distance"] < 100)
        & ((df["trip_distance"] / df["duration_hrs"]) <= 80)
    )
    df = df[valid_mask].copy()

    _, df["hour_of_week"] = localize(df["tpep_pickup_datetime"])

    # Aggregate total distance and total duration per hour of week
    grouped = df.groupby("hour_of_week")[["trip_distance", "duration_hrs"]].sum()
    baseline = (
        (grouped["trip_distance"] / grouped["duration_hrs"])
        .rename("baseline_mean_speed")
        .reset_index()
    )

    all_hours = pd.DataFrame({"hour_of_week": range(168)})
    baseline = all_hours.merge(baseline, on="hour_of_week", how="left")

    # Fallback to the global total_distance / total_duration if any hour is completely missing
    overall_speed = df["trip_distance"].sum() / df["duration_hrs"].sum()
    baseline["baseline_mean_speed"] = baseline["baseline_mean_speed"].fillna(
        overall_speed
    )

    baseline["window_end"] = BASELINE_STAMP

    return baseline[["hour_of_week", "window_end", "baseline_mean_speed"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    trips = load_raw()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zone = rolling_zone_features(trips)
    zone.to_parquet(OUT_DIR / "zone_stats.parquet", index=False)
    logger.info("zone_stats: %d rows", len(zone))

    baseline = hour_of_week_baseline(trips)
    baseline.to_parquet(OUT_DIR / "hour_of_week_baseline.parquet", index=False)
    logger.info("hour_of_week_baseline: %d rows", len(baseline))


if __name__ == "__main__":
    main()
