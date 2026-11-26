"""Training set: week 1's trips + Feast point-in-time features, one row per trip.

Run on the 1% sample first:
    RAW_DIR=../mlops-reproducible-pipeline/tests/data/smoke uv run python -m jobs.build_training_set
"""

import logging
import os
from pathlib import Path

import pandas as pd
from feast import FeatureStore
from taxi.features import features
from taxi.ingest import ingest
from taxi.validate import validate

from feature_store_realtime.entity_df import localize
from feature_store_realtime.impute import impute

logger = logging.getLogger(__name__)

RAW_DIR = Path(os.environ.get("RAW_DIR", "../mlops-reproducible-pipeline/data/raw"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
ZONE_REFS = [
    "zone_stats:trip_count",
    "zone_stats:mean_duration",
]
BASELINE_REFS = [
    "hour_of_week_baseline:baseline_mean_speed",
]
FEATURE_REFS = [*ZONE_REFS, *BASELINE_REFS]


def load_trips(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Week 1's ingest -> validate -> features, reused as is."""
    trips = features.fn(validate.fn(ingest.fn(raw_dir)))

    mask = (trips["tpep_pickup_datetime"] >= pd.Timestamp("2024-02-01")) & (
        trips["tpep_pickup_datetime"] < pd.Timestamp("2024-04-01")
    )
    return trips[mask].reset_index(drop=True)


def build_entity_df(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per trip: trip_id, PULocationID, hour_of_week, event_timestamp (UTC)."""
    event_timestamp, hour_of_week = localize(trips["tpep_pickup_datetime"])

    return pd.DataFrame(
        {
            "trip_id": trips.index.astype("int64"),
            "PULocationID": trips["PULocationID"].astype("int32"),
            "hour_of_week": hour_of_week,
            "event_timestamp": event_timestamp,
        }
    )


def fetch_features(
    fs: FeatureStore,
    entity_df: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Fetch zone and baseline views independently on unique keys, then left-merge onto entity_df."""
    # 1. Zone stats: deduplicate on (PULocationID, event_timestamp) so twin trips aren't dropped
    zone_keys = entity_df[["PULocationID", "event_timestamp"]].drop_duplicates()
    zone_cols = [ref.split(":")[1] for ref in ZONE_REFS]
    total_rows = len(zone_keys)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    zone_chunks: list[pd.DataFrame] = []

    for chunk_idx, start in enumerate(range(0, total_rows, chunk_size), start=1):
        key_chunk = zone_keys.iloc[start : start + chunk_size]
        hist_chunk = fs.get_historical_features(
            entity_df=key_chunk,
            features=ZONE_REFS,
        ).to_df()
        zone_chunks.append(hist_chunk[["PULocationID", "event_timestamp", *zone_cols]])
        logger.info(
            "Fetched zone chunk %d/%d (%d/%d unique keys)",
            chunk_idx,
            total_chunks,
            min(start + chunk_size, total_rows),
            total_rows,
        )

    zone_features = (
        pd.concat(zone_chunks, ignore_index=True)
        if zone_chunks
        else pd.DataFrame(columns=["PULocationID", "event_timestamp", *zone_cols])
    )

    # 2. Baseline: independent call so zone TTL expiry never drops baseline_mean_speed.
    # Deduplicating on hour_of_week alone assumes the baseline source holds one static timestamp.
    raw_source_path = Path(
        fs.get_feature_view("hour_of_week_baseline").batch_source.path
    )
    baseline_source_path = (
        raw_source_path
        if raw_source_path.is_absolute()
        else (Path(fs.repo_path) / raw_source_path).resolve()
    )
    assert (
        pd.read_parquet(baseline_source_path, columns=["window_end"])[
            "window_end"
        ].nunique()
        == 1
    ), "hour_of_week_baseline source must have exactly one static window_end timestamp"

    baseline_keys = entity_df[["hour_of_week", "event_timestamp"]].drop_duplicates(
        subset=["hour_of_week"]
    )
    baseline_cols = [ref.split(":")[1] for ref in BASELINE_REFS]
    baseline_features = fs.get_historical_features(
        entity_df=baseline_keys,
        features=BASELINE_REFS,
    ).to_df()[["hour_of_week", *baseline_cols]]

    # 3. Left-merge both back onto entity_df independently
    result = entity_df.merge(
        zone_features, on=["PULocationID", "event_timestamp"], how="left"
    )
    return result.merge(baseline_features, on="hour_of_week", how="left")


def assemble_training_set(trips: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    """Week 1's feature frame + the 3 Feast columns, imputed. Same rows, same order as `trips`.

    Requires `trips.index` to be a unique integer key matching `fetched["trip_id"]` (guaranteed by `load_trips`).
    """
    if not trips.index.is_unique:
        raise ValueError("trips.index must be unique to align with fetched['trip_id'].")
    if not fetched["trip_id"].is_unique:
        raise ValueError("fetched['trip_id'] contains duplicate keys.")

    feature_cols = [ref.split(":")[1] for ref in FEATURE_REFS]
    base = trips.copy()
    base["trip_id"] = base.index.astype("int64")

    merged = base.merge(
        fetched[["trip_id", *feature_cols]],
        on="trip_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["trip_id"])

    assert len(merged) == len(trips), f"Expected {len(trips)} rows, got {len(merged)}"
    return impute(merged)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    trips = load_trips()
    entity_df = build_entity_df(trips)
    logger.info("entity rows: %d", len(entity_df))
    fetched_df = fetch_features(FeatureStore("feature_repo"), entity_df)
    training_df = assemble_training_set(trips, fetched_df)
    logger.info(
        "training rows: %d, columns: %s", len(training_df), list(training_df.columns)
    )
