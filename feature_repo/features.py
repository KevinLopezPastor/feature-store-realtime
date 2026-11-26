"""Feast data sources and feature views, reading the Parquet files written by jobs/compute_aggregates.py."""

import os
from datetime import timedelta
from pathlib import Path

from entities import hour_of_week, pickup_zone
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int32

from feature_store_realtime.impute import ZONE_TTL

FEATURES_DIR = Path(os.environ.get("FEATURES_DIR", "../data/features"))

# --- Data Sources ---
# Timestamp field strictly maps to the window end in UTC, not the job execution time.

zone_stats_source = FileSource(
    name="zone_stats_source",
    path=str(FEATURES_DIR / "zone_stats.parquet"),
    timestamp_field="window_end",
)

hour_of_week_baseline_source = FileSource(
    name="hour_of_week_baseline_source",
    path=str(FEATURES_DIR / "hour_of_week_baseline.parquet"),
    timestamp_field="window_end",
)


# --- Feature Views ---

zone_stats_view = FeatureView(
    name="zone_stats",
    entities=[pickup_zone],
    ttl=ZONE_TTL,
    source=zone_stats_source,
    schema=[
        Field(name="trip_count", dtype=Int32),
        Field(name="mean_duration", dtype=Float32),
    ],
)

hour_of_week_baseline_view = FeatureView(
    name="hour_of_week_baseline",
    entities=[hour_of_week],
    ttl=timedelta(0),
    source=hour_of_week_baseline_source,
    schema=[
        Field(name="baseline_mean_speed", dtype=Float32),
    ],
)
