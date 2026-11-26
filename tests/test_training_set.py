"""Retrieval and assembly tests on hand-built feature Parquet (data/ is gitignored, CI has none).

Hand-built data is the only way to hit the edge cases on demand: exact stamp, exact TTL, twins, DST.
"""

import importlib
from pathlib import Path

import pandas as pd
import pytest
from feast import FeatureStore

from feature_store_realtime.impute import ZONE_TTL, impute
from jobs.build_training_set import FEATURE_REFS, assemble_training_set, fetch_features

UTC = "UTC"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A temporary FeatureStore over tiny zone_stats / hour_of_week_baseline Parquet files."""
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Hand-built zone_stats covering zone 161 at 12:00 and 12:15 UTC
    zone_df = pd.DataFrame(
        {
            "PULocationID": pd.array([161, 161], dtype="int32"),
            "window_end": pd.to_datetime(
                ["2024-02-10 12:00:00+00:00", "2024-02-10 12:15:00+00:00"], utc=True
            ),
            "trip_count": pd.array([10, 25], dtype="int32"),
            "mean_duration": pd.array([8.0, 14.5], dtype="float32"),
        }
    )
    zone_df.to_parquet(features_dir / "zone_stats.parquet", index=False)

    # Single static stamp for baseline (satisfies fetch_features nunique() == 1 assertion)
    baseline_df = pd.DataFrame(
        {
            "hour_of_week": pd.array([0], dtype="int32"),
            "window_end": pd.to_datetime(["2024-02-01 00:00:00+00:00"], utc=True),
            "baseline_mean_speed": pd.array([15.5], dtype="float32"),
        }
    )
    baseline_df.to_parquet(features_dir / "hour_of_week_baseline.parquet", index=False)

    # Point FEATURES_DIR to tmp_path/features and reload feature definitions
    monkeypatch.setenv("FEATURES_DIR", str(features_dir))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "feature_repo"))

    import entities
    import features

    importlib.reload(entities)
    importlib.reload(features)

    # Configure isolated Feast store in tmp_path
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "feature_store.yaml").write_text(
        f"project: test_taxi\n"
        f"registry: {repo_dir / 'registry.db'}\n"
        f"provider: local\n"
        f"online_store:\n"
        f"  type: sqlite\n"
        f"  path: {repo_dir / 'online.db'}\n"
        f"entity_key_serialization_version: 3\n"
    )

    fs = FeatureStore(repo_path=str(repo_dir))
    fs.apply(
        [
            entities.pickup_zone,
            entities.hour_of_week,
            features.zone_stats_view,
            features.hour_of_week_baseline_view,
        ]
    )
    return fs


def requests(*rows: tuple[int, str]) -> pd.DataFrame:
    """Entity rows: (zone, UTC timestamp string). trip_id is the row number; hour_of_week is a dummy."""
    return pd.DataFrame(
        {
            "trip_id": range(len(rows)),
            "PULocationID": pd.array([z for z, _ in rows], dtype="int32"),
            "hour_of_week": pd.array([0] * len(rows), dtype="int32"),
            "event_timestamp": pd.to_datetime([t for _, t in rows], utc=True),
        }
    )


def _causal_oracle(fs: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """Independent point-in-time lookup directly from Parquet sources."""
    zone_path = fs.get_feature_view("zone_stats").batch_source.path
    base_path = fs.get_feature_view("hour_of_week_baseline").batch_source.path

    zone_src = pd.read_parquet(zone_path)
    base_src = pd.read_parquet(base_path)

    rows = []
    for _, req in entity_df.iterrows():
        ts = req["event_timestamp"]
        valid_zone = zone_src[
            (zone_src["PULocationID"] == req["PULocationID"])
            & (zone_src["window_end"] <= ts)
            & ((ts - zone_src["window_end"]) <= ZONE_TTL)
        ].sort_values("window_end")

        if valid_zone.empty:
            trip_count, mean_duration = float("nan"), float("nan")
        else:
            latest = valid_zone.iloc[-1]
            trip_count = float(latest["trip_count"])
            mean_duration = float(latest["mean_duration"])

        valid_base = base_src[
            (base_src["hour_of_week"] == req["hour_of_week"])
            & (base_src["window_end"] <= ts)
        ].sort_values("window_end")
        baseline_speed = float(valid_base.iloc[-1]["baseline_mean_speed"])

        rows.append(
            {
                "trip_id": req["trip_id"],
                "trip_count": trip_count,
                "mean_duration": mean_duration,
                "baseline_mean_speed": baseline_speed,
            }
        )

    out = entity_df.merge(pd.DataFrame(rows), on="trip_id", how="left")
    return impute(out)


def test_retrieval_matches_causal_oracle(store):
    """Feast + merge-back == an independent lookup in the Parquet (latest stamp <= request, age <= TTL)."""
    entity_df = requests(
        (
            161,
            "2024-02-10 12:00:00+00:00",
        ),  # 0: exact stamp -> sees 12:00 (count=10), not 12:15
        (161, "2024-02-10 12:07:00+00:00"),  # 1: mid-window -> sees 12:00 (count=10)
        (
            161,
            "2024-02-10 12:30:00+00:00",
        ),  # 2: exactly 15m after 12:15 -> kept (count=25)
        (
            161,
            "2024-02-10 12:31:00+00:00",
        ),  # 3: 16m after 12:15 -> expired (count=0, dur=NaN)
        (
            161,
            "2024-02-10 12:20:00+00:00",
        ),  # 4: twin A (same zone, same second) -> count=25
        (
            161,
            "2024-02-10 12:20:00+00:00",
        ),  # 5: twin B (same zone, same second) -> count=25
    )

    actual = impute(fetch_features(store, entity_df))
    expected = _causal_oracle(store, entity_df)

    assert len(actual) == len(entity_df)
    assert actual["trip_count"].tolist() == [10, 10, 25, 0, 25, 25]
    assert pd.isna(actual["mean_duration"].iloc[3])
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-5, atol=1e-5)


def test_assemble_keeps_rows_and_exact_columns():
    """assemble_training_set: same row count and order, columns == week 1's + the 3 Feast ones, no join keys."""
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-02-10 12:00:00", "2024-02-10 12:31:00"]
            ),
            "VendorID": pd.Series([1, 2], dtype="category"),
            "PULocationID": pd.Series([161, 161], dtype="category"),
            "DOLocationID": pd.Series([237, 140], dtype="category"),
            "RatecodeID": pd.Series([1, 1], dtype="category"),
            "passenger_count": [1.0, 2.0],
            "trip_distance": [2.5, 4.1],
            "pickup_hour": [12, 12],
            "fare_amount": [14.0, 21.5],
        }
    )

    fetched = pd.DataFrame(
        {
            "trip_id": [1, 0],  # reversed order to verify alignment on trip_id
            "PULocationID": pd.array([161, 161], dtype="int32"),
            "hour_of_week": pd.array([0, 0], dtype="int32"),
            "event_timestamp": pd.to_datetime(
                ["2024-02-10 17:31:00+00:00", "2024-02-10 17:00:00+00:00"], utc=True
            ),
            "trip_count": [float("nan"), 10.0],
            "mean_duration": [float("nan"), 8.0],
            "baseline_mean_speed": [15.5, 15.5],
        }
    )

    out = assemble_training_set(trips, fetched)

    feature_cols = [ref.split(":")[1] for ref in FEATURE_REFS]
    assert list(out.columns) == [*trips.columns, *feature_cols]
    assert "trip_id" not in out.columns
    assert "hour_of_week" not in out.columns
    assert "event_timestamp" not in out.columns
    assert len(out) == len(trips)
    assert out["trip_count"].tolist() == [10, 0]
    assert out["fare_amount"].tolist() == [14.0, 21.5]

    for col in trips.columns:
        assert out[col].dtype == trips[col].dtype


def test_assemble_rejects_duplicate_trip_ids():
    """Duplicate trip_id in fetched must raise ValueError rather than silently multiplying rows."""
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-02-10 12:00:00", "2024-02-10 12:31:00"]
            ),
            "fare_amount": [14.0, 21.5],
        }
    )
    fetched = pd.DataFrame(
        {
            "trip_id": [0, 0, 1],
            "trip_count": [10.0, 12.0, 5.0],
            "mean_duration": [8.0, 9.0, 7.0],
            "baseline_mean_speed": [15.5, 15.5, 15.5],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        assemble_training_set(trips, fetched)
