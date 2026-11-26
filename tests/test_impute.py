"""Unit tests for the shared impute / freshness helper. Toy frames, answers worked out by hand."""

import pandas as pd
import pytest

from feature_store_realtime.impute import expire_stale, impute


def test_impute_count_to_zero_duration_stays_null():
    df = pd.DataFrame(
        {
            "trip_count": [None, 5.0],
            "mean_duration": [None, 12.0],
            "baseline_mean_speed": [15.0, 15.0],
        }
    )
    out = impute(df)

    assert out["trip_count"].tolist() == [0, 5]
    assert pd.isna(out["mean_duration"].iloc[0])
    assert out["mean_duration"].iloc[1] == 12.0
    assert out["trip_count"].dtype == "int64"
    assert out["mean_duration"].dtype == "float64"
    assert out["baseline_mean_speed"].dtype == "float64"


def test_impute_fails_loudly_on_null_baseline():
    df = pd.DataFrame(
        {"trip_count": [1.0], "mean_duration": [9.0], "baseline_mean_speed": [None]}
    )
    with pytest.raises(ValueError, match="baseline_mean_speed"):
        impute(df)


@pytest.mark.parametrize("age_minutes", [14, 15, 16])
def test_expire_stale_boundary(age_minutes):
    as_of = pd.Series([pd.Timestamp("2024-02-10 12:30", tz="UTC")])
    feature_ts = as_of - pd.Timedelta(minutes=age_minutes)
    df = pd.DataFrame({"trip_count": [7.0], "mean_duration": [11.0]})
    out = expire_stale(df, feature_ts, as_of)

    if age_minutes <= 15:
        assert out["trip_count"].iloc[0] == 7.0
        assert out["mean_duration"].iloc[0] == 11.0
    else:
        assert pd.isna(out["trip_count"].iloc[0])
        assert pd.isna(out["mean_duration"].iloc[0])


def test_expire_stale_and_impute_object_columns():
    as_of = pd.Series(
        [
            pd.Timestamp("2024-02-10 12:30", tz="UTC"),
            pd.Timestamp("2024-02-10 12:30", tz="UTC"),
            pd.Timestamp("2024-02-10 12:30", tz="UTC"),
        ]
    )
    feature_ts = pd.Series(
        [
            pd.Timestamp("2024-02-10 12:20", tz="UTC"),  # fresh (10 min)
            pd.Timestamp("2024-02-10 12:00", tz="UTC"),  # stale (30 min)
            pd.NaT,  # missing key from Redis
        ]
    )
    df = pd.DataFrame(
        {
            "trip_count": pd.Series([4, 9, None], dtype="object"),
            "mean_duration": pd.Series([8.5, 14.0, None], dtype="object"),
            "baseline_mean_speed": pd.Series([12.5, 12.5, 12.5], dtype="object"),
        }
    )

    out = impute(expire_stale(df, feature_ts, as_of))

    assert out["trip_count"].tolist() == [4, 0, 0]
    assert out["mean_duration"].iloc[0] == 8.5
    assert pd.isna(out["mean_duration"].iloc[1])
    assert pd.isna(out["mean_duration"].iloc[2])
    assert out["trip_count"].dtype == "int64"
    assert out["mean_duration"].dtype == "float64"
    assert out["baseline_mean_speed"].dtype == "float64"


def test_expire_stale_fails_on_nat_as_of():
    as_of = pd.Series([pd.NaT])
    feature_ts = pd.Series([pd.Timestamp("2024-02-10 12:20", tz="UTC")])
    df = pd.DataFrame({"trip_count": [7.0], "mean_duration": [11.0]})

    with pytest.raises(ValueError, match="as_of"):
        expire_stale(df, feature_ts, as_of)
