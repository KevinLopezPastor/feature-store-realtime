"""The one place that turns a naive NYC-local timestamp into (UTC, hour_of_week).

Used by jobs/compute_aggregates.py, jobs/build_training_set.py and, on Friday, service/app.py.
Training and serving must not derive these two values separately.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

LOCAL_TZ = "America/New_York"  # TLC pickup/dropoff times are NYC local, naive.


def localize(naive_local: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (utc, hour_of_week) from one NYC wall-clock Series.

    utc: tz-aware UTC. hour_of_week: int32, 0 = Monday 00:00 local, 0 to 167.
    """
    aware_local = naive_local.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward")
    shifted_count = (aware_local.dt.tz_localize(None) != naive_local).sum()
    if shifted_count > 0:
        logger.info(
            "Shifted %d timestamps falling in the DST gap for %s",
            shifted_count,
            naive_local.name,
        )

    utc = aware_local.dt.tz_convert("UTC")
    hour_of_week = (aware_local.dt.dayofweek * 24 + aware_local.dt.hour).astype("int32")
    return utc, hour_of_week
