"""Load the online store (Redis) from the offline Parquet files for a time window.

Run with `make materialize START=2024-01-01 END=2024-04-01`. Naive times are read as UTC.
"""

import argparse
from datetime import UTC, datetime

from feast import FeatureStore


def parse_utc(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_utc, required=True)
    parser.add_argument("--end", type=parse_utc, required=True)
    args = parser.parse_args()

    store = FeatureStore(repo_path="feature_repo")
    store.materialize(start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()
