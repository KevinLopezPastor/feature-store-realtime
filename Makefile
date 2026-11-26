SHELL := bash

# The window `materialize` copies into Redis. The online store keeps the latest row per key inside it.
START ?= 2024-01-01
END ?= 2024-04-01

.PHONY: up down features materialize train serve test lint loadtest demo

# Redis only. MLflow and Postgres come from `make up` in the week 1 repo (needs `aws sso login --profile ml-roadmap`).
up:
	docker compose up -d redis

down:
	docker compose down

# Aggregates to data/features/*.parquet, then register the definitions. `feast apply` moves no data.
features:
	uv run python -m jobs.compute_aggregates
	uv run feast -c feature_repo apply

# Offline Parquet to Redis for START..END (UTC).
materialize:
	uv run python -m jobs.materialize --start $(START) --end $(END)

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

# Later days: Thursday, Friday, Friday, Sunday.
train serve loadtest demo:
	@echo "make $@: not implemented yet (see docs/plan/week-02.md)"; exit 1
