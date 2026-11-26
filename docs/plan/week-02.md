# Week 2: Feature store and training/serving consistency

**Repo:** `feature-store-realtime`. Goal: one set of feature definitions that feeds both training
(point-in-time correct) and a low-latency online prediction API.

Problem for all of Phase 1: predict NYC taxi trip duration from the public
[NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
(yellow taxi, Jan to Mar 2024).

## Builds on week 1

- The registered model and pipeline from `mlops-reproducible-pipeline` (MLflow at the same tracking URI).
- The baseline MAE of week 1's `@champion`, which this week's model must be compared against (week 1 promotes on MAE, not RMSE).

## Sources

- [Feast docs](https://docs.feast.dev/): feature views, entities, `get_historical_features`, materialization, Redis online store.
- [Feast quickstart](https://docs.feast.dev/getting-started/quickstart), end to end before writing your own definitions.
- Chip Huyen, *Designing Machine Learning Systems*, chapter 5 (feature engineering) and chapter 7 (training/serving skew).
- [FastAPI docs](https://fastapi.tiangolo.com/): lifespan events and async endpoints.
- [k6 docs](https://grafana.com/docs/k6/latest/) for the load test.

## Features to build

| Feature | Entity | Window | Why |
| --- | --- | --- | --- |
| Trip count | pickup zone | rolling 1 hour | Demand pressure near the pickup |
| Mean trip duration | pickup zone | rolling 1 hour | Current congestion signal |
| Historic mean speed | hour of week (0 to 167) | Jan 2024 baseline | Typical traffic for that time slot |

## Day by day (about 15 hours)

| Day | Hours | Work |
| --- | --- | --- |
| Mon | 2 | Feast concepts and quickstart; understand point-in-time joins and why naive joins leak the future. |
| Tue | 2 | Design features: per pickup zone, rolling 1-hour trip count and mean duration; per hour-of-week, historic mean speed. |
| Wed | 2.5 | Batch job computes the aggregates to Parquet; Feast definitions; `feast apply` against local Redis. |
| Thu | 2.5 | Retrain week 1's model on `get_historical_features`; compare MAE against the week 1 baseline. |
| Fri | 2 | FastAPI `/predict` reads online features by zone id, calls the model, returns duration; load test with k6. |
| Sat | 2.5 | Skew test: for 1,000 sampled rows, offline and online feature values must match. CI runs it. Archify diagram. |
| Sun | 1.5 | README with p50/p95 latency table, demo GIF, publish, post. |

## Repo layout

```
feature-store-realtime/
├── feature_repo/{entities,features,feature_store.yaml}
├── jobs/compute_aggregates.py   # batch features to Parquet
├── jobs/materialize.py          # offline to Redis
├── service/app.py               # FastAPI /predict
├── tests/test_skew.py           # offline == online
├── loadtest/predict.js          # k6
├── docs/architecture.html
└── docker-compose.yml           # Redis + API + MLflow
```

## Archify diagram

Show: raw Parquet, the aggregates job, the Feast offline store and registry, materialization to Redis,
the training path (historical features to week 1's pipeline) and the serving path (request to FastAPI
to Redis to model). Follow-up prompt: "Show the training path and serving path side by side and mark
where they share feature definitions."

## Done when

- [ ] Training uses `get_historical_features` with no future leakage (a test asserts timestamps).
- [ ] `/predict` p95 is under 50 ms locally at 50 requests per second.
- [ ] The skew test passes in CI.
- [ ] The README shows the archify PNG and the latency table.

**Stretch:** add a streaming push source (Kafka or Kinesis into Feast's push API) so the rolling
counts update in near real time.
