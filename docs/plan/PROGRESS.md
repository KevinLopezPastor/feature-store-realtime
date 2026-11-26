# Week 2 progress

## Done when

- [ ] Training uses `get_historical_features` with no future leakage (test asserts timestamps)
- [ ] `/predict` p95 under 50 ms locally at 50 requests per second
- [ ] Skew test passes in CI
- [ ] README shows the archify PNG and the latency table

## Baseline from week 1

- Week 1 `@champion`: `nyc-taxi-duration-lgbm` v2 (run `62ed758d729043d78d87a0643b56346a`), test set = March 2024
  - MAE: **5.4802** (week 1's promotion metric; this week compares on MAE, a challenger must beat it by the 1% margin to be called better)
  - RMSE capped at 120 min: 6.9655 (secondary, report only)
  - RMSE raw: 31.0753 (dominated by the heavy tail; report only)
  - Lineage: data hash `1e82f1da...`, commit `6213abc`, `git_dirty=True` (so this run is not strictly reproducible)
- MLflow tracking URI: http://localhost:5000 (week 1's stack, artifacts in S3 via `--serve-artifacts`)
- Compare on the same split (train < 2024-03-01, test = March) and the same MAE from week 1's `evaluate`, or the comparison is meaningless.

## Carry-over from week 1

- Keep splitting by date (train before 2024-03-01, test = March), now also for the point-in-time join.
- SSO credential expiry: skipping the `~/.aws` mount fix. If a run freezes at `log_model` or MLflow logs `ExpiredToken`, run `aws sso login --profile ml-roadmap`, then `make down && make up` in week 1's repo.
- Week 1 finding: the Pandera gate checks duration and zone but not the pickup date, so rows dated 2003 and 2009 pass it (found in week 2's aggregates job).
- Registry: 20 versions of `nyc-taxi-duration-lgbm`, all MAE 5.4802, `@champion` = v2. Kept as lineage evidence. Always load `@champion`, never "latest".
- Week 1 packaging gap (found 2026-11-22): `taxi.validate` imports `schemas.trips`, but `schemas/` sits at week 1's repo root and is not installed with the dependency. Week 2 imports only work with `PYTHONPATH=../mlops-reproducible-pipeline`. Update 2026-11-24: this now blocks pytest (collecting `tests/test_training_set.py` fails with `No module named 'schemas'` unless `PYTHONPATH=../mlops-reproducible-pipeline`), and "skew test passes in CI" is a Done-when item. FIXED 2026-11-26 (Kevin chose (a)): week 1's `schemas/` moved to `src/taxi/schemas/` (import `from .schemas.trips`, `dvc.yaml` dep covered by `src/taxi`, README paths, pytest `pythonpath` crutch removed). Week 1: 13 tests pass without it. Week 2: after `uv sync --reinstall-package mlops-reproducible-pipeline`, `import taxi.validate` and the 10 tests work with no `PYTHONPATH`, so CI needs none. Not committed in week 1 yet; the next `dvc repro` there re-runs the training stage once (dvc.lock still lists the old `schemas` path).
- DECIDED 2026-11-26 (Kevin, option A): week 2's models live in their own entry `nyc-taxi-duration-lgbm-feast`. `@champion` there is gated by `decide_promotion` (beat week 1's re-scored `@champion` by the 1% margin on the same March rows) and stays unset until that happens. The API loads a separate alias `serving`, set by hand on the chosen version (currently v3). Friday's handler must load `models:/nyc-taxi-duration-lgbm-feast@serving`, never "latest".

## Feature design decisions

- Windowing: trip count by pickup time; mean duration by dropoff time. Each row stamped with the window END (data time, not job time), UTC.
- Cadence: 15-minute step, 1-hour trailing window.
- Baseline (historic mean speed per hour of week, NYC local time): computed from Jan 2024, stamped 2024-02-01 00:00 UTC, no TTL.
- Training on February only, test = March. Control run: week 1's 7-feature model on February only, so the effect of the Feast features is isolated. Document both in the README.
- DST (2024-03-10): `tz_localize(..., nonexistent="shift_forward")`. Count and log the shifted rows in the job.
- Empty zone-windows: sparse rows (no zero rows), TTL = cadence = 15 min for both rolling features, so an empty window expires to null (a longer TTL returns a stale row from an earlier window). Count nulls are imputed to 0 and duration nulls stay null, in one shared helper.
- Tested 2026-11-03 (offline, real data, zone 22): a request exactly 15 min after a stamp still gets the row (inclusive); at 16 min the ENTITY ROW IS DROPPED from `get_historical_features`, not returned as null. Training would silently lose requests in quiet zones. Decision (Kevin): A) keep rows sparse; after `get_historical_features`, left-merge the result back onto the entity dataframe and impute (count to 0, duration stays null), with a test asserting output rows == input rows. Not B (dense rows).
- Tested 2026-11-03 (online, after `make materialize`): Redis holds one latest row per key (430 = 262 zones + 168 hours). `get_online_features` does NOT apply the TTL: zone 22 still returns its last row (2024-04-01 01:45, trip_count 1) with the wall clock in 2026. Offline drops expired rows, online serves them forever: that is skew. Fix: the shared helper (imputation + freshness) compares the `__ts` event timestamp from the online response with the request's as-of time and treats an age over 15 min as missing, in training and in `/predict`.
- Friday decision: the freshness check needs a reference time. The data is from 2024, so wall-clock "now" would make everything stale; `/predict` takes an explicit as-of time (replay).
- Unknown zone (9999) returns nulls, not an error; the `/predict` handler should validate zone ids (1 to 265) instead of silently imputing 0.
- Imputation (count to 0, duration left null) lives in one shared helper used by training and `/predict`.
- Date bounds in the batch job: pickups strictly within Jan 1 to Mar 31, 2024, dropoffs before Apr 2, 2024. Raw files hold bogus years (2003, 2009) that made `resample` build empty 15-minute bins spanning decades.
- Baseline speed = total distance / total hours per hour of week (not the mean of per-trip speeds). Trips over 100 miles or 80 mph are dropped before summing; without the rule one hour came out at 161 mph. Result: 9.3 to 25.7 mph, median 12.9.
- Measured 2026-11-22/23 (Feast file offline store, `get_historical_features`): memory grows about 1 MB per entity row per call (1,000 rows peak 1.4 GB, 2,000 rows 2.3 GB, 8,000 or 65,898 in one call killed, exit 137, on a 7.6 GB box); time is about 3 s fixed plus 2.8 ms per row. A narrow time window did NOT reduce memory (no pushdown seen), so chunk by row count (`CHUNK_SIZE` = 1,000). Chunked output equals one-shot output (checked on 2 days). The 65,898-row smoke sample ran in 66 chunks, 4 min 22 s, rows in == rows out. The full 6.6 M rows would take about 5 h: decide before the full run (sample of trips, or fewer distinct entity keys).
- Smoke sample (65,898 trips) after the merge-back: 484 rows (0.73%) were dropped by Feast and come back with ALL features null, `baseline_mean_speed` included (null count identical to `trip_count`, every row), though the baseline view has no TTL. Another 171 rows have `trip_count` but null `mean_duration` (count by pickup time, duration by dropoff time). Serving would still return the baseline from Redis for those requests: skew unless the baseline is fetched so it cannot be dropped with the zone view.
- Found 2026-11-23: Feast collapses entity rows that share the same join key and event_timestamp (here zone + second) and returns ONE of them. Trip 21119 (zone 237, 18:11:26) came back null in a chunk that also held its twin 21652; alone, or with the twin removed, it came back with trip_count 365. With the merge-back on `trip_id` the lost twin would be imputed to count 0 while the real value is 365: silent corruption. 24 rows of the 65,898 sample are 2nd copies of a (zone, timestamp). Grows faster than linearly with trips per second; not measured at full scale. A null after the merge-back therefore means "expired" OR "collapsed twin" OR "dropped with another view"; the helper must not treat them all as expiry.
- Thursday results (2026-11-26), MAE on March unless noted. `jobs/retrain.py` logs the Feast model, the 7-feature control on the same rows, week 1's `@champion` re-scored on the same rows, and the paired per-row |error| difference (Feast minus control; negative = Feast better).

  | Run | Feast | Control | Champion on same rows | March paired diff (SE) | Feb holdout paired diff (SE) |
  | --- | --- | --- | --- | --- | --- |
  | Smoke, about 30 k train rows (registry v1, v2) | 6.8907 | 6.7555 | 5.3621 | +0.135 (0.031) | +0.225 (0.066) |
  | 100 k train, 50 k test (registry v3) | 6.2048 | 6.2275 | 5.3470 | -0.023 (0.018) | +0.067 (0.027) |

  Reading: Feast's effect on MAE at this scale is about zero (within roughly +/-0.04), not a gain. The gap closes as training rows grow (consistent with a learning-curve effect: 3 extra noisy columns overfit when rows are scarce; 3 points, one seed, not proven). The level-shift hypothesis (absolute `trip_count` higher in March) is not supported: Feast does not help in February and hurt in March. The SEs assume independent rows; trips in one zone-window share feature values, so they are optimistic. The champion still wins by about 0.86 because it trained on about 6 M rows; a fair comparison needs the full-data run (about 5 to 7 h of retrieval), parked as a stretch idea. The re-scored champion differs from its recorded 5.4802 by 0.12 on 35 k rows (sampling noise), 0.13 on 50 k.
- Dtypes to settle before Saturday: the job writes float64/int64, the feature views declare Float32/Int32. Float32 online vs float64 offline differs at about the 7th digit, so the skew test needs a tolerance or the views move to Float64.

## Stretch ideas (parked)

- Full-data run of `jobs/retrain.py` (Feb train, March test, about 6.6 M rows, 5 to 7 h of Feast retrieval) to make the champion comparison fair. Skipped 2026-11-26: the plan only asks for the comparison, not for beating the champion.
- Per-feature ablation (each Feast column alone) and feature importance, to see which of the 3 columns adds noise.
- Several seeds per sample size, to turn the 3-point learning curve into a real one.

- Mount `~/.aws` read-only with `AWS_PROFILE` in week 1's MLflow service so boto3 refreshes SSO credentials itself (skipped on 2026-11-01).

## Sessions

<!-- /checkpoint adds a new entry here, newest first -->

### 2026-11-26: Thu block (retrain on `get_historical_features`, MAE comparison)

**Done**
- Entity dataframe, chunked retrieval (`CHUNK_SIZE` 1,000), shared `localize` and `impute`, `assemble_training_set`; 10 tests pass (causal oracle, row preservation, impute edge cases).
- `jobs/retrain.py`: Feast model vs 7-feature control vs week 1 `@champion` on the same sampled rows; paired error statistics; February holdout; separate registry entry; promotion gate.
- Results and reading in "Feature design decisions" above. Registry: week 1's entry untouched (20 versions, `@champion` = v2); Feast entry has v1 to v3, no alias yet.

**Who wrote what**
- Kevin: `localize`, `impute`/`expire_stale`, `fetch_features`, `assemble_training_set`, tests, `_stratified_sample`, `sample_trips`, `score_champion`, `decide_promotion`, `paired_error_stats`, `feb_holdout`, the registry decision (own entry, option A).
- Claude: scaffolds, `params.yaml` copy, the `"split"` key fix in `feb_holdout` and the wiring in `main()`, probes, review, PROGRESS edits.

**Lessons**
- Feast silently drops expired rows and collapses twin entity rows; a merge-back with row-count and uniqueness checks was the only visible defence. Same family as week 1's silent corruption.
- Expired SSO token again surfaced as 500s from MLflow's artifact endpoint (week 1 carry-over). Run the decisive local experiments before the S3 step.
- A pilot on a small sample can show a sign that flips at 3x the data; pair the errors and read the SE before concluding.

**Open**
- Set `serving` alias on the chosen version (v3 now) before Friday; the API loads it.
- `localize()` NaT handling; `pd.to_numeric` before `fillna` in `impute`.
- Unit tests for the aggregates job (hand-computed answers); toy tests for `paired_error_stats` and `score_champion`.
- Packaging decision (a/b) before Saturday's CI; first git commit (lineage tag says `none`).
- Dtypes Float32 vs float64 for the skew test.

**Next step**
- Friday: FastAPI `/predict` with lifespan loading the `serving` model and the `FeatureStore`, as-of time replay, zone id validation (1 to 265), the shared helper, k6 script.

### 2026-11-03: Wed block done (scaffold, aggregates job, Feast definitions, apply, materialize, TTL probes)

**Done**
- Scaffold: `feature_repo/feature_store.yaml`, `jobs/`, `Makefile`, `.gitignore`, pytest `pythonpath`; `.env.example` corrected (model name, MinIO line removed).
- `jobs/compute_aggregates.py` (Kevin): 15-minute stamps, 1-hour trailing windows, window-END stamps, pickup-time counts and dropoff-time durations, DST `shift_forward`, date bounds, baseline = total distance / total hours with a per-trip outlier rule. Full data: 45 s, 962,605 zone rows, 168 baseline rows, 8 of 8 random rows match a brute-force recount.
- `feature_repo/entities.py`, `features.py` (Kevin): `feast apply` clean. `make materialize` loads 430 keys into Redis in 16 s.
- Probes: offline TTL is inclusive at 15 min and drops expired entity rows; online ignores TTL (see the decisions section).

**Who wrote what**
- Kevin: all aggregation logic, entities and feature views, the date and outlier rules, decision A.
- Claude: scaffold, brute-force check and TTL/online probe scripts (scratchpad, not in the repo), review, PROGRESS edits.

**Lessons**
- Offline and online disagree about TTL; the row count (requested vs returned) was the only visible symptom. Same family as week 1's silent corruption.
- `Entity(value_type=...)` takes `feast.value_type.ValueType`; `Field(dtype=...)` takes `feast.types`.
- Raw TLC files hold impossible dates (2003, 2009) and distances; week 1's gate does not check either.

**Open**
- Dtypes (Float32/Int32 views vs float64/int64 job) before the skew test.
- Unit tests for the aggregates (toy frame, hand-computed answers) not written yet.
- Registry name or alias for the retrained model (Thursday).

**Next step**
- Thursday: retrain on `get_historical_features`: entity dataframe from the trips, left-merge back, shared impute + freshness helper, a test that asserts feature timestamps <= event timestamps and output rows == input rows; compare MAE on March against 5.4802 (and the Feb-only control run).

### 2026-11-02: Mon block done (week 1 baseline, carry-over, Feast quickstart, point-in-time joins)

**Done**
- Baseline from the live registry: `@champion` v2, MAE 5.4802 (challenger must reach MAE below 5.4254). Plan switched from RMSE to MAE; MinIO references removed (S3 since week 1).
- Carry-over written. Registry's 20 versions kept as lineage evidence. `~/.aws` mount fix skipped and parked.
- Feast quickstart run in `/tmp/feast-scratch` (outside the repo): `feast apply`, `materialize-incremental`, `get_online_features`, `get_historical_features`.
- Point-in-time join understood: latest feature row with `feature_timestamp <= event_timestamp`; the 12:30 request returned the 12:00 row, as predicted. Online store returns the latest materialized row.

**Who wrote what**
- Kevin: the metric and MinIO decisions, the carry-over choices, the 12:30 prediction, the leakage paragraph, the first feature table.
- Claude: PROGRESS/plan/CLAUDE/SETUP/compose edits, the registry query, the quickstart commands.

**Lessons**
- Quickstart output of `materialize-incremental` shows no row counts; check with `get_online_features`.
- The feature timestamp means "known from this moment". A rolling aggregate must be stamped with its window end, or a correct join still leaks.
- `glob('*.parquet')[0]` is not deterministic; name the file.

**Open (Tuesday's design)**
- Mean duration: assign trips to windows by dropoff time, not pickup time. Trip count: pickup or dropoff?
- TTL = max staleness, not the window length. Pick the stamp cadence first.
- Baseline timestamp: a Feb 1 stamp gives January training rows null; an earlier stamp puts January trips in their own baseline. Choose the training range accordingly.
- Hour-of-week timezone: TLC times are NYC local, our timestamps are UTC. Same derivation in the job and the API.
- Entities need an explicit `value_type`.
- Thursday: registry name or alias for the retrained model.

**Next step**
- Tuesday: finalize the feature table above (timestamps, TTL, null handling), then Wednesday's aggregates job and Feast definitions.
