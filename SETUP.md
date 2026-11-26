# Week 2 Claude Code kit: feature store

Same mentor and reviewer as week 1, retargeted to Feast, plus the local environment for this week.
Assumes `mlops-reproducible-pipeline` (week 1) is cloned next to this repo, in the same parent folder.

## What changed from week 1

| File | Change |
| --- | --- |
| `CLAUDE.md` | Week 2 context, Feast conventions, new `make` targets, reuse of week 1's pipeline |
| `docs/plan/week-02.md` | The week 2 plan from the roadmap doc, plus the feature table and the week 1 baseline |
| `docs/plan/PROGRESS.md` | New log with "Baseline from week 1" and "Carry-over from week 1" sections |
| `.claude/output-styles/mentor.md` | The parts you write this week: feature views, aggregations, point-in-time retrieval, skew test, `/predict` |
| `.claude/agents/project-reviewer.md` | New "Feature store correctness" area; memory split into recurring patterns and open findings |
| `.claude/agent-memory/project-reviewer/MEMORY.md` | Starter memory where you paste week 1's lessons |
| `.claude/settings.json` | Read access to the week 1 repo; Feast, Redis and k6 commands allowed; `feast teardown` asks first |
| `docker-compose.yml`, `.env.example` | Redis now; the API service is ready to uncomment on Friday |

## Setup (about 30 minutes)

1. **Create the repo** next to week 1:
   ```bash
   cd <folder that holds mlops-reproducible-pipeline>
   gh repo create feature-store-realtime --public --clone && cd feature-store-realtime
   ```
2. **Copy the kit** into the repo root, hidden `.claude` folder included (`cp -r week2-claude-kit/. .`).
3. **Carry over the reviewer's lessons:** open
   `../mlops-reproducible-pipeline/.claude/agent-memory/project-reviewer/MEMORY.md`, and paste its
   recurring patterns and any findings you did not fix into this repo's
   `.claude/agent-memory/project-reviewer/MEMORY.md` under "Recurring patterns".
4. **Python project and dependencies:**
   ```bash
   uv init --python 3.12
   uv add "feast[redis]" fastapi "uvicorn[standard]" lightgbm mlflow pandas pyarrow
   uv add --dev pytest ruff httpx
   uv add "git+https://github.com/KevinLopezPastor/mlops-reproducible-pipeline"
   ```
   The last line reuses week 1's train, evaluate and register code. If week 1 is not packaged
   (no `[build-system]` in its `pyproject.toml`), use `uv add --editable ../mlops-reproducible-pipeline`
   for now and ask the mentor to help you package it; that is a good small fix to week 1 anyway.
5. **Local services:** run `make up` in the week 1 repo (MLflow, Postgres; artifacts go to S3, so run `aws sso login --profile ml-roadmap` first), then here
   `cp .env.example .env` and `docker compose up -d redis`. Check with
   `docker compose exec redis redis-cli ping` (expects `PONG`) and open http://localhost:5000 for MLflow.
6. **Feast works:** `uv run feast version`. For Monday's quickstart, run `uv run feast init quickstart`
   in a scratch folder outside this repo, so it does not end up in git.
7. **Start Claude Code** from this repo root (VS Code: File > Open Folder, then the Claude panel or
   `claude` in the terminal). Check: `/output-style` shows **Mentor**, `@` offers
   **project-reviewer (agent)**, and `/` shows `/today`, `/checkpoint`, `/review-week`.
   If something is missing, `/exit` and start `claude` again.
8. **First prompt:** `/today Fill in PROGRESS.md "Baseline from week 1" from the week 1 repo and MLflow,
   and help me write the carry-over notes.` The mentor can read the week 1 repo through
   `additionalDirectories` in `.claude/settings.json`.
9. Commit: `git add . && git commit -m "Add week 2 Claude Code kit"`.

## Daily loop (same as week 1)

1. `/today` to start.
2. Write your parts at each `TODO(human)` and say "done".
3. `/checkpoint` at the end.
4. `/review-week` on Friday and Saturday; `/review-week skew` or `/review-week latency` to focus it.

## Week 2 pitfalls the reviewer will look for

- Event timestamps missing or not timezone-aware, which silently breaks point-in-time joins.
- Rolling-window features computed with data after the event time (future leakage).
- Different code paths computing features for training and for serving.
- Forgetting `feast materialize-incremental` before testing `/predict`, so Redis returns nulls.
- Loading the model or `FeatureStore` on every request instead of once in the FastAPI lifespan.
