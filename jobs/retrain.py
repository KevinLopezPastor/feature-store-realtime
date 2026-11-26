"""Retrain on Feast features and compare with week 1's @champion on the SAME sampled March rows.

Pilot on the smoke sample first:
    PYTHONPATH=../mlops-reproducible-pipeline RAW_DIR=../mlops-reproducible-pipeline/tests/data/smoke \
        uv run python -m jobs.retrain
Needs week 1's MLflow stack up (`make up` there; `aws sso login --profile ml-roadmap` if a run freezes).
"""

import copy
import logging
import math
import os
import subprocess
from unittest.mock import patch

import mlflow
import mlflow.lightgbm
import pandas as pd
from feast import FeatureStore
from mlflow import MlflowClient
from taxi.evaluate import evaluate
from taxi.lineage import git_commit, git_dirty
from taxi.params import load_params
from taxi.register import CHAMPION_ALIAS, register
from taxi.train import train

from jobs.build_training_set import (
    FEATURE_REFS,
    assemble_training_set,
    build_entity_df,
    fetch_features,
    load_trips,
)

logger = logging.getLogger(__name__)

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "nyc-taxi-duration-feast"
WEEK1_MODEL = "nyc-taxi-duration-lgbm"  # week 1's registry entry: read only, never register into it
FEAST_COLS = [ref.split(":")[1] for ref in FEATURE_REFS]
SEED = 42
# None = use every row of that month (the smoke pilot). Set after measuring the pilot's spread.
N_TRAIN = int(os.environ["N_TRAIN"]) if "N_TRAIN" in os.environ else None
N_TEST = int(os.environ["N_TEST"]) if "N_TEST" in os.environ else None


RECORDED_CHAMPION_MAE = 5.4802


def _stratified_sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Proportional sample stratified by calendar day, keeping the original index."""
    if n is None or n >= len(df) or df.empty:
        return df

    days = df["tpep_pickup_datetime"].dt.date
    frac = n / len(df)
    sampled = df.groupby(days, group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed)
    )

    # Adjust rounding drift by topping up or trimming to exact size `n`
    if len(sampled) < n:
        remaining = df.drop(index=sampled.index)
        extra = remaining.sample(n=n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, extra])
    elif len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)

    return sampled


def sample_trips(
    trips: pd.DataFrame, n_train: int | None, n_test: int | None, seed: int
) -> pd.DataFrame:
    """Sample February trips (train side) and March trips (test side); keep the original index."""
    is_feb = trips["tpep_pickup_datetime"] < pd.Timestamp("2024-03-01")
    feb_trips = _stratified_sample(trips[is_feb], n_train, seed)
    mar_trips = _stratified_sample(trips[~is_feb], n_test, seed)
    return pd.concat([feb_trips, mar_trips])


def score_champion(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """MAE of week 1's @champion on exactly these rows."""
    model = mlflow.lightgbm.load_model(f"models:/{WEEK1_MODEL}@{CHAMPION_ALIAS}")
    X_champ = X_test[list(model.feature_name_)]
    metrics = evaluate.fn(model, X_champ, y_test)
    return float(metrics["mae"])


def decide_promotion(new_mae: float, champion_mae: float, margin: float) -> bool:
    """True if the Feast model should take @champion in the new registry entry."""
    if abs(champion_mae - RECORDED_CHAMPION_MAE) > RECORDED_CHAMPION_MAE * margin:
        logger.warning(
            "Re-scored champion MAE (%.4f) differs from recorded Week 1 MAE (%.4f) by more than margin (%.2f%%); "
            "consider increasing N_TEST to reduce sampling noise.",
            champion_mae,
            RECORDED_CHAMPION_MAE,
            margin * 100,
        )
    return bool(new_mae < champion_mae * (1.0 - margin))


FEB_HOLDOUT_CUTOFF = "2024-02-22"  # train on Feb 1-21, test on Feb 22-29 (same month, no count-level shift)


def paired_error_stats(model_a, X_a, model_b, X_b, y) -> dict[str, float]:
    """Per-row |error| difference (A minus B) on the SAME rows: mean, std, standard error, n."""
    if not (X_a.index.equals(X_b.index) and X_a.index.equals(y.index)):
        raise ValueError(
            "X_a, X_b, and y must have identical index alignment for paired comparison."
        )

    y_arr = y.to_numpy(dtype=float)
    err_a = abs(y_arr - model_a.predict(X_a))
    err_b = abs(y_arr - model_b.predict(X_b))
    diff = err_a - err_b

    n = len(diff)
    mean = float(diff.mean())
    std = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else float("nan")

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "n": float(n),
    }


def feb_holdout(df: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """Feast vs control trained on early February, tested on late February."""
    feb_df = df[df["tpep_pickup_datetime"] < pd.Timestamp("2024-03-01")].copy()

    holdout_cfg = copy.deepcopy(cfg)
    holdout_cfg["train"]["cutoff_date"] = FEB_HOLDOUT_CUTOFF

    with patch("taxi.train.load_params", return_value=holdout_cfg):
        model_feast, X_feast, y_feast = train.fn(feb_df)
        model_ctrl, X_ctrl, y_ctrl = train.fn(feb_df.drop(columns=FEAST_COLS))

    feast_metrics = evaluate.fn(model_feast, X_feast, y_feast)
    ctrl_metrics = evaluate.fn(model_ctrl, X_ctrl, y_ctrl)
    paired = paired_error_stats(model_feast, X_feast, model_ctrl, X_ctrl, y_feast)

    return {
        "feb_holdout_feast_mae": float(feast_metrics["mae"]),
        "feb_holdout_control_mae": float(ctrl_metrics["mae"]),
        **{f"feb_holdout_paired_{k}": v for k, v in paired.items()},
    }


def _git_tags() -> dict[str, str]:
    try:
        return {
            "lineage.git_commit": git_commit(),
            "lineage.git_dirty": str(git_dirty()),
        }
    except subprocess.CalledProcessError:  # this repo has no commit yet
        return {"lineage.git_commit": "none"}


def _latest_version(client: MlflowClient, name: str) -> str:
    return str(
        max(int(mv.version) for mv in client.search_model_versions(f"name='{name}'"))
    )


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    cfg = load_params()

    trips = sample_trips(load_trips(), N_TRAIN, N_TEST, SEED)
    fetched = fetch_features(FeatureStore("feature_repo"), build_entity_df(trips))
    df = assemble_training_set(trips, fetched)

    with mlflow.start_run():
        mlflow.set_tags(
            {**_git_tags(), "feature_set": "week1_7+feast_3", "primary_metric": "mae"}
        )
        mlflow.log_params(
            {f"train.lgbm.{k}": v for k, v in cfg["train"]["lgbm"].items()}
            | {"sample.n_train": N_TRAIN, "sample.n_test": N_TEST, "sample.seed": SEED}
        )

        # A: week 1's 7 features + the 3 Feast features.
        model, X_test, y_test = train.fn(df)
        metrics = evaluate.fn(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # B: control, the same rows without the Feast columns (isolates what Feast adds).
        control_model, X_ctrl, y_ctrl = train.fn(df.drop(columns=FEAST_COLS))
        control = evaluate.fn(control_model, X_ctrl, y_ctrl)
        mlflow.log_metrics({f"control_{k}": v for k, v in control.items()})

        # Paired Feast-vs-control error difference on March, then the February holdout.
        # Before the champion step so an S3/SSO failure cannot hide them.
        paired = paired_error_stats(model, X_test, control_model, X_ctrl, y_test)
        mlflow.log_metrics({f"paired_{k}": v for k, v in paired.items()})
        holdout = feb_holdout(df, cfg)
        mlflow.log_metrics(holdout)
        logger.info(
            "March paired diff (feast - control): mean=%.4f se=%.4f n=%d",
            paired["mean"],
            paired["se"],
            int(paired["n"]),
        )
        logger.info(
            "Feb holdout: feast=%.4f control=%.4f paired mean=%.4f se=%.4f",
            holdout["feb_holdout_feast_mae"],
            holdout["feb_holdout_control_mae"],
            holdout["feb_holdout_paired_mean"],
            holdout["feb_holdout_paired_se"],
        )

        # C: week 1's @champion, re-scored on these exact test rows.
        champion_mae = score_champion(X_test, y_test)
        mlflow.log_metric("champion_mae_on_sample", champion_mae)

        model_info = mlflow.lightgbm.log_model(lgb_model=model, name="model")
        register.fn(
            model_info.model_uri, metrics
        )  # auto_promote_first is false: no alias yet

        client = MlflowClient()
        name = cfg["register"]["model_name"]
        promote = decide_promotion(
            metrics["mae"], champion_mae, cfg["register"]["relative_margin"]
        )
        if promote:
            client.set_registered_model_alias(
                name, CHAMPION_ALIAS, _latest_version(client, name)
            )
        mlflow.set_tag("promoted", str(promote))
        logger.info(
            "MAE feast=%.4f control=%.4f champion(on sample)=%.4f promoted=%s",
            metrics["mae"],
            control["mae"],
            champion_mae,
            promote,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
