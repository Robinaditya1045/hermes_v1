from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import lightgbm as lgb
import pandas as pd
import polars as pl

from hermes_quant.features.technical import TECHNICAL_FEATURE_LOOKBACKS

MODEL_VERSION = "return_lgbm_technical_v1"
FEATURE_COLUMNS = tuple(TECHNICAL_FEATURE_LOOKBACKS)


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: tuple[date | datetime, ...]
    validation_dates: tuple[date | datetime, ...]
    test_dates: tuple[date | datetime, ...]

    @property
    def train_end(self):
        return self.train_dates[-1]

    @property
    def validation_start(self):
        return self.validation_dates[0]

    @property
    def validation_end(self):
        return self.validation_dates[-1]

    @property
    def test_start(self):
        return self.test_dates[0]


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pl.DataFrame
    folds: tuple[WalkForwardFold, ...]
    metrics: dict[str, float]


def build_labeled_dataset(
    technical_features: pl.DataFrame,
    market_bars: pl.DataFrame,
) -> pl.DataFrame:
    """Attach next-observation direction and return labels to technical features."""
    missing = set(FEATURE_COLUMNS) - set(technical_features.columns)
    if missing:
        raise ValueError(f"Missing technical features: {sorted(missing)}")
    if not {"symbol", "bar_date", "close"}.issubset(market_bars.columns):
        raise ValueError("Market bars must include symbol, bar_date, and close")

    labels = (
        market_bars.sort(["symbol", "bar_date"])
        .with_columns(
            pl.col("close").shift(-1).over("symbol").alias("next_close"),
        )
        .with_columns(
            ((pl.col("next_close") / pl.col("close")) - 1.0).alias("next_return"),
        )
        .select("symbol", "bar_date", "next_return")
        .with_columns((pl.col("next_return") > 0).cast(pl.Int8).alias("direction_up"))
    )
    return (
        technical_features.join(labels, on=["symbol", "bar_date"], how="inner")
        .filter(pl.col("next_return").is_not_null())
        .drop_nulls(subset=list(FEATURE_COLUMNS))
        .sort(["bar_date", "symbol"])
    )


def build_walk_forward_splits(
    dataset: pl.DataFrame,
    *,
    train_periods: int,
    validation_periods: int,
    test_periods: int,
    step_periods: int | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding, date-disjoint train/validation/test windows."""
    if min(train_periods, validation_periods, test_periods) < 1:
        raise ValueError("All walk-forward periods must be positive")

    dates = dataset.get_column("bar_date").unique().sort().to_list()
    step = step_periods or test_periods
    total_periods = train_periods + validation_periods + test_periods
    folds: list[WalkForwardFold] = []
    for start in range(0, len(dates) - total_periods + 1, step):
        train_end = start + train_periods
        validation_end = train_end + validation_periods
        test_end = validation_end + test_periods
        folds.append(
            WalkForwardFold(
                fold_id=len(folds),
                train_dates=tuple(dates[:train_end]),
                validation_dates=tuple(dates[train_end:validation_end]),
                test_dates=tuple(dates[validation_end:test_end]),
            )
        )
    if not folds:
        raise ValueError("Dataset does not contain enough dates for one walk-forward fold")
    return tuple(folds)


def _date_filter(frame: pl.DataFrame, dates: tuple[date | datetime, ...]) -> pl.DataFrame:
    return frame.filter(pl.col("bar_date").is_in(dates))


def _strategy_metrics(predictions: pl.DataFrame) -> dict[str, float]:
    daily = (
        predictions.group_by("decision_time")
        .agg(
            pl.col("next_return").mean().alias("buy_hold_return"),
            pl.when(pl.col("probability_up") >= 0.5)
            .then(pl.col("next_return"))
            .otherwise(0.0)
            .mean()
            .alias("model_return"),
        )
        .sort("decision_time")
    )
    buy_hold = (1.0 + daily.get_column("buy_hold_return")).product() - 1.0
    model_return = (1.0 + daily.get_column("model_return")).product() - 1.0
    return {
        "model_cumulative_return": float(model_return),
        "buy_and_hold_cumulative_return": float(buy_hold),
    }


def fit_walk_forward(
    dataset: pl.DataFrame,
    *,
    train_periods: int,
    validation_periods: int,
    test_periods: int,
    step_periods: int | None = None,
) -> WalkForwardResult:
    """Fit LightGBM only on each fold's past data and predict its future test data."""
    folds = build_walk_forward_splits(
        dataset,
        train_periods=train_periods,
        validation_periods=validation_periods,
        test_periods=test_periods,
        step_periods=step_periods,
    )
    predictions: list[pl.DataFrame] = []
    for fold in folds:
        train = _date_filter(dataset, fold.train_dates)
        validation = _date_filter(dataset, fold.validation_dates)
        test = _date_filter(dataset, fold.test_dates)
        train_x = pd.DataFrame(train.select(FEATURE_COLUMNS).to_numpy(), columns=FEATURE_COLUMNS)
        validation_x = pd.DataFrame(
            validation.select(FEATURE_COLUMNS).to_numpy(), columns=FEATURE_COLUMNS
        )
        test_x = pd.DataFrame(test.select(FEATURE_COLUMNS).to_numpy(), columns=FEATURE_COLUMNS)
        classifier = lgb.LGBMClassifier(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            random_state=fold.fold_id,
            verbosity=-1,
        )
        regressor = lgb.LGBMRegressor(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            random_state=fold.fold_id,
            verbosity=-1,
        )
        classifier.fit(
            train_x,
            train["direction_up"].to_numpy(),
            eval_set=[(validation_x, validation["direction_up"].to_numpy())],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        regressor.fit(
            train_x,
            train["next_return"].to_numpy(),
            eval_set=[(validation_x, validation["next_return"].to_numpy())],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        probability_up = classifier.predict_proba(test_x)[:, 1]
        predictions.append(
            test.select(
                pl.col("symbol").alias("asset"),
                pl.col("bar_date").alias("decision_time"),
                pl.lit("1d").alias("horizon"),
                pl.col("next_return"),
            ).with_columns(
                pl.Series("probability_up", probability_up),
                pl.Series(
                    "expected_return",
                    regressor.predict(test_x),
                ),
                pl.lit(MODEL_VERSION).alias("model_version"),
                pl.lit(fold.fold_id).alias("fold_id"),
            )
        )
    output = pl.concat(predictions).sort(["decision_time", "asset"])
    return WalkForwardResult(
        predictions=output,
        folds=folds,
        metrics=_strategy_metrics(output),
    )
