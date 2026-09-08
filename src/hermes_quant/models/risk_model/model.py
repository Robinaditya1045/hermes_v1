from __future__ import annotations

from datetime import datetime
from math import sqrt

import polars as pl

from hermes_quant.data.validation import validate_market_bars

RISK_MODEL_VERSION = "risk_state_technical_v1"


def calculate_risk_state(
    market_bars: pl.DataFrame,
    *,
    decision_at: datetime,
    outcome_at: datetime,
    volatility_window: int = 20,
    downside_window: int = 20,
) -> pl.DataFrame:
    """Calculate observed volatility, downside volatility, and drawdown state."""
    validate_market_bars(market_bars, decision_at=decision_at, outcome_at=outcome_at)
    if min(volatility_window, downside_window) < 2:
        raise ValueError("Risk windows must be at least two observations")

    bars = market_bars.sort(["symbol", "bar_date"])
    returns = pl.col("close").pct_change().over("symbol")
    return (
        bars.with_columns(
            returns.alias("daily_return"),
            pl.col("close").cum_max().over("symbol").alias("running_peak"),
        )
        .with_columns(
            pl.col("daily_return")
            .rolling_std(window_size=volatility_window, min_samples=volatility_window)
            .over("symbol")
            .mul(sqrt(252))
            .alias("realized_volatility"),
            pl.when(pl.col("daily_return") < 0)
            .then(pl.col("daily_return"))
            .otherwise(0.0)
            .rolling_std(window_size=downside_window, min_samples=downside_window)
            .over("symbol")
            .mul(sqrt(252))
            .alias("downside_volatility"),
            (pl.col("close") / pl.col("running_peak") - 1.0).alias("drawdown"),
        )
        .with_columns(
            pl.col("drawdown").cum_min().over("symbol").alias("max_drawdown"),
            pl.lit(RISK_MODEL_VERSION).alias("risk_model_version"),
        )
        .select(
            "symbol",
            "bar_date",
            "information_available_at",
            "realized_volatility",
            "downside_volatility",
            "drawdown",
            "max_drawdown",
            "risk_model_version",
        )
    )
