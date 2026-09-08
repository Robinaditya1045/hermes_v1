from __future__ import annotations

from datetime import datetime
from math import sqrt

import polars as pl

from hermes_quant.data.validation import validate_market_bars

TECHNICAL_FEATURE_VERSION = "technical.v1"
TECHNICAL_FEATURE_LOOKBACKS = {
    "return_1d": 1,
    "return_5d": 5,
    "return_20d": 20,
    "return_60d": 60,
    "sma_20d": 20,
    "sma_50d": 50,
    "sma_200d": 200,
    "ema_20d": 20,
    "ema_50d": 50,
    "ema_200d": 200,
    "atr_14d": 14,
    "realized_volatility_20d": 20,
    "volatility_percentile_252d": 252,
    "volume_z_score_20d": 20,
}


def calculate_technical_features(
    market_bars: pl.DataFrame,
    *,
    decision_at: datetime,
    outcome_at: datetime,
) -> pl.DataFrame:
    """Calculate leakage-safe technical features from validated daily market bars.

    Each output row carries the source ``information_available_at`` and an identical
    ``feature_timestamp``. Rolling features emit nulls until their documented lookback
    is available; callers must not silently fill those warmup values.
    """
    validate_market_bars(market_bars, decision_at=decision_at, outcome_at=outcome_at)

    bars = market_bars.sort(["symbol", "bar_date"])
    previous_close = pl.col("close").shift(1).over("symbol")
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )

    features = (
        bars.with_columns(
            *[
                pl.col("close").pct_change(window).over("symbol").alias(f"return_{window}d")
                for window in (1, 5, 20, 60)
            ],
            *[
                pl.col("close")
                .rolling_mean(window_size=window, min_samples=window)
                .over("symbol")
                .alias(f"sma_{window}d")
                for window in (20, 50, 200)
            ],
            *[
                pl.col("close")
                .ewm_mean(span=window, adjust=False, min_samples=window)
                .over("symbol")
                .alias(f"ema_{window}d")
                for window in (20, 50, 200)
            ],
            true_range.alias("true_range_1d"),
        )
        .with_columns(
            pl.col("true_range_1d")
            .rolling_mean(window_size=14, min_samples=14)
            .over("symbol")
            .alias("atr_14d"),
            pl.col("close")
            .pct_change()
            .over("symbol")
            .rolling_std(window_size=20, min_samples=20)
            .over("symbol")
            .mul(sqrt(252))
            .alias("realized_volatility_20d"),
            pl.col("volume")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("volume_mean_20d"),
            pl.col("volume")
            .rolling_std(window_size=20, min_samples=20)
            .over("symbol")
            .alias("volume_std_20d"),
        )
        .with_columns(
            pl.when(pl.col("volume_std_20d") > 0)
            .then((pl.col("volume") - pl.col("volume_mean_20d")) / pl.col("volume_std_20d"))
            .otherwise(0.0)
            .alias("volume_z_score_20d"),
            (
                pl.col("realized_volatility_20d")
                .rolling_rank(window_size=252, min_samples=20)
                .over("symbol")
                / pl.col("realized_volatility_20d")
                .is_not_null()
                .cast(pl.UInt32)
                .rolling_sum(window_size=252, min_samples=20)
                .over("symbol")
            ).alias("volatility_percentile_252d"),
        )
        .with_columns(
            pl.lit(TECHNICAL_FEATURE_VERSION).alias("feature_set_version"),
            pl.col("information_available_at").alias("feature_timestamp"),
        )
    )

    return features.drop(["true_range_1d", "volume_mean_20d", "volume_std_20d"]).select(
        [
            "symbol",
            "bar_date",
            "information_available_at",
            "feature_timestamp",
            "feature_set_version",
            *TECHNICAL_FEATURE_LOOKBACKS,
        ]
    )
