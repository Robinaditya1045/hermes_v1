from __future__ import annotations

import polars as pl

REGIME_MODEL_VERSION = "regime_rules_technical_v1"


def classify_regimes(technical_features: pl.DataFrame) -> pl.DataFrame:
    """Classify market context from technical features without forecasting prices."""
    required = {
        "symbol",
        "bar_date",
        "information_available_at",
        "feature_timestamp",
        "realized_volatility_20d",
        "volatility_percentile_252d",
        "return_20d",
        "ema_20d",
        "ema_50d",
    }
    missing = required - set(technical_features.columns)
    if missing:
        raise ValueError(f"Missing technical regime features: {sorted(missing)}")

    return (
        technical_features.with_columns(
            (pl.col("ema_20d") / pl.col("ema_50d") - 1.0).alias("trend_strength"),
        )
        .with_columns(
            pl.when(pl.col("volatility_percentile_252d") >= 0.8)
            .then(pl.lit("high_volatility"))
            .when((pl.col("return_20d") >= 0) & (pl.col("trend_strength") >= 0))
            .then(pl.lit("risk_on"))
            .when((pl.col("return_20d") < 0) & (pl.col("trend_strength") < 0))
            .then(pl.lit("risk_off"))
            .when(pl.col("trend_strength").abs() >= 0.02)
            .then(pl.lit("trending"))
            .otherwise(pl.lit("sideways"))
            .alias("regime"),
            pl.when(pl.col("volatility_percentile_252d").is_null())
            .then(0.0)
            .otherwise(
                pl.max_horizontal(
                    pl.col("trend_strength").abs() / 0.05,
                    pl.col("volatility_percentile_252d").fill_null(0.0),
                ).clip(0.0, 1.0)
            )
            .alias("regime_confidence"),
            pl.lit(REGIME_MODEL_VERSION).alias("regime_model_version"),
        )
        .select(
            [
                "symbol",
                "bar_date",
                "information_available_at",
                "feature_timestamp",
                "regime",
                "regime_confidence",
                "regime_model_version",
            ]
        )
    )
