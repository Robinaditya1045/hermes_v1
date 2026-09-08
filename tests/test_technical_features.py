from datetime import datetime, timedelta, timezone

import polars as pl

from hermes_quant.features.technical import (
    TECHNICAL_FEATURE_LOOKBACKS,
    calculate_technical_features,
)

DECISION_AT = datetime(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
OUTCOME_AT = datetime(2027, 1, 1, 20, 0, tzinfo=timezone.utc)


def market_bars(row_count: int = 260) -> pl.DataFrame:
    dates = [datetime(2025, 1, 1) + timedelta(days=index) for index in range(row_count)]
    closes = [100.0 + index * 0.1 for index in range(row_count)]
    return pl.DataFrame(
        {
            "symbol": ["AAPL"] * row_count,
            "bar_date": dates,
            "open": [close - 0.5 for close in closes],
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
            "close": closes,
            "volume": [1_000_000.0 + index * 1000 for index in range(row_count)],
            "information_available_at": [
                datetime(2025, 1, 1, 21, 0, tzinfo=timezone.utc) + timedelta(days=index)
                for index in range(row_count)
            ],
        }
    )


def test_technical_features_include_timestamps_and_lookback_outputs() -> None:
    features = calculate_technical_features(
        market_bars(), decision_at=DECISION_AT, outcome_at=OUTCOME_AT
    )

    assert set(TECHNICAL_FEATURE_LOOKBACKS).issubset(features.columns)
    assert features.get_column("feature_timestamp").equals(
        features.get_column("information_available_at")
    )
    assert features.get_column("feature_set_version").unique().to_list() == ["technical.v1"]
    assert features.get_column("sma_20d").drop_nulls().len() > 0
    assert features.get_column("atr_14d").drop_nulls().len() > 0
    assert features.get_column("volume_z_score_20d").drop_nulls().len() > 0
