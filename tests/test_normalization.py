from datetime import datetime, timezone

import polars as pl
import pytest

from hermes_quant.data.normalization import NormalizationError, normalize_daily_bars

RETRIEVED_AT = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)


def provider_bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Symbol": ["aapl"],
            "Date": ["2026-09-07"],
            "Open": [230.0],
            "High": [235.0],
            "Low": [228.0],
            "Close": [233.0],
            "Volume": [1_000_000.0],
        }
    )


def test_provider_columns_are_normalized() -> None:
    normalized = normalize_daily_bars(provider_bars(), retrieved_at=RETRIEVED_AT)

    assert normalized.columns == [
        "symbol",
        "bar_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "information_available_at",
    ]
    assert normalized.item(0, "symbol") == "AAPL"


def test_missing_provider_timestamp_fails_closed() -> None:
    with pytest.raises(NormalizationError, match="missing columns"):
        normalize_daily_bars(provider_bars().drop("Date"), retrieved_at=RETRIEVED_AT)


def test_malformed_provider_response_fails_closed() -> None:
    malformed = provider_bars().with_columns(pl.lit("not-a-price").alias("Close"))

    with pytest.raises(NormalizationError, match="malformed market data"):
        normalize_daily_bars(malformed, retrieved_at=RETRIEVED_AT)


def test_non_dataframe_provider_response_fails_closed() -> None:
    with pytest.raises(NormalizationError, match="Polars DataFrame"):
        normalize_daily_bars({"symbol": "AAPL"}, retrieved_at=RETRIEVED_AT)
