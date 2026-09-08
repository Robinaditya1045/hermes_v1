from datetime import datetime, timezone

import polars as pl
import pytest

from hermes_quant.data.validation import DataValidationError, validate_market_bars

DECISION_AT = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
OUTCOME_AT = datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc)


def valid_bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "bar_date": [datetime(2026, 9, 5), datetime(2026, 9, 5)],
            "open": [230.0, 500.0],
            "high": [235.0, 505.0],
            "low": [228.0, 498.0],
            "close": [233.0, 503.0],
            "volume": [1_000_000.0, 2_000_000.0],
            "information_available_at": [DECISION_AT, DECISION_AT],
        }
    )


def test_missing_timestamp_column_fails_closed() -> None:
    bars = valid_bars().drop("information_available_at")

    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_market_bars(bars, decision_at=DECISION_AT, outcome_at=OUTCOME_AT)


def test_null_timestamp_fails_closed() -> None:
    bars = valid_bars().with_columns(
        pl.when(pl.col("symbol") == "AAPL")
        .then(None)
        .otherwise(pl.col("information_available_at"))
        .alias("information_available_at")
    )

    with pytest.raises(DataValidationError, match="null values"):
        validate_market_bars(bars, decision_at=DECISION_AT, outcome_at=OUTCOME_AT)


def test_duplicate_symbol_and_date_fails_closed() -> None:
    bars = valid_bars().with_columns(pl.lit("AAPL").alias("symbol"))

    with pytest.raises(DataValidationError, match="Duplicate market bars"):
        validate_market_bars(bars, decision_at=DECISION_AT, outcome_at=OUTCOME_AT)


def test_information_available_at_decision_boundary_is_valid() -> None:
    validate_market_bars(valid_bars(), decision_at=DECISION_AT, outcome_at=OUTCOME_AT)


def test_decision_at_outcome_at_boundary_fails_closed() -> None:
    with pytest.raises(DataValidationError, match="strictly before"):
        validate_market_bars(valid_bars(), decision_at=DECISION_AT, outcome_at=DECISION_AT)


def test_future_information_fails_closed_instead_of_warning() -> None:
    bars = valid_bars().with_columns(
        pl.when(pl.col("symbol") == "AAPL")
        .then(datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc))
        .otherwise(pl.col("information_available_at"))
        .alias("information_available_at")
    )

    with pytest.raises(DataValidationError, match="Timestamp rule violated"):
        validate_market_bars(bars, decision_at=DECISION_AT, outcome_at=OUTCOME_AT)
