from datetime import datetime, timezone

import polars as pl
import pytest

from hermes_quant.data.normalization import normalize_daily_bars
from hermes_quant.data.validation import DataValidationError, validate_market_bars

DECISION_AT = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
OUTCOME_AT = datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc)


def raw_bar(date: str = "2026-09-07") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [date],
            "open": [230.0],
            "high": [235.0],
            "low": [228.0],
            "close": [233.0],
            "volume": [1_000_000.0],
        }
    )


def test_information_must_be_available_by_decision_time() -> None:
    bars = normalize_daily_bars(raw_bar(), retrieved_at=DECISION_AT)

    validate_market_bars(bars, decision_at=DECISION_AT, outcome_at=OUTCOME_AT)


def test_future_information_is_rejected() -> None:
    bars = normalize_daily_bars(raw_bar("2026-09-08"), retrieved_at=DECISION_AT)

    with pytest.raises(DataValidationError, match="Timestamp rule violated"):
        validate_market_bars(
            bars,
            decision_at=datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc),
            outcome_at=OUTCOME_AT,
        )


def test_outcome_must_be_strictly_after_decision() -> None:
    bars = normalize_daily_bars(raw_bar(), retrieved_at=DECISION_AT)

    with pytest.raises(DataValidationError, match="strictly before"):
        validate_market_bars(bars, decision_at=DECISION_AT, outcome_at=DECISION_AT)
