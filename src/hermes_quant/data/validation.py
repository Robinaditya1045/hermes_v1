from __future__ import annotations

from datetime import datetime

import polars as pl

REQUIRED_COLUMNS = {
    "symbol",
    "bar_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "information_available_at",
}


class DataValidationError(ValueError):
    """Raised when data cannot be trusted for research or execution."""


def validate_market_bars(
    bars: pl.DataFrame,
    *,
    decision_at: datetime,
    outcome_at: datetime,
) -> None:
    if decision_at >= outcome_at:
        raise DataValidationError("decision_at must be strictly before outcome_at")

    missing = REQUIRED_COLUMNS - set(bars.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")
    if bars.is_empty():
        raise DataValidationError("No market bars were returned")
    if bars.select(["symbol", "bar_date"]).is_duplicated().any():
        raise DataValidationError("Duplicate market bars for symbol and bar_date")

    if bars.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise DataValidationError("Market bars contain null values")
    if bars.select((pl.col("high") < pl.max_horizontal("open", "close")).any()).item():
        raise DataValidationError("A bar high is below its open or close")
    if bars.select((pl.col("low") > pl.min_horizontal("open", "close")).any()).item():
        raise DataValidationError("A bar low is above its open or close")
    if bars.select((pl.col("volume") < 0).any()).item():
        raise DataValidationError("A bar has negative volume")
    if bars.select(pl.col("symbol").n_unique() == 0).item():
        raise DataValidationError("No symbols were returned")

    leaking = bars.filter(pl.col("information_available_at") > decision_at)
    if not leaking.is_empty():
        raise DataValidationError(
            "Timestamp rule violated: information_available_at must be <= decision_at"
        )
