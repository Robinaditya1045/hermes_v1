from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import polars as pl

US_EASTERN = ZoneInfo("America/New_York")
SCHEMA_VERSION = "market_bars.v1"


class NormalizationError(ValueError):
    """Raised when a provider response cannot be normalized safely."""


def normalize_daily_bars(raw: pl.DataFrame, *, retrieved_at: datetime) -> pl.DataFrame:
    if not isinstance(raw, pl.DataFrame):
        raise NormalizationError("Provider response must be a Polars DataFrame")

    columns = {column.lower().replace(" ", "_"): column for column in raw.columns}
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(columns)
    if missing:
        raise NormalizationError(f"Provider response is missing columns: {sorted(missing)}")

    date_expression = (
        pl.col("date").str.to_date()
        if raw.schema[columns["date"]] == pl.String
        else pl.col("date").cast(pl.Date)
    )
    normalized = raw.rename({columns[name]: name for name in required}).select(
        "symbol", "date", "open", "high", "low", "close", "volume"
    )
    try:
        normalized = normalized.with_columns(
            pl.col("symbol").cast(pl.String).str.to_uppercase(),
            date_expression.alias("bar_date"),
            *[
                pl.col(name).cast(pl.Float64, strict=True)
                for name in ("open", "high", "low", "close", "volume")
            ],
        ).drop("date")
    except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as exc:
        raise NormalizationError("Provider response contains malformed market data") from exc

    normalized = normalized.with_columns(
        pl.struct("bar_date")
        .map_elements(
            lambda row: datetime.combine(row["bar_date"], time(16), tzinfo=US_EASTERN).astimezone(
                ZoneInfo("UTC")
            ),
            return_dtype=pl.Datetime(time_zone="UTC"),
        )
        .alias("information_available_at")
    )
    return normalized.select(
        "symbol",
        "bar_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "information_available_at",
    ).sort(["symbol", "bar_date"])
