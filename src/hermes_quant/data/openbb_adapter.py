from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from hermes_quant.data.models import IngestionBatch
from hermes_quant.data.normalization import normalize_daily_bars
from hermes_quant.data.validation import validate_market_bars

STARTER_UNIVERSE = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "AVGO",
    "BRK.B",
    "LLY",
    "JPM",
    "XOM",
    "UNH",
)


class OpenBBDataAdapter:
    endpoint = "obb.equity.price.historical"

    def __init__(self, *, provider: str | None = None, openbb_client=None) -> None:
        self.provider = provider
        self._obb = openbb_client

    def _client(self):
        if self._obb is None:
            from openbb import obb

            self._obb = obb
        return self._obb

    def fetch_daily_bars(
        self,
        *,
        start_date: str,
        end_date: str,
        symbols: tuple[str, ...] = STARTER_UNIVERSE,
        decision_at: datetime,
        outcome_at: datetime,
        retrieved_at: datetime,
        output_path: Path | None = None,
    ) -> tuple[pl.DataFrame, IngestionBatch]:
        parameters = {
            "symbol": ",".join(symbols),
            "start_date": start_date,
            "end_date": end_date,
            "interval": "1d",
        }
        if self.provider:
            parameters["provider"] = self.provider

        kwargs = {
            "symbol": parameters["symbol"],
            "start_date": start_date,
            "end_date": end_date,
            "interval": "1d",
        }
        if self.provider:
            kwargs["provider"] = self.provider
        response = self._client().equity.price.historical(**kwargs)
        raw = response.to_df() if hasattr(response, "to_df") else response
        if "date" not in raw.columns and hasattr(raw, "reset_index"):
            raw = raw.reset_index()
        if "symbol" not in raw.columns and len(symbols) == 1:
            raw["symbol"] = symbols[0]
        if not isinstance(raw, pl.DataFrame):
            raw = pl.from_pandas(raw)
        bars = normalize_daily_bars(raw, retrieved_at=retrieved_at)
        validate_market_bars(bars, decision_at=decision_at, outcome_at=outcome_at)

        batch = IngestionBatch.create(
            provider=self.provider or "openbb-default",
            endpoint=self.endpoint,
            request_parameters=parameters,
            retrieved_at=retrieved_at,
            schema_version="market_bars.v1",
            row_count=bars.height,
            source_metadata={"source": "OpenBB ODP", "interval": "1d"},
        )
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            bars.write_parquet(output_path)
            batch.write_json(output_path.with_suffix(".metadata.json"))
        return bars, batch
