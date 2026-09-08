from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hermes_quant.hermes.schemas.market_state import (
    FundamentalsState,
    MarketRegime,
    MarketState,
    NewsState,
    QuantPrediction,
    VolatilityState,
)


class ToolCallLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    request: dict[str, Any]
    response: Any = None
    latency_ms: float = Field(ge=0.0)
    status: str


@dataclass
class ReadOnlyToolServer:
    market_states: dict[str, MarketState] = field(default_factory=dict)
    quant_predictions: dict[str, QuantPrediction] = field(default_factory=dict)
    market_regimes: dict[str, MarketRegime] = field(default_factory=dict)
    risk_assessments: dict[str, VolatilityState] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    trade_history: list[dict[str, Any]] = field(default_factory=list)
    logs: list[ToolCallLog] = field(default_factory=list)

    READ_ONLY_TOOLS = (
        "get_market_state",
        "get_quant_prediction",
        "get_fundamentals",
        "get_recent_news",
        "get_market_regime",
        "get_risk_assessment",
        "get_portfolio",
        "get_trade_history",
    )

    def _call(self, tool: str, request: dict[str, Any], function: Callable[[], Any]) -> Any:
        started = monotonic()
        try:
            response = function()
            response_snapshot = (
                response.model_dump(mode="json") if isinstance(response, BaseModel) else response
            )
            self.logs.append(
                ToolCallLog(
                    tool=tool,
                    request=request,
                    response=copy.deepcopy(response_snapshot),
                    latency_ms=(monotonic() - started) * 1000,
                    status="success",
                )
            )
            return response
        except Exception as exc:
            self.logs.append(
                ToolCallLog(
                    tool=tool,
                    request=request,
                    response={"error": str(exc)},
                    latency_ms=(monotonic() - started) * 1000,
                    status="error",
                )
            )
            raise

    def get_market_state(self, symbol: str) -> MarketState:
        return self._call(
            "get_market_state", {"symbol": symbol}, lambda: self.market_states[symbol]
        )

    def get_quant_prediction(self, symbol: str) -> QuantPrediction:
        return self._call(
            "get_quant_prediction",
            {"symbol": symbol},
            lambda: self.quant_predictions[symbol],
        )

    def get_fundamentals(self, symbol: str) -> FundamentalsState:
        return self._call(
            "get_fundamentals",
            {"symbol": symbol},
            lambda: FundamentalsState(available=False, metrics={}),
        )

    def get_recent_news(self, symbol: str) -> NewsState:
        return self._call(
            "get_recent_news",
            {"symbol": symbol},
            lambda: NewsState(available=False, sentiment=None, impact="unknown", items=0),
        )

    def get_market_regime(self, symbol: str) -> MarketRegime:
        return self._call(
            "get_market_regime", {"symbol": symbol}, lambda: self.market_regimes[symbol]
        )

    def get_risk_assessment(self, symbol: str) -> VolatilityState:
        return self._call(
            "get_risk_assessment",
            {"symbol": symbol},
            lambda: self.risk_assessments[symbol],
        )

    def get_portfolio(self) -> dict[str, Any]:
        return self._call("get_portfolio", {}, lambda: copy.deepcopy(self.portfolio))

    def get_trade_history(self) -> list[dict[str, Any]]:
        return self._call("get_trade_history", {}, lambda: copy.deepcopy(self.trade_history))
