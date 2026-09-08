from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MARKET_STATE_SCHEMA_VERSION = "market_state.v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TechnicalState(StrictModel):
    feature_timestamp: datetime
    feature_set_version: str
    features: dict[str, float | None]


class QuantPrediction(StrictModel):
    probability_up: float = Field(ge=0.0, le=1.0)
    expected_return: float
    model_version: str


class VolatilityState(StrictModel):
    realized_20d: float | None
    downside_volatility: float | None
    max_drawdown: float | None
    risk_model_version: str


class MarketRegime(StrictModel):
    state: Literal["risk_on", "risk_off", "high_volatility", "trending", "sideways"]
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str


class FundamentalsState(StrictModel):
    available: bool
    metrics: dict[str, float | None]


class NewsState(StrictModel):
    available: bool
    sentiment: float | None = Field(default=None, ge=-1.0, le=1.0)
    impact: Literal["low", "medium", "high", "unknown"]
    items: int = Field(ge=0)


class MarketState(StrictModel):
    schema_version: Literal[MARKET_STATE_SCHEMA_VERSION] = MARKET_STATE_SCHEMA_VERSION
    asset: str = Field(min_length=1)
    decision_time: datetime
    horizon: Literal["1d"]
    technical: TechnicalState
    market_prediction: QuantPrediction
    volatility: VolatilityState
    market_regime: MarketRegime
    fundamentals: FundamentalsState
    news: NewsState


def build_market_state(
    *,
    asset: str,
    decision_time: datetime,
    technical: dict,
    market_prediction: dict,
    volatility: dict,
    market_regime: dict,
    fundamentals: dict,
    news: dict,
) -> MarketState:
    """Assemble and validate the complete state contract before Hermes sees it."""
    return MarketState.model_validate(
        {
            "schema_version": MARKET_STATE_SCHEMA_VERSION,
            "asset": asset,
            "decision_time": decision_time,
            "horizon": "1d",
            "technical": technical,
            "market_prediction": market_prediction,
            "volatility": volatility,
            "market_regime": market_regime,
            "fundamentals": fundamentals,
            "news": news,
        }
    )
