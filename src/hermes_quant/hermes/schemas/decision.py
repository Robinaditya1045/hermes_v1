from __future__ import annotations

from typing import Literal

from pydantic import Field

from hermes_quant.hermes.schemas.market_state import StrictModel


class HermesDecision(StrictModel):
    asset: str = Field(min_length=1)
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: list[str] = Field(min_length=1)
    risks: list[str] = Field(min_length=1)
    evidence_used: list[str] = Field(min_length=1)
    requested_notional: float = Field(ge=0.0)
