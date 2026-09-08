from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from hermes_quant.trading.orders import PaperOrder


@dataclass(frozen=True)
class FillModelConfig:
    fill_model_version: str = "constant_fill_v1"
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    fees_per_share: float = 0.005
    fees_bps: float = 0.0


@dataclass(frozen=True)
class Fill:
    fill_id: UUID
    order_id: UUID
    quantity: float
    price: float
    fees: float
    slippage: float
    fill_model_version: str
    fill_model_config: dict[str, float | str]
    occurred_at: datetime


@dataclass(frozen=True)
class FillMarketData:
    price: float
    requested_quantity: float | None = None


class FillModel(Protocol):
    def simulate_fill(self, order: PaperOrder, market_data: FillMarketData) -> Fill: ...


@dataclass(frozen=True)
class ConstantFillModel:
    config: FillModelConfig

    def simulate_fill(self, order: PaperOrder, market_data: FillMarketData) -> Fill:
        quantity = min(market_data.requested_quantity or order.quantity, order.quantity)
        direction = 1.0 if order.action == "BUY" else -1.0
        spread = market_data.price * self.config.spread_bps / 20_000.0
        slippage = market_data.price * self.config.slippage_bps / 10_000.0
        price = market_data.price + direction * (spread + slippage)
        notional = price * quantity
        fees = quantity * self.config.fees_per_share + notional * self.config.fees_bps / 10_000.0
        return Fill(
            fill_id=uuid4(),
            order_id=order.order_id,
            quantity=quantity,
            price=price,
            fees=fees,
            slippage=slippage,
            fill_model_version=self.config.fill_model_version,
            fill_model_config=asdict(self.config),
            occurred_at=datetime.now(timezone.utc),
        )
