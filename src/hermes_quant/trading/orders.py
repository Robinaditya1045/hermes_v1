from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from hermes_quant.risk.engine import RiskDecision


class OrderCreationError(ValueError):
    pass


@dataclass(frozen=True)
class PaperOrder:
    order_id: UUID
    decision_id: str
    symbol: str
    action: str
    quantity: float
    notional: float
    status: str
    created_at: datetime

    def __init__(self, *, _factory_token: object, risk_decision: RiskDecision):
        if _factory_token is not _ORDER_FACTORY_TOKEN:
            raise TypeError("PaperOrder must be created from an approved RiskDecision")
        if not risk_decision.approved:
            raise OrderCreationError("Cannot create an order from a rejected RiskDecision")
        if not risk_decision.decision_id or not risk_decision.symbol or not risk_decision.action:
            raise OrderCreationError("Approved RiskDecision requires trade identity")
        object.__setattr__(self, "order_id", uuid4())
        object.__setattr__(self, "decision_id", risk_decision.decision_id)
        object.__setattr__(self, "symbol", risk_decision.symbol)
        object.__setattr__(self, "action", risk_decision.action)
        object.__setattr__(self, "quantity", risk_decision.approved_quantity)
        object.__setattr__(self, "notional", risk_decision.approved_notional)
        object.__setattr__(self, "status", "CREATED")
        object.__setattr__(self, "created_at", datetime.now(timezone.utc))

    @classmethod
    def _from_persisted(
        cls,
        *,
        order_id: UUID,
        decision_id: str,
        symbol: str,
        action: str,
        quantity: float,
        notional: float,
        status: str,
        created_at: datetime,
    ) -> PaperOrder:
        order = object.__new__(cls)
        for name, value in {
            "order_id": order_id,
            "decision_id": decision_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "notional": notional,
            "status": status,
            "created_at": created_at,
        }.items():
            object.__setattr__(order, name, value)
        return order


_ORDER_FACTORY_TOKEN = object()


def create_order_from_risk_decision(risk_decision: RiskDecision) -> PaperOrder:
    if not risk_decision.approved:
        raise OrderCreationError("Cannot create an order from a rejected RiskDecision")
    return PaperOrder(_factory_token=_ORDER_FACTORY_TOKEN, risk_decision=risk_decision)
