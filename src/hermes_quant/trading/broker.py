from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from hermes_quant.risk.engine import RiskDecision
from hermes_quant.trading.fills import Fill, FillMarketData, FillModel
from hermes_quant.trading.orders import PaperOrder, create_order_from_risk_decision


class TradeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    action: str
    requested_quantity: float = Field(gt=0)
    requested_notional: float = Field(gt=0)


@dataclass(frozen=True)
class OrderTransition:
    order_id: Any
    from_status: str | None
    to_status: str
    occurred_at: datetime


@dataclass(frozen=True)
class BrokerResult:
    status: str
    risk_decision: RiskDecision
    order: PaperOrder | None = None
    fill: Fill | None = None


class InMemoryOrderStore:
    def __init__(self) -> None:
        self.orders: dict[str, PaperOrder] = {}
        self.transitions: list[OrderTransition] = []
        self.fills: list[Fill] = []

    def get_by_decision_id(self, decision_id: str) -> BrokerResult | None:
        order = self.orders.get(decision_id)
        if order is None:
            return None
        fill = next((item for item in self.fills if item.order_id == order.order_id), None)
        return BrokerResult(
            "ALREADY_PROCESSED",
            RiskDecision(
                True,
                order.quantity,
                order.notional,
                "approved",
                decision_id,
                None,
                order.symbol,
                order.action,
            ),
            order,
            fill,
        )

    def save_order(self, order: PaperOrder) -> None:
        if order.decision_id in self.orders:
            raise RuntimeError("unique decision_id constraint violated")
        self.orders[order.decision_id] = order

    def save_transition(self, transition: OrderTransition) -> None:
        self.transitions.append(transition)

    def save_fill(self, fill: Fill) -> None:
        self.fills.append(fill)


class SQLiteOrderStore(InMemoryOrderStore):
    """File-backed order identity store for restart-safe idempotency."""

    def __init__(self, database_path: str) -> None:
        super().__init__()
        self.connection = sqlite3.connect(database_path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS orders ("
            "order_id TEXT PRIMARY KEY, decision_id TEXT UNIQUE NOT NULL, "
            "symbol TEXT NOT NULL, action TEXT NOT NULL, quantity REAL NOT NULL, "
            "notional REAL NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS order_transitions ("
            "order_id TEXT NOT NULL, from_status TEXT, to_status TEXT NOT NULL, "
            "occurred_at TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS fills ("
            "fill_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, quantity REAL NOT NULL, "
            "price REAL NOT NULL, fees REAL NOT NULL, slippage REAL NOT NULL, "
            "fill_model_version TEXT NOT NULL, fill_model_config TEXT NOT NULL, "
            "occurred_at TEXT NOT NULL)"
        )
        self.connection.commit()

    def get_by_decision_id(self, decision_id: str) -> BrokerResult | None:
        row = self.connection.execute(
            "SELECT order_id, symbol, action, quantity, notional, status, created_at "
            "FROM orders WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        if row is None:
            return None
        order = PaperOrder._from_persisted(
            order_id=UUID(row[0]),
            decision_id=decision_id,
            symbol=row[1],
            action=row[2],
            quantity=row[3],
            notional=row[4],
            status=row[5],
            created_at=datetime.fromisoformat(row[6]),
        )
        fill_row = self.connection.execute(
            "SELECT fill_id, quantity, price, fees, slippage, fill_model_version, "
            "fill_model_config, occurred_at FROM fills WHERE order_id = ?",
            (str(order.order_id),),
        ).fetchone()
        fill = None
        if fill_row is not None:
            fill = Fill(
                fill_id=UUID(fill_row[0]),
                order_id=order.order_id,
                quantity=fill_row[1],
                price=fill_row[2],
                fees=fill_row[3],
                slippage=fill_row[4],
                fill_model_version=fill_row[5],
                fill_model_config=json.loads(fill_row[6]),
                occurred_at=datetime.fromisoformat(fill_row[7]),
            )
        return BrokerResult(
            "ALREADY_PROCESSED",
            RiskDecision(
                True,
                order.quantity,
                order.notional,
                "approved",
                decision_id,
                None,
                order.symbol,
                order.action,
            ),
            order,
            fill,
        )

    def save_order(self, order: PaperOrder) -> None:
        try:
            self.connection.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(order.order_id),
                    order.decision_id,
                    order.symbol,
                    order.action,
                    order.quantity,
                    order.notional,
                    order.status,
                    order.created_at.isoformat(),
                    order.created_at.isoformat(),
                ),
            )
            self.connection.commit()
        except sqlite3.IntegrityError as exc:
            raise RuntimeError("unique decision_id constraint violated") from exc

    def save_transition(self, transition: OrderTransition) -> None:
        self.connection.execute(
            "INSERT INTO order_transitions VALUES (?, ?, ?, ?)",
            (
                str(transition.order_id),
                transition.from_status,
                transition.to_status,
                transition.occurred_at.isoformat(),
            ),
        )
        self.connection.execute(
            "UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?",
            (transition.to_status, transition.occurred_at.isoformat(), str(transition.order_id)),
        )
        self.connection.commit()

    def save_fill(self, fill: Fill) -> None:
        self.connection.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(fill.fill_id),
                str(fill.order_id),
                fill.quantity,
                fill.price,
                fill.fees,
                fill.slippage,
                fill.fill_model_version,
                json.dumps(fill.fill_model_config),
                fill.occurred_at.isoformat(),
            ),
        )
        self.connection.commit()


@dataclass
class PaperBroker:
    store: InMemoryOrderStore = field(default_factory=InMemoryOrderStore)
    fill_model: FillModel | None = None

    def submit(
        self,
        proposal: TradeProposal,
        risk_decision: RiskDecision,
        *,
        market_data: float,
        fill_quantity: float | None = None,
    ) -> BrokerResult:
        existing = self.store.get_by_decision_id(proposal.decision_id)
        if existing is not None:
            return existing
        for identity_field, decision_value, proposal_value in (
            ("decision_id", risk_decision.decision_id, proposal.decision_id),
            ("symbol", risk_decision.symbol, proposal.symbol),
            ("action", risk_decision.action, proposal.action),
        ):
            if identity_field == "decision_id" and decision_value is None:
                raise ValueError("RiskDecision decision_id is required")
            if decision_value is not None and decision_value != proposal_value:
                raise ValueError(f"RiskDecision {identity_field} does not match TradeProposal")
        risk_decision = replace(
            risk_decision,
            decision_id=risk_decision.decision_id,
            symbol=risk_decision.symbol or proposal.symbol,
            action=risk_decision.action or proposal.action,
        )
        if not risk_decision.approved:
            return BrokerResult("RISK_REJECTED", risk_decision)
        order = create_order_from_risk_decision(risk_decision)
        try:
            self.store.save_order(order)
        except RuntimeError:
            existing = self.store.get_by_decision_id(proposal.decision_id)
            if existing is not None:
                return existing
            raise
        self.store.save_transition(
            OrderTransition(order.order_id, None, "CREATED", datetime.now(timezone.utc))
        )
        self._transition(order, "VALIDATING")
        self._transition(order, "RISK_APPROVED")
        self._transition(order, "SUBMITTED")
        if self.fill_model is None:
            raise RuntimeError("fill_model is required")
        fill = self.fill_model.simulate_fill(
            order,
            FillMarketData(price=market_data, requested_quantity=fill_quantity),
        )
        self.store.save_fill(fill)
        if fill.quantity < order.quantity:
            self._transition(order, "PARTIALLY_FILLED")
        else:
            self._transition(order, "FILLED")
            self._transition(order, "CLOSED")
        return BrokerResult(order.status, risk_decision, order, fill)

    def _transition(self, order: PaperOrder, status: str) -> None:
        previous = order.status
        object.__setattr__(order, "status", status)
        self.store.save_transition(
            OrderTransition(order.order_id, previous, status, datetime.now(timezone.utc))
        )
