from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from hermes_quant.experience.record import TradeExperience
from hermes_quant.experience.store import ExperienceStore


@dataclass(frozen=True)
class TradeAuditBundle:
    """Section 23.2 complete cryptographic audit bundle for a historical trade."""

    trade_id: str
    decision_id: str
    decision_time: datetime
    asset: str
    provenance: dict[str, Any]
    agent_decision: dict[str, Any]
    risk_decision: dict[str, Any]
    order: dict[str, Any] | None
    order_transitions: list[dict[str, Any]]
    fills: list[dict[str, Any]]
    outcome: dict[str, Any]
    reconstruction_hash: str
    reconstructed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision_time"] = self.decision_time.isoformat()
        d["reconstructed_at"] = self.reconstructed_at.isoformat()
        return d


class AuditService:
    """Reconstructs historical trades end-to-end without agent memory (Section 23.2)."""

    def __init__(
        self,
        experience_store: ExperienceStore,
        broker_store: Any = None,
        orchestrator_store: Any = None,
    ) -> None:
        self.experience_store = experience_store
        self.broker_store = broker_store
        self.orchestrator_store = orchestrator_store

    def reconstruct_trade(self, identifier: str) -> TradeAuditBundle | None:
        # Try lookup by trade_id, then decision_id
        exp: TradeExperience | None = self.experience_store.get(identifier)
        if exp is None:
            exp = self.experience_store.get_by_decision_id(identifier)
        if exp is None:
            return None

        decision_id = exp.decision_id

        # 1. Provenance
        provenance = {
            "market_state_version": exp.market_state_version,
            "feature_version": exp.feature_version,
            "model_versions": exp.model_versions,
            "hermes_model": exp.hermes_model,
            "hermes_prompt_version": exp.hermes_prompt_version,
        }

        # 2. Agent Decision
        agent_decision = {
            "asset": exp.asset,
            "action": exp.action,
            "confidence": exp.confidence,
        }

        # 3. Risk Decision
        risk_decision = {
            "risk_decision": exp.risk_decision,
            "approved_quantity": exp.approved_quantity,
        }

        # 4. Broker Order & Fills
        order_dict = None
        transitions_list: list[dict[str, Any]] = []
        fills_list: list[dict[str, Any]] = []

        if self.broker_store is not None:
            # Check InMemoryOrderStore or SQLiteOrderStore
            if hasattr(self.broker_store, "get_by_decision_id"):
                res = self.broker_store.get_by_decision_id(decision_id)
                if res and res.order:
                    o = res.order
                    order_dict = {
                        "order_id": str(o.order_id),
                        "decision_id": o.decision_id,
                        "symbol": o.symbol,
                        "action": o.action,
                        "quantity": o.quantity,
                        "notional": o.notional,
                        "status": o.status,
                        "created_at": o.created_at.isoformat(),
                    }
                if res and res.fill:
                    f = res.fill
                    fills_list.append(
                        {
                            "fill_id": str(f.fill_id),
                            "order_id": str(f.order_id),
                            "quantity": f.quantity,
                            "price": f.price,
                            "fees": f.fees,
                            "slippage": f.slippage,
                            "fill_model_version": f.fill_model_version,
                            "occurred_at": f.occurred_at.isoformat(),
                        }
                    )
            if hasattr(self.broker_store, "get_transitions_by_order_id") and order_dict:
                transitions_list.extend(
                    self.broker_store.get_transitions_by_order_id(order_dict.get("order_id"))
                )
            elif hasattr(self.broker_store, "transitions"):
                for t in self.broker_store.transitions:
                    if order_dict and str(t.order_id) == order_dict.get("order_id"):
                        transitions_list.append(
                            {
                                "order_id": str(t.order_id),
                                "from_status": t.from_status,
                                "to_status": t.to_status,
                                "occurred_at": t.occurred_at.isoformat(),
                            }
                        )

        # 5. Outcome
        outcome = {
            "entry_price": exp.entry_price,
            "exit_price": exp.exit_price,
            "fees": exp.fees,
            "slippage": exp.slippage,
            "realized_pnl": exp.realized_pnl,
            "net_return": exp.net_return,
            "holding_period_bars": exp.holding_period_bars,
            "outcome_label": exp.outcome_label,
            "regime_at_decision": exp.regime_at_decision,
            "post_trade_notes": exp.post_trade_notes,
            "recorded_at": exp.recorded_at.isoformat(),
        }

        # 6. Canonical Hash Digest
        hash_payload = {
            "trade_id": exp.trade_id,
            "decision_id": exp.decision_id,
            "decision_time": exp.decision_time.isoformat(),
            "asset": exp.asset,
            "provenance": provenance,
            "agent_decision": agent_decision,
            "risk_decision": risk_decision,
            "order": order_dict,
            "transitions": transitions_list,
            "fills": fills_list,
            "outcome": outcome,
        }
        reconstruction_hash = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        return TradeAuditBundle(
            trade_id=exp.trade_id,
            decision_id=exp.decision_id,
            decision_time=exp.decision_time,
            asset=exp.asset,
            provenance=provenance,
            agent_decision=agent_decision,
            risk_decision=risk_decision,
            order=order_dict,
            order_transitions=transitions_list,
            fills=fills_list,
            outcome=outcome,
            reconstruction_hash=reconstruction_hash,
        )

    def list_auditable_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        records = self.experience_store.list_experiences()
        recent = sorted(records, key=lambda x: x.decision_time, reverse=True)[:limit]
        return [
            {
                "trade_id": r.trade_id,
                "decision_id": r.decision_id,
                "decision_time": r.decision_time.isoformat(),
                "asset": r.asset,
                "action": r.action,
                "confidence": r.confidence,
                "outcome_label": r.outcome_label,
                "realized_pnl": r.realized_pnl,
                "net_return": r.net_return,
            }
            for r in recent
        ]
