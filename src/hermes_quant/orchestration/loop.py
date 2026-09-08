from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from hermes_quant.hermes.schemas import HermesDecision
from hermes_quant.risk.engine import (
    PortfolioRiskState,
    RiskConfig,
    RiskDecision,
    RiskProposal,
    evaluate_proposal,
)
from hermes_quant.trading.broker import (
    BrokerResult,
    PaperBroker,
    TradeProposal,
)


def _uuid7() -> str:
    """Generate a time-ordered UUIDv7-compatible identifier on Python 3.10+."""
    timestamp_ms = int(time.time() * 1000)
    random_bits = UUID(int=__import__("secrets").randbits(128))
    value = (timestamp_ms << 80) | (random_bits.int & ((1 << 76) - 1))
    value &= ~(0xF << 76)
    value |= 0x7 << 76
    value &= ~(0x3 << 62)
    value |= 0x2 << 62
    return str(UUID(int=value))


@dataclass(frozen=True)
class HaltConfig:
    consecutive_data_failures: int = 3
    consecutive_risk_rejections: int = 3
    risk_rejection_scope: str = "portfolio"
    stale_data_seconds: float = 300.0
    daily_loss_limit: float = 500.0
    max_drawdown: float = 0.20


@dataclass(frozen=True)
class HaltTriggerInput:
    data_validation_failed: bool = False
    risk_rejected_symbol: str | None = None
    information_available_at: datetime | None = None
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    unexpected_position_state: bool = False


@dataclass(frozen=True)
class ExperienceRecord:
    trade_id: str
    decision_id: str
    asset: str
    action: str
    outcome_status: str
    recorded_at: datetime


@dataclass(frozen=True)
class OrchestratorResult:
    status: str
    decision_id: str | None
    dedup_key: str
    risk_decision: RiskDecision | None = None
    broker_result: BrokerResult | None = None
    reason: str | None = None


class InMemoryOrchestratorStore:
    def __init__(self) -> None:
        self.dedup: dict[str, OrchestratorResult] = {}
        self.halted = False
        self.halt_reason: str | None = None
        self.experience: list[ExperienceRecord] = []

    def get_dedup(self, key: str) -> OrchestratorResult | None:
        return self.dedup.get(key)

    def save_dedup(self, key: str, result: OrchestratorResult) -> None:
        self.dedup[key] = result

    def set_halted(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def reset_halt(self) -> None:
        self.halted = False
        self.halt_reason = None

    def save_experience(self, record: ExperienceRecord) -> None:
        self.experience.append(record)


class SQLiteOrchestratorStore(InMemoryOrchestratorStore):
    """Crash-durable deduplication, halt state, and experience records."""

    def __init__(self, database_path: str) -> None:
        super().__init__()
        self.connection = sqlite3.connect(database_path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS orchestrator_requests ("
            "dedup_key TEXT PRIMARY KEY, decision_id TEXT, result_json TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS trading_state ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), halted INTEGER NOT NULL, reason TEXT)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS experience ("
            "trade_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL, asset TEXT NOT NULL, "
            "action TEXT NOT NULL, outcome_status TEXT NOT NULL, recorded_at TEXT NOT NULL)"
        )
        row = self.connection.execute(
            "SELECT halted, reason FROM trading_state WHERE id = 1"
        ).fetchone()
        if row is None:
            self.connection.execute("INSERT INTO trading_state VALUES (1, 0, NULL)")
        else:
            self.halted, self.halt_reason = bool(row[0]), row[1]
        self.connection.commit()

    def get_dedup(self, key: str) -> OrchestratorResult | None:
        row = self.connection.execute(
            "SELECT result_json FROM orchestrator_requests WHERE dedup_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return OrchestratorResult(
            "ALREADY_PROCESSED", data["decision_id"], key, reason=data["reason"]
        )

    def save_dedup(self, key: str, result: OrchestratorResult) -> None:
        self.connection.execute(
            "INSERT INTO orchestrator_requests VALUES (?, ?, ?)",
            (
                key,
                result.decision_id,
                json.dumps({"decision_id": result.decision_id, "reason": result.reason}),
            ),
        )
        self.connection.commit()

    def set_halted(self, reason: str) -> None:
        super().set_halted(reason)
        self.connection.execute(
            "UPDATE trading_state SET halted = 1, reason = ? WHERE id = 1", (reason,)
        )
        self.connection.commit()

    def reset_halt(self) -> None:
        super().reset_halt()
        self.connection.execute("UPDATE trading_state SET halted = 0, reason = NULL WHERE id = 1")
        self.connection.commit()

    def save_experience(self, record: ExperienceRecord) -> None:
        super().save_experience(record)
        self.connection.execute(
            "INSERT INTO experience VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.trade_id,
                record.decision_id,
                record.asset,
                record.action,
                record.outcome_status,
                record.recorded_at.isoformat(),
            ),
        )
        self.connection.commit()


@dataclass
class Orchestrator:
    broker: PaperBroker
    risk_config: RiskConfig
    halt_config: HaltConfig = field(default_factory=HaltConfig)
    store: InMemoryOrchestratorStore = field(default_factory=InMemoryOrchestratorStore)
    data_failure_count: int = 0
    portfolio_risk_rejection_count: int = 0
    symbol_risk_rejections: dict[str, int] = field(default_factory=dict)

    def submit(
        self,
        decision: HermesDecision,
        *,
        price: float,
        estimated_volatility: float,
        average_daily_volume: float,
        portfolio: PortfolioRiskState,
        market_data_at: datetime,
        now: datetime | None = None,
        drawdown: float = 0.0,
        fill_quantity: float | None = None,
    ) -> OrchestratorResult:
        dedup_key = self._dedup_key(decision, price, estimated_volatility, average_daily_volume)
        existing = self.store.get_dedup(dedup_key)
        if existing is not None:
            return existing
        if self.store.halted:
            return OrchestratorResult(
                "TRADING_HALTED", None, dedup_key, reason=self.store.halt_reason
            )
        if not math.isfinite(price) or price <= 0:
            self.observe(
                HaltTriggerInput(data_validation_failed=True),
                now=now,
            )
            status = "TRADING_HALTED" if self.store.halted else "DATA_REJECTED"
            result = OrchestratorResult(status, None, dedup_key, reason="invalid_price")
            self.store.save_dedup(dedup_key, result)
            return result
        self.observe(
            HaltTriggerInput(
                information_available_at=market_data_at,
                daily_pnl=portfolio.daily_pnl,
                drawdown=drawdown,
            ),
            now=now,
        )
        if self.store.halted:
            result = OrchestratorResult(
                "TRADING_HALTED", None, dedup_key, reason=self.store.halt_reason
            )
            self.store.save_dedup(dedup_key, result)
            return result
        if decision.action == "HOLD":
            result = OrchestratorResult("HOLD", None, dedup_key, reason="no_order_requested")
            self.store.save_dedup(dedup_key, result)
            return result

        decision_id = _uuid7()
        quantity = decision.requested_notional / price
        risk_proposal = RiskProposal(
            decision.asset,
            decision_id,
            price,
            quantity,
            estimated_volatility,
            average_daily_volume,
            decision.action,
        )
        risk_decision = evaluate_proposal(risk_proposal, portfolio, self.risk_config)
        proposal = TradeProposal(
            decision_id=decision_id,
            symbol=decision.asset,
            action=decision.action,
            requested_quantity=quantity,
            requested_notional=decision.requested_notional,
        )
        if not risk_decision.approved:
            self._record_risk_rejection(decision.asset)
            if (
                self.halt_config.risk_rejection_scope == "portfolio"
                and self.portfolio_risk_rejection_count
                >= self.halt_config.consecutive_risk_rejections
            ):
                self._halt("consecutive_risk_rejections")
            result = OrchestratorResult(
                "RISK_REJECTED",
                decision_id,
                dedup_key,
                risk_decision,
                reason=risk_decision.reason,
            )
            self.store.save_dedup(dedup_key, result)
            return result
        try:
            broker_result = self.broker.submit(
                proposal, risk_decision, market_data=price, fill_quantity=fill_quantity
            )
        except Exception:
            self.store.set_halted("paper_broker_error")
            raise
        result = OrchestratorResult(
            broker_result.status, decision_id, dedup_key, risk_decision, broker_result
        )
        self.store.save_dedup(dedup_key, result)
        self.store.save_experience(
            ExperienceRecord(
                decision_id,
                decision_id,
                decision.asset,
                decision.action,
                broker_result.status,
                datetime.now(timezone.utc),
            )
        )
        return result

    def observe(self, trigger: HaltTriggerInput, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if trigger.data_validation_failed:
            self.data_failure_count += 1
        else:
            self.data_failure_count = 0
        if trigger.risk_rejected_symbol:
            self._record_risk_rejection(trigger.risk_rejected_symbol)
        if (
            trigger.information_available_at is not None
            and (now - trigger.information_available_at).total_seconds()
            > self.halt_config.stale_data_seconds
        ):
            return self._halt("stale_market_data")
        if trigger.daily_pnl <= -abs(self.halt_config.daily_loss_limit):
            return self._halt("daily_loss_threshold")
        if trigger.drawdown <= -abs(self.halt_config.max_drawdown):
            return self._halt("drawdown_threshold")
        if trigger.unexpected_position_state:
            return self._halt("unexpected_position_state")
        if self.data_failure_count >= self.halt_config.consecutive_data_failures:
            return self._halt("consecutive_data_failures")
        if (
            self.halt_config.risk_rejection_scope == "portfolio"
            and self.portfolio_risk_rejection_count >= self.halt_config.consecutive_risk_rejections
        ):
            return self._halt("consecutive_risk_rejections")
        return self.store.halted

    def manual_reset(self) -> None:
        self.store.reset_halt()
        self.data_failure_count = 0
        self.portfolio_risk_rejection_count = 0
        self.symbol_risk_rejections.clear()

    def _record_risk_rejection(self, symbol: str) -> None:
        self.portfolio_risk_rejection_count += 1
        self.symbol_risk_rejections[symbol] = self.symbol_risk_rejections.get(symbol, 0) + 1
        if (
            self.halt_config.risk_rejection_scope == "symbol"
            and self.symbol_risk_rejections[symbol] >= self.halt_config.consecutive_risk_rejections
        ):
            self._halt("consecutive_symbol_risk_rejections")

    def _halt(self, reason: str) -> bool:
        self.store.set_halted(reason)
        return True

    @staticmethod
    def _dedup_key(decision: HermesDecision, price: float, volatility: float, volume: float) -> str:
        payload = {
            "decision": decision.model_dump(mode="json"),
            "price": price,
            "volatility": volatility,
            "volume": volume,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
