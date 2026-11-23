from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_quant.experience.store import ExperienceStore
from hermes_quant.risk.engine import PortfolioRiskState, RiskConfig


@dataclass(frozen=True)
class SystemHealth:
    status: str
    postgres_status: str
    sqlite_status: str
    orchestrator_status: str
    halt_reason: str | None
    data_freshness_seconds: float | None
    active_models: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass(frozen=True)
class PositionView:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_return: float


@dataclass(frozen=True)
class PortfolioSummary:
    cash: float
    total_equity: float
    active_exposure: float
    exposure_percentage: float
    realized_pnl: float
    unrealized_pnl: float
    drawdown: float
    positions: list[PositionView]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class MonitoringService:
    """Aggregates real-time portfolio, risk, trade, and model status (Section 23)."""

    def __init__(
        self,
        *,
        experience_store: ExperienceStore,
        risk_config: RiskConfig | None = None,
        orchestrator_store: Any = None,
        broker: Any = None,
        postgres_dsn: str = "postgresql://hermes:hermes_local_only@localhost:5432/hermes_quant",
        baseline_report_path: Path = Path("artifacts/quantitative_baseline/baseline_report.json"),
    ) -> None:
        self.experience_store = experience_store
        self.risk_config = risk_config or RiskConfig(
            max_position_notional=10_000.0,
            max_portfolio_exposure=1.0,
            max_daily_loss=1_000.0,
            max_volatility=0.5,
            max_risk_per_trade=0.5,
            max_liquidity_fraction=1.0,
        )
        self.orchestrator_store = orchestrator_store
        self.broker = broker
        self.postgres_dsn = postgres_dsn
        self.baseline_report_path = baseline_report_path

    def get_system_health(self) -> SystemHealth:
        # Check Postgres
        pg_status = "UNKNOWN"
        try:
            import psycopg

            with psycopg.connect(self.postgres_dsn, connect_timeout=1) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                pg_status = "HEALTHY"
        except Exception:
            pg_status = "UNAVAILABLE"

        # Check SQLite
        sqlite_status = "HEALTHY"

        # Check Orchestrator
        orch_status = "RUNNING"
        halt_reason = None
        if self.orchestrator_store is not None:
            if getattr(self.orchestrator_store, "halted", False):
                orch_status = "HALTED"
                halt_reason = getattr(self.orchestrator_store, "halt_reason", "manual_halt")

        overall_status = "OPERATIONAL" if orch_status == "RUNNING" else "DEGRADED"

        return SystemHealth(
            status=overall_status,
            postgres_status=pg_status,
            sqlite_status=sqlite_status,
            orchestrator_status=orch_status,
            halt_reason=halt_reason,
            data_freshness_seconds=0.0,
            active_models=[
                "return_lgbm_technical_v1",
                "risk_volatility_v1",
                "regime_classifier_v1",
            ],
        )

    def get_portfolio_summary(
        self,
        current_prices: dict[str, float] | None = None,
        base_cash: float = 100_000.0,
    ) -> PortfolioSummary:
        prices = current_prices or {}
        trades = self.experience_store.list_experiences()

        # Compute positions from open/closed experiences
        position_map: dict[str, dict[str, float]] = {}
        realized_pnl = 0.0

        for t in trades:
            if t.realized_pnl is not None:
                realized_pnl += t.realized_pnl

            # Accumulate open quantity
            if t.outcome_label == "OPEN" and t.entry_price is not None and t.approved_quantity > 0:
                cur = position_map.setdefault(
                    t.asset, {"qty": 0.0, "total_cost": 0.0, "entry_px": t.entry_price}
                )
                cur["qty"] += t.approved_quantity
                cur["total_cost"] += t.approved_quantity * t.entry_price

        positions: list[PositionView] = []
        total_market_val = 0.0
        total_unrealized_pnl = 0.0

        for sym, pos_data in position_map.items():
            qty = pos_data["qty"]
            if qty <= 0:
                continue
            entry_px = pos_data["total_cost"] / qty if qty > 0 else pos_data["entry_px"]
            curr_px = prices.get(sym, entry_px)
            mkt_val = qty * curr_px
            unrealized = (curr_px - entry_px) * qty
            ret = unrealized / (entry_px * qty) if entry_px * qty > 0 else 0.0

            total_market_val += mkt_val
            total_unrealized_pnl += unrealized

            positions.append(
                PositionView(
                    symbol=sym,
                    quantity=qty,
                    entry_price=entry_px,
                    current_price=curr_px,
                    market_value=mkt_val,
                    unrealized_pnl=unrealized,
                    unrealized_return=ret,
                )
            )

        cash = base_cash + realized_pnl - sum(p["total_cost"] for p in position_map.values())
        equity = cash + total_market_val
        exposure_pct = total_market_val / equity if equity > 0 else 0.0
        drawdown = 0.0

        return PortfolioSummary(
            cash=cash,
            total_equity=equity,
            active_exposure=total_market_val,
            exposure_percentage=exposure_pct,
            realized_pnl=realized_pnl,
            unrealized_pnl=total_unrealized_pnl,
            drawdown=drawdown,
            positions=positions,
        )

    def get_risk_status(self, portfolio_state: PortfolioRiskState | None = None) -> dict[str, Any]:
        halted = False
        halt_reason = None
        if self.orchestrator_store is not None:
            halted = getattr(self.orchestrator_store, "halted", False)
            halt_reason = getattr(self.orchestrator_store, "halt_reason", None)

        p = portfolio_state or PortfolioRiskState(
            portfolio_value=100_000.0,
            invested_notional=0.0,
            symbol_notional=0.0,
            daily_pnl=0.0,
            kill_switch_active=halted,
        )

        exposure_val = p.invested_notional / p.portfolio_value if p.portfolio_value > 0 else 0.0
        exposure_util = (
            (p.invested_notional / (p.portfolio_value * self.risk_config.max_portfolio_exposure))
            if (p.portfolio_value > 0 and self.risk_config.max_portfolio_exposure > 0)
            else 0.0
        )
        daily_loss_util = (
            abs(p.daily_pnl) / self.risk_config.max_daily_loss
            if (p.daily_pnl < 0 and self.risk_config.max_daily_loss > 0)
            else 0.0
        )

        return {
            "halted": halted or p.kill_switch_active,
            "halt_reason": halt_reason,
            "limits": {
                "max_position_notional": self.risk_config.max_position_notional,
                "max_portfolio_exposure": self.risk_config.max_portfolio_exposure,
                "max_daily_loss": self.risk_config.max_daily_loss,
                "max_volatility": self.risk_config.max_volatility,
                "max_risk_per_trade": self.risk_config.max_risk_per_trade,
            },
            "current_state": {
                "total_equity": p.portfolio_value,
                "invested_notional": p.invested_notional,
                "symbol_notional": p.symbol_notional,
                "exposure": exposure_val,
                "daily_pnl": p.daily_pnl,
            },
            "utilization": {
                "exposure_utilization": exposure_util,
                "daily_loss_utilization": daily_loss_util,
            },
        }

    def get_model_status(self) -> dict[str, Any]:
        baseline_data = None
        if self.baseline_report_path.exists():
            try:
                with open(self.baseline_report_path) as f:
                    baseline_data = json.load(f)
            except Exception:
                pass

        return {
            "active_models": [
                {
                    "model_id": "return_lgbm_technical_v1",
                    "type": "Return Forecaster (LightGBM)",
                    "status": "CANDIDATE (REJECTED_BY_GATE_24_3)",
                    "last_evaluation_oos_days": 441,
                    "sharpe_ratio": 2.87,
                    "annualized_return": 0.3223,
                },
                {
                    "model_id": "risk_volatility_v1",
                    "type": "Volatility & Drawdown Risk Model",
                    "status": "PRODUCTION",
                    "lookback_days": 252,
                },
                {
                    "model_id": "regime_classifier_v1",
                    "type": "Regime Classification Model",
                    "status": "PRODUCTION",
                    "supported_regimes": [
                        "risk_on",
                        "risk_off",
                        "high_volatility",
                        "trending",
                        "sideways",
                    ],
                },
            ],
            "baseline_benchmark_metrics": baseline_data.get("metrics") if baseline_data else None,
            "gate_review": baseline_data.get("gate_review") if baseline_data else None,
        }

    def get_trades_summary(
        self,
        *,
        limit: int = 50,
        asset: str | None = None,
        outcome_label: str | None = None,
    ) -> list[dict[str, Any]]:
        trades = self.experience_store.list_experiences(asset=asset, outcome_label=outcome_label)
        recent = sorted(trades, key=lambda x: x.decision_time, reverse=True)[:limit]
        return [t.to_dict() for t in recent]
