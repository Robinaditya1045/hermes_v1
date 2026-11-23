from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from apps.api.app import AppState, create_api_app
from apps.dashboard.app import create_dashboard_app
from hermes_quant.audit.service import AuditService, TradeAuditBundle
from hermes_quant.experience.record import TradeExperience
from hermes_quant.experience.store import InMemoryExperienceStore
from hermes_quant.monitoring.service import MonitoringService, SystemHealth
from hermes_quant.risk.engine import RiskConfig, RiskDecision
from hermes_quant.trading.broker import (
    InMemoryOrderStore,
    PaperBroker,
    TradeProposal,
)
from hermes_quant.trading.fills import ConstantFillModel, FillModelConfig


@pytest.fixture
def memory_experience_store() -> InMemoryExperienceStore:
    store = InMemoryExperienceStore()
    now = datetime(2026, 9, 10, 0, 0, 0, tzinfo=timezone.utc)
    # Seed 3 sample trades
    t1 = TradeExperience(
        trade_id="trade-win-1",
        decision_id="dec-win-1",
        decision_time=now,
        asset="AAPL",
        action="BUY",
        confidence=0.82,
        risk_decision="APPROVED",
        approved_quantity=20.0,
        entry_price=150.0,
        exit_price=165.0,
        fees=2.0,
        slippage=1.5,
        realized_pnl=296.5,
        net_return=0.0988,
        holding_period_bars=5,
        outcome_label="WIN",
        regime_at_decision="trending",
        post_trade_notes="Strong momentum breakout",
    )
    t2 = TradeExperience(
        trade_id="trade-loss-1",
        decision_id="dec-loss-1",
        decision_time=now,
        asset="TSLA",
        action="BUY",
        confidence=0.58,
        risk_decision="APPROVED",
        approved_quantity=15.0,
        entry_price=220.0,
        exit_price=205.0,
        fees=1.8,
        slippage=2.2,
        realized_pnl=-229.0,
        net_return=-0.0694,
        holding_period_bars=3,
        outcome_label="LOSS",
        regime_at_decision="high_volatility",
        post_trade_notes="Dip buy failed",
    )
    t3 = TradeExperience(
        trade_id="trade-open-1",
        decision_id="dec-open-1",
        decision_time=now,
        asset="MSFT",
        action="BUY",
        confidence=0.75,
        risk_decision="APPROVED",
        approved_quantity=10.0,
        entry_price=400.0,
        fees=1.0,
        slippage=0.5,
        outcome_label="OPEN",
        regime_at_decision="risk_on",
        post_trade_notes="Holding for breakout",
    )
    store.save(t1)
    store.save(t2)
    store.save(t3)
    return store


@pytest.fixture
def mock_orchestrator_store() -> Any:
    class MockStore:
        def __init__(self) -> None:
            self.halted = False
            self.halt_reason: str | None = None

        def set_halted(self, reason: str) -> None:
            self.halted = True
            self.halt_reason = reason

        def reset_halt(self) -> None:
            self.halted = False
            self.halt_reason = None

    return MockStore()


@pytest.fixture
def test_app_state(
    memory_experience_store: InMemoryExperienceStore,
    mock_orchestrator_store: Any,
) -> AppState:
    broker_store = InMemoryOrderStore()
    risk_config = RiskConfig(
        max_position_notional=5_000.0,
        max_portfolio_exposure=1.0,
        max_daily_loss=500.0,
        max_volatility=0.5,
        max_risk_per_trade=0.5,
        max_liquidity_fraction=1.0,
    )
    return AppState(
        experience_store=memory_experience_store,
        broker_store=broker_store,
        risk_config=risk_config,
    )


# ASGI Request Helper
def run_request(
    app: Any,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    async def _do_req() -> tuple[int, dict[str, Any] | str]:
        body_bytes = json.dumps(body).encode("utf-8") if body is not None else b""
        response_body: list[bytes] = []
        response_status = 500

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            nonlocal response_status, response_body
            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "http.response.body":
                response_body.append(message.get("body", b""))

        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": method.upper(),
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"host", b"testserver"),
            ],
            "app": app,
        }
        await app(scope, receive, send)
        raw_text = b"".join(response_body).decode("utf-8")
        try:
            parsed: dict[str, Any] | str = json.loads(raw_text)
        except Exception:
            parsed = raw_text
        return response_status, parsed

    return asyncio.run(_do_req())


# ==============================================================================
# 1. AuditService Unit Tests (Section 23.2)
# ==============================================================================


def test_audit_service_reconstruction(memory_experience_store: InMemoryExperienceStore) -> None:
    audit_service = AuditService(experience_store=memory_experience_store)

    bundle = audit_service.reconstruct_trade("trade-win-1")
    assert bundle is not None
    assert isinstance(bundle, TradeAuditBundle)
    assert bundle.trade_id == "trade-win-1"
    assert bundle.decision_id == "dec-win-1"
    assert bundle.asset == "AAPL"
    assert bundle.agent_decision["confidence"] == 0.82
    assert bundle.risk_decision["risk_decision"] == "APPROVED"
    assert bundle.outcome["outcome_label"] == "WIN"
    assert bundle.outcome["realized_pnl"] == 296.5

    # Cryptographic SHA-256 hash
    assert len(bundle.reconstruction_hash) == 64
    int(bundle.reconstruction_hash, 16)  # Verifies hex format


def test_audit_service_hash_determinism(memory_experience_store: InMemoryExperienceStore) -> None:
    audit_service = AuditService(experience_store=memory_experience_store)

    bundle1 = audit_service.reconstruct_trade("trade-win-1")
    bundle2 = audit_service.reconstruct_trade("trade-win-1")
    assert bundle1 is not None and bundle2 is not None
    assert bundle1.reconstruction_hash == bundle2.reconstruction_hash


def test_audit_service_missing_trade_returns_none(
    memory_experience_store: InMemoryExperienceStore,
) -> None:
    audit_service = AuditService(experience_store=memory_experience_store)
    assert audit_service.reconstruct_trade("non-existent-id") is None


def test_audit_service_reconstruction_with_broker_order() -> None:
    exp_store = InMemoryExperienceStore()
    broker_store = InMemoryOrderStore()
    broker = PaperBroker(
        store=broker_store,
        fill_model=ConstantFillModel(config=FillModelConfig()),
    )
    decision_id = f"dec-{uuid4().hex[:8]}"

    # Submit broker order
    proposal = TradeProposal(
        decision_id=decision_id,
        symbol="NVDA",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1200.0,
    )
    risk_dec = RiskDecision(
        approved=True,
        approved_quantity=10.0,
        approved_notional=1200.0,
        reason="approved",
        decision_id=decision_id,
        symbol="NVDA",
        action="BUY",
    )
    broker_res = broker.submit(proposal, risk_dec, market_data=120.0)

    # Record experience
    exp = TradeExperience(
        trade_id=decision_id,
        decision_id=decision_id,
        decision_time=datetime.now(timezone.utc),
        asset="NVDA",
        action="BUY",
        confidence=0.88,
        risk_decision="APPROVED",
        approved_quantity=10.0,
        entry_price=broker_res.fill.price if broker_res.fill else 120.0,
        outcome_label="OPEN",
        post_trade_notes="AI Momentum Thesis",
    )
    exp_store.save(exp)

    # Reconstruct audit
    audit_service = AuditService(experience_store=exp_store, broker_store=broker_store)
    bundle = audit_service.reconstruct_trade(decision_id)

    assert bundle is not None
    assert bundle.order is not None
    assert bundle.order["symbol"] == "NVDA"
    assert len(bundle.order_transitions) >= 3
    assert len(bundle.fills) == 1
    assert bundle.fills[0]["price"] > 0
    assert len(bundle.reconstruction_hash) == 64


def test_audit_service_list_auditable_trades(
    memory_experience_store: InMemoryExperienceStore,
) -> None:
    audit_service = AuditService(experience_store=memory_experience_store)
    trades = audit_service.list_auditable_trades(limit=10)
    assert len(trades) == 3
    assert trades[0]["trade_id"] in {"trade-win-1", "trade-loss-1", "trade-open-1"}


# ==============================================================================
# 2. MonitoringService Unit Tests
# ==============================================================================


def test_monitoring_service_health(
    memory_experience_store: InMemoryExperienceStore,
    mock_orchestrator_store: Any,
) -> None:
    mon = MonitoringService(
        experience_store=memory_experience_store,
        orchestrator_store=mock_orchestrator_store,
    )
    health = mon.get_system_health()
    assert isinstance(health, SystemHealth)
    assert health.status == "OPERATIONAL"
    assert "return_lgbm_technical_v1" in health.active_models

    mock_orchestrator_store.set_halted("risk_circuit_breaker")
    health_halted = mon.get_system_health()
    assert health_halted.status == "DEGRADED"
    assert health_halted.orchestrator_status == "HALTED"
    assert health_halted.halt_reason == "risk_circuit_breaker"


def test_monitoring_service_portfolio_summary(
    memory_experience_store: InMemoryExperienceStore,
) -> None:
    mon = MonitoringService(experience_store=memory_experience_store)
    summary = mon.get_portfolio_summary(current_prices={"MSFT": 420.0}, base_cash=100_000.0)

    # Realized PnL: 296.5 + (-229.0) = 67.5
    assert abs(summary.realized_pnl - 67.5) < 1e-4
    # Positions: MSFT has 10 units with entry 400.0, current 420.0
    assert len(summary.positions) == 1
    pos = summary.positions[0]
    assert pos.symbol == "MSFT"
    assert pos.quantity == 10.0
    assert pos.current_price == 420.0
    assert pos.market_value == 4200.0
    assert pos.unrealized_pnl == 200.0
    assert abs(pos.unrealized_return - 0.05) < 1e-4
    assert summary.active_exposure == 4200.0


def test_monitoring_service_risk_and_model_status(
    memory_experience_store: InMemoryExperienceStore,
) -> None:
    mon = MonitoringService(experience_store=memory_experience_store)
    risk_status = mon.get_risk_status()
    assert "limits" in risk_status
    assert risk_status["limits"]["max_position_notional"] == 10_000.0
    assert risk_status["halted"] is False

    model_status = mon.get_model_status()
    assert len(model_status["active_models"]) == 3
    assert model_status["active_models"][0]["model_id"] == "return_lgbm_technical_v1"


# ==============================================================================
# 3. FastAPI REST Endpoints & Dashboard Web UI Tests
# ==============================================================================


def test_api_health_endpoint(test_app_state: AppState) -> None:
    app = create_api_app(test_app_state)
    status, body = run_request(app, "GET", "/api/v1/health")
    assert status == 200
    assert isinstance(body, dict)
    assert body["status"] == "OPERATIONAL"
    assert "return_lgbm_technical_v1" in body["active_models"]

    # Root alias /health
    status_root, body_root = run_request(app, "GET", "/health")
    assert status_root == 200
    assert isinstance(body_root, dict)
    assert body_root["status"] == "OPERATIONAL"


def test_api_portfolio_endpoint(test_app_state: AppState) -> None:
    app = create_api_app(test_app_state)
    status, body = run_request(app, "GET", "/api/v1/portfolio")
    assert status == 200
    assert isinstance(body, dict)
    assert "total_equity" in body
    assert "cash" in body
    assert "positions" in body


def test_api_risk_status_and_kill_switch(test_app_state: AppState) -> None:
    app = create_api_app(test_app_state)

    # Initial risk status
    status, body = run_request(app, "GET", "/api/v1/risk/status")
    assert status == 200
    assert isinstance(body, dict)
    assert body["halted"] is False

    # Trigger emergency halt
    status_halt, body_halt = run_request(
        app,
        "POST",
        "/api/v1/risk/kill-switch",
        body={"action": "halt", "reason": "operator_circuit_trip"},
    )
    assert status_halt == 200
    assert isinstance(body_halt, dict)
    assert body_halt["halted"] is True
    assert body_halt["halt_reason"] == "operator_circuit_trip"

    # Verify risk status reflects halt
    _, body_check = run_request(app, "GET", "/api/v1/risk/status")
    assert isinstance(body_check, dict)
    assert body_check["halted"] is True

    # Resume trading
    status_resume, body_resume = run_request(
        app,
        "POST",
        "/api/v1/risk/kill-switch",
        body={"action": "resume"},
    )
    assert status_resume == 200
    assert isinstance(body_resume, dict)
    assert body_resume["halted"] is False


def test_api_trades_and_audit(test_app_state: AppState) -> None:
    app = create_api_app(test_app_state)

    # List trades
    status, body = run_request(app, "GET", "/api/v1/trades")
    assert status == 200
    assert isinstance(body, list)
    assert len(body) == 3

    # Audit valid trade
    status_audit, body_audit = run_request(app, "GET", "/api/v1/audit/trade-win-1")
    assert status_audit == 200
    assert isinstance(body_audit, dict)
    assert body_audit["trade_id"] == "trade-win-1"
    assert len(body_audit["reconstruction_hash"]) == 64

    # Audit invalid trade -> 404
    status_404, _ = run_request(app, "GET", "/api/v1/audit/unknown-trade-xyz")
    assert status_404 == 404


def test_api_models_and_experience_report(test_app_state: AppState) -> None:
    app = create_api_app(test_app_state)

    # Models
    status_m, body_m = run_request(app, "GET", "/api/v1/models")
    assert status_m == 200
    assert isinstance(body_m, dict)
    assert "active_models" in body_m

    # Experience report
    status_e, body_e = run_request(app, "GET", "/api/v1/experience/report")
    assert status_e == 200
    assert isinstance(body_e, dict)
    assert "calibration_bins" in body_e
    assert "regime_breakdown" in body_e
    assert "improvement_proposals" in body_e


def test_api_paper_orders_approved_and_rejected(test_app_state: AppState) -> None:
    app = create_api_app(test_app_state)

    # 1. Approved order
    order_payload = {
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 10.0,
        "price": 150.0,
        "confidence": 0.75,
        "thesis": "Breakout above 50 SMA",
    }
    status_ok, body_ok = run_request(app, "POST", "/api/v1/paper/orders", body=order_payload)
    assert status_ok == 200
    assert isinstance(body_ok, dict)
    assert body_ok["approved"] is True
    assert "order_id" in body_ok
    assert "fill" in body_ok

    # Verify order is recorded in experience store
    assert test_app_state.experience_store.get(body_ok["decision_id"]) is not None

    # 2. Risk rejected order (estimated volatility 0.85 > max_volatility 0.50)
    high_vol_order = {
        "symbol": "NVDA",
        "action": "BUY",
        "quantity": 10.0,
        "price": 200.0,
        "confidence": 0.60,
        "estimated_volatility": 0.85,
    }
    status_rej, body_rej = run_request(app, "POST", "/api/v1/paper/orders", body=high_vol_order)
    assert status_rej == 200
    assert isinstance(body_rej, dict)
    assert body_rej["approved"] is False
    assert body_rej["status"] == "RISK_REJECTED"
    assert body_rej["reason"] == "volatility_limit"


def test_dashboard_web_ui_serving() -> None:
    app = create_dashboard_app()

    # GET /
    status_root, body_root = run_request(app, "GET", "/")
    assert status_root == 200
    assert isinstance(body_root, str)
    assert "HERMES QUANT" in body_root
    assert "Cryptographic Audit" in body_root

    # GET /dashboard
    status_dash, body_dash = run_request(app, "GET", "/dashboard")
    assert status_dash == 200
    assert isinstance(body_dash, str)
    assert "HERMES QUANT" in body_dash
