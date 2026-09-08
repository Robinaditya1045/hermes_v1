from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hermes_quant.hermes.schemas import HermesDecision
from hermes_quant.orchestration import (
    HaltConfig,
    HaltTriggerInput,
    Orchestrator,
    SQLiteOrchestratorStore,
)
from hermes_quant.risk import PortfolioRiskState, RiskConfig
from hermes_quant.trading import ConstantFillModel, FillModelConfig, PaperBroker

NOW = datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc)


def risk_config(**overrides) -> RiskConfig:
    values = {
        "max_position_notional": 10_000.0,
        "max_portfolio_exposure": 1.0,
        "max_daily_loss": 1_000.0,
        "max_volatility": 0.5,
        "max_risk_per_trade": 0.5,
        "max_liquidity_fraction": 1.0,
    }
    values.update(overrides)
    return RiskConfig(**values)


def decision() -> HermesDecision:
    return HermesDecision(
        asset="AAPL",
        action="BUY",
        confidence=0.8,
        thesis=["technical signal"],
        risks=["volatility"],
        evidence_used=["market_state"],
        requested_notional=1_000.0,
    )


def orchestrator(store=None, **halt_overrides) -> Orchestrator:
    return Orchestrator(
        broker=PaperBroker(fill_model=ConstantFillModel(FillModelConfig())),
        risk_config=risk_config(),
        halt_config=HaltConfig(**halt_overrides),
        store=store or SQLiteOrchestratorStore(":memory:"),
    )


def submit(orchestrator: Orchestrator, *, now=NOW, **kwargs):
    return orchestrator.submit(
        decision(),
        price=100.0,
        estimated_volatility=0.1,
        average_daily_volume=1_000.0,
        portfolio=PortfolioRiskState(10_000.0, 0.0, 0.0, 0.0),
        market_data_at=NOW,
        now=now,
        **kwargs,
    )


def test_full_hermes_to_experience_path_and_orchestrator_dedup() -> None:
    instance = orchestrator()
    first = submit(instance)
    second = submit(instance)

    assert first.status == "CLOSED"
    assert first.decision_id is not None
    assert second.status == "ALREADY_PROCESSED"
    assert second.decision_id == first.decision_id
    assert len(instance.store.experience) == 1
    assert [item.to_status for item in instance.broker.store.transitions] == [
        "CREATED",
        "VALIDATING",
        "RISK_APPROVED",
        "SUBMITTED",
        "FILLED",
        "CLOSED",
    ]


def test_hermes_surface_has_no_raw_broker_submission_path() -> None:
    from hermes_quant.hermes.tools import ReadOnlyToolServer

    assert not hasattr(ReadOnlyToolServer, "submit_paper_order")
    assert not hasattr(ReadOnlyToolServer, "submit")


def test_risk_rejection_is_structured_and_can_halt() -> None:
    instance = orchestrator(consecutive_risk_rejections=1)
    instance.risk_config = risk_config(max_volatility=0.01)

    result = submit(instance)

    assert result.status == "RISK_REJECTED"
    assert instance.store.halted is True
    assert instance.store.halt_reason == "consecutive_risk_rejections"


def test_invalid_price_is_structured_data_rejection_and_counts_toward_halt() -> None:
    instance = orchestrator(consecutive_data_failures=2)

    result = instance.submit(
        decision(),
        price=0.0,
        estimated_volatility=0.1,
        average_daily_volume=1_000.0,
        portfolio=PortfolioRiskState(10_000.0, 0.0, 0.0, 0.0),
        market_data_at=NOW,
        now=NOW,
    )

    assert result.status == "DATA_REJECTED"
    assert result.reason == "invalid_price"
    assert instance.data_failure_count == 1
    assert instance.store.halted is False

    instance.observe(HaltTriggerInput(data_validation_failed=True), now=NOW)
    assert instance.data_failure_count == 2
    assert instance.store.halted is True
    assert instance.store.halt_reason == "consecutive_data_failures"


@pytest.mark.parametrize(
    ("trigger", "expected"),
    [
        ("data", "consecutive_data_failures"),
        ("symbol_risk", "consecutive_symbol_risk_rejections"),
        ("stale", "stale_market_data"),
        ("daily_loss", "daily_loss_threshold"),
        ("drawdown", "drawdown_threshold"),
        ("position", "unexpected_position_state"),
    ],
)
def test_each_automatic_halt_trigger(trigger: str, expected: str) -> None:
    instance = orchestrator(
        consecutive_data_failures=2,
        consecutive_risk_rejections=2,
        risk_rejection_scope="symbol",
        stale_data_seconds=60.0,
        daily_loss_limit=500.0,
        max_drawdown=0.2,
    )
    if trigger == "data":
        instance.observe(HaltTriggerInput(data_validation_failed=True), now=NOW)
        halted = instance.observe(HaltTriggerInput(data_validation_failed=True), now=NOW)
    elif trigger == "symbol_risk":
        instance.observe(HaltTriggerInput(risk_rejected_symbol="AAPL"), now=NOW)
        halted = instance.observe(HaltTriggerInput(risk_rejected_symbol="AAPL"), now=NOW)
    elif trigger == "stale":
        halted = instance.observe(
            HaltTriggerInput(information_available_at=NOW - timedelta(seconds=61)), now=NOW
        )
    elif trigger == "daily_loss":
        halted = instance.observe(HaltTriggerInput(daily_pnl=-500.0), now=NOW)
    elif trigger == "drawdown":
        halted = instance.observe(HaltTriggerInput(drawdown=-0.2), now=NOW)
    else:
        halted = instance.observe(HaltTriggerInput(unexpected_position_state=True), now=NOW)

    assert halted is True
    assert instance.store.halt_reason == expected


def test_halted_submission_is_structured_and_manual_reset_is_required() -> None:
    instance = orchestrator()
    instance.observe(HaltTriggerInput(unexpected_position_state=True), now=NOW)

    halted = submit(instance)
    assert halted.status == "TRADING_HALTED"
    assert instance.broker.store.orders == {}

    instance.manual_reset()
    resumed = submit(instance)
    assert resumed.status == "CLOSED"


def test_halt_survives_store_restart(tmp_path: Path) -> None:
    database = str(tmp_path / "orchestrator.sqlite")
    first = orchestrator(SQLiteOrchestratorStore(database))
    first.observe(HaltTriggerInput(unexpected_position_state=True), now=NOW)

    restarted = orchestrator(SQLiteOrchestratorStore(database))
    result = submit(restarted)

    assert result.status == "TRADING_HALTED"
    assert result.reason == "unexpected_position_state"


def test_broker_exception_halts_and_reraises() -> None:
    instance = orchestrator()
    instance.broker.fill_model = None

    with pytest.raises(RuntimeError, match="fill_model"):
        submit(instance)

    assert instance.store.halted is True
    assert instance.store.halt_reason == "paper_broker_error"
