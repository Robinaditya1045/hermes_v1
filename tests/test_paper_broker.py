from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_quant.risk import (
    PortfolioRiskState,
    RiskConfig,
    RiskDecision,
    RiskProposal,
    evaluate_proposal,
)
from hermes_quant.trading import (
    ConstantFillModel,
    FillMarketData,
    FillModelConfig,
    PaperBroker,
    SQLiteOrderStore,
    TradeProposal,
    create_order_from_risk_decision,
)
from hermes_quant.trading.orders import OrderCreationError, PaperOrder


def approved_risk(decision_id: str = "decision-1") -> RiskDecision:
    return RiskDecision(True, 10.0, 1_000.0, "approved", decision_id, None, "AAPL", "BUY")


def risk_config() -> RiskConfig:
    return RiskConfig(10_000.0, 1.0, 1_000.0, 0.50, 0.50, 1.0)


def test_order_requires_approved_risk_decision_and_has_no_raw_constructor() -> None:
    with pytest.raises(TypeError):
        PaperOrder(symbol="AAPL", quantity=10.0)  # type: ignore[call-arg]
    assert not hasattr(PaperOrder, "from_persisted")
    with pytest.raises(OrderCreationError):
        create_order_from_risk_decision(
            RiskDecision(False, 0.0, 0.0, "daily_loss", "decision-1", "DAILY_LOSS")
        )


def test_full_proposal_risk_order_fill_path_records_lifecycle() -> None:
    risk_proposal = RiskProposal("AAPL", "decision-1", 100.0, 10.0, 0.10, 1_000.0, action="BUY")
    risk = evaluate_proposal(
        risk_proposal,
        PortfolioRiskState(10_000.0, 0.0, 0.0, 0.0),
        risk_config(),
    )
    proposal = TradeProposal(
        decision_id="decision-1",
        symbol="AAPL",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )
    fill_config = FillModelConfig(spread_bps=4.0, slippage_bps=3.0, fees_per_share=0.01)
    broker = PaperBroker(fill_model=ConstantFillModel(fill_config))

    result = broker.submit(proposal, risk, market_data=100.0)

    assert result.status == "CLOSED"
    assert result.order is not None
    assert result.order.symbol == "AAPL"
    assert result.fill is not None
    assert result.fill.fill_model_version == "constant_fill_v1"
    assert result.fill.fill_model_config == {
        "fill_model_version": "constant_fill_v1",
        "spread_bps": 4.0,
        "slippage_bps": 3.0,
        "fees_per_share": 0.01,
        "fees_bps": 0.0,
    }
    assert [item.to_status for item in broker.store.transitions] == [
        "CREATED",
        "VALIDATING",
        "RISK_APPROVED",
        "SUBMITTED",
        "FILLED",
        "CLOSED",
    ]


def test_rejected_risk_returns_structured_result_and_never_creates_order() -> None:
    broker = PaperBroker(fill_model=ConstantFillModel(FillModelConfig()))
    proposal = TradeProposal(
        decision_id="decision-rejected",
        symbol="AAPL",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )
    rejected = RiskDecision(
        False,
        0.0,
        0.0,
        "daily_loss_limit",
        decision_id="decision-rejected",
        risk_code="DAILY_LOSS",
    )

    result = broker.submit(proposal, rejected, market_data=100.0)

    assert result.status == "RISK_REJECTED"
    assert result.risk_decision.approved is False
    assert result.risk_decision.reason == "daily_loss_limit"
    assert result.risk_decision.risk_code == "DAILY_LOSS"
    assert result.order is None


def test_mismatched_risk_identity_raises_before_order_creation() -> None:
    broker = PaperBroker(fill_model=ConstantFillModel(FillModelConfig()))
    proposal = TradeProposal(
        decision_id="decision-mismatch",
        symbol="MSFT",
        action="SELL",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )
    mismatched = RiskDecision(
        True, 10.0, 1_000.0, "approved", "decision-mismatch", None, "AAPL", "BUY"
    )

    with pytest.raises(ValueError, match="does not match"):
        broker.submit(proposal, mismatched, market_data=100.0)

    assert broker.store.orders == {}


def test_cross_submitted_risk_decision_id_raises() -> None:
    risk_proposal_a = RiskProposal("AAPL", "decision-a", 100.0, 10.0, 0.10, 1_000.0, action="BUY")
    risk_proposal_b = RiskProposal("AAPL", "decision-b", 100.0, 10.0, 0.10, 1_000.0, action="BUY")
    portfolio = PortfolioRiskState(10_000.0, 0.0, 0.0, 0.0)
    risk_a = evaluate_proposal(risk_proposal_a, portfolio, risk_config())
    evaluate_proposal(risk_proposal_b, portfolio, risk_config())
    proposal_b = TradeProposal(
        decision_id="decision-b",
        symbol="AAPL",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )
    broker = PaperBroker(fill_model=ConstantFillModel(FillModelConfig()))

    with pytest.raises(ValueError, match="decision_id"):
        broker.submit(proposal_b, risk_a, market_data=100.0)


def test_decision_id_is_idempotent_across_store_reconstruction(tmp_path: Path) -> None:
    database = str(tmp_path / "orders.sqlite")
    proposal = TradeProposal(
        decision_id="decision-retry",
        symbol="AAPL",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )
    first_broker = PaperBroker(SQLiteOrderStore(database), ConstantFillModel(FillModelConfig()))
    first = first_broker.submit(proposal, approved_risk("decision-retry"), market_data=100.0)

    retry = TradeProposal(
        decision_id="decision-retry",
        symbol="AAPL",
        action="BUY",
        requested_quantity=1.0,
        requested_notional=100.0,
    )
    second_broker = PaperBroker(SQLiteOrderStore(database), ConstantFillModel(FillModelConfig()))
    second = second_broker.submit(retry, approved_risk("decision-retry"), market_data=999.0)

    assert first.order is not None and second.order is not None
    assert second.status == "ALREADY_PROCESSED"
    assert second.order.order_id == first.order.order_id
    assert second.order.quantity == 10.0
    assert second.fill is not None and first.fill is not None
    assert second.fill.fill_id == first.fill.fill_id
    assert second.fill.fill_model_config == first.fill.fill_model_config


def test_partial_fill_stops_at_partially_filled() -> None:
    broker = PaperBroker(fill_model=ConstantFillModel(FillModelConfig()))
    proposal = TradeProposal(
        decision_id="decision-partial",
        symbol="AAPL",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )

    result = broker.submit(
        proposal,
        approved_risk("decision-partial"),
        market_data=100.0,
        fill_quantity=4.0,
    )

    assert result.status == "PARTIALLY_FILLED"
    assert result.order is not None and result.order.status == "PARTIALLY_FILLED"
    assert [item.to_status for item in broker.store.transitions][-1] == "PARTIALLY_FILLED"


@dataclass(frozen=True)
class DummyFillModel:
    def simulate_fill(self, order: PaperOrder, market_data: FillMarketData):
        from hermes_quant.trading.fills import Fill

        return Fill(
            order_id=order.order_id,
            fill_id=order.order_id,
            quantity=order.quantity,
            price=market_data.price,
            fees=0.0,
            slippage=0.0,
            fill_model_version="dummy_v1",
            fill_model_config={"mode": "dummy"},
            occurred_at=datetime.now(timezone.utc),
        )


def test_fill_model_interface_is_swappable() -> None:
    broker = PaperBroker(fill_model=DummyFillModel())
    proposal = TradeProposal(
        decision_id="decision-dummy",
        symbol="AAPL",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )

    result = broker.submit(proposal, approved_risk("decision-dummy"), market_data=101.0)

    assert result.fill is not None
    assert result.fill.fill_model_version == "dummy_v1"


class FailingStore:
    def get_by_decision_id(self, decision_id: str):
        return None

    def save_order(self, order: PaperOrder) -> None:
        raise OSError("database unavailable")


class RaceStore(FailingStore):
    def __init__(self, result):
        self.result = result

    def get_by_decision_id(self, decision_id: str):
        return self.result

    def save_order(self, order: PaperOrder) -> None:
        raise RuntimeError("unique decision_id constraint violated")


def test_unique_constraint_race_retries_idempotency_lookup() -> None:
    proposal = TradeProposal(
        decision_id="decision-race",
        symbol="AAPL",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )
    original = PaperBroker(fill_model=ConstantFillModel(FillModelConfig())).submit(
        proposal, approved_risk("decision-race"), market_data=100.0
    )
    broker = PaperBroker(RaceStore(original), ConstantFillModel(FillModelConfig()))

    result = broker.submit(proposal, approved_risk("decision-race"), market_data=999.0)

    assert result is original


def test_genuine_persistence_failure_raises() -> None:
    broker = PaperBroker(FailingStore(), ConstantFillModel(FillModelConfig()))  # type: ignore[arg-type]
    proposal = TradeProposal(
        decision_id="decision-db-failure",
        symbol="AAPL",
        action="BUY",
        requested_quantity=10.0,
        requested_notional=1_000.0,
    )

    with pytest.raises(OSError, match="database unavailable"):
        broker.submit(proposal, approved_risk("decision-db-failure"), market_data=100.0)
