import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from hermes_quant.models.regime_model import classify_regimes
from hermes_quant.models.risk_model import calculate_risk_state
from hermes_quant.risk import (
    PortfolioRiskState,
    RiskConfig,
    RiskProposal,
    evaluate_proposal,
)

DECISION_AT = datetime(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
OUTCOME_AT = datetime(2027, 1, 1, 20, 0, tzinfo=timezone.utc)


def proposal(
    *,
    price: float = 100.0,
    requested_quantity: float = 10.0,
    estimated_volatility: float = 0.10,
    average_daily_volume: float = 1_000.0,
) -> RiskProposal:
    return RiskProposal(
        "AAPL", price, requested_quantity, estimated_volatility, average_daily_volume
    )


def portfolio(
    *,
    portfolio_value: float = 10_000.0,
    invested_notional: float = 0.0,
    symbol_notional: float = 0.0,
    daily_pnl: float = 0.0,
    kill_switch_active: bool = False,
) -> PortfolioRiskState:
    return PortfolioRiskState(
        portfolio_value,
        invested_notional,
        symbol_notional,
        daily_pnl,
        kill_switch_active,
    )


def config(
    *,
    max_position_notional: float = 2_000.0,
    max_portfolio_exposure: float = 0.80,
    max_daily_loss: float = 500.0,
    max_volatility: float = 0.50,
    max_risk_per_trade: float = 0.02,
    max_liquidity_fraction: float = 1.0,
) -> RiskConfig:
    return RiskConfig(
        max_position_notional,
        max_portfolio_exposure,
        max_daily_loss,
        max_volatility,
        max_risk_per_trade,
        max_liquidity_fraction,
    )


def test_position_limit_resizes_exactly() -> None:
    decision = evaluate_proposal(
        proposal(requested_quantity=100.0),
        portfolio(invested_notional=1_000.0, symbol_notional=500.0),
        config(max_risk_per_trade=0.02),
    )

    assert decision.approved is True
    assert decision.approved_quantity == 15.0
    assert decision.approved_notional == 1_500.0
    assert decision.reason == "resized_by_position_limit"


def test_portfolio_exposure_is_binding_resize_reason() -> None:
    decision = evaluate_proposal(
        proposal(requested_quantity=100.0),
        portfolio(invested_notional=5_000.0),
        config(max_position_notional=9_000.0, max_risk_per_trade=0.50),
    )

    assert decision.approved_quantity == 30.0
    assert decision.approved_notional == 3_000.0
    assert decision.reason == "resized_by_portfolio_exposure"


def test_liquidity_limit_is_binding_resize_reason() -> None:
    decision = evaluate_proposal(
        proposal(requested_quantity=100.0, average_daily_volume=100.0),
        portfolio(),
        config(
            max_position_notional=9_000.0,
            max_risk_per_trade=0.20,
            max_liquidity_fraction=0.10,
        ),
    )

    assert decision.approved_quantity == 10.0
    assert decision.approved_notional == 1_000.0
    assert decision.reason == "resized_by_liquidity_limit"


def test_risk_budget_is_binding_resize_reason() -> None:
    decision = evaluate_proposal(
        proposal(requested_quantity=100.0),
        portfolio(),
        config(
            max_position_notional=9_000.0,
            max_risk_per_trade=0.01,
        ),
    )

    assert decision.approved_quantity == 10.0
    assert decision.approved_notional == 1_000.0
    assert decision.reason == "resized_by_risk_budget"


def test_daily_loss_limit_rejects_exactly() -> None:
    decision = evaluate_proposal(
        proposal(),
        portfolio(daily_pnl=-500.0),
        config(),
    )

    assert decision.approved is False
    assert decision.approved_quantity == 0.0
    assert decision.reason == "daily_loss_limit"


def test_kill_switch_rejects_exactly() -> None:
    decision = evaluate_proposal(
        proposal(),
        portfolio(kill_switch_active=True),
        config(),
    )

    assert decision.approved is False
    assert decision.approved_notional == 0.0
    assert decision.reason == "kill_switch_active"


@pytest.mark.parametrize(
    ("proposal_kwargs", "portfolio_kwargs"),
    [
        ({"price": math.nan}, {}),
        ({"estimated_volatility": math.nan}, {}),
        ({}, {"portfolio_value": math.inf}),
        ({}, {"daily_pnl": math.inf}),
    ],
)
def test_non_finite_inputs_fail_closed(
    proposal_kwargs: dict[str, float], portfolio_kwargs: dict[str, float]
) -> None:
    decision = evaluate_proposal(
        proposal(**proposal_kwargs), portfolio(**portfolio_kwargs), config()
    )

    assert decision.approved is False
    assert decision.approved_quantity == 0.0
    assert decision.approved_notional == 0.0
    assert decision.reason == "invalid_input"


def test_regime_and_risk_models_emit_versioned_timestamped_state() -> None:
    dates = [datetime(2026, 1, 1) + timedelta(days=index) for index in range(3)]
    technical = pl.DataFrame(
        {
            "symbol": ["AAPL"] * 3,
            "bar_date": dates,
            "information_available_at": [
                datetime(2026, 1, 1, 21, tzinfo=timezone.utc) + timedelta(days=index)
                for index in range(3)
            ],
            "feature_timestamp": [
                datetime(2026, 1, 1, 21, tzinfo=timezone.utc) + timedelta(days=index)
                for index in range(3)
            ],
            "realized_volatility_20d": [0.2, 0.3, 0.4],
            "volatility_percentile_252d": [0.2, 0.5, 0.9],
            "return_20d": [0.1, -0.1, 0.0],
            "ema_20d": [101.0, 99.0, 100.0],
            "ema_50d": [100.0, 100.0, 100.0],
        }
    )
    regimes = classify_regimes(technical)
    bars = technical.select("symbol", "bar_date", "information_available_at").with_columns(
        pl.lit(100.0).alias("open"),
        pl.Series("high", [101.0, 100.0, 102.0]),
        pl.Series("low", [99.0, 97.0, 99.0]),
        pl.Series("close", [100.0, 98.0, 101.0]),
        pl.lit(1_000.0).alias("volume"),
    )
    risk = calculate_risk_state(bars, decision_at=DECISION_AT, outcome_at=OUTCOME_AT)

    assert regimes.get_column("regime").to_list() == ["risk_on", "risk_off", "high_volatility"]
    assert regimes.get_column("regime_model_version").n_unique() == 1
    assert {"realized_volatility", "downside_volatility", "max_drawdown"}.issubset(risk.columns)
    assert risk.get_column("risk_model_version").n_unique() == 1
