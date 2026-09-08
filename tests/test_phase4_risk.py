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


def test_position_limit_resizes_exactly() -> None:
    decision = evaluate_proposal(
        RiskProposal("AAPL", 100.0, 100.0, 0.10),
        PortfolioRiskState(10_000.0, 1_000.0, 500.0, 0.0),
        RiskConfig(2_000.0, 0.80, 500.0, 0.50, 0.02),
    )

    assert decision.approved is True
    assert decision.approved_quantity == pytest.approx(15.0)
    assert decision.approved_notional == pytest.approx(1_500.0)
    assert decision.reason == "resized_to_limits"


def test_daily_loss_limit_rejects_exactly() -> None:
    decision = evaluate_proposal(
        RiskProposal("AAPL", 100.0, 10.0, 0.10),
        PortfolioRiskState(10_000.0, 0.0, 0.0, -500.0),
        RiskConfig(2_000.0, 0.80, 500.0, 0.50, 0.02),
    )

    assert decision.approved is False
    assert decision.approved_quantity == 0.0
    assert decision.reason == "daily_loss_limit"


def test_kill_switch_rejects_exactly() -> None:
    decision = evaluate_proposal(
        RiskProposal("AAPL", 100.0, 10.0, 0.10),
        PortfolioRiskState(10_000.0, 0.0, 0.0, 0.0, kill_switch_active=True),
        RiskConfig(2_000.0, 0.80, 500.0, 0.50, 0.02),
    )

    assert decision.approved is False
    assert decision.approved_notional == 0.0
    assert decision.reason == "kill_switch_active"


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
