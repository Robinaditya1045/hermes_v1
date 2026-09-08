from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hermes_quant.hermes.schemas import HermesDecision, MarketState, build_market_state
from hermes_quant.hermes.tools import ReadOnlyToolServer

DECISION_TIME = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)


def market_state() -> MarketState:
    return build_market_state(
        asset="AAPL",
        decision_time=DECISION_TIME,
        technical={
            "feature_timestamp": DECISION_TIME,
            "feature_set_version": "technical.v1",
            "features": {"return_20d": 0.12, "volume_z_score_20d": 1.4},
        },
        market_prediction={
            "probability_up": 0.67,
            "expected_return": 0.012,
            "model_version": "return_lgbm_technical_v1",
        },
        volatility={
            "realized_20d": 0.42,
            "downside_volatility": 0.30,
            "max_drawdown": -0.08,
            "risk_model_version": "risk_state_technical_v1",
        },
        market_regime={
            "state": "risk_on",
            "confidence": 0.82,
            "model_version": "regime_rules_technical_v1",
        },
        fundamentals={"available": False, "metrics": {}},
        news={"available": False, "sentiment": None, "impact": "unknown", "items": 0},
    )


def test_market_state_rejects_missing_required_field() -> None:
    state = market_state().model_dump()
    del state["market_prediction"]

    with pytest.raises(ValidationError):
        MarketState.model_validate(state)


def test_read_only_tools_return_typed_data_and_log_calls() -> None:
    state = market_state()
    server = ReadOnlyToolServer(
        market_states={"AAPL": state},
        quant_predictions={"AAPL": state.market_prediction},
        market_regimes={"AAPL": state.market_regime},
        risk_assessments={"AAPL": state.volatility},
        portfolio={"cash": 10_000.0, "positions": []},
        trade_history=[],
    )

    assert server.get_market_state("AAPL").asset == "AAPL"
    assert server.get_quant_prediction("AAPL").probability_up == 0.67
    assert server.get_fundamentals("AAPL").available is False
    assert server.get_recent_news("AAPL").items == 0
    assert server.get_market_regime("AAPL").state == "risk_on"
    assert server.get_risk_assessment("AAPL").max_drawdown == -0.08
    assert server.get_portfolio()["cash"] == 10_000.0
    assert server.get_trade_history() == []

    assert len(server.logs) == 8
    assert {entry.status for entry in server.logs} == {"success"}
    assert all(entry.latency_ms >= 0.0 for entry in server.logs)
    assert {entry.tool for entry in server.logs} == set(server.READ_ONLY_TOOLS)


def test_failed_read_tool_is_logged_and_not_silently_coerced() -> None:
    server = ReadOnlyToolServer()

    with pytest.raises(KeyError):
        server.get_market_state("MISSING")

    assert server.logs[-1].tool == "get_market_state"
    assert server.logs[-1].status == "error"


def test_read_only_portfolio_and_trade_history_return_copies() -> None:
    server = ReadOnlyToolServer(
        portfolio={"cash": 10_000.0, "positions": [{"symbol": "AAPL"}]},
        trade_history=[{"trade_id": "trade-1", "pnl": 25.0}],
    )

    portfolio_snapshot = server.get_portfolio()
    portfolio_snapshot["cash"] = 0.0
    portfolio_snapshot["positions"].append({"symbol": "MSFT"})
    trade_history_snapshot = server.get_trade_history()
    trade_history_snapshot[0]["pnl"] = -999.0
    trade_history_snapshot.append({"trade_id": "trade-2"})

    assert server.get_portfolio() == {
        "cash": 10_000.0,
        "positions": [{"symbol": "AAPL"}],
    }
    assert server.get_trade_history() == [{"trade_id": "trade-1", "pnl": 25.0}]


def test_logs_store_immutable_response_snapshots() -> None:
    server = ReadOnlyToolServer(portfolio={"cash": 10_000.0, "positions": []})

    server.get_portfolio()
    server.portfolio["cash"] = 0.0
    server.portfolio["positions"].append({"symbol": "AAPL"})

    assert server.logs[0].response == {"cash": 10_000.0, "positions": []}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence", 1.4),
        ("thesis", []),
        ("requested_notional", -1.0),
        ("action", "INVALID"),
    ],
)
def test_decision_schema_rejects_each_malformed_field(field: str, value: object) -> None:
    decision = {
        "asset": "AAPL",
        "action": "BUY",
        "confidence": 0.8,
        "thesis": ["positive model signal"],
        "risks": ["volatility"],
        "evidence_used": ["market_state"],
        "requested_notional": 1_000.0,
    }
    decision[field] = value

    with pytest.raises(ValidationError):
        HermesDecision.model_validate(decision)
