from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hermes_quant.hermes.schemas import HermesDecision, MarketState, build_market_state
from hermes_quant.hermes.tools import ReadOnlyToolServer

READ_ONLY_TOOLS = list(ReadOnlyToolServer.READ_ONLY_TOOLS)
FORBIDDEN_TOOLS = {
    "submit_paper_order",
    "submit",
    "evaluate_proposal",
    "create_order_from_risk_decision",
}


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    symbol: str = Field(min_length=1)


class ToolObservation(BaseModel):
    tool: str
    arguments: dict[str, Any]
    response: Any = None
    status: str
    timestamp: datetime
    latency_ms: float
    invalid_arguments: bool = False


class HarnessRun(BaseModel):
    scenario: str
    market_state: dict[str, Any]
    available_tools: list[str]
    tool_calls: list[ToolObservation]
    final_decision: dict[str, Any] | None
    decision_validated_first_attempt: bool
    decision_repair_attempts: int
    runtime_available: bool
    token_usage: dict[str, int] | None
    total_latency_ms: float


class HermesRuntime(Protocol):
    runtime_available: bool
    token_usage: dict[str, int] | None

    def decide(self, market_state: MarketState, call_tool) -> Any: ...


@dataclass(frozen=True)
class Scenario:
    name: str
    asset: str
    state: MarketState
    missing_symbol_probe: bool = False


def _state(
    *,
    asset: str,
    probability_up: float,
    expected_return: float,
    realized_volatility: float,
    downside_volatility: float,
    max_drawdown: float,
    regime: str,
    regime_confidence: float,
    return_20d: float,
    volume_z_score: float,
) -> MarketState:
    decision_time = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    return build_market_state(
        asset=asset,
        decision_time=decision_time,
        technical={
            "feature_timestamp": decision_time,
            "feature_set_version": "technical.v1",
            "features": {
                "return_1d": return_20d / 20,
                "return_5d": return_20d / 4,
                "return_20d": return_20d,
                "return_60d": return_20d * 2.2,
                "sma_20d": 100.0,
                "sma_50d": 98.0,
                "sma_200d": 91.0,
                "ema_20d": 101.0,
                "ema_50d": 98.0,
                "ema_200d": 92.0,
                "atr_14d": realized_volatility * 100 / 16,
                "realized_volatility_20d": realized_volatility,
                "volatility_percentile_252d": min(1.0, realized_volatility / 0.6),
                "volume_z_score_20d": volume_z_score,
            },
        },
        market_prediction={
            "probability_up": probability_up,
            "expected_return": expected_return,
            "model_version": "return_lgbm_technical_v1",
        },
        volatility={
            "realized_20d": realized_volatility,
            "downside_volatility": downside_volatility,
            "max_drawdown": max_drawdown,
            "risk_model_version": "risk_state_technical_v1",
        },
        market_regime={
            "state": regime,
            "confidence": regime_confidence,
            "model_version": "regime_rules_technical_v1",
        },
        fundamentals={"available": False, "metrics": {}},
        news={"available": False, "sentiment": None, "impact": "unknown", "items": 0},
    )


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "clearly_bullish",
            "AAPL",
            _state(
                asset="AAPL",
                probability_up=0.82,
                expected_return=0.018,
                realized_volatility=0.22,
                downside_volatility=0.14,
                max_drawdown=-0.05,
                regime="risk_on",
                regime_confidence=0.91,
                return_20d=0.14,
                volume_z_score=1.2,
            ),
        ),
        Scenario(
            "bearish_high_risk",
            "NVDA",
            _state(
                asset="NVDA",
                probability_up=0.24,
                expected_return=-0.021,
                realized_volatility=0.58,
                downside_volatility=0.49,
                max_drawdown=-0.24,
                regime="high_volatility",
                regime_confidence=0.94,
                return_20d=-0.18,
                volume_z_score=2.8,
            ),
        ),
        Scenario(
            "ambiguous_conflicting_signals",
            "MSFT",
            _state(
                asset="MSFT",
                probability_up=0.51,
                expected_return=0.001,
                realized_volatility=0.27,
                downside_volatility=0.25,
                max_drawdown=-0.09,
                regime="sideways",
                regime_confidence=0.46,
                return_20d=0.01,
                volume_z_score=-0.2,
            ),
        ),
        Scenario(
            "missing_symbol_tool_error",
            "GOOGL",
            _state(
                asset="GOOGL",
                probability_up=0.59,
                expected_return=0.006,
                realized_volatility=0.25,
                downside_volatility=0.18,
                max_drawdown=-0.07,
                regime="trending",
                regime_confidence=0.68,
                return_20d=0.04,
                volume_z_score=0.5,
            ),
            missing_symbol_probe=True,
        ),
    )


class ScriptedHermesRuntime:
    """Deterministic manual observer used because no Hermes runtime is installed locally."""

    runtime_available = False
    token_usage = None

    def decide(self, market_state: MarketState, call_tool) -> dict[str, Any]:
        symbol = market_state.asset
        call_tool("get_market_state", {"symbol": symbol})
        call_tool("get_quant_prediction", {"symbol": symbol})
        call_tool("get_market_regime", {"symbol": symbol})
        call_tool("get_risk_assessment", {"symbol": symbol})
        call_tool("get_fundamentals", {"symbol": symbol})
        call_tool("get_recent_news", {"symbol": symbol})
        call_tool("get_portfolio", {})
        call_tool("get_trade_history", {})
        action = "BUY" if market_state.market_prediction.probability_up > 0.6 else "HOLD"
        return {
            "asset": symbol,
            "action": action,
            "confidence": market_state.market_prediction.probability_up,
            "thesis": [
                f"Model probability_up={market_state.market_prediction.probability_up:.2f}",
                f"Regime={market_state.market_regime.state}",
            ],
            "risks": [
                f"Realized volatility={market_state.volatility.realized_20d:.2f}",
                "Fundamentals and news tools reported unavailable",
            ],
            "evidence_used": [
                "market_state",
                "quant_prediction",
                "market_regime",
                "risk_assessment",
            ],
            "requested_notional": 1_000.0 if action == "BUY" else 0.0,
        }


class ObservationHarness:
    def __init__(self, runtime: HermesRuntime | None = None) -> None:
        self.runtime = runtime or ScriptedHermesRuntime()
        self._assert_surface()

    @staticmethod
    def _assert_surface() -> None:
        assert set(READ_ONLY_TOOLS).isdisjoint(FORBIDDEN_TOOLS)
        assert not hasattr(ReadOnlyToolServer, "submit_paper_order")
        assert not hasattr(ReadOnlyToolServer, "submit")
        assert not hasattr(ReadOnlyToolServer, "evaluate_proposal")
        assert not hasattr(ReadOnlyToolServer, "create_order_from_risk_decision")

    def run(self, scenario: Scenario) -> HarnessRun:
        server = ReadOnlyToolServer(
            market_states={scenario.asset: scenario.state},
            quant_predictions={scenario.asset: scenario.state.market_prediction},
            market_regimes={scenario.asset: scenario.state.market_regime},
            risk_assessments={scenario.asset: scenario.state.volatility},
            portfolio={"cash": 10_000.0, "positions": []},
            trade_history=[],
        )
        observations: list[ToolObservation] = []

        def call_tool(name: str, arguments: dict[str, Any]) -> Any:
            started = time.perf_counter()
            timestamp = datetime.now(timezone.utc)
            invalid = False
            try:
                if name not in READ_ONLY_TOOLS:
                    raise ValueError(f"tool is not available: {name}")
                if name in {"get_portfolio", "get_trade_history"}:
                    if arguments:
                        raise ValueError("tool takes no arguments")
                    response = getattr(server, name)()
                else:
                    parsed = ToolArguments.model_validate(arguments)
                    response = getattr(server, name)(parsed.symbol)
                response_json = (
                    response.model_dump(mode="json")
                    if isinstance(response, BaseModel)
                    else response
                )
                observations.append(
                    ToolObservation(
                        tool=name,
                        arguments=arguments,
                        response=response_json,
                        status="success",
                        timestamp=timestamp,
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                )
                return response
            except (ValidationError, ValueError, KeyError) as exc:
                invalid = isinstance(exc, ValidationError) or name not in READ_ONLY_TOOLS
                observations.append(
                    ToolObservation(
                        tool=name,
                        arguments=arguments,
                        response={"error": str(exc)},
                        status="error",
                        timestamp=timestamp,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        invalid_arguments=invalid,
                    )
                )
                raise

        started = time.perf_counter()
        raw_decision = None
        first_attempt = True
        repair_attempts = 0
        try:
            if scenario.missing_symbol_probe:
                try:
                    call_tool("get_market_state", {"symbol": "UNKNOWN"})
                except KeyError:
                    pass
            raw_decision = self.runtime.decide(scenario.state, call_tool)
            try:
                validated = HermesDecision.model_validate(raw_decision)
            except ValidationError:
                first_attempt = False
                repair_attempts = 1
                raise
        except ValidationError:
            validated = None
        return HarnessRun(
            scenario=scenario.name,
            market_state=scenario.state.model_dump(mode="json"),
            available_tools=READ_ONLY_TOOLS,
            tool_calls=observations,
            final_decision=validated.model_dump(mode="json") if validated else raw_decision,
            decision_validated_first_attempt=first_attempt and validated is not None,
            decision_repair_attempts=repair_attempts,
            runtime_available=self.runtime.runtime_available,
            token_usage=self.runtime.token_usage,
            total_latency_ms=(time.perf_counter() - started) * 1000,
        )


def write_observations(path: Path, runs: tuple[HarnessRun, ...] | list[HarnessRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for run in runs:
            stream.write(run.model_dump_json() + "\n")
