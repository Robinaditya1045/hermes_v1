from __future__ import annotations

import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from hermes_quant.experience import (
    ControlledLearningEngine,
    ControlledSkillCandidate,
    ExperienceAnalyzer,
    OutcomeLabel,
    PostgresExperienceStore,
    SQLiteExperienceStore,
    TradeExperience,
    calculate_realized_trade_outcome,
    label_open_trades_with_market_bars,
)
from hermes_quant.hermes.schemas import HermesDecision
from hermes_quant.orchestration import (
    Orchestrator,
    SQLiteOrchestratorStore,
)
from hermes_quant.risk import PortfolioRiskState, RiskConfig
from hermes_quant.trading import ConstantFillModel, FillModelConfig, PaperBroker

NOW = datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc)


def sample_experience(
    trade_id: str = "trade-001",
    action: str = "BUY",
    confidence: float = 0.8,
    entry_price: float = 100.0,
    quantity: float = 10.0,
    regime: str = "trending",
) -> TradeExperience:
    return TradeExperience(
        trade_id=trade_id,
        decision_id=f"dec-{trade_id}",
        decision_time=NOW,
        asset="AAPL",
        action=action,
        confidence=confidence,
        risk_decision="APPROVED",
        approved_quantity=quantity,
        entry_price=entry_price,
        fees=1.5,
        slippage=0.5,
        regime_at_decision=regime,
        outcome_label="OPEN",
    )


def test_trade_experience_record_creation_and_serialization() -> None:
    exp = sample_experience()
    assert exp.asset == "AAPL"
    assert exp.action == "BUY"
    assert exp.confidence == 0.8
    assert exp.outcome_status == "OPEN"
    assert exp.market_state_version == "market_state.v1"
    assert exp.feature_version == "technical.v1"

    d = exp.to_dict()
    assert d["trade_id"] == "trade-001"
    assert isinstance(d["decision_time"], str)

    reconstructed = TradeExperience.from_dict(d)
    assert reconstructed.trade_id == exp.trade_id
    assert reconstructed.entry_price == exp.entry_price
    assert reconstructed.decision_time == exp.decision_time


def test_outcome_labeler_win_and_loss_classification() -> None:
    # 1. Long position: Entry 100, Exit 110 -> WIN
    long_exp = sample_experience(entry_price=100.0, quantity=10.0)
    win_exp = calculate_realized_trade_outcome(
        long_exp,
        exit_price=110.0,
        exit_fees=1.5,
        exit_slippage=0.5,
        holding_period_bars=3,
        benchmark_return=0.02,
    )
    assert win_exp.outcome_label == OutcomeLabel.WIN.value
    assert win_exp.exit_price == 110.0
    # Gross PnL: (110 - 100) * 10 = 100. Total fees: 1.5 + 1.5 = 3.0. Net PnL = 97.0
    assert win_exp.realized_pnl == pytest.approx(97.0)
    assert win_exp.net_return == pytest.approx(97.0 / 1000.0)
    assert win_exp.holding_period_bars == 3
    assert "OUTPERFORMED" in str(win_exp.post_trade_notes)

    # 2. Long position: Entry 100, Exit 90 -> LOSS
    loss_exp = calculate_realized_trade_outcome(
        long_exp,
        exit_price=90.0,
        exit_fees=1.5,
        exit_slippage=0.5,
        holding_period_bars=1,
    )
    assert loss_exp.outcome_label == OutcomeLabel.LOSS.value
    assert loss_exp.realized_pnl < 0.0

    # 3. Short position: Entry 100, Exit 90 -> WIN for Short
    short_exp = sample_experience(action="SELL", entry_price=100.0, quantity=10.0)
    short_win = calculate_realized_trade_outcome(short_exp, exit_price=90.0, exit_fees=1.0)
    assert short_win.outcome_label == OutcomeLabel.WIN.value
    assert short_win.realized_pnl > 0.0

    # 4. Breakeven test: Exit 100.16 (net return within 5 bps band)
    be_exp = calculate_realized_trade_outcome(
        long_exp, exit_price=100.16, exit_fees=0.0, exit_slippage=0.0
    )

    assert be_exp.outcome_label == OutcomeLabel.BREAKEVEN.value


def test_batch_label_open_trades_with_market_bars() -> None:
    exp = sample_experience(trade_id="t-batch", entry_price=150.0, quantity=10.0)
    bars = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "bar_date": [date(2026, 9, 11), date(2026, 9, 12)],
            "close": [155.0, 160.0],
        }
    )

    labeled = label_open_trades_with_market_bars([exp], bars, horizon_bars=1)
    assert len(labeled) == 1
    assert labeled[0].outcome_label == OutcomeLabel.WIN.value
    assert labeled[0].exit_price == 155.0
    assert labeled[0].holding_period_bars == 1


def test_in_memory_and_sqlite_experience_stores() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = str(Path(tmpdir) / "test_exp.db")
        store = SQLiteExperienceStore(db_file)

        exp1 = sample_experience(trade_id="t-1", confidence=0.7)
        exp2 = sample_experience(trade_id="t-2", confidence=0.9)
        store.save(exp1)
        store.save(exp2)

        assert store.get("t-1") is not None
        assert store.get_by_decision_id("dec-t-1") is not None
        assert len(store.list_experiences()) == 2

        # Test outcome resolution update
        resolved = calculate_realized_trade_outcome(exp1, exit_price=120.0)
        store.save(resolved)
        assert store.get("t-1").outcome_label == "WIN"
        assert store.get("t-1").exit_price == 120.0

        # Reload from SQLite disk
        disk_store = SQLiteExperienceStore(db_file)
        assert disk_store.get("t-1").outcome_label == "WIN"
        assert disk_store.get("t-1").exit_price == 120.0

        # Test Polars export
        df = disk_store.to_polars()
        assert len(df) == 2
        assert "outcome_label" in df.columns
        assert "realized_pnl" in df.columns


def test_postgres_experience_store_integration() -> None:
    try:
        pg_store = PostgresExperienceStore()
    except Exception as e:
        pytest.skip(f"Postgres not accessible: {e}")

    exp = sample_experience(trade_id=f"pg-{NOW.timestamp()}", confidence=0.85)
    pg_store.save(exp)

    retrieved = pg_store.get(exp.trade_id)
    assert retrieved is not None
    assert retrieved.asset == "AAPL"
    assert retrieved.confidence == 0.85

    # Update with realized outcome
    labeled = calculate_realized_trade_outcome(retrieved, exit_price=115.0)
    pg_store.save(labeled)

    updated = pg_store.get(exp.trade_id)
    assert updated.outcome_label == "WIN"
    assert updated.exit_price == 115.0


def test_controlled_learning_and_pattern_detection() -> None:
    analyzer = ExperienceAnalyzer()

    # Create synthetic experiences:
    # 2 high confidence wins, 2 high confidence losses (overconfidence trap)
    # 2 high_volatility losses (regime trap)
    exps: list[TradeExperience] = []

    # High conf wins (0.8 confidence)
    for i in range(2):
        base = sample_experience(trade_id=f"win-{i}", confidence=0.85)
        exps.append(calculate_realized_trade_outcome(base, exit_price=110.0))

    # High conf losses (0.8 confidence)
    for i in range(2):
        base = sample_experience(trade_id=f"loss-conf-{i}", confidence=0.85)
        exps.append(calculate_realized_trade_outcome(base, exit_price=90.0))

    # High volatility regime losses
    for i in range(2):
        base = sample_experience(trade_id=f"loss-vol-{i}", confidence=0.6, regime="high_volatility")
        exps.append(calculate_realized_trade_outcome(base, exit_price=85.0))

    report = analyzer.analyze(exps)
    assert report.total_trades == 6
    assert report.closed_trades == 6
    assert len(report.calibration_bins) > 0
    assert "high_volatility" in report.regime_breakdown
    assert len(report.failure_modes) >= 2

    mode_names = [m.name for m in report.failure_modes]
    assert "overconfident_blindspot" in mode_names
    assert "counter_regime_long_trap" in mode_names

    # Controlled Learning Engine
    engine = ControlledLearningEngine(analyzer)
    candidates = engine.generate_improvement_candidates(exps)
    assert len(candidates) >= 2
    for cand in candidates:
        assert cand.status == "PROPOSED"
        assert len(cand.evidence_trade_ids) > 0


def test_section_17_2_promotion_guard_invariant() -> None:
    # Invariant: Cannot promote a candidate skill without promotion notes
    with pytest.raises(ValueError, match="Violation of Section 17.2"):
        ControlledSkillCandidate(
            candidate_id="cand-fail",
            target_component="agent_skill",
            hypothesis="Test",
            proposed_rule="Test rule",
            evidence_trade_ids=("t1",),
            status="PROMOTED",
            promotion_notes=None,  # Fails guard
        )


def test_orchestrator_submits_and_persists_full_trade_experience() -> None:
    store = SQLiteOrchestratorStore(":memory:")
    broker = PaperBroker(fill_model=ConstantFillModel(FillModelConfig()))
    risk = RiskConfig(
        max_position_notional=10_000.0,
        max_portfolio_exposure=1.0,
        max_daily_loss=1_000.0,
        max_volatility=0.5,
        max_risk_per_trade=0.5,
        max_liquidity_fraction=1.0,
    )
    orch = Orchestrator(broker=broker, risk_config=risk, store=store)

    decision = HermesDecision(
        asset="AAPL",
        action="BUY",
        confidence=0.75,
        thesis=["Technical trend"],
        risks=["Macro vol"],
        evidence_used=["market_state"],
        requested_notional=1_000.0,
    )

    result = orch.submit(
        decision,
        price=100.0,
        estimated_volatility=0.15,
        average_daily_volume=50_000.0,
        portfolio=PortfolioRiskState(10_000.0, 0.0, 0.0, 0.0),
        market_data_at=NOW,
        now=NOW,
    )

    assert result.status == "CLOSED"
    assert len(store.experience) == 1
    exp = store.experience[0]

    # Verify rich Section 17.1 fields recorded from execution
    assert isinstance(exp, TradeExperience)
    assert exp.trade_id == result.decision_id
    assert exp.asset == "AAPL"
    assert exp.action == "BUY"
    assert exp.confidence == 0.75
    assert exp.entry_price is not None
    assert exp.entry_price > 100.0  # Fill slippage & spread added
    assert exp.fees > 0.0
    assert exp.slippage > 0.0
    assert exp.approved_quantity > 0.0
    assert exp.risk_decision == "APPROVED"
