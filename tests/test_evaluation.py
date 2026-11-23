from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from hermes_quant.models.evaluation import (
    PerformanceMetrics,
    calculate_metrics,
    generate_buy_and_hold_weights,
    generate_model_direction_weights,
    generate_model_top_k_weights,
    generate_momentum_weights,
    generate_moving_average_weights,
    generate_random_weights,
    simulate_random_baseline_monte_carlo,
    simulate_strategy_returns,
)


def test_calculate_metrics_computes_all_section_16_2_fields() -> None:
    # 252 days of +1% daily return
    returns = np.full(252, 0.01)
    metrics = calculate_metrics(returns)

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_return > 10.0  # (1.01)^252 - 1 ≈ 11.27
    assert metrics.annualized_return > 10.0
    assert metrics.win_rate == 1.0
    assert metrics.max_drawdown == 0.0
    assert metrics.profit_factor == float("inf")
    assert metrics.tail_loss_var_95 == pytest.approx(0.01)
    assert metrics.tail_loss_cvar_95 == pytest.approx(0.01)


def test_calculate_metrics_with_drawdown_and_benchmark() -> None:
    r = np.array([0.05, -0.05, 0.02, -0.03, 0.04])
    bench = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
    turnover = np.array([1.0, 0.2, 0.0, 0.5, 0.1])
    exposure = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    metrics = calculate_metrics(
        r,
        benchmark_returns=bench,
        daily_turnover=turnover,
        daily_exposure=exposure,
    )

    assert metrics.max_drawdown < 0.0
    assert metrics.turnover == pytest.approx(float(np.mean(turnover)))
    assert metrics.exposure == pytest.approx(1.0)
    assert metrics.tracking_error > 0.0
    assert "tracking_error" in metrics.to_dict()


def test_simulate_strategy_returns_handles_turnover_and_costs() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    weights = pl.DataFrame(
        {
            "decision_time": [dates[0], dates[0], dates[1]],
            "asset": ["AAPL", "MSFT", "AAPL"],
            "weight": [0.5, 0.5, 1.0],
        }
    )
    returns = pl.DataFrame(
        {
            "decision_time": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "asset": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "next_return": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
        }
    )

    daily, joined = simulate_strategy_returns(weights, returns, transaction_cost_bps=10.0)

    assert len(daily) == 3
    assert daily.get_column("decision_time").to_list() == dates
    # Day 0 turnover is 1.0 (inception)
    assert daily.get_column("turnover")[0] == pytest.approx(1.0)
    # Day 1 turnover: AAPL went 0.5->1.0 (0.5), MSFT went 0.5->0.0 (0.5), total 1.0
    assert daily.get_column("turnover")[1] == pytest.approx(1.0)
    # Day 2 turnover: AAPL went 1.0->0.0 (1.0), total 1.0
    assert daily.get_column("turnover")[2] == pytest.approx(1.0)
    # Cost per day is 10 bps * 1.0 = 0.0010
    assert daily.get_column("costs")[0] == pytest.approx(0.001)
    assert daily.get_column("net_return")[0] == pytest.approx(-0.005 - 0.001)


def test_simulate_strategy_returns_empty_weights() -> None:
    returns = pl.DataFrame(
        {
            "decision_time": [date(2024, 1, 2), date(2024, 1, 3)],
            "asset": ["AAPL", "AAPL"],
            "next_return": [0.01, -0.02],
        }
    )
    empty_weights = pl.DataFrame(
        schema={"decision_time": pl.Date, "asset": pl.Utf8, "weight": pl.Float64}
    )
    daily, joined = simulate_strategy_returns(empty_weights, returns)
    assert (daily.get_column("net_return") == 0.0).all()
    assert (daily.get_column("exposure") == 0.0).all()


def test_baseline_generators() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    df = pl.DataFrame(
        {
            "decision_time": [dates[0], dates[0], dates[1], dates[1]],
            "asset": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "next_return": [0.01, -0.01, 0.02, 0.03],
            "close": [150.0, 300.0, 155.0, 290.0],
            "sma_50d": [140.0, 310.0, 150.0, 300.0],
            "return_20d": [0.05, -0.02, 0.03, 0.04],
            "probability_up": [0.65, 0.40, 0.55, 0.70],
        }
    )

    # 1. Buy and hold: 50% each
    bh = generate_buy_and_hold_weights(df)
    assert len(bh) == 4
    assert bh.get_column("weight").to_list() == [0.5, 0.5, 0.5, 0.5]

    # 2. Momentum: Day 1 only AAPL (0.05 > 0), Day 2 both AAPL & MSFT
    mom = generate_momentum_weights(df)
    day1_mom = mom.filter(pl.col("decision_time") == dates[0])
    day2_mom = mom.filter(pl.col("decision_time") == dates[1])
    assert day1_mom.get_column("asset").to_list() == ["AAPL"]
    assert day1_mom.get_column("weight").to_list() == [1.0]
    assert len(day2_mom) == 2

    # 3. Moving average: Day 1 only AAPL (150 > 140), Day 2 only AAPL (155 > 150)
    ma = generate_moving_average_weights(df)
    assert (ma.get_column("asset") == "AAPL").all()

    # 4. Random baseline
    rnd = generate_random_weights(df, seed=42)
    assert len(rnd) > 0
    # On each day, weights sum to 1.0
    for dt in dates:
        day_rnd = rnd.filter(pl.col("decision_time") == dt)
        assert day_rnd.get_column("weight").sum() == pytest.approx(1.0)

    # 5. Model direction (prob >= 0.5): Day 1 only AAPL (0.65), Day 2 both
    md = generate_model_direction_weights(df, threshold=0.5)
    day1_md = md.filter(pl.col("decision_time") == dates[0])
    assert day1_md.get_column("asset").to_list() == ["AAPL"]
    assert day1_md.get_column("weight").to_list() == [1.0]

    # 6. Model top-k (k=1)
    top1 = generate_model_top_k_weights(df, top_k=1, score_col="probability_up")
    assert len(top1) == 2
    assert top1.filter(pl.col("decision_time") == dates[0]).get_column("asset").to_list() == [
        "AAPL"
    ]
    assert top1.filter(pl.col("decision_time") == dates[1]).get_column("asset").to_list() == [
        "MSFT"
    ]


def test_simulate_random_baseline_monte_carlo() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    universe = pl.DataFrame(
        {
            "decision_time": [dates[0], dates[0], dates[1], dates[1]],
            "asset": ["AAPL", "MSFT", "AAPL", "MSFT"],
        }
    )
    returns = pl.DataFrame(
        {
            "decision_time": [dates[0], dates[0], dates[1], dates[1]],
            "asset": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "next_return": [0.01, -0.01, 0.02, 0.03],
        }
    )
    mc_metrics = simulate_random_baseline_monte_carlo(
        universe, returns, n_simulations=10, base_seed=42
    )
    assert "total_return" in mc_metrics
    assert "total_return_std" in mc_metrics
    assert "sharpe_ratio" in mc_metrics
