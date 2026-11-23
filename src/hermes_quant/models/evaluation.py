from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PerformanceMetrics:
    """Section 16.2 required performance, risk, and attribution metrics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    average_trade_return: float
    turnover: float
    exposure: float
    tail_loss_var_95: float
    tail_loss_cvar_95: float
    excess_annualized_return: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_metrics(
    daily_returns: np.ndarray | pl.Series,
    *,
    benchmark_returns: np.ndarray | pl.Series | None = None,
    daily_turnover: np.ndarray | pl.Series | None = None,
    daily_exposure: np.ndarray | pl.Series | None = None,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> PerformanceMetrics:
    """Compute complete Section 16.2 evaluation metrics for a daily return series."""
    if isinstance(daily_returns, pl.Series):
        r = daily_returns.to_numpy()
    else:
        r = np.asarray(daily_returns, dtype=np.float64)

    t_obs = len(r)
    if t_obs == 0:
        raise ValueError("Cannot calculate metrics on empty return series")

    # 1. Total & Annualized Return
    cum_wealth = np.cumprod(1.0 + r)
    total_return = float(cum_wealth[-1] - 1.0)
    if total_return > -1.0:
        annualized_return = float((1.0 + total_return) ** (annualization_factor / t_obs) - 1.0)
    else:
        annualized_return = -1.0

    # 2. Volatility
    daily_vol = float(np.std(r, ddof=1)) if t_obs > 1 else 0.0
    annualized_volatility = float(daily_vol * sqrt(annualization_factor))

    # 3. Sharpe Ratio
    excess_return = annualized_return - risk_free_rate
    if annualized_volatility > 1e-8:
        sharpe_ratio = float(excess_return / annualized_volatility)
    else:
        sharpe_ratio = 0.0

    # 4. Downside Volatility & Sortino Ratio
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    negative_excess = np.minimum(r - rf_daily, 0.0)
    downside_var = np.mean(negative_excess**2)
    downside_vol = float(sqrt(downside_var) * sqrt(annualization_factor))
    if downside_vol > 1e-8:
        sortino_ratio = float(excess_return / downside_vol)
    else:
        sortino_ratio = 0.0

    # 5. Maximum Drawdown & Calmar Ratio
    running_max = np.maximum.accumulate(cum_wealth)
    drawdowns = (cum_wealth - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))
    if abs(max_drawdown) > 1e-8:
        calmar_ratio = float(annualized_return / abs(max_drawdown))
    else:
        calmar_ratio = 0.0

    # 6. Win Rate & Profit Factor
    active_days = r[np.abs(r) > 1e-8]
    if len(active_days) > 0:
        win_rate = float(np.sum(active_days > 0) / len(active_days))
        average_trade_return = float(np.mean(active_days))
    else:
        win_rate = 0.0
        average_trade_return = 0.0

    gross_gains = float(np.sum(r[r > 0]))
    gross_losses = float(np.abs(np.sum(r[r < 0])))
    if gross_losses > 1e-8:
        profit_factor = float(gross_gains / gross_losses)
    elif gross_gains > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # 7. Turnover & Exposure
    if daily_turnover is not None:
        to_arr = (
            daily_turnover.to_numpy()
            if isinstance(daily_turnover, pl.Series)
            else np.asarray(daily_turnover, dtype=np.float64)
        )
        turnover = float(np.mean(to_arr))
    else:
        turnover = 0.0

    if daily_exposure is not None:
        exp_arr = (
            daily_exposure.to_numpy()
            if isinstance(daily_exposure, pl.Series)
            else np.asarray(daily_exposure, dtype=np.float64)
        )
        exposure = float(np.mean(exp_arr))
    else:
        exposure = float(np.mean(np.abs(r) > 1e-8))

    # 8. Tail Loss (VaR 95% and CVaR / Expected Shortfall 95%)
    var_95 = float(np.percentile(r, 5.0))
    tail_returns = r[r <= var_95]
    cvar_95 = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_95

    # 9. Benchmark-relative Performance
    excess_ann_return = 0.0
    tracking_error = 0.0
    information_ratio = 0.0
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, pl.Series):
            bench_r = benchmark_returns.to_numpy()
        else:
            bench_r = np.asarray(benchmark_returns, dtype=np.float64)

        if len(bench_r) == t_obs:
            bench_cum = np.cumprod(1.0 + bench_r)
            bench_tot = float(bench_cum[-1] - 1.0)
            bench_ann = (
                float((1.0 + bench_tot) ** (annualization_factor / t_obs) - 1.0)
                if bench_tot > -1.0
                else -1.0
            )
            excess_ann_return = annualized_return - bench_ann
            active_diff = r - bench_r
            te_daily = float(np.std(active_diff, ddof=1)) if t_obs > 1 else 0.0
            tracking_error = float(te_daily * sqrt(annualization_factor))
            if tracking_error > 1e-8:
                information_ratio = float(excess_ann_return / tracking_error)

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        profit_factor=profit_factor,
        average_trade_return=average_trade_return,
        turnover=turnover,
        exposure=exposure,
        tail_loss_var_95=var_95,
        tail_loss_cvar_95=cvar_95,
        excess_annualized_return=excess_ann_return,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
    )


def simulate_strategy_returns(
    weights_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    *,
    transaction_cost_bps: float = 5.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Simulate daily portfolio returns with turnover and transaction costs.

    weights_df: columns [decision_time, asset, weight]
    returns_df: columns [decision_time, asset, next_return]
    """
    if len(returns_df) == 0:
        raise ValueError("Cannot simulate returns on empty returns DataFrame")

    unique_dates = returns_df.select("decision_time").unique().sort("decision_time")
    dates_list = unique_dates.get_column("decision_time").to_list()

    if len(weights_df) == 0:
        daily = unique_dates.with_columns(
            pl.lit(0.0).alias("gross_return"),
            pl.lit(0.0).alias("exposure"),
            pl.lit(0.0).alias("turnover"),
            pl.lit(0.0).alias("costs"),
            pl.lit(0.0).alias("net_return"),
        )
        empty_joined = returns_df.with_columns(pl.lit(0.0).alias("weight"))
        return daily, empty_joined

    joined = weights_df.join(
        returns_df,
        on=["decision_time", "asset"],
        how="inner",
    ).sort(["decision_time", "asset"])

    daily_gross = (
        unique_dates.join(
            joined.group_by("decision_time").agg(
                (pl.col("weight") * pl.col("next_return")).sum().alias("gross_return"),
                pl.col("weight").abs().sum().alias("exposure"),
            ),
            on="decision_time",
            how="left",
        )
        .with_columns(
            pl.col("gross_return").fill_null(0.0),
            pl.col("exposure").fill_null(0.0),
        )
        .sort("decision_time")
    )

    asset_pivoted = (
        unique_dates.join(
            weights_df.pivot(index="decision_time", on="asset", values="weight"),
            on="decision_time",
            how="left",
        )
        .fill_null(0.0)
        .sort("decision_time")
    )

    weight_cols = [c for c in asset_pivoted.columns if c != "decision_time"]
    weight_matrix = asset_pivoted.select(weight_cols).to_numpy()
    turnovers = [float(np.sum(np.abs(weight_matrix[0])))]
    for idx in range(1, len(dates_list)):
        diff = np.sum(np.abs(weight_matrix[idx] - weight_matrix[idx - 1]))
        turnovers.append(float(diff))

    turnover_frame = pl.DataFrame(
        {
            "decision_time": dates_list,
            "turnover": turnovers,
        }
    )

    cost_multiplier = transaction_cost_bps / 10_000.0
    daily = (
        daily_gross.join(turnover_frame, on="decision_time", how="left")
        .with_columns(
            (pl.col("turnover") * cost_multiplier).alias("costs"),
        )
        .with_columns(
            (pl.col("gross_return") - pl.col("costs")).alias("net_return"),
        )
        .sort("decision_time")
    )

    return daily, joined


def generate_buy_and_hold_weights(universe_df: pl.DataFrame) -> pl.DataFrame:
    """Equal-weight portfolio of all active universe assets on each decision date."""
    counts = universe_df.group_by("decision_time").agg(pl.len().alias("_count"))
    return (
        universe_df.join(counts, on="decision_time", how="inner")
        .with_columns((1.0 / pl.col("_count")).alias("weight"))
        .select("decision_time", "asset", "weight")
        .sort(["decision_time", "asset"])
    )


def generate_momentum_weights(
    features_df: pl.DataFrame,
    *,
    momentum_col: str = "return_20d",
) -> pl.DataFrame:
    """Long assets with positive momentum (return_20d > 0), equal-weighted."""
    positive = features_df.filter(pl.col(momentum_col) > 0.0)
    if len(positive) == 0:
        return pl.DataFrame(
            schema={"decision_time": pl.Date, "asset": pl.Utf8, "weight": pl.Float64}
        )
    counts = positive.group_by("decision_time").agg(pl.len().alias("_pos_count"))
    return (
        positive.join(counts, on="decision_time", how="inner")
        .with_columns((1.0 / pl.col("_pos_count")).alias("weight"))
        .select("decision_time", "asset", "weight")
        .sort(["decision_time", "asset"])
    )


def generate_moving_average_weights(
    features_df: pl.DataFrame,
    *,
    price_col: str = "close",
    sma_col: str = "sma_50d",
) -> pl.DataFrame:
    """Long assets with close > sma_50d (trend following), equal-weighted."""
    above = features_df.filter(pl.col(price_col) > pl.col(sma_col))
    if len(above) == 0:
        return pl.DataFrame(
            schema={"decision_time": pl.Date, "asset": pl.Utf8, "weight": pl.Float64}
        )
    counts = above.group_by("decision_time").agg(pl.len().alias("_above_count"))
    return (
        above.join(counts, on="decision_time", how="inner")
        .with_columns((1.0 / pl.col("_above_count")).alias("weight"))
        .select("decision_time", "asset", "weight")
        .sort(["decision_time", "asset"])
    )


def generate_random_weights(
    universe_df: pl.DataFrame,
    *,
    seed: int = 42,
    selection_prob: float = 0.5,
) -> pl.DataFrame:
    """Random portfolio baseline selecting a random subset of assets each day."""
    rng = np.random.default_rng(seed)
    unique_dates = (
        universe_df.select("decision_time")
        .unique()
        .sort("decision_time")
        .get_column("decision_time")
        .to_list()
    )
    rows: list[dict[str, Any]] = []
    for dt in unique_dates:
        day_assets = universe_df.filter(pl.col("decision_time") == dt).get_column("asset").to_list()
        n = len(day_assets)
        if n == 0:
            continue
        mask = rng.random(n) < selection_prob
        selected_idx = np.where(mask)[0]
        if len(selected_idx) == 0:
            selected_idx = np.array([rng.integers(0, n)])
        w = 1.0 / len(selected_idx)
        for idx in selected_idx:
            rows.append({"decision_time": dt, "asset": day_assets[idx], "weight": float(w)})

    if not rows:
        return pl.DataFrame(
            schema={"decision_time": pl.Date, "asset": pl.Utf8, "weight": pl.Float64}
        )
    return pl.DataFrame(rows).sort(["decision_time", "asset"])


def generate_model_direction_weights(
    predictions_df: pl.DataFrame,
    *,
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Long assets where model predicted probability_up >= threshold, equal-weighted."""
    selected = predictions_df.filter(pl.col("probability_up") >= threshold)
    if len(selected) == 0:
        return pl.DataFrame(
            schema={"decision_time": pl.Date, "asset": pl.Utf8, "weight": pl.Float64}
        )
    counts = selected.group_by("decision_time").agg(pl.len().alias("_count"))
    return (
        selected.join(counts, on="decision_time", how="inner")
        .with_columns((1.0 / pl.col("_count")).alias("weight"))
        .select("decision_time", "asset", "weight")
        .sort(["decision_time", "asset"])
    )


def generate_model_top_k_weights(
    predictions_df: pl.DataFrame,
    *,
    top_k: int = 5,
    score_col: str = "probability_up",
) -> pl.DataFrame:
    """Long top-k ranked assets by predicted score (e.g. probability_up or expected_return)."""
    ranked = (
        predictions_df.sort(["decision_time", score_col], descending=[False, True])
        .with_columns(
            pl.int_range(0, pl.len()).over("decision_time").alias("rank"),
        )
        .filter(pl.col("rank") < top_k)
        .with_columns(pl.lit(1.0 / top_k).alias("weight"))
        .select("decision_time", "asset", "weight")
        .sort(["decision_time", "asset"])
    )
    return ranked


def simulate_random_baseline_monte_carlo(
    universe_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    *,
    n_simulations: int = 50,
    transaction_cost_bps: float = 5.0,
    benchmark_returns: pl.Series | None = None,
    base_seed: int = 100,
) -> dict[str, float]:
    """Simulate N Monte Carlo random paths and compute average Section 16.2 metrics."""
    sim_metrics: list[PerformanceMetrics] = []
    for i in range(n_simulations):
        r_weights = generate_random_weights(universe_df, seed=base_seed + i)
        daily, _ = simulate_strategy_returns(
            r_weights, returns_df, transaction_cost_bps=transaction_cost_bps
        )
        m = calculate_metrics(
            daily.get_column("net_return"),
            benchmark_returns=benchmark_returns,
            daily_turnover=daily.get_column("turnover"),
            daily_exposure=daily.get_column("exposure"),
        )
        sim_metrics.append(m)

    metric_names = list(sim_metrics[0].to_dict().keys())
    averaged: dict[str, float] = {}
    for name in metric_names:
        values = [getattr(m, name) for m in sim_metrics]
        finite_values = [v for v in values if np.isfinite(v)]
        if finite_values:
            averaged[name] = float(np.mean(finite_values))
            averaged[f"{name}_std"] = (
                float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0
            )
        else:
            averaged[name] = float(values[0]) if values else 0.0
            averaged[f"{name}_std"] = 0.0
    return averaged
