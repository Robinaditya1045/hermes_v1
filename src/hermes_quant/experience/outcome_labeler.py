from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

import polars as pl

from hermes_quant.experience.record import TradeExperience

if TYPE_CHECKING:
    pass


class OutcomeLabel(str, Enum):
    """Deterministic trade outcome categories."""

    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    RISK_REJECTED = "RISK_REJECTED"
    OPEN = "OPEN"
    EXPIRED = "EXPIRED"
    UNREALIZED = "UNREALIZED"


def calculate_realized_trade_outcome(
    experience: TradeExperience,
    *,
    exit_price: float,
    exit_fees: float = 0.0,
    exit_slippage: float = 0.0,
    holding_period_bars: int = 1,
    breakeven_bps: float = 5.0,
    benchmark_return: float | None = None,
    notes: str | None = None,
) -> TradeExperience:
    """Calculate realized PnL, net return, and assign deterministic outcome label.

    Produces a new immutable TradeExperience instance preserving full auditability.
    """
    if experience.entry_price is None or experience.approved_quantity <= 0:
        return replace(
            experience,
            outcome_label=OutcomeLabel.RISK_REJECTED.value
            if experience.risk_decision == "REJECTED"
            else OutcomeLabel.EXPIRED.value,
            post_trade_notes=notes or "No executed entry position to evaluate",
        )

    qty = experience.approved_quantity
    entry_px = experience.entry_price
    direction = 1.0 if experience.action == "BUY" else -1.0

    gross_pnl = direction * (exit_price - entry_px) * qty
    total_fees = experience.fees + exit_fees
    total_slippage = experience.slippage + exit_slippage
    realized_pnl = gross_pnl - total_fees

    capital_committed = entry_px * qty
    net_return = realized_pnl / capital_committed if capital_committed > 0 else 0.0

    threshold = breakeven_bps / 10_000.0
    if net_return > threshold:
        label = OutcomeLabel.WIN.value
    elif net_return < -threshold:
        label = OutcomeLabel.LOSS.value
    else:
        label = OutcomeLabel.BREAKEVEN.value

    attribution_notes = []
    if notes:
        attribution_notes.append(notes)
    if benchmark_return is not None:
        beat = net_return > benchmark_return
        attribution_notes.append(
            f"Benchmark comparison: {'OUTPERFORMED' if beat else 'UNDERPERFORMED'} "
            f"(Trade: {net_return * 100:+.2f}%, Benchmark: {benchmark_return * 100:+.2f}%)"
        )
    attribution_notes.append(
        f"Realized PnL: ${realized_pnl:,.2f} ({net_return * 100:+.2f}%) "
        f"over {holding_period_bars} bar(s)"
    )

    return replace(
        experience,
        exit_price=exit_price,
        fees=total_fees,
        slippage=total_slippage,
        realized_pnl=realized_pnl,
        net_return=net_return,
        holding_period_bars=holding_period_bars,
        outcome_label=label,
        post_trade_notes=" | ".join(attribution_notes),
    )


def label_open_trades_with_market_bars(
    experiences: list[TradeExperience],
    market_bars: pl.DataFrame,
    *,
    horizon_bars: int = 1,
    cost_bps: float = 5.0,
) -> list[TradeExperience]:
    """Batch-label open trade experiences against subsequent verified market bars."""
    sorted_bars = market_bars.sort(["symbol", "bar_date"])
    labeled: list[TradeExperience] = []

    for exp in experiences:
        if exp.outcome_label not in (OutcomeLabel.OPEN.value, "CLOSED"):
            labeled.append(exp)
            continue
        if exp.entry_price is None or exp.approved_quantity <= 0:
            labeled.append(exp)
            continue

        dec_date = (
            exp.decision_time.date() if hasattr(exp.decision_time, "date") else exp.decision_time
        )
        future_bars = sorted_bars.filter(
            (pl.col("symbol") == exp.asset) & (pl.col("bar_date") > dec_date)
        )
        if len(future_bars) < horizon_bars:
            labeled.append(exp)
            continue

        target_bar = future_bars.row(horizon_bars - 1, named=True)
        exit_price = float(target_bar["close"])
        exit_slippage = exit_price * (cost_bps / 10_000.0)
        exit_fees = exp.approved_quantity * 0.005

        labeled_exp = calculate_realized_trade_outcome(
            exp,
            exit_price=exit_price,
            exit_fees=exit_fees,
            exit_slippage=exit_slippage,
            holding_period_bars=horizon_bars,
        )
        labeled.append(labeled_exp)

    return labeled
