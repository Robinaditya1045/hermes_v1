from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    max_position_notional: float
    max_portfolio_exposure: float
    max_daily_loss: float
    max_volatility: float
    max_risk_per_trade: float
    max_liquidity_fraction: float
    horizon_factor: float = 1.0


@dataclass(frozen=True)
class PortfolioRiskState:
    portfolio_value: float
    invested_notional: float
    symbol_notional: float
    daily_pnl: float
    kill_switch_active: bool = False


@dataclass(frozen=True)
class RiskProposal:
    symbol: str
    price: float
    requested_quantity: float
    estimated_volatility: float
    average_daily_volume: float


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    approved_quantity: float
    approved_notional: float
    reason: str


def _is_invalid_number(value: float) -> bool:
    return math.isnan(value) or math.isinf(value)


def evaluate_proposal(
    proposal: RiskProposal,
    portfolio: PortfolioRiskState,
    config: RiskConfig,
) -> RiskDecision:
    """Apply deterministic hard controls; this function has no model or agent inputs."""
    numeric_inputs = (
        proposal.price,
        proposal.requested_quantity,
        proposal.estimated_volatility,
        proposal.average_daily_volume,
        portfolio.portfolio_value,
        portfolio.invested_notional,
        portfolio.symbol_notional,
        portfolio.daily_pnl,
        config.max_position_notional,
        config.max_portfolio_exposure,
        config.max_daily_loss,
        config.max_volatility,
        config.max_risk_per_trade,
        config.max_liquidity_fraction,
        config.horizon_factor,
    )
    if any(_is_invalid_number(value) for value in numeric_inputs):
        return RiskDecision(False, 0.0, 0.0, "invalid_input")
    if portfolio.kill_switch_active:
        return RiskDecision(False, 0.0, 0.0, "kill_switch_active")
    if portfolio.daily_pnl <= -abs(config.max_daily_loss):
        return RiskDecision(False, 0.0, 0.0, "daily_loss_limit")
    if proposal.price <= 0 or proposal.requested_quantity <= 0:
        return RiskDecision(False, 0.0, 0.0, "invalid_order")
    if proposal.estimated_volatility > config.max_volatility:
        return RiskDecision(False, 0.0, 0.0, "volatility_limit")

    remaining_portfolio = max(
        0.0,
        portfolio.portfolio_value * config.max_portfolio_exposure - portfolio.invested_notional,
    )
    remaining_symbol = max(0.0, config.max_position_notional - portfolio.symbol_notional)
    risk_budget = portfolio.portfolio_value * config.max_risk_per_trade
    unit_risk = proposal.price * proposal.estimated_volatility * config.horizon_factor
    raw_quantity = risk_budget / unit_risk if unit_risk > 0 else 0.0
    requested_notional = proposal.price * proposal.requested_quantity
    constraints = {
        "portfolio_exposure": remaining_portfolio,
        "position_limit": remaining_symbol,
        "risk_budget": raw_quantity * proposal.price,
        "liquidity_limit": (
            proposal.average_daily_volume * proposal.price * config.max_liquidity_fraction
        ),
    }
    approved_notional = min(requested_notional, *constraints.values())
    approved_quantity = approved_notional / proposal.price
    if approved_quantity <= 0:
        return RiskDecision(False, 0.0, 0.0, "exposure_limit")
    if approved_quantity >= proposal.requested_quantity:
        reason = "approved"
    else:
        binding_constraint = min(constraints, key=constraints.get)
        reason = f"resized_by_{binding_constraint}"
    return RiskDecision(True, approved_quantity, approved_notional, reason)
