from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from hermes_quant.experience.record import TradeExperience


@dataclass(frozen=True)
class ConfidenceCalibrationBin:
    bin_label: str
    min_confidence: float
    max_confidence: float
    trade_count: int
    win_count: int
    win_rate: float
    average_return: float
    expected_confidence: float


@dataclass(frozen=True)
class FailureModePattern:
    name: str
    description: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    matched_trade_ids: list[str]
    sample_size: int
    average_loss: float
    suggested_guardrail: str


@dataclass(frozen=True)
class ExperienceAnalysisReport:
    total_trades: int
    closed_trades: int
    open_trades: int
    risk_rejected_trades: int
    win_rate: float
    profit_factor: float
    total_realized_pnl: float
    average_trade_return: float
    calibration_bins: list[ConfidenceCalibrationBin]
    regime_breakdown: dict[str, dict[str, float]]
    failure_modes: list[FailureModePattern]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledSkillCandidate:
    """A proposed skill/prompt rule generated from experience analysis.

    Per Section 17.2, must NOT be automatically promoted to production.
    """

    candidate_id: str
    target_component: Literal["hermes_prompt", "agent_skill", "model_filter"]
    hypothesis: str
    proposed_rule: str
    evidence_trade_ids: tuple[str, ...]
    status: Literal["PROPOSED", "VALIDATED", "REJECTED", "PROMOTED"] = "PROPOSED"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promotion_notes: str | None = None

    def __post_init__(self) -> None:
        if self.status == "PROMOTED" and not self.promotion_notes:
            raise ValueError(
                "Violation of Section 17.2: Cannot promote a skill/model candidate "
                "without explicit promotion review notes"
            )


class ExperienceAnalyzer:
    """Analyzes trade experience to detect miscalibration, regime weaknesses, and traps."""

    def analyze(self, experiences: list[TradeExperience]) -> ExperienceAnalysisReport:
        total = len(experiences)
        closed = [e for e in experiences if e.outcome_label in ("WIN", "LOSS", "BREAKEVEN")]
        open_trades = [e for e in experiences if e.outcome_label == "OPEN"]
        rejected = [e for e in experiences if e.outcome_label == "RISK_REJECTED"]

        if not closed:
            return ExperienceAnalysisReport(
                total_trades=total,
                closed_trades=0,
                open_trades=len(open_trades),
                risk_rejected_trades=len(rejected),
                win_rate=0.0,
                profit_factor=0.0,
                total_realized_pnl=0.0,
                average_trade_return=0.0,
                calibration_bins=[],
                regime_breakdown={},
                failure_modes=[],
            )

        returns = [e.net_return for e in closed if e.net_return is not None]
        pnls = [e.realized_pnl for e in closed if e.realized_pnl is not None]
        wins = [e for e in closed if e.outcome_label == "WIN"]
        losses = [e for e in closed if e.outcome_label == "LOSS"]

        win_rate = len(wins) / len(closed) if closed else 0.0
        tot_pnl = sum(pnls)
        avg_ret = float(np.mean(returns)) if returns else 0.0

        gross_gains = sum(e.realized_pnl for e in wins if e.realized_pnl is not None)
        gross_losses = abs(sum(e.realized_pnl for e in losses if e.realized_pnl is not None))
        profit_factor = (
            gross_gains / gross_losses
            if gross_losses > 1e-8
            else (float("inf") if gross_gains > 0 else 0.0)
        )

        # 1. Calibration Bins
        calibration_bins = self._compute_calibration(closed)

        # 2. Regime Breakdown
        regime_breakdown = self._compute_regime_breakdown(closed)

        # 3. Failure Mode Pattern Detection
        failure_modes = self._detect_failure_modes(closed)

        return ExperienceAnalysisReport(
            total_trades=total,
            closed_trades=len(closed),
            open_trades=len(open_trades),
            risk_rejected_trades=len(rejected),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_realized_pnl=tot_pnl,
            average_trade_return=avg_ret,
            calibration_bins=calibration_bins,
            regime_breakdown=regime_breakdown,
            failure_modes=failure_modes,
        )

    def _compute_calibration(self, closed: list[TradeExperience]) -> list[ConfidenceCalibrationBin]:
        bounds = [
            ("Low (0.0 - 0.5)", 0.0, 0.5),
            ("Moderate (0.5 - 0.7)", 0.5, 0.7),
            ("High (0.7 - 0.85)", 0.7, 0.85),
            ("Very High (0.85 - 1.0)", 0.85, 1.0),
        ]
        bins: list[ConfidenceCalibrationBin] = []
        for label, low, high in bounds:
            bin_trades = [
                e
                for e in closed
                if (low <= e.confidence < high) or (high == 1.0 and e.confidence == 1.0)
            ]
            if not bin_trades:
                continue
            b_wins = sum(1 for e in bin_trades if e.outcome_label == "WIN")
            b_rets = [e.net_return for e in bin_trades if e.net_return is not None]
            b_confs = [e.confidence for e in bin_trades]
            bins.append(
                ConfidenceCalibrationBin(
                    bin_label=label,
                    min_confidence=low,
                    max_confidence=high,
                    trade_count=len(bin_trades),
                    win_count=b_wins,
                    win_rate=b_wins / len(bin_trades),
                    average_return=float(np.mean(b_rets)) if b_rets else 0.0,
                    expected_confidence=float(np.mean(b_confs)) if b_confs else (low + high) / 2,
                )
            )
        return bins

    def _compute_regime_breakdown(
        self, closed: list[TradeExperience]
    ) -> dict[str, dict[str, float]]:
        regimes: dict[str, list[TradeExperience]] = {}
        for e in closed:
            r = e.regime_at_decision or "unknown"
            regimes.setdefault(r, []).append(e)

        breakdown: dict[str, dict[str, float]] = {}
        for r, trades in regimes.items():
            wins = sum(1 for t in trades if t.outcome_label == "WIN")
            rets = [t.net_return for t in trades if t.net_return is not None]
            pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
            breakdown[r] = {
                "count": float(len(trades)),
                "win_rate": float(wins / len(trades)),
                "average_return": float(np.mean(rets)) if rets else 0.0,
                "total_pnl": float(sum(pnls)),
            }
        return breakdown

    def _detect_failure_modes(self, closed: list[TradeExperience]) -> list[FailureModePattern]:
        patterns: list[FailureModePattern] = []

        # Trap 1: High-Confidence Blindspot (Confidence >= 0.75 but Loss)
        high_conf_losses = [e for e in closed if e.confidence >= 0.75 and e.outcome_label == "LOSS"]
        if high_conf_losses:
            loss_rets = [e.net_return for e in high_conf_losses if e.net_return is not None]
            patterns.append(
                FailureModePattern(
                    name="overconfident_blindspot",
                    description="Trades executed with >= 0.75 confidence resulted in loss.",
                    severity="HIGH",
                    matched_trade_ids=[e.trade_id for e in high_conf_losses],
                    sample_size=len(high_conf_losses),
                    average_loss=float(np.mean(loss_rets)) if loss_rets else 0.0,
                    suggested_guardrail=(
                        "Require at least 2 distinct counter-evidence checks "
                        "when confidence exceeds 0.75."
                    ),
                )
            )

        # Trap 2: Adverse Volatility Drag (Regime = high_volatility / risk_off losses)
        vol_losses = [
            e
            for e in closed
            if e.regime_at_decision in ("high_volatility", "risk_off")
            and e.action == "BUY"
            and e.outcome_label == "LOSS"
        ]
        if vol_losses:
            vol_rets = [e.net_return for e in vol_losses if e.net_return is not None]
            patterns.append(
                FailureModePattern(
                    name="counter_regime_long_trap",
                    description="Long positions in high_volatility or risk_off regimes failed.",
                    severity="CRITICAL",
                    matched_trade_ids=[e.trade_id for e in vol_losses],
                    sample_size=len(vol_losses),
                    average_loss=float(np.mean(vol_rets)) if vol_rets else 0.0,
                    suggested_guardrail=(
                        "Enforce 50% notional penalty or require explicit trend confirmation "
                        "before buying in high_volatility."
                    ),
                )
            )

        return patterns


class ControlledLearningEngine:
    """Implements Section 17.2 controlled learning loop.

    Generates hypothesis-driven candidates from experience without allowing
    unilateral production modifications.
    """

    def __init__(self, analyzer: ExperienceAnalyzer | None = None) -> None:
        self.analyzer = analyzer or ExperienceAnalyzer()

    def generate_improvement_candidates(
        self, experiences: list[TradeExperience]
    ) -> list[ControlledSkillCandidate]:
        report = self.analyzer.analyze(experiences)
        candidates: list[ControlledSkillCandidate] = []

        for pattern in report.failure_modes:
            if pattern.name == "overconfident_blindspot":
                candidates.append(
                    ControlledSkillCandidate(
                        candidate_id=f"cand-{uuid4().hex[:8]}",
                        target_component="hermes_prompt",
                        hypothesis="High-confidence decisions suffer from single indicator spikes.",
                        proposed_rule=(
                            "PROMPT_RULE: When proposing confidence >= 0.75, the agent must "
                            "explicitly verify that recent price is above both the 20d and 50d "
                            "SMAs and volatility percentile is below 80%."
                        ),
                        evidence_trade_ids=tuple(pattern.matched_trade_ids),
                        status="PROPOSED",
                    )
                )
            elif pattern.name == "counter_regime_long_trap":
                candidates.append(
                    ControlledSkillCandidate(
                        candidate_id=f"cand-{uuid4().hex[:8]}",
                        target_component="agent_skill",
                        hypothesis="Buying dips during risk_off or high-volatility regimes fails.",
                        proposed_rule=(
                            "SKILL_RULE: In risk_off or high_volatility regimes, BUY proposals "
                            "require explicit confirmation that 5-day return is positive, or "
                            "requested_notional must be reduced by at least 50%."
                        ),
                        evidence_trade_ids=tuple(pattern.matched_trade_ids),
                        status="PROPOSED",
                    )
                )

        return candidates
