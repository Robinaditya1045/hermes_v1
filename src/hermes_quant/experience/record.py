from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TradeExperience:
    """Immutable Section 17.1 Trade Experience Record."""

    trade_id: str
    decision_id: str
    decision_time: datetime
    asset: str
    market_state_version: str = "market_state.v1"
    feature_version: str = "technical.v1"
    model_versions: dict[str, str] = field(default_factory=dict)
    hermes_model: str = "llama3.2:1b"
    hermes_prompt_version: str = "hermes_v1"
    action: str = "BUY"
    confidence: float = 0.5
    risk_decision: str = "APPROVED"
    approved_quantity: float = 0.0
    entry_price: float | None = None
    exit_price: float | None = None
    fees: float = 0.0
    slippage: float = 0.0
    realized_pnl: float | None = None
    net_return: float | None = None
    holding_period_bars: int | None = None
    outcome_label: str = "OPEN"
    regime_at_decision: str | None = None
    post_trade_notes: str | None = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def outcome_status(self) -> str:
        """Backward-compatibility alias for Phase 7 Orchestrator ExperienceRecord."""
        return self.outcome_label

    def to_dict(self) -> dict[str, Any]:
        """Convert experience record to a JSON-serializable dictionary."""
        d = asdict(self)
        d["decision_time"] = self.decision_time.isoformat()
        d["recorded_at"] = self.recorded_at.isoformat()
        d["model_versions"] = json.dumps(self.model_versions, sort_keys=True)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradeExperience:
        """Reconstruct experience record from a dictionary."""
        d = dict(data)
        if isinstance(d.get("decision_time"), str):
            d["decision_time"] = datetime.fromisoformat(d["decision_time"])
        if isinstance(d.get("recorded_at"), str):
            d["recorded_at"] = datetime.fromisoformat(d["recorded_at"])
        if isinstance(d.get("model_versions"), str):
            try:
                d["model_versions"] = json.loads(d["model_versions"])
            except Exception:
                d["model_versions"] = {}
        return cls(**d)
