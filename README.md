# Hermes Quant

Phases 0 through 7 of the research-first paper-trading platform described in the system design document.

## Scope

This slice contains the repository foundation, fail-closed OpenBB daily OHLCV ingestion, technical features, a walk-forward return model, regime/risk models, deterministic risk controls, read-only Hermes integration, paper execution, and the isolated Hermes-to-experience orchestrator. It does not include a scheduled/live agent loop.

The ingestion contract is:

`information_available_at <= decision_at < outcome_at`

Provider selection is configurable through `OpenBBDataAdapter(provider=...)`; no provider credentials are committed to the repository.

## Local checks

```powershell
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

Start the local TimescaleDB service with `docker compose up -d postgres`.

## Kill switch status

The deterministic risk engine accepts `kill_switch_active` as explicit portfolio state and fails closed when it is true. Phase 7 adds automatic trigger evaluation and crash-durable `TRADING_HALTED` persistence at the orchestrator boundary. Manual reset is required after a halt.

## Phase 7 safety notes

The orchestrator deduplication key is a SHA-256 hash of the validated Hermes decision plus the exact price, volatility, and volume inputs. This is an explicit V1 limitation: a retry with changed market inputs is treated as a new execution context. A future market-session identity can narrow this behavior after the scheduler and market-session model exist.

`unexpected_position_state` is an untested-reconciliation placeholder, not a completed reconciliation system. The eventual implementation must reconcile persisted orders and fills against the expected portfolio state before setting this trigger automatically.

## Technical debt

See the canonical [technical debt register](docs/technical_debt.md). Any deliberate scope-deferral decision must be logged there when it is made, not retroactively. This includes the kill-switch and deduplication decisions made during Phases 4 through 7.
